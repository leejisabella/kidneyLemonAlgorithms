#!/usr/bin/env python3
"""
Patient-level fairness analysis using common random numbers.

Generates one shared "world" (patient arrivals with thresholds, kidney
arrivals with qualities). Runs each mechanism on this fixed world, with
shared acceptance-decision RNG when possible. Records per-patient outcomes.

Then asks:
  1. Distribution of allocation depth (queue position at allocation)
  2. Pareto comparison: which patients are better/worse off under each mechanism
     compared to each other?
"""

import os
import sys
sys.path.insert(0, '/home/claude')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from collections import deque
from threshold_sim import (
    sigmoid, p_of_theta, declare_bucket, bucket_of_theta,
    THRESHOLD_BETA_ALPHA, THRESHOLD_BETA_BETA,
    MAX_HOURS, RHO, SEED, Patient,
)

OUTPUT_DIR = '/mnt/user-data/outputs'

# CRN simulation parameters
N_KIDNEYS = 1500
N_WARMUP = 300
INITIAL_QUEUE = 150
N_REPS = 8


@dataclass
class WorldState:
    """Pre-generated world: patient arrivals, thresholds, kidney qualities, and
    a pre-generated random number stream for acceptance decisions."""
    initial_thresholds: np.ndarray         # shape (INITIAL_QUEUE,)
    kidney_qualities: np.ndarray            # shape (N_KIDNEYS,)
    arrival_counts: np.ndarray              # shape (N_KIDNEYS,) - how many patients arrive after each kidney
    arrival_thresholds: list                # list of arrays, one per kidney
    # For each (kidney_idx, queue_position), a pre-drawn uniform [0,1] random number.
    # Used to determine accept/reject given P(accept).
    # We store it as a 2D array indexed by (kidney_idx, offer_index_within_kidney).
    accept_draws: np.ndarray                # shape (N_KIDNEYS, MAX_HOURS * MAX_BATCH)


def generate_world(seed):
    """Generate one CRN world."""
    rng = np.random.default_rng(seed)
    initial_thresholds = rng.beta(THRESHOLD_BETA_ALPHA, THRESHOLD_BETA_BETA, INITIAL_QUEUE)
    kidney_qualities = rng.random(N_KIDNEYS)

    # Determine patient arrivals. Use integer arrivals with fractional accumulator
    # for consistency across mechanisms.
    arrival_counts = np.zeros(N_KIDNEYS, dtype=int)
    fractional = 0.0
    for k in range(N_KIDNEYS):
        fractional += RHO
        arrival_counts[k] = int(fractional)
        fractional -= arrival_counts[k]
    total_arrivals = arrival_counts.sum()

    # Pre-draw all arrival thresholds
    arrival_thresholds_flat = rng.beta(THRESHOLD_BETA_ALPHA, THRESHOLD_BETA_BETA, total_arrivals)
    arrival_thresholds = []
    idx = 0
    for k in range(N_KIDNEYS):
        n = arrival_counts[k]
        arrival_thresholds.append(arrival_thresholds_flat[idx:idx+n])
        idx += n

    # Pre-draw acceptance decisions
    # Maximum offers per kidney: max(MAX_HOURS * batch_size). With batch=5 and MAX_HOURS=12, that's 60.
    # For threshold mechanisms with sequential offers, also up to MAX_HOURS = 12.
    # Use 60 to be safe (covers all mechanisms).
    max_offers_per_kidney = 60
    accept_draws = rng.random((N_KIDNEYS, max_offers_per_kidney))

    return WorldState(
        initial_thresholds=initial_thresholds,
        kidney_qualities=kidney_qualities,
        arrival_counts=arrival_counts,
        arrival_thresholds=arrival_thresholds,
        accept_draws=accept_draws,
    )


def run_mechanism_on_world(world, mechanism, delta, K_cutoffs=None, batch_m=5):
    """Run a mechanism on a CRN world. Returns per-patient outcomes and per-kidney metrics."""
    queue = deque()
    next_pid = 0

    # Initialize queue
    for thr in world.initial_thresholds:
        p = Patient(pid=next_pid, threshold=float(thr), arrival_kidney=-1,
                    initial_priority=next_pid + 1)
        if mechanism in ('threshold', 'threshold_batched'):
            p.declared_bucket = declare_bucket(p.threshold, K_cutoffs)
        queue.append(p)
        next_pid += 1

    kidney_discarded = np.zeros(N_KIDNEYS, dtype=bool)
    kidney_placement = np.full(N_KIDNEYS, -1, dtype=np.int32)
    all_patients = list(queue)

    for k_idx in range(N_KIDNEYS):
        theta = float(world.kidney_qualities[k_idx])
        # accept_draws[k_idx, offer_index] is the uniform draw for the offer_index-th offer
        # of this kidney
        draws = world.accept_draws[k_idx]
        offer_count = 0  # index into draws

        allocated = False
        winner_pos = -1

        if len(queue) > 0:
            if mechanism == 'omniscient':
                for i, pat in enumerate(queue):
                    if pat.threshold <= theta:
                        winner_pos = i
                        allocated = True
                        break

            elif mechanism in ('sequential', 'batching'):
                bs = 1 if mechanism == 'sequential' else batch_m
                alpha_base = np.log(0.9 / 0.1)
                beta_pos = 0.10
                for hour in range(MAX_HOURS):
                    start = hour * bs
                    if start >= len(queue):
                        break
                    end = min(start + bs, len(queue))
                    k_declines = hour * bs
                    accepts = []
                    for pos in range(start, end):
                        pat = queue[pos]
                        pos1 = pos + 1
                        # Consume one random draw per offer attempt (whether the patient
                        # is threshold-eligible or not, to keep CRN aligned across mechanisms)
                        u = draws[offer_count]
                        offer_count += 1
                        if pat.threshold > theta:
                            continue   # auto-decline
                        p_acc = sigmoid(alpha_base + beta_pos * (pos1 - 1) - delta * k_declines)
                        if u < p_acc:
                            accepts.append(pos)
                    if accepts:
                        winner_pos = min(accepts)
                        allocated = True
                        break

            elif mechanism == 'threshold':
                target_bucket = bucket_of_theta(theta, K_cutoffs)
                alpha_base = np.log(0.9 / 0.1)
                beta_pos = 0.10
                k_declines = 0
                offers_made = 0
                i = 0
                while i < len(queue) and offers_made < MAX_HOURS:
                    pat = queue[i]
                    if K_cutoffs is None:
                        eligible = pat.threshold <= theta
                    else:
                        eligible = pat.declared_bucket <= target_bucket
                    if not eligible:
                        i += 1
                        continue
                    offers_made += 1
                    u = draws[offer_count]
                    offer_count += 1
                    pos1 = i + 1
                    if pat.threshold > theta:
                        k_declines += 1
                        i += 1
                        continue
                    p_acc = sigmoid(alpha_base + beta_pos * (pos1 - 1) - delta * k_declines)
                    if u < p_acc:
                        winner_pos = i
                        allocated = True
                        break
                    else:
                        k_declines += 1
                        i += 1

        if allocated:
            kidney_placement[k_idx] = winner_pos + 1
            queue[winner_pos].received = True
            queue[winner_pos].received_at = k_idx
            del queue[winner_pos]
        else:
            kidney_discarded[k_idx] = True

        # Patient arrivals
        for thr in world.arrival_thresholds[k_idx]:
            p = Patient(pid=next_pid, threshold=float(thr), arrival_kidney=k_idx,
                        initial_priority=len(queue) + 1)
            if mechanism in ('threshold', 'threshold_batched'):
                p.declared_bucket = declare_bucket(p.threshold, K_cutoffs)
            queue.append(p)
            all_patients.append(p)
            next_pid += 1

    return {
        'kidney_discarded': kidney_discarded,
        'kidney_placement': kidney_placement,
        'patients': all_patients,
    }


def per_patient_outcomes(result):
    """Extract per-patient outcomes: pid -> (received_bool, received_at_kidney)."""
    return {p.pid: (p.received, p.received_at, p.arrival_kidney, p.threshold,
                     p.initial_priority)
            for p in result['patients']}


def run_crn_comparison():
    mechanisms = [
        ('sequential', None, 'Sequential'),
        ('batching', None, 'Batching m=5'),
        ('threshold', None, 'Threshold continuous (K=∞)'),
        ('threshold', [0.15], 'Threshold K=2 (OPTN)'),
    ]
    deltas = [0.0, 0.3, 1.0]

    all_position_records = []   # per-allocation depth
    all_patient_records = []    # per-patient outcomes

    for rep in range(N_REPS):
        print(f'rep {rep+1}/{N_REPS}')
        world_seed = SEED + rep * 1000
        world = generate_world(world_seed)

        for delta in deltas:
            for mech, kcuts, label in mechanisms:
                result = run_mechanism_on_world(world, mech, delta, K_cutoffs=kcuts)

                # Position distribution
                placements = result['kidney_placement'][N_WARMUP:]
                qualities = world.kidney_qualities[N_WARMUP:]
                for k_off, (pos, q) in enumerate(zip(placements, qualities)):
                    if pos > 0:  # allocated
                        all_position_records.append({
                            'rep': rep, 'mechanism': label, 'delta': delta,
                            'kidney_idx': N_WARMUP + k_off,
                            'kidney_quality': float(q),
                            'placement': int(pos),
                        })

                # Per-patient outcomes
                outcomes = per_patient_outcomes(result)
                for pid, (received, recv_at, arr_k, thr, init_prio) in outcomes.items():
                    if arr_k < N_WARMUP or arr_k >= N_KIDNEYS - 100:
                        continue   # skip warmup and late arrivals
                    all_patient_records.append({
                        'rep': rep, 'mechanism': label, 'delta': delta,
                        'pid': pid, 'arrival_kidney': arr_k, 'threshold': thr,
                        'initial_priority': init_prio,
                        'received': received,
                        'received_at': recv_at if received else -1,
                        'wait': (recv_at - arr_k) if received else -1,
                    })

    pos_df = pd.DataFrame(all_position_records)
    pat_df = pd.DataFrame(all_patient_records)
    pos_df.to_csv(os.path.join(OUTPUT_DIR, 'crn_positions.csv'), index=False)
    pat_df.to_csv(os.path.join(OUTPUT_DIR, 'crn_patients.csv'), index=False)
    print(f'wrote crn_positions.csv ({len(pos_df)} rows), crn_patients.csv ({len(pat_df)} rows)')

    return pos_df, pat_df


def pareto_analysis(pat_df):
    """For each (rep, delta), pivot patient outcomes across mechanisms.
    Count Pareto violations: patients transplanted under A but not under B."""
    deltas = sorted(pat_df['delta'].unique())
    mechanisms = sorted(pat_df['mechanism'].unique())

    print('\n=== Pareto comparison: patients transplanted under A but NOT under B ===')
    print('(over all reps; patient identified by (rep, pid))')

    results = []
    for delta in deltas:
        sub = pat_df[pat_df['delta'] == delta]
        # Pivot: rows are (rep, pid), columns are mechanism, value is received bool
        pivot = sub.pivot_table(index=['rep', 'pid'], columns='mechanism',
                                  values='received', observed=True, aggfunc='first')
        for a in mechanisms:
            for b in mechanisms:
                if a == b:
                    continue
                # Patients received under a but not under b
                only_a = pivot[(pivot[a] == True) & (pivot[b] == False)]
                only_b = pivot[(pivot[a] == False) & (pivot[b] == True)]
                both = pivot[(pivot[a] == True) & (pivot[b] == True)]
                neither = pivot[(pivot[a] == False) & (pivot[b] == False)]
                results.append({
                    'delta': delta,
                    'A': a, 'B': b,
                    'received_A_not_B': len(only_a),
                    'received_B_not_A': len(only_b),
                    'both': len(both),
                    'neither': len(neither),
                })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'crn_pareto.csv'), index=False)
    return df


def position_summary(pos_df):
    """Summary statistics on allocation depth by mechanism × delta."""
    summary = pos_df.groupby(['mechanism', 'delta'], observed=True).agg(
        n=('placement', 'count'),
        mean_placement=('placement', 'mean'),
        median_placement=('placement', 'median'),
        p25=('placement', lambda s: s.quantile(0.25)),
        p75=('placement', lambda s: s.quantile(0.75)),
        p90=('placement', lambda s: s.quantile(0.90)),
        max_placement=('placement', 'max'),
    ).reset_index()
    summary.to_csv(os.path.join(OUTPUT_DIR, 'crn_position_summary.csv'), index=False)
    return summary


def main():
    print('Running CRN comparison')
    pos_df, pat_df = run_crn_comparison()

    pos_summary = position_summary(pos_df)
    print('\n=== Position summary ===')
    print(pos_summary.round(1).to_string(index=False))

    pareto_df = pareto_analysis(pat_df)
    print('\n=== Pareto pairs (most informative comparisons) ===')
    interesting = pareto_df[
        ((pareto_df['A'] == 'Batching m=5') & (pareto_df['B'].str.startswith('Threshold'))) |
        ((pareto_df['B'] == 'Batching m=5') & (pareto_df['A'].str.startswith('Threshold')))
    ]
    print(interesting.to_string(index=False))


if __name__ == '__main__':
    main()
