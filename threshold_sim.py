#!/usr/bin/env python3
"""
Threshold mechanism simulation for kidney allocation
====================================================

Compares mechanisms for allocating deceased-donor kidneys in a multi-kidney
open-queue setting with observational learning (lemons) effects.

Mechanisms:
    omniscient    : first-best benchmark. Kidney goes to the highest-priority
                    patient whose true threshold accepts the kidney quality.
    sequential    : single-offer-per-hour FCFS down the priority list,
                    with lemons cascade.
    batching      : m=5 simultaneous offers per hour, lemons signal frozen
                    within batch.
    threshold_K   : patients pre-declare which of K quality buckets they
                    accept. Kidney goes to highest-priority patient who
                    pre-declared they would accept this quality.

The threshold mechanism with K=infinity is the Su-Zenios continuous optimum.
K=2 (OPTN-style) uses a fixed cutoff at theta=0.15 corresponding to KDPI 85.
K=2_opt finds the optimal binary cutoff.

Parameters calibrated to OPTN data:
  - theta ~ Uniform[0,1], where theta = 1 - KDPI/100
  - p(theta) calibrated to discard rates by KDPI tier (Mohan et al. 2018)
  - patient threshold ~ Beta(2,5) calibrated so ~30% accept a marginal kidney
  - rho = 1.6 patients per kidney (real US ratio)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation parameters
N_KIDNEYS = 1500          # total kidneys per run
N_WARMUP = 300            # discard first N for steady-state stats
INITIAL_QUEUE = 150
RHO = 1.6                 # patients per kidney
MAX_HOURS = 12
SEED = 42
N_REPS = 15

# Beta(2,5) gives mean=2/7 ~ 0.29, matching ~30% acceptance of KDPI-90 kidneys
THRESHOLD_BETA_ALPHA = 2.0
THRESHOLD_BETA_BETA = 5.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def acceptance_prob_logistic(p, beta, delta, position, k_declines):
    """Position+lemons logistic from Part 1 model. P(accept | position, prior declines)."""
    p_clipped = np.clip(p, 1e-9, 1 - 1e-9)
    alpha = np.log(p_clipped / (1 - p_clipped))
    return sigmoid(alpha + beta * (position - 1) - delta * k_declines)


# OPTN-anchored p(theta) calibration
# Three anchor points (theta, target_p):
#   theta=0.15 (KDPI 85, marginal):  p ~ 0.05
#   theta=0.50 (KDPI 50, average):   p ~ 0.30
#   theta=0.85 (KDPI 15, premium):   p ~ 0.70
# Logistic fit: p(theta) = sigmoid(a + b*theta)
# Solving for these anchors: a ~ -3.4, b ~ 5.7
P_THETA_A = -3.4
P_THETA_B = 5.7


def p_of_theta(theta):
    """Baseline per-offer acceptance probability as a function of kidney quality."""
    return sigmoid(P_THETA_A + P_THETA_B * theta)


# ---------- Patient and Kidney structures ----------

@dataclass
class Patient:
    pid: int
    threshold: float           # true willingness threshold (private)
    arrival_kidney: int        # which kidney index they arrived at
    initial_priority: int      # queue position at arrival
    declared_bucket: Optional[int] = None  # for threshold mechanisms, which K-bucket they opted into
    received: bool = False
    received_at: int = -1      # kidney index when received


# ---------- Mechanism implementations ----------

def simulate_run(mechanism, rng, n_kidneys=N_KIDNEYS, initial_queue=INITIAL_QUEUE,
                  rho=RHO, max_hours=MAX_HOURS, p_baseline=0.10, beta_pos=0.10,
                  delta_lemons=0.0, batch_m=5, K_bins=None, K_cutoffs=None):
    """
    Run one full simulation.

    mechanism in {'omniscient', 'sequential', 'batching', 'threshold'}
    For threshold: K_bins is number of buckets, K_cutoffs is the breakpoints.
        K=2 OPTN: K_cutoffs = [0.15]
        K=2 opt:  K_cutoffs = [optimal cutoff, found by sweep]
        K=k uniform: K_cutoffs = [k/K_bins for k in 1..K_bins-1]
        K=infinity:  K_cutoffs = None (use continuous threshold)
    """
    # Initialize queue
    queue = deque()
    next_pid = 0
    for _ in range(initial_queue):
        thr = float(rng.beta(THRESHOLD_BETA_ALPHA, THRESHOLD_BETA_BETA))
        p = Patient(pid=next_pid, threshold=thr, arrival_kidney=-1,
                    initial_priority=next_pid+1)
        if mechanism in ('threshold', 'threshold_batched'):
            p.declared_bucket = declare_bucket(thr, K_cutoffs)
        queue.append(p)
        next_pid += 1

    kidney_discarded = np.zeros(n_kidneys, dtype=bool)
    kidney_placement = np.full(n_kidneys, -1, dtype=np.int32)
    kidney_quality = np.zeros(n_kidneys)

    fractional_arrivals = 0.0
    all_patients = list(queue)  # track for end-of-run stats

    for k_idx in range(n_kidneys):
        theta = float(rng.random())   # uniform [0,1]
        kidney_quality[k_idx] = theta

        allocated = False
        if len(queue) > 0:
            if mechanism == 'omniscient':
                # First-best: find highest-priority patient with threshold <= theta
                for i, pat in enumerate(queue):
                    if pat.threshold <= theta:
                        kidney_placement[k_idx] = i + 1
                        pat.received = True
                        pat.received_at = k_idx
                        del queue[i]
                        allocated = True
                        break

            elif mechanism == 'sequential':
                # m=1, walk down queue with lemons cascade
                allocated, pos = run_offer_round(queue, theta, p_baseline, beta_pos,
                                                  delta_lemons, batch_size=1,
                                                  max_hours=max_hours, rng=rng)
                if allocated:
                    kidney_placement[k_idx] = pos + 1
                    queue[pos].received = True
                    queue[pos].received_at = k_idx
                    del queue[pos]

            elif mechanism == 'batching':
                # m=batch_m, lemons frozen within batch
                allocated, pos = run_offer_round(queue, theta, p_baseline, beta_pos,
                                                  delta_lemons, batch_size=batch_m,
                                                  max_hours=max_hours, rng=rng)
                if allocated:
                    kidney_placement[k_idx] = pos + 1
                    queue[pos].received = True
                    queue[pos].received_at = k_idx
                    del queue[pos]

            elif mechanism == 'threshold_batched':
                # Hybrid: K-bucket self-selection PLUS m=5 batching within eligible patients.
                # Unified acceptance: threshold gate + position+lemons logistic.
                target_bucket = bucket_of_theta(theta, K_cutoffs)
                alpha_base = np.log(0.9 / 0.1)
                beta_pos_threshold = 0.10
                eligible_indices = [i for i, pat in enumerate(queue)
                                     if (K_cutoffs is None) or (pat.declared_bucket <= target_bucket)]
                winner_pos = -1
                for hour in range(max_hours):
                    start = hour * batch_m
                    end = min(start + batch_m, len(eligible_indices))
                    if start >= len(eligible_indices):
                        break
                    k_declines = hour * batch_m
                    batch_accepts = []
                    for ei in range(start, end):
                        pat_idx = eligible_indices[ei]
                        pat = queue[pat_idx]
                        if pat.threshold > theta:
                            continue   # would decline; not an accepter (but doesn't update k within batch)
                        pos_1indexed = pat_idx + 1
                        p_accept = sigmoid(alpha_base
                                            + beta_pos_threshold * (pos_1indexed - 1)
                                            - delta_lemons * k_declines)
                        if rng.random() < p_accept:
                            batch_accepts.append(pat_idx)
                    if batch_accepts:
                        winner_pos = min(batch_accepts)
                        break
                if winner_pos >= 0:
                    kidney_placement[k_idx] = winner_pos + 1
                    queue[winner_pos].received = True
                    queue[winner_pos].received_at = k_idx
                    del queue[winner_pos]
                    allocated = True

            elif mechanism == 'threshold':
                # Su-Zenios-style: mechanism knows declared buckets and offers ONLY to eligible
                # patients. Non-eligible patients are skipped without cost (we know in advance
                # they won't accept based on their declaration). The 12-hour budget applies to
                # the number of offers made to eligible patients.
                # Unified acceptance: threshold gate + position+lemons logistic.
                target_bucket = bucket_of_theta(theta, K_cutoffs)
                alpha_base = np.log(0.9 / 0.1)
                beta_pos_threshold = 0.10
                k_declines = 0
                offers_made = 0
                i = 0
                winner_pos = -1
                while i < len(queue) and offers_made < max_hours:
                    pat = queue[i]
                    if K_cutoffs is None:
                        # Continuous: mechanism uses true threshold directly.
                        eligible = pat.threshold <= theta
                    else:
                        # Coarse: mechanism uses declared bucket.
                        eligible = pat.declared_bucket <= target_bucket
                    if not eligible:
                        i += 1
                        continue
                    # Offer this patient the kidney (counts toward 12-hour budget)
                    offers_made += 1
                    pos_1indexed = i + 1
                    # Under coarse mechanism, patient may still decline based on true threshold
                    if pat.threshold > theta:
                        k_declines += 1
                        i += 1
                        continue
                    # Quality acceptable; check position+lemons-modulated acceptance
                    p_accept = sigmoid(alpha_base + beta_pos_threshold * (pos_1indexed - 1)
                                       - delta_lemons * k_declines)
                    if rng.random() < p_accept:
                        winner_pos = i
                        break
                    else:
                        k_declines += 1
                        i += 1
                if winner_pos >= 0:
                    kidney_placement[k_idx] = winner_pos + 1
                    queue[winner_pos].received = True
                    queue[winner_pos].received_at = k_idx
                    del queue[winner_pos]
                    allocated = True

        if not allocated:
            kidney_discarded[k_idx] = True

        # Patient arrivals between this kidney and the next
        fractional_arrivals += rho
        n_arr = int(fractional_arrivals)
        fractional_arrivals -= n_arr
        for _ in range(n_arr):
            thr = float(rng.beta(THRESHOLD_BETA_ALPHA, THRESHOLD_BETA_BETA))
            p = Patient(pid=next_pid, threshold=thr, arrival_kidney=k_idx,
                        initial_priority=len(queue)+1)
            if mechanism in ('threshold', 'threshold_batched'):
                p.declared_bucket = declare_bucket(thr, K_cutoffs)
            queue.append(p)
            all_patients.append(p)
            next_pid += 1

    return {
        'kidney_discarded': kidney_discarded,
        'kidney_placement': kidney_placement,
        'kidney_quality': kidney_quality,
        'patients': all_patients,
        'queue_end_size': len(queue),
        'total_arrivals': next_pid,
    }


def run_offer_round(queue, theta, p_baseline, beta_pos, delta_lemons,
                     batch_size, max_hours, rng):
    """Run hourly offer rounds with UNIFIED preference-based acceptance.

    A patient accepts iff their true threshold <= theta AND a random draw from
    sigmoid(alpha_base + beta_pos*(n-1) - delta*k). Threshold-violating offers
    auto-decline (P=0). Both threshold-mismatch and random declines count toward
    the lemons cascade (subsequent patients don't see WHY someone declined).

    alpha_base = logit(0.9) so an eligible patient at position 1 with no priors
    accepts with 90% probability. This matches the threshold mechanism's baseline.

    Position effect still operates among eligible patients.
    Returns (allocated, winner_pos).
    """
    alpha_base = np.log(0.9 / 0.1)   # logit(0.9)

    queue_len = len(queue)
    for hour in range(max_hours):
        start = hour * batch_size
        if start >= queue_len:
            return False, -1
        end = min(start + batch_size, queue_len)
        k_declines = hour * batch_size

        accepts_in_batch = []
        for pos in range(start, end):
            pat = queue[pos]
            pos_1indexed = pos + 1
            if pat.threshold > theta:
                # Threshold violated; patient declines with certainty.
                # Decline still counts toward lemons (subsequent patients don't see why).
                # Within a batch, declines don't propagate (lemons frozen within batch).
                continue
            # Threshold met; check noisy acceptance
            p_accept = sigmoid(alpha_base + beta_pos * (pos_1indexed - 1)
                               - delta_lemons * k_declines)
            if rng.random() < p_accept:
                accepts_in_batch.append(pos)

        if accepts_in_batch:
            return True, min(accepts_in_batch)

    return False, -1


def declare_bucket(threshold, K_cutoffs):
    """Patient declares the bucket containing their true threshold (Option II aggressive).
    They will receive offers from this bucket and above. Their actual acceptance
    of any individual offer is determined later by checking against true threshold.

    K_cutoffs are quality cutoffs (sorted, ascending).
    Buckets: bucket 0 = [0, c_0), bucket 1 = [c_0, c_1), ..., bucket K-1 = [c_{K-2}, 1]
    Patient with threshold thr is in bucket b where c_{b-1} <= thr < c_b
    (using convention c_{-1} = 0 and c_{K-1} = 1).

    If K_cutoffs is None, continuous mechanism (no bucketing).
    """
    if K_cutoffs is None:
        return None
    for b, cut in enumerate(K_cutoffs):
        if threshold < cut:
            return b
    return len(K_cutoffs)   # falls in the topmost bucket


def bucket_of_theta(theta, K_cutoffs):
    """Return the bucket index of a kidney with quality theta."""
    if K_cutoffs is None:
        return 0
    for b in range(len(K_cutoffs)):
        if theta < K_cutoffs[b]:
            return b
    return len(K_cutoffs)


# For continuous threshold (K=infinity), use the threshold mechanism with K_cutoffs=None
# (handled in the main simulate_run function). The old simulate_run_continuous_threshold
# function is retained but unused; kept for backwards compatibility.
def simulate_run_continuous_threshold(rng, **kwargs):
    """DEPRECATED: use simulate_run with mechanism='threshold' and K_cutoffs=None.
    Kept for reference but should not be called.
    """
    raise RuntimeError("Use simulate_run with mechanism='threshold' and K_cutoffs=None")


def aggregate(result, n_warmup=N_WARMUP, n_kidneys=N_KIDNEYS):
    """Compute headline metrics from a single run."""
    kd = result['kidney_discarded'][n_warmup:]
    kp = result['kidney_placement'][n_warmup:]
    allocated = kp != -1
    discard_rate = float(kd.mean())
    mean_placement = float(kp[allocated].mean()) if allocated.any() else np.nan

    # Patient-level: include patients who arrived in steady state
    cutoff_low = n_warmup
    cutoff_high = n_kidneys - 100  # exclude very late arrivals
    elig = [p for p in result['patients']
            if cutoff_low <= p.arrival_kidney < cutoff_high]
    if len(elig) > 30:
        transplant_rate = float(np.mean([p.received for p in elig]))
        waits = [p.received_at - p.arrival_kidney for p in elig if p.received]
        mean_wait = float(np.mean(waits)) if waits else np.nan
    else:
        transplant_rate = np.nan
        mean_wait = np.nan

    return {
        'discard_rate': discard_rate,
        'mean_placement': mean_placement,
        'transplant_rate': transplant_rate,
        'mean_wait': mean_wait,
        'queue_end_size': result['queue_end_size'],
        'n_eligible_patients': len(elig),
    }


def run_cell(mechanism, delta_lemons, K_cutoffs=None, n_reps=N_REPS, base_seed=SEED,
              p_baseline=0.10, beta_pos=0.10, batch_m=5):
    """Run many reps at one parameter cell and return mean metrics."""
    rng_master = np.random.default_rng(
        base_seed + hash((mechanism, delta_lemons, str(K_cutoffs))) % 10_000)
    metrics_list = []
    for r in range(n_reps):
        seed = int(rng_master.integers(0, 1_000_000))
        rng = np.random.default_rng(seed)
        result = simulate_run(mechanism, rng, p_baseline=p_baseline, beta_pos=beta_pos,
                               delta_lemons=delta_lemons, batch_m=batch_m,
                               K_cutoffs=K_cutoffs)
        metrics_list.append(aggregate(result))

    keys = metrics_list[0].keys()
    avg = {k: float(np.nanmean([m[k] for m in metrics_list])) for k in keys}
    se = {f'{k}_se': float(np.nanstd([m[k] for m in metrics_list]) / np.sqrt(n_reps))
          for k in keys}
    avg.update(se)
    avg['mechanism'] = mechanism
    avg['delta_lemons'] = delta_lemons
    avg['K_cutoffs'] = str(K_cutoffs)
    return avg


# ---------- Run sweep ----------

def main():
    print('Threshold mechanism comparison sweep')
    print(f'  {N_REPS} reps × {N_KIDNEYS} kidneys × multiple mechanisms')

    delta_values = [0.0, 0.3, 1.0]
    rows = []

    for delta in delta_values:
        print(f'\ndelta = {delta}')

        # Omniscient first-best (delta is irrelevant for omniscient)
        if delta == 0.0:  # only run once since omniscient is delta-invariant
            print('  omniscient', end='', flush=True)
            r = run_cell('omniscient', 0.0)
            for d in delta_values:
                r2 = dict(r)
                r2['delta_lemons'] = d
                r2['mechanism_label'] = 'omniscient'
                rows.append(r2)
            print(' done')

        # Sequential FCFS with lemons
        print('  sequential', end='', flush=True)
        r = run_cell('sequential', delta)
        r['mechanism_label'] = 'sequential'
        rows.append(r)
        print(' done')

        # Batching m=5 with lemons
        print('  batching m=5', end='', flush=True)
        r = run_cell('batching', delta, batch_m=5)
        r['mechanism_label'] = 'batching_m5'
        rows.append(r)
        print(' done')

        # Threshold continuous (K=infinity)
        print('  threshold continuous', end='', flush=True)
        r = run_cell('threshold', delta, K_cutoffs=None)
        r['mechanism_label'] = 'threshold_continuous'
        rows.append(r)
        print(' done')

        # Threshold K=10
        K = 10
        cuts = [(i+1)/K for i in range(K-1)]
        print(f'  threshold K=10', end='', flush=True)
        r = run_cell('threshold', delta, K_cutoffs=cuts)
        r['mechanism_label'] = 'threshold_K10'
        rows.append(r)
        print(' done')

        # Threshold K=5
        K = 5
        cuts = [(i+1)/K for i in range(K-1)]
        print(f'  threshold K=5', end='', flush=True)
        r = run_cell('threshold', delta, K_cutoffs=cuts)
        r['mechanism_label'] = 'threshold_K5'
        rows.append(r)
        print(' done')

        # Threshold K=3
        K = 3
        cuts = [(i+1)/K for i in range(K-1)]
        print(f'  threshold K=3', end='', flush=True)
        r = run_cell('threshold', delta, K_cutoffs=cuts)
        r['mechanism_label'] = 'threshold_K3'
        rows.append(r)
        print(' done')

        # Threshold K=2 OPTN-realistic (cutoff at theta=0.15, KDPI 85)
        print(f'  threshold K=2 OPTN', end='', flush=True)
        r = run_cell('threshold', delta, K_cutoffs=[0.15])
        r['mechanism_label'] = 'threshold_K2_OPTN'
        rows.append(r)
        print(' done')

        # Hybrid: threshold K=2 OPTN with m=5 batching within eligible patients
        print(f'  threshold_batched K=2 OPTN m=5', end='', flush=True)
        r = run_cell('threshold_batched', delta, K_cutoffs=[0.15], batch_m=5)
        r['mechanism_label'] = 'threshold_batched_K2_OPTN_m5'
        rows.append(r)
        print(' done')

        # Threshold K=2 with optimized cutoff (search a few candidates)
        # We'll just try several and pick the best on discard rate later
        for cutoff in [0.20, 0.30, 0.40, 0.50]:
            print(f'  threshold K=2 cutoff={cutoff}', end='', flush=True)
            r = run_cell('threshold', delta, K_cutoffs=[cutoff])
            r['mechanism_label'] = f'threshold_K2_cut{cutoff}'
            rows.append(r)
            print(' done')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'threshold_results.csv'), index=False)
    print(f'\nwrote threshold_results.csv  ({len(df)} rows)')

    # Print summary table
    print('\n=== Headline summary ===')
    cols = ['mechanism_label', 'delta_lemons', 'discard_rate', 'mean_placement',
            'transplant_rate', 'mean_wait', 'queue_end_size']
    print(df[cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
