#!/usr/bin/env python3
"""
Marginal kidney analysis: who gets marginal kidneys, how often, and how well-matched.

Marginal kidneys: theta <= 0.15 (KDPI >= 85), corresponding to current OPTN
high-KDPI category.

For each mechanism x delta:
  - Allocation rate among marginal kidneys
  - Mean position at allocation
  - Mean recipient threshold
  - Mean match quality (theta - threshold for recipient)
  - Aggregate welfare per marginal kidney (counting discards as 0 utility)
"""

import os
import sys
sys.path.insert(0, '/home/claude')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from threshold_sim import (
    sigmoid, p_of_theta, declare_bucket, bucket_of_theta,
    THRESHOLD_BETA_ALPHA, THRESHOLD_BETA_BETA,
    MAX_HOURS, RHO, SEED, Patient,
)

OUTPUT_DIR = '/mnt/user-data/outputs'
N_KIDNEYS = 1500
N_WARMUP = 300
INITIAL_QUEUE = 150
N_REPS = 10

# Define what counts as "marginal" (OPTN-aligned: KDPI >= 85, theta <= 0.15)
MARGINAL_THRESHOLD = 0.15
VERY_MARGINAL_THRESHOLD = 0.05  # KDPI 95-100, the most marginal tier


def run_with_marginal_tracking(world_seed, mechanism, delta, K_cutoffs=None, batch_m=5):
    """Run a mechanism and track marginal kidney outcomes including recipient info."""
    rng = np.random.default_rng(world_seed)

    queue = deque()
    next_pid = 0
    for _ in range(INITIAL_QUEUE):
        thr = float(rng.beta(THRESHOLD_BETA_ALPHA, THRESHOLD_BETA_BETA))
        p = Patient(pid=next_pid, threshold=thr, arrival_kidney=-1,
                    initial_priority=next_pid + 1)
        if mechanism in ('threshold', 'threshold_batched'):
            p.declared_bucket = declare_bucket(thr, K_cutoffs)
        queue.append(p)
        next_pid += 1

    records = []  # one row per kidney
    fractional = 0.0

    for k_idx in range(N_KIDNEYS):
        theta = float(rng.random())

        allocated = False
        winner_pos = -1
        winner_thr = np.nan

        if len(queue) > 0:
            if mechanism == 'omniscient':
                for i, pat in enumerate(queue):
                    if pat.threshold <= theta:
                        winner_pos = i
                        winner_thr = pat.threshold
                        allocated = True
                        break

            elif mechanism in ('sequential', 'batching'):
                bs = 1 if mechanism == 'sequential' else batch_m
                alpha_base = np.log(0.9 / 0.1)   # logit(0.9), unified baseline
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
                        if pat.threshold > theta:
                            continue   # threshold violated; auto-decline (no random draw)
                        p_acc = sigmoid(alpha_base + beta_pos * (pos1 - 1) - delta * k_declines)
                        if rng.random() < p_acc:
                            accepts.append(pos)
                    if accepts:
                        winner_pos = min(accepts)
                        winner_thr = queue[winner_pos].threshold
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
                    pos1 = i + 1
                    if pat.threshold > theta:
                        k_declines += 1
                        i += 1
                        continue
                    p_acc = sigmoid(alpha_base + beta_pos * (pos1 - 1) - delta * k_declines)
                    if rng.random() < p_acc:
                        winner_pos = i
                        winner_thr = pat.threshold
                        allocated = True
                        break
                    else:
                        k_declines += 1
                        i += 1

        # Record this kidney's outcome
        records.append({
            'kidney_idx': k_idx,
            'theta': theta,
            'allocated': allocated,
            'position': winner_pos + 1 if allocated else -1,
            'recipient_threshold': winner_thr,
            'match_quality': (theta - winner_thr) if allocated else np.nan,
            'queue_len_at_offer': len(queue),
        })

        if allocated:
            queue[winner_pos].received = True
            queue[winner_pos].received_at = k_idx
            del queue[winner_pos]

        # Patient arrivals
        fractional += RHO
        n_arr = int(fractional)
        fractional -= n_arr
        for _ in range(n_arr):
            thr = float(rng.beta(THRESHOLD_BETA_ALPHA, THRESHOLD_BETA_BETA))
            p = Patient(pid=next_pid, threshold=thr, arrival_kidney=k_idx,
                        initial_priority=len(queue) + 1)
            if mechanism in ('threshold', 'threshold_batched'):
                p.declared_bucket = declare_bucket(thr, K_cutoffs)
            queue.append(p)
            next_pid += 1

    return pd.DataFrame(records)


def main():
    mechanisms = [
        ('omniscient', None, 'Omniscient'),
        ('sequential', None, 'Sequential'),
        ('batching', None, 'Batching m=5'),
        ('threshold', [0.15], 'Threshold K=2 (OPTN)'),
        ('threshold', [(i+1)/5 for i in range(4)], 'Threshold K=5'),
        ('threshold', [(i+1)/10 for i in range(9)], 'Threshold K=10'),
        ('threshold', None, 'Threshold continuous (K=∞)'),
    ]
    deltas = [0.0, 0.3, 1.0]

    all_records = []
    for rep in range(N_REPS):
        print(f'rep {rep+1}/{N_REPS}')
        for delta in deltas:
            for mech, kcuts, label in mechanisms:
                seed = SEED + rep * 1000 + int(delta * 100) + hash(label) % 10000
                df = run_with_marginal_tracking(seed, mech, delta, K_cutoffs=kcuts)
                df = df[df['kidney_idx'] >= N_WARMUP].copy()
                df['rep'] = rep
                df['delta'] = delta
                df['mechanism'] = label
                all_records.append(df)

    full = pd.concat(all_records, ignore_index=True)
    full['is_marginal'] = full['theta'] <= MARGINAL_THRESHOLD
    full['is_very_marginal'] = full['theta'] <= VERY_MARGINAL_THRESHOLD
    full.to_csv(os.path.join(OUTPUT_DIR, 'marginal_kidneys.csv'), index=False)

    # Compute welfare: u = max(theta - threshold, 0) if allocated, else 0
    full['welfare'] = np.where(
        full['allocated'],
        np.maximum(full['theta'] - full['recipient_threshold'], 0),
        0.0,
    )

    # Aggregate by mechanism × delta, separately for marginal kidneys
    print('\n=== Marginal kidneys (theta <= 0.15) ===')
    sub = full[full['is_marginal']]
    summary_marginal = sub.groupby(['mechanism', 'delta'], observed=True).agg(
        n=('theta', 'count'),
        allocation_rate=('allocated', 'mean'),
        discard_rate=('allocated', lambda s: 1 - s.mean()),
        mean_position=('position', lambda s: s[s > 0].mean()),
        median_position=('position', lambda s: s[s > 0].median()),
        mean_recipient_threshold=('recipient_threshold', lambda s: s.dropna().mean()),
        mean_match_quality=('match_quality', lambda s: s.dropna().mean()),
        mean_welfare=('welfare', 'mean'),  # averages over all kidneys including discards
    ).reset_index()
    print(summary_marginal.round(3).to_string(index=False))
    summary_marginal.to_csv(os.path.join(OUTPUT_DIR, 'marginal_summary.csv'), index=False)

    # Same for very marginal
    print('\n=== Very marginal kidneys (theta <= 0.05) ===')
    sub2 = full[full['is_very_marginal']]
    summary_vmarg = sub2.groupby(['mechanism', 'delta'], observed=True).agg(
        n=('theta', 'count'),
        allocation_rate=('allocated', 'mean'),
        discard_rate=('allocated', lambda s: 1 - s.mean()),
        mean_position=('position', lambda s: s[s > 0].mean()),
        mean_recipient_threshold=('recipient_threshold', lambda s: s.dropna().mean()),
        mean_welfare=('welfare', 'mean'),
    ).reset_index()
    print(summary_vmarg.round(3).to_string(index=False))
    summary_vmarg.to_csv(os.path.join(OUTPUT_DIR, 'very_marginal_summary.csv'), index=False)

    # All kidneys summary for comparison
    print('\n=== All kidneys ===')
    summary_all = full.groupby(['mechanism', 'delta'], observed=True).agg(
        n=('theta', 'count'),
        allocation_rate=('allocated', 'mean'),
        discard_rate=('allocated', lambda s: 1 - s.mean()),
        mean_welfare=('welfare', 'mean'),
    ).reset_index()
    print(summary_all.round(3).to_string(index=False))
    summary_all.to_csv(os.path.join(OUTPUT_DIR, 'all_kidneys_summary.csv'), index=False)


if __name__ == '__main__':
    main()
