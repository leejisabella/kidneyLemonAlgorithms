#!/usr/bin/env python3
"""
Fairness analysis for Part 2 threshold mechanism comparison.

Breaks patient outcomes by:
  - Initial priority tercile (queue position at arrival)
  - True threshold tercile (low/mid/high willingness)
  - Both jointly (2D heatmap)

For each cell: transplant rate, mean wait, mean kidney quality received.

Compares four mechanisms at three lemons strengths:
  sequential, batching m=5, threshold continuous, threshold K=2 OPTN
"""

import os
import sys
sys.path.insert(0, '/home/claude')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threshold_sim import (
    simulate_run, SEED, N_KIDNEYS, N_WARMUP,
)

OUTPUT_DIR = '/mnt/user-data/outputs'
N_REPS_FAIRNESS = 10   # fewer reps for fairness since we already have point estimates


def collect_patient_outcomes(mechanism, delta, K_cutoffs, n_reps, base_seed=SEED,
                              batch_m=5):
    """Run reps and return concatenated patient records with their final outcomes."""
    rng_master = np.random.default_rng(
        base_seed + hash((mechanism, delta, str(K_cutoffs), 'fair')) % 10_000)
    all_records = []
    for r in range(n_reps):
        seed = int(rng_master.integers(0, 1_000_000))
        rng = np.random.default_rng(seed)
        result = simulate_run(mechanism, rng, delta_lemons=delta,
                               K_cutoffs=K_cutoffs, batch_m=batch_m)
        # Eligible patients: arrived after warmup, before late horizon
        cutoff_low = N_WARMUP
        cutoff_high = N_KIDNEYS - 100
        kidney_quality = result['kidney_quality']
        for p in result['patients']:
            if cutoff_low <= p.arrival_kidney < cutoff_high:
                received_quality = (kidney_quality[p.received_at]
                                     if p.received and p.received_at >= 0
                                     else np.nan)
                wait = p.received_at - p.arrival_kidney if p.received else np.nan
                all_records.append({
                    'rep': r,
                    'pid': p.pid,
                    'arrival_kidney': p.arrival_kidney,
                    'initial_priority': p.initial_priority,
                    'threshold': p.threshold,
                    'received': p.received,
                    'wait': wait,
                    'received_quality': received_quality,
                })
    return pd.DataFrame(all_records)


def add_terciles(df):
    """Add priority and threshold tercile columns."""
    df = df.copy()
    # Priority: low number = high priority. Reverse for intuition.
    # Use percentile rank within run for stability
    df['priority_pct'] = df.groupby('rep')['initial_priority'].rank(pct=True, method='first')
    df['threshold_pct'] = df.groupby('rep')['threshold'].rank(pct=True, method='first')
    # Tercile labels
    df['priority_tercile'] = pd.cut(df['priority_pct'], bins=[0, 1/3, 2/3, 1.0001],
                                      labels=['high', 'mid', 'low'],
                                      include_lowest=True)
    df['threshold_tercile'] = pd.cut(df['threshold_pct'], bins=[0, 1/3, 2/3, 1.0001],
                                       labels=['low', 'mid', 'high'],
                                       include_lowest=True)
    return df


def tercile_summary(df, by='priority_tercile'):
    """Aggregate by single tercile dimension."""
    grouped = df.groupby(by, observed=True)
    summary = grouped.agg(
        n=('pid', 'count'),
        transplant_rate=('received', 'mean'),
        mean_wait=('wait', lambda s: s.dropna().mean()),
        mean_quality_received=('received_quality', lambda s: s.dropna().mean()),
    ).reset_index()
    return summary


def joint_summary(df):
    """Aggregate by priority × threshold tercile."""
    grouped = df.groupby(['priority_tercile', 'threshold_tercile'], observed=True)
    summary = grouped.agg(
        n=('pid', 'count'),
        transplant_rate=('received', 'mean'),
        mean_wait=('wait', lambda s: s.dropna().mean()),
    ).reset_index()
    return summary


def main():
    print('Collecting patient outcomes per mechanism...')
    mechanisms = [
        ('sequential', None, 'Sequential'),
        ('batching', None, 'Batching m=5'),
        ('threshold', None, 'Threshold continuous (K=∞)'),
        ('threshold', [0.15], 'Threshold K=2 (OPTN)'),
    ]
    deltas = [0.0, 0.3, 1.0]

    all_priority_records = []
    all_threshold_records = []
    all_joint_records = []

    for mech, kcuts, label in mechanisms:
        for delta in deltas:
            print(f'  {label}, δ={delta}')
            df = collect_patient_outcomes(mech, delta, kcuts, N_REPS_FAIRNESS)
            df = add_terciles(df)

            # Priority tercile summary
            ps = tercile_summary(df, by='priority_tercile')
            ps['mechanism'] = label
            ps['delta'] = delta
            all_priority_records.append(ps)

            # Threshold tercile summary
            ts = tercile_summary(df, by='threshold_tercile')
            ts['mechanism'] = label
            ts['delta'] = delta
            all_threshold_records.append(ts)

            # Joint summary
            js = joint_summary(df)
            js['mechanism'] = label
            js['delta'] = delta
            all_joint_records.append(js)

    priority_df = pd.concat(all_priority_records, ignore_index=True)
    threshold_df = pd.concat(all_threshold_records, ignore_index=True)
    joint_df = pd.concat(all_joint_records, ignore_index=True)

    priority_df.to_csv(os.path.join(OUTPUT_DIR, 'fairness_priority.csv'), index=False)
    threshold_df.to_csv(os.path.join(OUTPUT_DIR, 'fairness_threshold.csv'), index=False)
    joint_df.to_csv(os.path.join(OUTPUT_DIR, 'fairness_joint.csv'), index=False)
    print('wrote fairness_*.csv')

    # Print summaries
    print('\n=== Priority tercile summary ===')
    print(priority_df.round(3).to_string(index=False))
    print('\n=== Threshold tercile summary ===')
    print(threshold_df.round(3).to_string(index=False))
    print('\n=== Joint summary ===')
    print(joint_df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
