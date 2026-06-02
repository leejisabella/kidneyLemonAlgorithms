#!/usr/bin/env python3
"""Plots for CRN fairness analysis."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUTPUT_DIR = '/mnt/user-data/outputs'
pos_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'crn_positions.csv'))
pareto_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'crn_pareto.csv'))

MECHS = ['Sequential', 'Batching m=5', 'Threshold K=2 (OPTN)', 'Threshold continuous (K=∞)']
COLORS = {'Sequential': '#888888', 'Batching m=5': 'tab:blue',
          'Threshold K=2 (OPTN)': 'tab:orange',
          'Threshold continuous (K=∞)': 'tab:red'}
DELTAS = [0.0, 0.3, 1.0]


def plot_position_cdf():
    """Plot CDF of allocation position by mechanism × delta."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, sharey=True)
    max_x = 30
    for ax, delta in zip(axes, DELTAS):
        for mech in MECHS:
            sub = pos_df[(pos_df['mechanism'] == mech) & (pos_df['delta'] == delta)]
            if len(sub) == 0:
                continue
            placements = sub['placement'].values
            placements = np.clip(placements, 0, max_x + 5)
            xs = np.sort(placements)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=COLORS[mech], lw=2, label=mech)
        ax.set_xlabel('Queue position at allocation')
        ax.set_xlim(0, max_x)
        ax.set_title(f'δ = {delta}')
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel('Fraction of allocated kidneys')
            ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('CDF of allocation depth by mechanism\n'
                  'Reading: at position p, what fraction of allocations have occurred?',
                 fontsize=12, y=1.04)
    fig.savefig(os.path.join(OUTPUT_DIR, 'crn_position_cdf.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def plot_position_histogram_log():
    """Plot histogram of allocation position with log y-axis to show the long tail."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True,
                              sharex=True, sharey=True)
    for col, delta in enumerate(DELTAS):
        # Row 0: batching vs threshold-K2
        ax = axes[0, col]
        for mech in ['Batching m=5', 'Threshold K=2 (OPTN)']:
            sub = pos_df[(pos_df['mechanism'] == mech) & (pos_df['delta'] == delta)]
            if len(sub) == 0:
                continue
            placements = sub['placement'].values
            bins = np.arange(0, 100, 2)
            ax.hist(placements, bins=bins, alpha=0.5, color=COLORS[mech], label=mech)
        ax.set_title(f'δ = {delta}: batching vs K=2')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Count (log scale)')
            ax.legend(fontsize=9)

        # Row 1: batching vs threshold continuous
        ax = axes[1, col]
        for mech in ['Batching m=5', 'Threshold continuous (K=∞)']:
            sub = pos_df[(pos_df['mechanism'] == mech) & (pos_df['delta'] == delta)]
            if len(sub) == 0:
                continue
            placements = sub['placement'].values
            bins = np.arange(0, 100, 2)
            ax.hist(placements, bins=bins, alpha=0.5, color=COLORS[mech], label=mech)
        ax.set_title(f'δ = {delta}: batching vs continuous')
        ax.set_yscale('log')
        ax.set_xlabel('Queue position at allocation')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Count (log scale)')
            ax.legend(fontsize=9)

    fig.suptitle('Distribution of allocation depth (log scale)\n'
                  'Both threshold mechanisms have similar peaks; threshold continuous has longer tail',
                 fontsize=12, y=1.03)
    fig.savefig(os.path.join(OUTPUT_DIR, 'crn_position_hist.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def plot_position_vs_quality():
    """Allocation depth as a function of kidney quality. This is the key plot:
    threshold mechanisms reach deep only for marginal kidneys."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True,
                              sharex=True, sharey=True)
    quality_bins = np.linspace(0, 1, 11)
    bin_centers = (quality_bins[:-1] + quality_bins[1:]) / 2

    for col, delta in enumerate(DELTAS):
        # Row 0: batching vs threshold-K2
        ax = axes[0, col]
        for mech in ['Batching m=5', 'Threshold K=2 (OPTN)']:
            sub = pos_df[(pos_df['mechanism'] == mech) & (pos_df['delta'] == delta)]
            if len(sub) == 0:
                continue
            sub = sub.copy()
            sub['q_bin'] = pd.cut(sub['kidney_quality'], bins=quality_bins,
                                    labels=bin_centers)
            stats = sub.groupby('q_bin', observed=True)['placement'].agg(['mean', 'median', 'count'])
            ax.plot(stats.index.astype(float), stats['mean'], '-o', color=COLORS[mech],
                    label=f'{mech} (mean)', lw=2, ms=6)
            ax.plot(stats.index.astype(float), stats['median'], '--', color=COLORS[mech],
                    label=f'{mech} (median)', lw=1.5, alpha=0.6)
        ax.set_title(f'δ = {delta}: batching vs K=2')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Allocation position')
            ax.legend(fontsize=8)

        # Row 1: batching vs threshold continuous
        ax = axes[1, col]
        for mech in ['Batching m=5', 'Threshold continuous (K=∞)']:
            sub = pos_df[(pos_df['mechanism'] == mech) & (pos_df['delta'] == delta)]
            if len(sub) == 0:
                continue
            sub = sub.copy()
            sub['q_bin'] = pd.cut(sub['kidney_quality'], bins=quality_bins,
                                    labels=bin_centers)
            stats = sub.groupby('q_bin', observed=True)['placement'].agg(['mean', 'median', 'count'])
            ax.plot(stats.index.astype(float), stats['mean'], '-o', color=COLORS[mech],
                    label=f'{mech} (mean)', lw=2, ms=6)
            ax.plot(stats.index.astype(float), stats['median'], '--', color=COLORS[mech],
                    label=f'{mech} (median)', lw=1.5, alpha=0.6)
        ax.set_yscale('log')
        ax.set_xlabel('Kidney quality θ (higher = better)')
        ax.set_title(f'δ = {delta}: batching vs continuous')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Allocation position (log)')
            ax.legend(fontsize=8)

    fig.suptitle('Allocation depth by kidney quality\n'
                  'Threshold mechanisms reach deep only for marginal kidneys (low θ)',
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(OUTPUT_DIR, 'crn_position_by_quality.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def plot_pareto_bars():
    """Bar plot: of all patients in the simulation, how many are uniquely
    helped by each mechanism vs another?"""
    mechs_to_compare = [('Batching m=5', 'Threshold K=2 (OPTN)'),
                         ('Batching m=5', 'Threshold continuous (K=∞)')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, (A, B) in zip(axes, mechs_to_compare):
        # For each delta, show counts: only A, only B, both, neither
        deltas = sorted(pareto_df['delta'].unique())
        only_a = []
        only_b = []
        both = []
        neither = []
        for delta in deltas:
            row = pareto_df[(pareto_df['delta'] == delta)
                          & (pareto_df['A'] == A) & (pareto_df['B'] == B)]
            if len(row) == 0:
                continue
            only_a.append(int(row['received_A_not_B'].values[0]))
            only_b.append(int(row['received_B_not_A'].values[0]))
            both.append(int(row['both'].values[0]))
            neither.append(int(row['neither'].values[0]))

        x = np.arange(len(deltas))
        width = 0.6

        # Stack: both at bottom (green), then only A (blue), then only B (red), then neither (gray)
        ax.bar(x, both, width, color='tab:green', label='received under both')
        ax.bar(x, only_a, width, bottom=both,
                color=COLORS[A], label=f'received only under {A}')
        ax.bar(x, only_b, width, bottom=[b+a for b, a in zip(both, only_a)],
                color=COLORS[B], label=f'received only under {B}')
        ax.bar(x, neither, width,
                bottom=[b+a+c for b, a, c in zip(both, only_a, only_b)],
                color='lightgray', label='received under neither')

        # Annotate the "only" bars
        for i, (a, b) in enumerate(zip(only_a, only_b)):
            if a > 50:
                ax.text(x[i], both[i] + a/2, str(a), ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')
            if b > 50:
                ax.text(x[i], both[i] + only_a[i] + b/2, str(b), ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'δ={d}' for d in deltas])
        ax.set_title(f'{A} vs {B}')
        ax.set_ylabel('Number of patients (across all reps)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Patient-level Pareto comparison under common random numbers\n'
                  'No "only A" or "only B" overlap means strict Pareto dominance',
                 fontsize=12, y=1.05)
    fig.savefig(os.path.join(OUTPUT_DIR, 'crn_pareto_bars.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_position_cdf()
    plot_position_histogram_log()
    plot_position_vs_quality()
    plot_pareto_bars()
    print('CRN plots written.')
