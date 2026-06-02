#!/usr/bin/env python3
"""Plots for threshold mechanism simulation results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = '/mnt/user-data/outputs'
df = pd.read_csv(os.path.join(OUTPUT_DIR, 'threshold_results.csv'))


# Headline plot 1: Pareto frontier (discard vs mean placement) at each delta
def plot_pareto():
    deltas = sorted(df['delta_lemons'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Highlight headline mechanisms with distinct colors
    color_map = {
        'omniscient': ('black', 'X', 200),
        'sequential': ('#888888', 'D', 100),
        'batching_m5': ('tab:blue', 's', 150),
        'threshold_continuous': ('tab:red', '*', 250),
        'threshold_K10': ('#cc4444', 'o', 80),
        'threshold_K5': ('#dd6666', 'o', 80),
        'threshold_K3': ('#ee8888', 'o', 80),
        'threshold_K2_OPTN': ('tab:orange', 'P', 150),
        'threshold_batched_K2_OPTN_m5': ('tab:green', 'h', 150),
    }

    for ax, delta in zip(axes, deltas):
        sub = df[df['delta_lemons'] == delta]
        for label, (color, marker, size) in color_map.items():
            row = sub[sub['mechanism_label'] == label]
            if len(row) == 0:
                continue
            x = row['discard_rate'].values[0]
            y = row['mean_placement'].values[0]
            ax.scatter(x, y, color=color, marker=marker, s=size,
                       edgecolor='black', linewidth=0.5, label=label,
                       zorder=3)
        ax.set_xlabel('Discard rate')
        ax.set_ylabel('Mean placement | allocated')
        ax.set_title(f'δ = {delta} (lemons strength)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.01, max(0.5, sub['discard_rate'].max() + 0.02))
        if delta == deltas[0]:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)

    fig.suptitle('Pareto frontier: discard rate vs mean placement, by mechanism and lemons strength',
                 fontsize=13, y=1.03)
    fig.savefig(os.path.join(OUTPUT_DIR, 'threshold_pareto.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


# Plot 2: Discard rate vs delta, one line per mechanism
def plot_discard_vs_delta():
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    headline_labels = ['omniscient', 'sequential', 'batching_m5',
                        'threshold_continuous', 'threshold_K10', 'threshold_K5',
                        'threshold_K3', 'threshold_K2_OPTN',
                        'threshold_batched_K2_OPTN_m5']
    label_display = {
        'omniscient': 'omniscient',
        'sequential': 'sequential (m=1)',
        'batching_m5': 'batching m=5',
        'threshold_continuous': 'threshold K=∞',
        'threshold_K10': 'threshold K=10',
        'threshold_K5': 'threshold K=5',
        'threshold_K3': 'threshold K=3',
        'threshold_K2_OPTN': 'threshold K=2 (OPTN)',
        'threshold_batched_K2_OPTN_m5': 'threshold K=2 + batching',
    }
    colors = {
        'omniscient': 'black',
        'sequential': '#888888',
        'batching_m5': 'tab:blue',
        'threshold_continuous': 'tab:red',
        'threshold_K10': '#cc4444',
        'threshold_K5': '#dd6666',
        'threshold_K3': '#ee8888',
        'threshold_K2_OPTN': 'tab:orange',
        'threshold_batched_K2_OPTN_m5': 'tab:green',
    }
    for label in headline_labels:
        sub = df[df['mechanism_label'] == label].sort_values('delta_lemons')
        ax.plot(sub['delta_lemons'], sub['discard_rate'], '-o',
                color=colors[label], label=label_display[label], lw=2, ms=7)
    ax.set_xlabel('δ (lemons strength)')
    ax.set_ylabel('Discard rate')
    ax.set_title('Discard rate vs lemons strength, by mechanism')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    fig.savefig(os.path.join(OUTPUT_DIR, 'threshold_discard_vs_delta.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


# Plot 3: Coarsening effect — discard rate vs K (with delta as separate lines)
def plot_coarsening():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    # K-labeled threshold mechanisms only
    k_map = {
        'threshold_K2_OPTN': 2,
        'threshold_K3': 3,
        'threshold_K5': 5,
        'threshold_K10': 10,
        'threshold_continuous': 100,   # treat as 'infinity' on log scale
    }
    deltas = sorted(df['delta_lemons'].unique())
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(deltas)))

    # Panel A: discard rate vs K
    ax = axes[0]
    for j, delta in enumerate(deltas):
        sub = df[df['delta_lemons'] == delta]
        xs, ys = [], []
        for label, K in sorted(k_map.items(), key=lambda x: x[1]):
            row = sub[sub['mechanism_label'] == label]
            if len(row):
                xs.append(K)
                ys.append(row['discard_rate'].values[0])
        ax.plot(xs, ys, '-o', color=cmap[j], lw=2, ms=7,
                label=f'δ = {delta}')
    # Add reference lines for omniscient and batching
    omn = df[df['mechanism_label'] == 'omniscient']['discard_rate'].iloc[0]
    ax.axhline(omn, color='black', linestyle=':', lw=1.5,
                label=f'omniscient ({omn:.1%})')
    for j, delta in enumerate(deltas):
        bat = df[(df['mechanism_label'] == 'batching_m5')
                 & (df['delta_lemons'] == delta)]['discard_rate'].iloc[0]
        ax.axhline(bat, color=cmap[j], linestyle='--', lw=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xticks([2, 3, 5, 10, 100])
    ax.set_xticklabels(['2 (OPTN)', '3', '5', '10', '∞'])
    ax.set_xlabel('Number of buckets K')
    ax.set_ylabel('Discard rate')
    ax.set_title('Coarsening cost: discard rate vs K')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: efficiency loss vs omniscient
    ax = axes[1]
    for j, delta in enumerate(deltas):
        sub = df[df['delta_lemons'] == delta]
        xs, ys = [], []
        for label, K in sorted(k_map.items(), key=lambda x: x[1]):
            row = sub[sub['mechanism_label'] == label]
            if len(row):
                xs.append(K)
                ys.append(row['discard_rate'].values[0] - omn)
        ax.plot(xs, ys, '-o', color=cmap[j], lw=2, ms=7,
                label=f'δ = {delta}')
    ax.set_xscale('log')
    ax.set_xticks([2, 3, 5, 10, 100])
    ax.set_xticklabels(['2 (OPTN)', '3', '5', '10', '∞'])
    ax.set_xlabel('Number of buckets K')
    ax.set_ylabel('Discard rate above omniscient')
    ax.set_title('Efficiency loss from coarsening')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Coarsening cost grows with lemons strength',
                 fontsize=13, y=1.04)
    fig.savefig(os.path.join(OUTPUT_DIR, 'threshold_coarsening.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


# Plot 4: Optimal K=2 cutoff comparison
def plot_K2_cutoffs():
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    cutoffs = [0.15, 0.20, 0.30, 0.40, 0.50]
    labels = ['K2_OPTN', 'K2_cut0.2', 'K2_cut0.3', 'K2_cut0.4', 'K2_cut0.5']
    deltas = sorted(df['delta_lemons'].unique())
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(deltas)))
    for j, delta in enumerate(deltas):
        sub = df[df['delta_lemons'] == delta]
        ys = []
        for label, cutoff in zip(labels, cutoffs):
            target = 'threshold_' + label if label == 'K2_OPTN' else f'threshold_{label}'
            row = sub[sub['mechanism_label'] == target]
            if len(row):
                ys.append(row['discard_rate'].values[0])
            else:
                ys.append(np.nan)
        ax.plot(cutoffs, ys, '-o', color=cmap[j], lw=2, ms=7,
                label=f'δ = {delta}')
    ax.axvline(0.15, color='black', linestyle=':', lw=1, label='OPTN cutoff (KDPI 85)')
    ax.set_xlabel('Binary cutoff θ (1 - KDPI/100)')
    ax.set_ylabel('Discard rate')
    ax.set_title('Discard rate vs binary cutoff (K=2 only)')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, 'threshold_K2_cutoffs.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_pareto()
    plot_discard_vs_delta()
    plot_coarsening()
    plot_K2_cutoffs()
    print('Plots written.')
