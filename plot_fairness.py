#!/usr/bin/env python3
"""Plots for fairness analysis."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUTPUT_DIR = '/mnt/user-data/outputs'
pdf = pd.read_csv(os.path.join(OUTPUT_DIR, 'fairness_priority.csv'))
tdf = pd.read_csv(os.path.join(OUTPUT_DIR, 'fairness_threshold.csv'))
jdf = pd.read_csv(os.path.join(OUTPUT_DIR, 'fairness_joint.csv'))

MECHS = ['Sequential', 'Batching m=5', 'Threshold K=2 (OPTN)', 'Threshold continuous (K=∞)']
DELTAS = [0.0, 0.3, 1.0]


def plot_transplant_by_priority():
    """Transplant rate by priority tercile, faceted by delta."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, sharey=True)
    terciles = ['high', 'mid', 'low']
    colors = ['#888888', 'tab:blue', 'tab:orange', 'tab:red']
    x = np.arange(len(terciles))
    width = 0.20

    for ax, delta in zip(axes, DELTAS):
        for j, mech in enumerate(MECHS):
            ys = []
            for tert in terciles:
                row = pdf[(pdf['mechanism'] == mech) & (pdf['delta'] == delta)
                         & (pdf['priority_tercile'] == tert)]
                ys.append(row['transplant_rate'].values[0] if len(row) else np.nan)
            ax.bar(x + (j - 1.5) * width, ys, width, color=colors[j], label=mech,
                   edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(terciles)
        ax.set_xlabel('Initial priority tercile')
        ax.set_title(f'δ = {delta}')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        if ax is axes[0]:
            ax.set_ylabel('Transplant rate')
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Transplant rate by initial priority tercile, by mechanism',
                 fontsize=14, y=1.04)
    fig.savefig(os.path.join(OUTPUT_DIR, 'fairness_priority.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_transplant_by_threshold():
    """Transplant rate by threshold tercile, faceted by delta."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, sharey=True)
    terciles = ['low', 'mid', 'high']   # low threshold = accepts more
    colors = ['#888888', 'tab:blue', 'tab:orange', 'tab:red']
    x = np.arange(len(terciles))
    width = 0.20

    for ax, delta in zip(axes, DELTAS):
        for j, mech in enumerate(MECHS):
            ys = []
            for tert in terciles:
                row = tdf[(tdf['mechanism'] == mech) & (tdf['delta'] == delta)
                         & (tdf['threshold_tercile'] == tert)]
                ys.append(row['transplant_rate'].values[0] if len(row) else np.nan)
            ax.bar(x + (j - 1.5) * width, ys, width, color=colors[j], label=mech,
                   edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['low\n(accepts marginal)', 'mid', 'high\n(picky)'])
        ax.set_xlabel('Threshold tercile')
        ax.set_title(f'δ = {delta}')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 0.7)
        if ax is axes[0]:
            ax.set_ylabel('Transplant rate')
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Transplant rate by patient threshold tercile, by mechanism',
                 fontsize=14, y=1.04)
    fig.savefig(os.path.join(OUTPUT_DIR, 'fairness_threshold.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_joint_heatmap():
    """2D heatmap: transplant rate by priority × threshold tercile, per mechanism × delta."""
    fig, axes = plt.subplots(len(MECHS), len(DELTAS), figsize=(12, 14),
                              constrained_layout=True)
    priority_terciles = ['high', 'mid', 'low']
    threshold_terciles = ['low', 'mid', 'high']

    vmax = 1.0
    for i, mech in enumerate(MECHS):
        for j, delta in enumerate(DELTAS):
            ax = axes[i, j]
            mat = np.full((len(priority_terciles), len(threshold_terciles)), np.nan)
            for pi, pt in enumerate(priority_terciles):
                for ti, tt in enumerate(threshold_terciles):
                    row = jdf[(jdf['mechanism'] == mech) & (jdf['delta'] == delta)
                             & (jdf['priority_tercile'] == pt)
                             & (jdf['threshold_tercile'] == tt)]
                    if len(row):
                        mat[pi, ti] = row['transplant_rate'].values[0]
            im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=vmax, aspect='auto')
            for pi in range(3):
                for ti in range(3):
                    if not np.isnan(mat[pi, ti]):
                        v = mat[pi, ti]
                        txt_color = 'white' if 0.3 < v < 0.7 else 'black'
                        ax.text(ti, pi, f'{v:.2f}', ha='center', va='center',
                                color=txt_color, fontsize=10, fontweight='bold')
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(['low', 'mid', 'high'], fontsize=9)
            ax.set_yticklabels(priority_terciles, fontsize=9)
            if i == 0:
                ax.set_title(f'δ = {delta}', fontsize=11)
            if j == 0:
                ax.set_ylabel(f'{mech}\n(priority)', fontsize=10)
            if i == len(MECHS) - 1:
                ax.set_xlabel('Threshold tercile', fontsize=9)

    fig.suptitle('Transplant rate by priority × threshold tercile',
                 fontsize=14, y=1.02)
    fig.savefig(os.path.join(OUTPUT_DIR, 'fairness_joint.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_priority_summary():
    """A single clean summary plot: priority-band transplant rate, lines across delta."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, sharey=True)
    terciles = ['high', 'mid', 'low']
    colors = {'Sequential': '#888888', 'Batching m=5': 'tab:blue',
              'Threshold K=2 (OPTN)': 'tab:orange',
              'Threshold continuous (K=∞)': 'tab:red'}
    for ax, tert in zip(axes, terciles):
        for mech in MECHS:
            ys, xs = [], []
            for delta in DELTAS:
                row = pdf[(pdf['mechanism'] == mech) & (pdf['delta'] == delta)
                         & (pdf['priority_tercile'] == tert)]
                if len(row):
                    xs.append(delta)
                    ys.append(row['transplant_rate'].values[0])
            ax.plot(xs, ys, '-o', color=colors[mech], lw=2.2, ms=8, label=mech)
        ax.set_xlabel('δ (lemons strength)')
        ax.set_title(f'{tert}-priority tercile')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.05)
        if ax is axes[0]:
            ax.set_ylabel('Transplant rate')
            ax.legend(loc='best', fontsize=8.5)
    fig.suptitle('Transplant rate by priority tercile vs lemons strength',
                 fontsize=13, y=1.04)
    fig.savefig(os.path.join(OUTPUT_DIR, 'fairness_priority_lines.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_transplant_by_priority()
    plot_transplant_by_threshold()
    plot_joint_heatmap()
    plot_priority_summary()
    print('Fairness plots written.')
