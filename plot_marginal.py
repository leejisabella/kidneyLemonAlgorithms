#!/usr/bin/env python3
"""Marginal kidney analysis plots."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = '/mnt/user-data/outputs'

marg = pd.read_csv(os.path.join(OUTPUT_DIR, 'marginal_summary.csv'))
vmarg = pd.read_csv(os.path.join(OUTPUT_DIR, 'very_marginal_summary.csv'))
allk = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_kidneys_summary.csv'))

# Standard ordering & colors
MECHS = ['Omniscient', 'Sequential', 'Batching m=5',
         'Threshold K=2 (OPTN)', 'Threshold K=5', 'Threshold K=10',
         'Threshold continuous (K=∞)']
COLORS = {
    'Omniscient': 'black',
    'Sequential': '#888888',
    'Batching m=5': 'tab:blue',
    'Threshold K=2 (OPTN)': 'tab:orange',
    'Threshold K=5': '#dd6666',
    'Threshold K=10': '#cc4444',
    'Threshold continuous (K=∞)': 'tab:red',
}
DELTAS = [0.0, 0.3, 1.0]


def _bar_panel(ax, df, value_col, ylabel, title, ylim=None):
    """Helper: bar plot of value_col by mechanism, faceted-ready."""
    x = np.arange(len(DELTAS))
    width = 0.11
    for j, mech in enumerate(MECHS):
        ys = []
        for delta in DELTAS:
            row = df[(df['mechanism'] == mech) & (df['delta'] == delta)]
            ys.append(row[value_col].values[0] if len(row) else np.nan)
        offset = (j - (len(MECHS) - 1) / 2) * width
        ax.bar(x + offset, ys, width, color=COLORS[mech], label=mech,
                edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'δ={d}' for d in DELTAS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    if ylim:
        ax.set_ylim(*ylim)


def plot_marginal_discard_welfare():
    """Two-panel: discard rate and welfare for marginal kidneys."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    _bar_panel(axes[0], marg, 'discard_rate', 'Discard rate',
                'Discard rate among marginal kidneys (θ ≤ 0.15)',
                ylim=(0, 1.0))
    _bar_panel(axes[1], marg, 'mean_welfare', 'Mean welfare per marginal kidney',
                'Welfare per marginal kidney (max(θ − threshold, 0))',
                ylim=(0, 0.025))
    axes[0].legend(loc='upper left', fontsize=8, ncol=2)
    fig.suptitle('Marginal kidneys: lower discards, higher welfare under threshold mechanisms',
                 fontsize=13, y=1.04)
    fig.savefig(os.path.join(OUTPUT_DIR, 'marginal_discard_welfare.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_marginal_recipient_match():
    """Two-panel: mean recipient threshold and match quality for marginal kidneys."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    _bar_panel(axes[0], marg, 'mean_recipient_threshold',
                'Mean recipient threshold $\\tilde\\theta_i$',
                'Recipient threshold for marginal kidneys\n(low = recipient wanted marginal; high = was reluctant)')
    _bar_panel(axes[1], marg, 'mean_match_quality',
                'Mean match quality θ − $\\tilde\\theta_i$',
                'Match quality for marginal kidneys\n(positive = patient would have accepted; negative = coerced)')
    axes[1].axhline(0, color='black', linewidth=1)
    axes[0].legend(loc='best', fontsize=8, ncol=2)
    fig.suptitle('Who gets marginal kidneys, and how well-matched are they?',
                 fontsize=13, y=1.04)
    fig.savefig(os.path.join(OUTPUT_DIR, 'marginal_recipient_match.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_coarsening_marginal():
    """Coarsening effect specifically on marginal kidneys.
    Show discard, welfare, position as functions of K, faceted by delta."""
    k_map = {
        'Threshold K=2 (OPTN)': 2,
        'Threshold K=5': 5,
        'Threshold K=10': 10,
        'Threshold continuous (K=∞)': 100,
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(DELTAS)))

    # Panel A: discard rate vs K
    ax = axes[0]
    for j, delta in enumerate(DELTAS):
        xs, ys = [], []
        for label, K in sorted(k_map.items(), key=lambda x: x[1]):
            row = marg[(marg['mechanism'] == label) & (marg['delta'] == delta)]
            if len(row):
                xs.append(K)
                ys.append(row['discard_rate'].values[0])
        ax.plot(xs, ys, '-o', color=cmap[j], lw=2.2, ms=8, label=f'δ = {delta}')
    # Reference lines
    omn = marg[marg['mechanism'] == 'Omniscient']['discard_rate'].iloc[0]
    ax.axhline(omn, color='black', linestyle=':', lw=1.5, label='omniscient')
    for j, delta in enumerate(DELTAS):
        bat = marg[(marg['mechanism'] == 'Batching m=5')
                   & (marg['delta'] == delta)]['discard_rate'].iloc[0]
        ax.axhline(bat, color=cmap[j], linestyle='--', lw=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 100])
    ax.set_xticklabels(['2 (OPTN)', '5', '10', '∞'])
    ax.set_xlabel('Number of buckets K')
    ax.set_ylabel('Discard rate')
    ax.set_title('Marginal kidney discard rate\n(dashed = batching reference)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # Panel B: welfare vs K
    ax = axes[1]
    for j, delta in enumerate(DELTAS):
        xs, ys = [], []
        for label, K in sorted(k_map.items(), key=lambda x: x[1]):
            row = marg[(marg['mechanism'] == label) & (marg['delta'] == delta)]
            if len(row):
                xs.append(K)
                ys.append(row['mean_welfare'].values[0])
        ax.plot(xs, ys, '-o', color=cmap[j], lw=2.2, ms=8, label=f'δ = {delta}')
    omn_w = marg[marg['mechanism'] == 'Omniscient']['mean_welfare'].iloc[0]
    ax.axhline(omn_w, color='black', linestyle=':', lw=1.5, label='omniscient')
    for j, delta in enumerate(DELTAS):
        bat_w = marg[(marg['mechanism'] == 'Batching m=5')
                     & (marg['delta'] == delta)]['mean_welfare'].iloc[0]
        ax.axhline(bat_w, color=cmap[j], linestyle='--', lw=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 100])
    ax.set_xticklabels(['2 (OPTN)', '5', '10', '∞'])
    ax.set_xlabel('Number of buckets K')
    ax.set_ylabel('Mean welfare per marginal kidney')
    ax.set_title('Marginal kidney welfare\n(dashed = batching reference)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # Panel C: mean position vs K
    ax = axes[2]
    for j, delta in enumerate(DELTAS):
        xs, ys = [], []
        for label, K in sorted(k_map.items(), key=lambda x: x[1]):
            row = marg[(marg['mechanism'] == label) & (marg['delta'] == delta)]
            if len(row):
                xs.append(K)
                ys.append(row['mean_position'].values[0])
        ax.plot(xs, ys, '-o', color=cmap[j], lw=2.2, ms=8, label=f'δ = {delta}')
    omn_p = marg[marg['mechanism'] == 'Omniscient']['mean_position'].iloc[0]
    ax.axhline(omn_p, color='black', linestyle=':', lw=1.5,
                label=f'omniscient ({omn_p:.0f})')
    for j, delta in enumerate(DELTAS):
        bat_p = marg[(marg['mechanism'] == 'Batching m=5')
                     & (marg['delta'] == delta)]['mean_position'].iloc[0]
        ax.axhline(bat_p, color=cmap[j], linestyle='--', lw=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 100])
    ax.set_xticklabels(['2 (OPTN)', '5', '10', '∞'])
    ax.set_xlabel('Number of buckets K')
    ax.set_ylabel('Mean queue position at allocation')
    ax.set_title('Allocation depth for marginal kidneys\n(dashed = batching reference)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    fig.suptitle('Coarsening effect on marginal kidneys: discard, welfare, depth',
                 fontsize=13, y=1.04)
    fig.savefig(os.path.join(OUTPUT_DIR, 'coarsening_marginal.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_overall_vs_marginal():
    """Compare overall (all kidneys) vs marginal kidneys welfare and discard.
    Shows that marginal kidneys are where the action is."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Panel A: discard rate, marginal vs all kidneys
    ax = axes[0]
    width = 0.35
    x = np.arange(len(MECHS))
    delta_focus = 1.0  # use most extreme lemons
    marg_y = []
    all_y = []
    for mech in MECHS:
        m_row = marg[(marg['mechanism'] == mech) & (marg['delta'] == delta_focus)]
        a_row = allk[(allk['mechanism'] == mech) & (allk['delta'] == delta_focus)]
        marg_y.append(m_row['discard_rate'].values[0] if len(m_row) else np.nan)
        all_y.append(a_row['discard_rate'].values[0] if len(a_row) else np.nan)
    ax.bar(x - width/2, all_y, width, color='steelblue',
            label='All kidneys', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, marg_y, width, color='tab:red',
            label='Marginal kidneys only', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n', 1) for m in MECHS],
                       rotation=0, fontsize=8)
    ax.set_ylabel('Discard rate')
    ax.set_title(f'Discard rate at δ = {delta_focus}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)

    # Panel B: welfare, marginal vs all kidneys
    ax = axes[1]
    marg_y = []
    all_y = []
    for mech in MECHS:
        m_row = marg[(marg['mechanism'] == mech) & (marg['delta'] == delta_focus)]
        a_row = allk[(allk['mechanism'] == mech) & (allk['delta'] == delta_focus)]
        marg_y.append(m_row['mean_welfare'].values[0] if len(m_row) else np.nan)
        all_y.append(a_row['mean_welfare'].values[0] if len(a_row) else np.nan)
    ax.bar(x - width/2, all_y, width, color='steelblue',
            label='All kidneys', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, marg_y, width, color='tab:red',
            label='Marginal kidneys only', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n', 1) for m in MECHS],
                       rotation=0, fontsize=8)
    ax.set_ylabel('Mean welfare per kidney')
    ax.set_title(f'Welfare per kidney at δ = {delta_focus}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    fig.suptitle(f'All kidneys vs marginal kidneys: marginal is where mechanisms differ',
                 fontsize=13, y=1.04)
    fig.savefig(os.path.join(OUTPUT_DIR, 'marginal_vs_all.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_marginal_discard_welfare()
    plot_marginal_recipient_match()
    plot_coarsening_marginal()
    plot_overall_vs_marginal()
    print('Marginal plots written.')
