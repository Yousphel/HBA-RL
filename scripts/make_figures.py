"""
Paper figures as vector PDFs (section 11.3 of the revision guide).

  fig3_training_curves.pdf   2 x 2 panels (agent x dataset): mean across seeds, shaded min-max band, markers for
                             the mean best-checkpoint episode and for early stops. Reward is the TRAINING reward
                             as logged (environment reward + routing shaping for HBA-RL / matched Flat PPO);
                             the second figure shows the environment reward only (reward_env).
  fig4_profit.pdf            one panel per dataset, all methods; bars = mean, dots = seed means (learned agents).
                             Error bars: SD across seed means (learned agents) / per-episode SD (heuristics).
  fig_routing_share.pdf      evaluation-only routing share per node (R4)

No in-plot titles, consistent method colours, sentence-case axis labels with units.
    python scripts/make_figures.py
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'HBA-RL': '#1f4e79', 'Flat PPO': '#c0504d', 'Flat PPO (matched)': '#e8a13b',
          'First-Fit': '#9bbb59', 'Best-Fit': '#8064a2', 'Priority-Based': '#4bacc6', 'Cost-Aware': '#f79646',
          'SJF': '#7f7f7f', 'Round-Robin': '#bfbfbf'}
LEARNED = ['HBA-RL', 'Flat PPO', 'Flat PPO (matched)']
DS_NAME = {'synth': 'Synthetic', 'criteo': 'Criteo evaluation split'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def load_eval(logs):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(logs, 'eval_*.csv')))]
    if not frames:
        raise SystemExit('No eval logs found.')
    ep = pd.concat(frames, ignore_index=True)
    ep['profit'] = ep.revenue - ep.penalties - ep.cost
    return ep


def fig_profit(ep, out):
    dss = [d for d in ('synth', 'criteo') if d in set(ep.dataset)]
    fig, axes = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 4), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        d = ep[ep.dataset == ds]
        methods = [m for m in COLORS if m in set(d.method)]
        for i, m in enumerate(methods):
            dm = d[d.method == m]
            per_seed = dm.groupby('seed').profit.mean()
            if m in LEARNED and len(per_seed) > 1:
                mean, err = per_seed.mean(), per_seed.std(ddof=1)
                ax.scatter(np.full(len(per_seed), i) + np.linspace(-0.15, 0.15, len(per_seed)), per_seed.values,
                           s=14, color='black', zorder=3)
            else:
                mean, err = dm.profit.mean(), dm.profit.std(ddof=1)
            ax.bar(i, mean, color=COLORS[m], alpha=0.85)
            ax.errorbar(i, mean, yerr=err, color='black', capsize=3, lw=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha='right')
        ax.set_ylabel('Profit per episode ($)')
        ax.text(0.02, 0.97, f'({chr(97 + dss.index(ds))})', transform=ax.transAxes, va='top', fontsize=9, fontweight='bold')   # panel letter; meaning in the caption
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig4_profit.pdf'))
    plt.close()


def fig_routing(ep, out):
    d = ep[ep.method.isin(LEARNED)]
    keys = [(m, ds) for ds in ('synth', 'criteo') for m in LEARNED if ((d.method == m) & (d.dataset == ds)).any()]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(1.4 * len(keys) + 2, 3.8))
    bottoms = np.zeros(len(keys))
    cmap = plt.get_cmap('Blues')
    for n in range(5):
        vals = []
        for m, ds in keys:
            x = d[(d.method == m) & (d.dataset == ds)]
            tot = x[[f'route_n{i}' for i in range(5)]].values.sum()
            vals.append(100 * x[f'route_n{n}'].sum() / max(tot, 1))
        ax.bar(range(len(keys)), vals, bottom=bottoms, color=cmap(0.25 + 0.15 * n), edgecolor='white',
               label=f'Node {n}' + (' (high-performance)' if n < 2 else ''))
        bottoms += np.array(vals)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f'{m}\n{DS_NAME[ds].split()[0]}' for m, ds in keys], fontsize=8)
    ax.set_ylabel('Routing decisions (%)')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig_routing_share.pdf'))
    plt.close()


def fig_training(logs, root, out):
    files = glob.glob(os.path.join(logs, 'training', 'train_*.csv'))
    if not files:
        print('No training logs; skipping fig3.')
        return
    runs = []
    for f in files:
        d = pd.read_csv(f)
        base = os.path.basename(f)[6:-4]
        ds = base.split('_')[0]
        seed = int(base.split('_seed')[-1])
        label = base[len(ds) + 1: base.rindex('_seed')]
        method = {'HBA_RL': 'HBA-RL', 'Flat_PPO': 'Flat PPO', 'Flat_PPO__matched_': 'Flat PPO (matched)'}.get(label)
        if method:
            runs.append((method, ds, seed, d))
    best = {}
    for f in glob.glob(os.path.join(root, '**', 'training_summary.json'), recursive=True):
        s = json.load(open(f))
        best[(s['method'], s['dataset'], s['seed'])] = s.get('best_checkpoint_episode')
    for col, suffix, ylabel in (('reward_total', '', 'Training reward per episode (50-episode mean)'),
                                ('reward_env', '_env', 'Environment reward per episode (50-episode mean)')):
        agents = [a for a in LEARNED if any(r[0] == a for r in runs)]
        dss = [d for d in ('synth', 'criteo') if any(r[1] == d for r in runs)]
        fig, axes = plt.subplots(len(agents), len(dss), figsize=(5 * len(dss), 3 * len(agents)), squeeze=False,
                                 sharey='col')
        for i, a in enumerate(agents):
            for j, ds in enumerate(dss):
                ax = axes[i][j]
                sel = [(s, d) for (m, dd, s, d) in runs if m == a and dd == ds]
                if not sel:
                    ax.axis('off'); continue
                n = max(len(d) for _, d in sel)
                grid = np.full((len(sel), n), np.nan)
                for k, (s, d) in enumerate(sel):
                    grid[k, :len(d)] = d[col].rolling(50, min_periods=10).mean().values
                    ax.axvline(len(d), color=COLORS[a], alpha=0.25, lw=0.8)             # early stop / end of run
                    b = best.get((a, ds, s))
                    if b:
                        ax.plot(b, grid[k, min(b, len(d)) - 1], marker='o', ms=3.5, color='black', zorder=4)
                x = np.arange(1, n + 1)
                ax.fill_between(x, np.nanmin(grid, 0), np.nanmax(grid, 0), color=COLORS[a], alpha=0.2)
                ax.plot(x, np.nanmean(grid, 0), color=COLORS[a], lw=1.5)
                ax.set_xlabel('Training episode')
                ax.set_ylabel(ylabel if j == 0 else '')
                ax.text(0.02, 0.96, f'({chr(97 + i * len(dss) + j)})', transform=ax.transAxes, va='top', fontsize=9,
                        fontweight='bold')   # panel letter; meaning in figure_captions.md
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'fig3_training_curves{suffix}.pdf'))
        plt.close()


def write_captions(ep, out):
    """Caption drafts that state the error type and the panel mapping (A4, A11, S7)."""
    dss = [d for d in ('synth', 'criteo') if d in set(ep.dataset)]
    k = ep[ep.method == 'HBA-RL'].groupby('dataset').seed.nunique().to_dict()
    panels = ', '.join(f'({chr(97 + i)}) {DS_NAME[d].lower()}' for i, d in enumerate(dss))
    n = ', '.join(f'{DS_NAME[d].split()[0].lower()} n = {k.get(d, "?")}' for d in dss)
    txt = (f'Fig. 4  Mean profit per episode for every method on {panels}. Bars are means; dots are individual training '
           f'seeds (learned agents; {n}). Error bars show the SD across seed means for learned agents and the '
           f'per-episode SD over 100 evaluation episodes for heuristics (deterministic, one run per episode seed).\n\n'
           f'Fig. 3  Training reward (50-episode moving mean) by agent and dataset; panel order: ' +
           ', '.join(f'({chr(97 + i * len(dss) + j)}) agent {i + 1} / {"synthetic" if d == "synth" else "Criteo training split"}' for i in range(3) for j, d in enumerate(dss)) +
           '. Lines are means across seeds, bands the min-max across seeds, vertical lines the stop episode of each run, '
           'dots the best-checkpoint episode. fig3_training_curves.pdf shows the logged training reward (environment '
           'reward plus routing shaping where used); fig3_training_curves_env.pdf shows the environment reward only. '
           'Agent order (rows): HBA-RL, Flat PPO, Flat PPO (matched). Neither plot uses normalised reward.\n')
    open(os.path.join(out, 'figure_captions.md'), 'w').write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default=os.path.join('results', 'logs'))
    ap.add_argument('--root', default='businessAware')
    ap.add_argument('--out', default=os.path.join('results', 'figures'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ep = load_eval(args.logs)
    fig_profit(ep, args.out)
    fig_routing(ep, args.out)
    fig_training(args.logs, args.root, args.out)
    write_captions(ep, args.out)
    print('Figures written to', args.out)


if __name__ == '__main__':
    main()
