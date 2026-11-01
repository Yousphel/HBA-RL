"""
Table M3 (mapped workload attributes per dataset and split) and the load level (A6, A16, D4).

For each dataset/split it generates N episodes with the environment itself (episode seeds 42 + e for the
evaluation split, 1000 + e for the training split) and reports, over every job that arrived:
    median, quartiles, IQR, coefficient of variation, and the share of jobs sitting at the lower / upper
    clip bound (Criteo only; the synthetic generator has no clip).
It also reports the OFFERED LOAD per resource: sum over arrived jobs of (demand x duration) divided by
the cluster's capacity x episode length. Values near 0 mean the cluster is under-loaded (A16).

    python scripts/workload_stats.py                      # synthetic only
    python scripts/workload_stats.py /path/to/criteo      # synthetic + Criteo train + Criteo eval
    python scripts/workload_stats.py /path/to/criteo --arrival-probs 0.3 0.6 0.9   # load for ABL-4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from common import make_env

ATTRS = ['value', 'penalty', 'duration', 'cpu', 'gpu', 'memory', 'slack', 'priority']
LABELS = {'value': 'Business value ($)', 'penalty': 'SLA penalty ($)', 'duration': 'Duration (steps)',
          'cpu': 'CPU demand', 'gpu': 'GPU demand', 'memory': 'Memory demand (GB)',
          'slack': 'Deadline slack (x duration)', 'priority': 'Priority'}


def collect(env, seeds):
    rows, loads = [], []
    for s in seeds:
        env.reset(seed=s)
        cap = {r: sum(n.total_resources[r] for n in env.nodes) * env.episode_length for r in ('cpu', 'gpu', 'memory')}
        offered = {r: 0.0 for r in cap}
        for t, j in env._schedule.items():
            rows.append({'value': j.revenue_value, 'penalty': j.penalty_cost, 'duration': j.estimated_duration,
                         'cpu': j.resource_requirements['cpu'], 'gpu': j.resource_requirements['gpu'],
                         'memory': j.resource_requirements['memory'],
                         'slack': j.original_deadline / max(j.estimated_duration, 1e-9), 'priority': j.priority})
            for r in offered:
                offered[r] += j.resource_requirements[r] * j.estimated_duration
        loads.append({**{f'offered_{r}': offered[r] / cap[r] for r in cap}, 'arrivals': len(env._schedule)})
    return pd.DataFrame(rows), pd.DataFrame(loads)


def describe(df, dataset, split, bounds):
    out = []
    for a in ATTRS:
        x = df[a].to_numpy()
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        lo, hi = bounds.get(a, (None, None))
        eps = 1e-9
        out.append({'dataset': dataset, 'split': split, 'attribute': a, 'n_jobs': len(x),
                    'median': med, 'q25': q1, 'q75': q3, 'iqr': q3 - q1,
                    'cv': x.std(ddof=1) / x.mean() if x.mean() else np.nan,
                    'pct_at_lower_clip': 100 * np.mean(x <= lo + eps) if lo is not None else np.nan,
                    'pct_at_upper_clip': 100 * np.mean(x >= hi - eps) if hi is not None else np.nan})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('criteo_path', nargs='?', default=None)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--arrival-probs', type=float, nargs='+', default=[0.3])
    ap.add_argument('--out', default=os.path.join('results', 'tables'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from criteo import CLIP
    crit = {k: CLIP[k] for k in ('value', 'duration', 'gpu', 'cpu', 'memory', 'priority')}

    stats, load = [], []
    for p in args.arrival_probs:
        jobs = [('synth', 'n/a', make_env(None, arrival_prob=p)[0], {}, 42)]
        if args.criteo_path:
            for split, base in (('train', 1000), ('eval', 42)):
                jobs.append(('criteo', split, make_env(args.criteo_path, split=split, arrival_prob=p)[0], crit, base))
        for ds, split, env, bounds, base in jobs:
            df, ld = collect(env, [base + e for e in range(args.episodes)])
            if p == args.arrival_probs[0]:
                stats += describe(df, ds, split, bounds)
            row = {'dataset': ds, 'split': split, 'arrival_prob': p, 'episodes': args.episodes,
                   'mean_arrivals': ld.arrivals.mean()}
            for r in ('cpu', 'gpu', 'memory'):
                row[f'offered_load_{r}_mean'] = ld[f'offered_{r}'].mean()
                row[f'offered_load_{r}_max'] = ld[f'offered_{r}'].max()
            if ds == 'criteo':
                data = env.criteo_loader.load_sample()
                row['click_rate_pct'] = 100 * float(data['label'].mean())
                row['rows'] = len(data)
            load.append(row)

    m3 = pd.DataFrame(stats)
    m3.to_csv(os.path.join(args.out, 'M3_workload_attributes.csv'), index=False)
    pd.DataFrame(load).to_csv(os.path.join(args.out, 'M3_load_level.csv'), index=False)
    pd.set_option('display.width', 220)
    print(m3.round(3).to_string(index=False))
    print('\nOffered load (1.0 = cluster fully busy if jobs ran back to back):')
    print(pd.DataFrame(load).round(3).to_string(index=False))

    if args.criteo_path:   # histograms of the mapped Criteo attributes (support for Table M3)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        env = make_env(args.criteo_path, split='eval')[0]
        df, _ = collect(env, [42 + e for e in range(args.episodes)])
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        for ax, a in zip(axes.ravel(), ATTRS):
            ax.hist(df[a], bins=40, color='#4C72B0')
            ax.set_xlabel(LABELS[a])
            ax.set_ylabel('Jobs')
            if a in ('value', 'penalty'):
                ax.set_yscale('log')
        plt.tight_layout()
        os.makedirs(os.path.join('results', 'figures'), exist_ok=True)
        plt.savefig(os.path.join('results', 'figures', 'M3_criteo_attribute_histograms.pdf'))
        print('\nWrote results/figures/M3_criteo_attribute_histograms.pdf')


if __name__ == '__main__':
    main()
