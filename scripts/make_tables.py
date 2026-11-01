"""
Tables R4 (routing share), R5 (training behaviour), R6 (ablations), M1 (hyper-parameters), M2 (environment
parameters) and the stop-episode columns of R2 (D3, D14, D15, D17, A13).

  R4  evaluation episodes ONLY (not training): share of macro routing decisions per node, and share of jobs
      actually placed per node, per agent, dataset and seed. Source: results/logs/eval_*.csv
  R5  episodes run, stop reason, best-checkpoint episode, wall-clock minutes per run.
      Source: businessAware/**/training_summary.json
  M1  hyper-parameters side by side. Source: businessAware/**/run_config.json (written by every training run).
      Rows whose values differ between columns are flagged.

    python scripts/make_tables.py
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

AGENTS = ['HBA-RL', 'Flat PPO', 'Flat PPO (matched)']

# Architecture rows that differ between the macro and the micro networks of HBA-RL (read from hrl_agent.py).
LEVEL_ROWS = {
    'network': ('50 -> 256 -> 256 -> 128 trunk; 12 subgoal logits + 5 routing logits + value',
                '62 (state 50 + subgoal 12) -> 128 -> 128 -> 64; 2 action means + value'),
    'initial_log_sd': ('n/a (categorical heads)', '0.0 (state-independent, clamped to [-3, 1])'),
    'one_policy_per': ('cluster', 'node (5 policies)'),
}


def table_r4(logs):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(logs, 'eval_*.csv')))]
    if not frames:
        return pd.DataFrame()
    ep = pd.concat(frames, ignore_index=True)
    ep = ep[ep.method.isin(AGENTS)]
    rows = []
    for (m, ds, s), d in ep.groupby(['method', 'dataset', 'seed']):
        r = d[[f'route_n{i}' for i in range(5)]].sum()
        p = d[[f'placed_n{i}' for i in range(5)]].sum()
        row = {'agent': m, 'dataset': ds, 'seed': s}
        for i in range(5):
            row[f'route_pct_n{i}'] = 100 * r.iloc[i] / max(r.sum(), 1)
        for i in range(5):
            row[f'placed_pct_n{i}'] = 100 * p.iloc[i] / max(p.sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def table_r5(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, '**', 'training_summary.json'), recursive=True)):
        d = json.load(open(f))
        rows.append({'agent': d['method'], 'dataset': d['dataset'], 'seed': d['seed'],
                     'episodes_run': d['episodes_run'], 'best_checkpoint_episode': d['best_checkpoint_episode'],
                     'wall_clock_min': d['wall_clock_minutes'], 'stop_reason': d['stop_reason'],
                     'folder': os.path.dirname(f)})
    return pd.DataFrame(rows)


def table_r6(logs, alpha_note=True):
    """Ablations (ABL-1..4): variant vs the matching HBA-RL reference; Welch t on seed means."""
    from scipy import stats
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(logs, 'eval_*.csv')))]
    if not frames:
        return pd.DataFrame()
    ep = pd.concat(frames, ignore_index=True)
    sm = ep.groupby(['dataset', 'method', 'seed']).profit.mean().reset_index()
    rows = []
    for ds in sm.dataset.unique():
        for m in sorted(sm[sm.dataset == ds].method.unique()):
            if m == 'HBA-RL' or not (m.startswith('Flat PPO (matched)') or '[' in m):
                continue
            if m.startswith('HBA-RL ['):
                if m.startswith('HBA-RL [abl4'):
                    continue                       # ABL-4 HBA-RL runs are the references for the load comparisons
                ref = 'HBA-RL'                      # ABL-2 / ABL-3 are compared with nominal-load HBA-RL
            elif m.startswith('Flat PPO (matched)'):
                ref = 'HBA-RL' + m[len('Flat PPO (matched)'):]     # same arrival probability / tag
            else:
                ref = 'HBA-RL' + m[len('Flat PPO'):]
            a = sm[(sm.dataset == ds) & (sm.method == ref)].profit.to_numpy()
            b = sm[(sm.dataset == ds) & (sm.method == m)].profit.to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            t = stats.ttest_ind(a, b, equal_var=False)
            rows.append({'variant': m, 'reference': ref, 'dataset': ds, 'n_seeds': len(b),
                         'profit_mean': b.mean(), 'sd_seed_means': b.std(ddof=1),
                         'reference_mean': a.mean(), 'gap_vs_reference': b.mean() - a.mean(),
                         'gap_pct': 100 * (b.mean() - a.mean()) / a.mean(), 'welch_t': t.statistic, 'p': t.pvalue})
    return pd.DataFrame(rows)


def table_m2():
    """Environment parameters, read from the code so the table cannot drift from it."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import common
    from common import make_env
    env = make_env(None)[0]
    rows = [('episode length (steps)', common.EPISODE_LENGTH), ('queue capacity', common.MAX_WORKLOADS),
            ('arrival probability per step', 0.3), ('number of nodes', common.NUM_NODES)]
    rows += [('nodes 0-1 (high-performance): CPU / GPU / memory (GB) / bandwidth, cost per hour',
              'U(32,64) / U(4,8) / U(128,256) / U(10,20), U(2.0,3.0)  [continuous]'),
             ('nodes 2-4 (standard): CPU / GPU / memory (GB) / bandwidth, cost per hour',
              'U(8,16) / U(2,4) / U(32,64) / U(5,10), U(0.5,1.0)  [continuous]'),
             ('operating cost accrual', 'cost_per_hour / 60 per running job per step'),
             ('synthetic jobs by model type (CPU, GPU, memory, duration)',
              'Transformer: U(8,16), U(2,4), U(32,64), U(10,20); CNN or RNN: U(4,8), U(1,2), U(16,32), U(5,15); LinearModel: U(2,4), U(0,1), U(8,16), U(3,8) (model type uniform)')]
    rows += [('deadline', 'duration x U(2, 4) steps'), ('priority', 'Beta(2, 5)'),
             ('synthetic revenue', 'Exponential(mean 500) x (1 + priority), NOT clipped'),
             ('SLA penalty constant', 'penalty = 0.2 x revenue; late job pays penalty x min(lateness ratio, 1), expired job pays the full penalty'),
             ('reward constants', '+2 x priority when scheduled; -1 failed placement; +revenue/1000 on time; -penalty/1000 late or expired; +0.5 x mean CPU utilisation'),
             ('training-only shaping (HBA-RL, matched Flat PPO)', 'admitted jobs only: +0.5 if p > 0.6 on node 0/1; -0.3 if p > 0.6 on node 2-4; +0.2 if p <= 0.6 on node 2-4')]
    return pd.DataFrame(rows, columns=['parameter', 'value'])


def table_m1(root):
    cols = {}
    for f in sorted(glob.glob(os.path.join(root, '**', 'run_config.json'), recursive=True)):
        c = json.load(open(f))
        if c['method'] in AGENTS and c['method'] not in cols:
            cols[c['method']] = c['hyperparameters']
    if not cols:
        return pd.DataFrame()
    keys = list(dict.fromkeys(k for h in cols.values() for k in h))
    rows = []
    for k in keys + [k for k in LEVEL_ROWS if k not in keys]:
        vals = {m: json.dumps(h.get(k, 'n/a'), default=str) if isinstance(h.get(k), (dict, list)) else h.get(k, 'n/a')
                for m, h in cols.items()}
        hba = vals.pop('HBA-RL', 'n/a')
        macro, micro = LEVEL_ROWS.get(k, (hba, hba))
        if k in LEVEL_ROWS:
            macro, micro = LEVEL_ROWS[k]
        row = {'hyperparameter': k, 'HBA-RL macro': macro if k in LEVEL_ROWS else hba,
               'HBA-RL micro': micro if k in LEVEL_ROWS else hba, **vals}
        row['differs_between_agents'] = len({str(v) for v in [hba, *vals.values()]}) > 1
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default=os.path.join('results', 'logs'))
    ap.add_argument('--root', default='businessAware')
    ap.add_argument('--out', default=os.path.join('results', 'tables'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option('display.width', 250)
    r5 = table_r5(args.root)
    r2_path = os.path.join(args.out, 'R2_per_seed.csv')
    if os.path.exists(r2_path) and not r5.empty:   # add the stop episode to R2 (spec: Agent|Dataset|Seed|Stop episode|...)
        r2 = pd.read_csv(r2_path).drop(columns=[c for c in ('stop_episode', 'best_checkpoint_episode') if c in pd.read_csv(r2_path).columns])
        m = r5.rename(columns={'agent': 'method', 'episodes_run': 'stop_episode'})[
            ['method', 'dataset', 'seed', 'stop_episode', 'best_checkpoint_episode']]
        r2.merge(m, on=['method', 'dataset', 'seed'], how='left').to_csv(r2_path, index=False)
        print('R2_per_seed: stop_episode and best_checkpoint_episode added')
    try:
        m2 = table_m2()
    except Exception as e:   # needs the environment importable
        print('M2 skipped:', e)
        m2 = pd.DataFrame()
    for name, df in (('R4_routing_share', table_r4(args.logs)), ('R5_training_behaviour', r5),
                     ('R6_ablations', table_r6(args.logs)), ('M1_hyperparameters', table_m1(args.root)),
                     ('M2_environment', m2)):
        if df.empty:
            print(f'{name}: no input found, skipped')
            continue
        df.to_csv(os.path.join(args.out, f'{name}.csv'), index=False)
        print(f'\n== {name} ==')
        print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
