#!/usr/bin/env python3
"""Export the per-episode training rewards of the three learned agents to one long table.

Reads  results/logs/training/train_<dataset>_<agent>_seed<N>.csv  (column reward_total)
Writes results/tables/training_rewards_long.csv  with columns agent, dataset, seed, episode, reward.
plot_training_curves.py draws Fig. 3 from this table.
"""
import argparse, glob, os
import pandas as pd

AGENTS = {'HBA_RL': 'HBA-RL', 'Flat_PPO': 'Flat PPO', 'Flat_PPO__matched': 'Flat PPO (matched)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='results/logs/training')
    ap.add_argument('--out', default='results/tables/training_rewards_long.csv')
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.logs, 'train_*.csv'))):
        base = os.path.basename(f)[len('train_'):-len('.csv')]
        dataset = base.split('_')[0]
        seed = int(base.rsplit('_seed', 1)[1])
        label = base[len(dataset) + 1:base.rindex('_seed')]
        agent = AGENTS.get(label)
        if agent is None:               # ablation and load-sensitivity runs are not part of Fig. 3
            continue
        d = pd.read_csv(f)
        ep = d['episode'].to_numpy() if 'episode' in d.columns else range(1, len(d) + 1)
        rows.append(pd.DataFrame({'agent': agent, 'dataset': dataset, 'seed': seed,
                                  'episode': ep, 'reward': d['reward_total'].to_numpy()}))
    out = pd.concat(rows)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'wrote {args.out}: {len(out)} rows, {out.groupby(["agent", "dataset"]).seed.nunique().to_dict()}')


if __name__ == '__main__':
    main()
