"""
Common evaluation protocol (revision 1; fixes A1, A3, D8, D11).

EVERY method (six heuristics, HBA-RL, Flat PPO, matched Flat PPO, ablations) is
evaluated by `evaluate()` below:
  * the same environment class and, for Criteo, the same evaluation split;
  * the same 100 episode seeds, 42 + e for e = 0..99;
  * the same job stream per episode seed (the environment generates it at reset());
  * deterministic actions for learned agents (macro: argmax of each head; micro and
    flat policy: the mean action) - see the act_fn built in each training script;
  * unshaped environment reward and the unified profit / SLA definitions.
One CSV per (method, dataset, seed) is written to results/logs/ with one row per episode.
"""
import os
import time

import numpy as np
import pandas as pd

from common import EVAL_SEED_BASE, N_EVAL

LOG_DIR = os.path.join('results', 'logs')


def slug(text):
    return ''.join(c if c.isalnum() else '_' for c in text).strip('_')


def log_path(method, dataset, seed, tag=None, root=LOG_DIR):
    name = f"eval_{dataset}_{slug(method)}_seed{seed}.csv"
    return os.path.join(root, name)


def run_episode(env, act_fn, episode_seed, on_start=None):
    """Play one episode. act_fn(state, env) -> (env_action, routed_node_index)."""
    state, _ = env.reset(seed=episode_seed)
    if on_start is not None:
        on_start()
    done, ep_reward = False, 0.0
    route = np.zeros(env.num_nodes, dtype=int)
    info = {}
    while not done:
        action, node = act_fn(state, env)
        state, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        route[node] += 1
        done = terminated or truncated
    return info, ep_reward, route


def episode_row(method, dataset, split, seed, episode_seed, arrival_prob, info, ep_reward, route):
    row = {
        'method': method, 'dataset': dataset, 'split': split, 'seed': seed,
        'episode_seed': episode_seed, 'arrival_prob': arrival_prob,
        'revenue': info['total_revenue'], 'penalties': info['total_penalties'],
        'cost': info['total_cost'], 'profit': info['profit'],
        'completed': info['n_ontime'],            # finished on time
        'violated': info['n_late'],               # finished late
        'expired': info['n_expired'],             # dropped from the queue
        'sla_pct': 100.0 * info['sla_compliance'],
        'failed_placements': info['failed_placements'],
        'dropped_arrivals': info['dropped_arrivals'], 'arrivals': info['n_arrivals'],
        'mean_cpu_util': info['mean_cpu_util'], 'mean_completion_time': info['mean_completion_time'],
        'reward': ep_reward,
    }
    for k, v in info['reward_parts'].items():
        row[k] = v
    for i, v in enumerate(route):
        row[f'route_n{i}'] = int(v)                          # macro routing DECISIONS per node
    for i, v in enumerate(info['placed_by_node']):
        row[f'placed_n{i}'] = int(v)                         # jobs actually placed per node
    for r, v in info['offered_load'].items():
        row[f'offered_{r}'] = v
    return row


def evaluate(env, act_fn, method, dataset, seed, split='-', episodes=N_EVAL, base_seed=EVAL_SEED_BASE,
             arrival_prob=0.3, on_start=None, out_csv=None, verbose=True):
    """Evaluate one policy; returns the per-episode DataFrame (and writes it if out_csv is set)."""
    rows, t0 = [], time.time()
    for e in range(episodes):
        ep_seed = base_seed + e
        info, ep_reward, route = run_episode(env, act_fn, ep_seed, on_start)
        rows.append(episode_row(method, dataset, split, seed, ep_seed, arrival_prob, info, ep_reward, route))
    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        df.to_csv(out_csv, index=False)
    if verbose:
        print(f"  [{method} | {dataset} | seed {seed}] {episodes} episodes in {time.time() - t0:.0f}s | "
              f"profit {df.profit.mean():,.0f} (episode SD {df.profit.std(ddof=1):,.0f}) | "
              f"SLA {df.sla_pct.mean():.2f}% | late+expired "
              f"{(df.violated + df.expired).mean():.2f} | wrote {out_csv}")
    return df
