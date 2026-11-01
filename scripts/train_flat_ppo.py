"""
Train and evaluate the Flat PPO ablation (revision 1)
=====================================================
    python train_flat_ppo.py --seed 7                              # original Flat PPO, synthetic
    python train_flat_ppo.py /path/to/criteo --seed 7              # original Flat PPO, Criteo
    python train_flat_ppo.py /path/to/criteo --seed 7 --matched    # ABL-1: matched Flat PPO
    python train_flat_ppo.py ... --eval-only                       # reload checkpoint_best.pt and re-evaluate

Two configurations (D7: full specification)
-------------------------------------------
                         default ("original")               --matched  (ABL-1)
 network                 256-wide MLP, 113,287 parameters   416-wide MLP + categorical node head, ~284,800 parameters
 action                  3 continuous outputs; node =       2 continuous outputs (workload, allocation) +
                         int(a1 * 5)                        5-way categorical node head
 routing shaping         none                               same rule as HBA-RL (common.routing_shaping)
 KL threshold            0.01                               0.05
 early stopping          patience 300 EPISODES, checked     patience 600 episodes, checked every 50 episodes on the
                         after EVERY episode: the counter   50-episode mean reward; improvement must exceed 0.1
                         resets only when an episode's raw  (stop after 12 failed checks)  = HBA-RL's rule
                         reward beats the best single
                         episode so far. Stops are therefore
                         not multiples of 50 (feedback D7).
 checkpoint rule         raw reward of one episode (the     mean reward of the last 10 episodes at each update,
                         last of each 10), highest wins     smoothed over the last 10 updates = HBA-RL's rule
Both: PPO with GAE, clip 0.1, 4 epochs, lr 3e-4 cosine, entropy 0.05 -> 0.001, up to 2000 episodes.

Evaluation (A1, D8): identical to every other method - evaluation.evaluate(), 100 episodes, seeds 42 + e,
deterministic actions (mean / argmax), unshaped reward, Criteo EVALUATION split.

Extra switches: --tag, --arrival-prob (ABL-4), --legacy-shaping, --legacy-observation, --output.
"""

import os as _os
import sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_R] + [_os.path.join(_R, _d) for _d in ('env', 'agents', 'baselines')]
import argparse
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import parse_with_config, EVAL_SEED_BASE, N_EVAL, dump_json, make_env, routing_shaping, run_metadata
from evaluation import evaluate, log_path, slug
from flat_ppo import FlatPPOAgent
from seed_utils import get_env_seed, set_global_seed


def make_act_fn(agent):
    def act(state, env):
        action, _, _, _ = agent.select_action(state, deterministic=True)
        node = min(int(action[1] * env.num_nodes), env.num_nodes - 1)
        return action, node
    return act


class FlatPPOTrainer:
    def __init__(self, env, agent, dataset, seed, output_dir, method, num_episodes=2000, update_frequency=10,
                 patience=300, matched=False, legacy_shaping=False):
        self.env, self.agent = env, agent
        self.dataset, self.seed, self.method = dataset, seed, method
        self.output_dir = output_dir
        self.num_episodes, self.update_frequency, self.patience = num_episodes, update_frequency, patience
        self.matched, self.legacy_shaping = matched, legacy_shaping
        os.makedirs(output_dir, exist_ok=True)
        self.rows = []
        self.best_episode = None
        self.stop_reason = 'reached episode cap'
        self._best_ma, self._no_improve_checks = -np.inf, 0           # matched rule
        self._best_single, self._episodes_no_improve = -np.inf, 0     # original rule

    def train(self):
        t0 = time.time()
        print(f"\n{'=' * 60}\n  Flat PPO | {self.method} | {self.dataset} | seed {self.seed}\n{'=' * 60}")
        print(f"  episodes <= {self.num_episodes} | patience {self.patience} | "
              f"{'MATCHED (ABL-1)' if self.matched else 'original'} | parameters {self.agent.n_parameters():,}")

        for episode in tqdm(range(1, self.num_episodes + 1), desc="Training"):
            state, _ = self.env.reset(seed=get_env_seed(self.seed, episode))
            ep_reward = ep_shaping = 0.0
            done = False
            route = np.zeros(self.env.num_nodes, dtype=int)
            while not done:
                action, raw, log_prob, value = self.agent.select_action(state)
                node = min(int(action[1] * self.env.num_nodes), self.env.num_nodes - 1)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                shaped = routing_shaping(self.env, node, legacy=self.legacy_shaping) if self.matched else 0.0
                reward += shaped

                self.agent.store_transition(state, raw, log_prob, value, reward, done)
                ep_reward += reward
                ep_shaping += shaped
                route[node] += 1
                state = next_state

            row = {'episode': episode, 'reward_total': ep_reward, 'reward_shaping': ep_shaping,
                   'reward_env': ep_reward - ep_shaping, 'revenue': info['total_revenue'],
                   'penalties': info['total_penalties'], 'cost': info['total_cost'], 'profit': info['profit'],
                   'violations': info['sla_violations'], 'sla_pct': 100 * info['sla_compliance'],
                   'completed': info['n_ontime'], 'failed_placements': info['failed_placements']}
            row.update(info['reward_parts'])
            row.update({f'route_n{i}': int(v) for i, v in enumerate(route)})
            self.rows.append(row)

            if episode % self.update_frequency == 0:
                before = self.agent.best_reward
                if self.matched:    # HBA-RL's checkpoint input: mean of the last 10 episodes
                    score = float(np.mean([r['reward_total'] for r in self.rows[-self.update_frequency:]]))
                else:               # original: the single last episode
                    score = ep_reward
                self.agent.update_policy(episode_reward=score)
                if self.agent.best_reward > before:
                    self.best_episode = episode

            # ── Early stopping ───────────────────────────────────────
            if self.matched:
                if episode % 50 == 0:
                    ma = float(np.mean([r['reward_total'] for r in self.rows[-50:]]))
                    if ma > self._best_ma + 0.1:
                        self._best_ma, self._no_improve_checks = ma, 0
                    else:
                        self._no_improve_checks += 1
                    if self._no_improve_checks >= self.patience // 50:
                        self.stop_reason = f'early stop: no improvement for {self.patience} episodes (50-episode mean)'
                        print(f"\n[Early Stop] {self.stop_reason}")
                        break
            else:
                if ep_reward > self._best_single:
                    self._best_single, self._episodes_no_improve = ep_reward, 0
                else:
                    self._episodes_no_improve += 1
                if self._episodes_no_improve >= self.patience:
                    self.stop_reason = (f'early stop: {self.patience} episodes without a new best single-episode '
                                        f'reward (stopped at episode {episode})')
                    print(f"\n[Early Stop] {self.stop_reason}")
                    break

            if episode % 50 == 0:
                last = pd.DataFrame(self.rows[-50:])
                tqdm.write(f"\nEpisode {episode}/{self.num_episodes} | reward {last.reward_total.mean():.2f} | "
                           f"profit ${last.profit.mean():,.0f} | penalties ${last.penalties.mean():,.0f} | "
                           f"cost ${last.cost.mean():,.1f} | SLA {last.sla_pct.mean():.1f}% | "
                           f"entropy coef {self.agent.entropy_coef:.4f}")

        self.minutes = (time.time() - t0) / 60
        self.agent.restore_best()
        self.agent.save_model(os.path.join(self.output_dir, 'checkpoint_best.pt'))
        os.makedirs(os.path.join('results', 'logs', 'training'), exist_ok=True)
        path = os.path.join('results', 'logs', 'training',
                            f"train_{self.dataset}_{slug(self.method)}_seed{self.seed}.csv")
        pd.DataFrame(self.rows).to_csv(path, index=False)
        print(f"  [Training log: {path}]")


def evaluate_agent(agent, dataset, criteo_path, seed, method, output_dir, arrival_prob, observe_all, extra=None):
    eval_env, _ = make_env(criteo_path, split='eval', arrival_prob=arrival_prob, observe_all_nodes=observe_all)
    split = 'eval' if dataset == 'criteo' else 'n/a'
    df = evaluate(eval_env, make_act_fn(agent), method, dataset, seed, split=split,
                  arrival_prob=arrival_prob, out_csv=log_path(method, dataset, seed))
    viol = df.violated + df.expired
    report = {
        'seed': seed, 'dataset': dataset, 'method': method, 'split': split,
        'episode_seeds': [EVAL_SEED_BASE, EVAL_SEED_BASE + N_EVAL - 1], 'actions': 'deterministic',
        'final_metrics': {
            'avg_profit': df.profit.mean(), 'std_profit': df.profit.std(ddof=1),
            'avg_revenue': df.revenue.mean(), 'avg_penalties': df.penalties.mean(), 'avg_cost': df.cost.mean(),
            'avg_completed': df.completed.mean(), 'avg_violated': df.violated.mean(), 'avg_expired': df.expired.mean(),
            'avg_sla_violations': viol.mean(), 'sla_compliance_pct': df.sla_pct.mean(),
            'avg_failed_placements': df.failed_placements.mean(), 'avg_cpu_util': df.mean_cpu_util.mean(),
            'avg_completion_time': df.mean_completion_time.mean()},
    }
    if extra:
        report.update(extra)
    dump_json(report, os.path.join(output_dir, 'flat_ppo_eval_results.json'))
    return df


def main():
    parser = argparse.ArgumentParser(description='Flat PPO ablation: train and evaluate')
    parser.add_argument('criteo_path', nargs='?', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--matched', action='store_true', help='ABL-1: capacity, categorical node head, shaping, KL, patience, checkpoint rule')
    parser.add_argument('--output', type=str, default='businessAware')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--arrival-prob', type=float, default=0.3, help='ABL-4')
    parser.add_argument('--legacy-shaping', action='store_true')
    parser.add_argument('--legacy-observation', action='store_true')
    parser.add_argument('--force', action='store_true', help='retrain even if this run already finished (default: skip it)')
    parser.add_argument('--eval-only', action='store_true')
    args = parse_with_config(parser)

    if args.arrival_prob != 0.3 and not args.tag:
        parser.error('--arrival-prob needs --tag so that results do not overwrite the main runs')
    observe_all = not args.legacy_observation
    base = 'Flat PPO (matched)' if args.matched else 'Flat PPO'
    method = f"{base} [{args.tag}]" if args.tag else base

    set_global_seed(args.seed)
    env, dataset = make_env(args.criteo_path, split='train', arrival_prob=args.arrival_prob,
                            observe_all_nodes=observe_all)
    root = os.path.join(args.output, 'ablations', args.tag) if args.tag else args.output
    output_dir = os.path.join(root, 'flat_ppo', ('matched_' if args.matched else '') + dataset, f'seed_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)

    agent = FlatPPOAgent(
        state_dim=50, lr=3e-4, gamma=0.99, gae_lambda=0.95, eps_clip=0.1, k_epochs=4, entropy_coef=0.05,
        entropy_coef_min=0.001, value_loss_coef=0.5, max_grad_norm=0.5,
        target_kl=0.05 if args.matched else 0.01, normalize_rewards=True, total_episodes=args.episodes,
        seed=args.seed, hidden_dim=416 if args.matched else 256, smoothed_best=args.matched,
        categorical_node=args.matched)

    config = {'method': method, 'dataset': dataset, 'seed': args.seed, 'matched': args.matched,
              'patience_episodes': 600 if args.matched else 300,
              'early_stopping': ('every 50 episodes on the 50-episode mean, improvement > 0.1' if args.matched
                                 else 'after every episode: counter resets only on a new best single-episode reward'),
              'shaping': args.matched, 'legacy_shaping': args.legacy_shaping, 'arrival_prob': args.arrival_prob,
              'observed_nodes': env.observe_nodes,
              'evaluation': {'episodes': N_EVAL, 'episode_seeds': f'{EVAL_SEED_BASE} + e', 'actions': 'deterministic',
                             'split': 'eval' if dataset == 'criteo' else 'n/a'},
              'hyperparameters': agent.hparams(), 'cli': vars(args)}

    if not args.eval_only and not args.force and os.path.exists(os.path.join(output_dir, 'training_summary.json')):
        print(f'Already trained: {output_dir} (use --force to retrain). Skipping.')
        return

    if args.eval_only:
        ckpt = os.path.join(output_dir, 'checkpoint_best.pt')
        if not os.path.exists(ckpt):
            raise SystemExit(f'--eval-only: no checkpoint at {ckpt}')
        agent.load_model(ckpt)
        evaluate_agent(agent, dataset, args.criteo_path, args.seed, method, output_dir, args.arrival_prob, observe_all)
        return

    trainer = FlatPPOTrainer(env, agent, dataset, args.seed, output_dir, method, num_episodes=args.episodes,
                             update_frequency=10, patience=600 if args.matched else 300,
                             matched=args.matched, legacy_shaping=args.legacy_shaping)
    trainer.train()

    config['software'] = run_metadata()
    dump_json(config, os.path.join(output_dir, 'run_config.json'))
    dump_json({'method': method, 'dataset': dataset, 'seed': args.seed, 'episodes_run': len(trainer.rows),
               'stop_reason': trainer.stop_reason, 'best_checkpoint_episode': trainer.best_episode,
               'wall_clock_minutes': round(trainer.minutes, 2)}, os.path.join(output_dir, 'training_summary.json'))

    print(f"\n  Evaluating the best checkpoint ({N_EVAL} episodes, seeds {EVAL_SEED_BASE}+e, deterministic) ...")
    evaluate_agent(agent, dataset, args.criteo_path, args.seed, method, output_dir, args.arrival_prob,
                   observe_all, extra={'training_episodes': len(trainer.rows)})
    print(f"\n  Results: {output_dir}\n  Per-episode logs: results/logs/")


if __name__ == '__main__':
    main()
