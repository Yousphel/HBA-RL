"""
Train and evaluate HBA-RL (revision 1)
======================================
One seed per call:

    python train_hrl_scheduler.py --seed 7                          # synthetic
    python train_hrl_scheduler.py /path/to/criteo --seed 7          # Criteo
    python train_hrl_scheduler.py ... --eval-only                   # reload checkpoint_best.pt and re-evaluate

Ablations (ABL-2, ABL-3, ABL-4) - every ablation needs --tag so it cannot overwrite the main runs:
    --no-shaping            ABL-2  no routing-shaping term during training
    --zero-subgoal          ABL-3  micro-controllers receive g = 0
    --arrival-prob 0.6      ABL-4  heavier load (also pass the same value to the heuristics and Flat PPO)
    --tag ABL-2_no_shaping

Other switches
    --legacy-shaping        reproduce the earlier shaping rule (applied on every non-empty-queue step to the
                            latest running job). Default is the rule in Algorithm 1: admitted jobs only.
    --legacy-observation    observe nodes 0-2 only (the earlier state vector)

Evaluation protocol (A1, D8): after training the best checkpoint is restored and evaluated by
evaluation.evaluate(): 100 episodes, seeds 42 + e, deterministic actions, unshaped reward, on the Criteo
EVALUATION split (or the synthetic generator). Exactly the protocol used for the heuristics.

Outputs
    businessAware/<dataset>/seed_<N>/            (ablations: businessAware/ablations/<tag>/<dataset>/seed_<N>/)
        checkpoint_best.pt  run_config.json  training_summary.json  evaluation_report.json
    results/logs/eval_<dataset>_<method>_seed<N>.csv       per-episode evaluation log (all fields of D11)
    results/logs/training/train_<dataset>_<method>_seed<N>.csv   per-episode training log with reward components
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
from hrl_agent import HierarchicalRLAgent
from seed_utils import get_env_seed, set_global_seed


def make_act_fn(agent, deterministic=True):
    """(state, env) -> (env_action, routed node). Deterministic = argmax macro, mean micro."""
    def act(state, env):
        high = agent.select_high_level_action(state, deterministic=deterministic)
        node = high['node_idx']
        micro, _, _, _ = agent.select_low_level_action(state, high, deterministic=deterministic)
        env_action = np.array([micro[0], (node + 0.5) / env.num_nodes, micro[1]], dtype=np.float32)
        return env_action, node
    return act


class TrainingManager:
    def __init__(self, env, agent, dataset, seed, output_dir, method, num_episodes=2000,
                 update_frequency=10, patience=600, shaping=True, legacy_shaping=False):
        self.env, self.agent = env, agent
        self.dataset, self.seed, self.method = dataset, seed, method
        self.output_dir = output_dir
        self.num_episodes, self.update_frequency, self.patience = num_episodes, update_frequency, patience
        self.shaping, self.legacy_shaping = shaping, legacy_shaping
        os.makedirs(output_dir, exist_ok=True)
        self.rows = []                      # one dict per training episode
        self._best_ma, self._no_improve = -np.inf, 0
        self.best_episode = None
        self.stop_reason = 'reached episode cap'

    # ------------------------------------------------------------------
    def train(self):
        t0 = time.time()
        print(f"\n{'=' * 60}\n  HBA-RL | {self.method} | {self.dataset} | seed {self.seed}\n{'=' * 60}")
        print(f"  episodes <= {self.num_episodes} | update every {self.update_frequency} | patience {self.patience}"
              f" | shaping {'ON' if self.shaping else 'OFF'}"
              f"{' (legacy rule)' if self.shaping and self.legacy_shaping else ''}"
              f" | zero_subgoal {self.agent.zero_subgoal}")
        print(f"  parameters: {self.agent.n_parameters()}")

        for episode in tqdm(range(self.num_episodes), desc=f"seed={self.seed}"):
            state, _ = self.env.reset(seed=get_env_seed(self.seed, episode))
            ep_reward = ep_shaping = 0.0
            route = np.zeros(self.env.num_nodes, dtype=int)
            done = False
            while not done:
                high = self.agent.select_high_level_action(state)
                micro, raw, log_prob, value = self.agent.select_low_level_action(state, high)
                node = high['node_idx']
                env_action = np.array([micro[0], (node + 0.5) / self.env.num_nodes, micro[1]], dtype=np.float32)

                next_state, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated

                shaped = 0.0
                if self.shaping:
                    shaped = routing_shaping(self.env, node, legacy=self.legacy_shaping)
                reward += shaped

                self.agent.store_transition(state=state, high_action=high, low_action_raw=raw,
                                            low_log_prob=log_prob, low_value=value, reward=reward,
                                            done=float(done))
                ep_reward += reward
                ep_shaping += shaped
                route[node] += 1
                state = next_state

            row = {'episode': episode + 1, 'reward_total': ep_reward, 'reward_shaping': ep_shaping,
                   'reward_env': ep_reward - ep_shaping,
                   'revenue': info['total_revenue'], 'penalties': info['total_penalties'],
                   'cost': info['total_cost'], 'profit': info['profit'],
                   'violations': info['sla_violations'], 'sla_pct': 100 * info['sla_compliance'],
                   'completed': info['n_ontime'], 'failed_placements': info['failed_placements']}
            row.update(info['reward_parts'])
            row.update({f'route_n{i}': int(v) for i, v in enumerate(route)})
            self.rows.append(row)

            if (episode + 1) % self.update_frequency == 0:
                before = self.agent.best_reward
                recent = float(np.mean([r['reward_total'] for r in self.rows[-self.update_frequency:]]))
                self.agent.update_policies(episode_reward=recent)
                if self.agent.best_reward > before:
                    self.best_episode = episode + 1

            if (episode + 1) % 50 == 0:
                last = pd.DataFrame(self.rows[-50:])
                ma_r = last.reward_total.mean()
                r_pct = 100 * last[[f'route_n{i}' for i in range(self.env.num_nodes)]].sum() / max(
                    last[[f'route_n{i}' for i in range(self.env.num_nodes)]].values.sum(), 1)
                print(f"\n  ep {episode + 1:4d}/{self.num_episodes} | reward {ma_r:7.2f} (best smoothed "
                      f"{self.agent.best_reward:.2f}) | profit ${last.profit.mean():,.0f} | penalties "
                      f"${last.penalties.mean():,.0f} | cost ${last.cost.mean():,.1f} | SLA {last.sla_pct.mean():.1f}% | "
                      f"entropy coef {self.agent.entropy_coef:.4f}")
                print("    routing (training episodes): " + " ".join(f"n{i}={v:.0f}%" for i, v in enumerate(r_pct)))

                if ma_r > self._best_ma + 0.1:      # early-stopping rule of Algorithm 1 lines 23-26
                    self._best_ma, self._no_improve = ma_r, 0
                else:
                    self._no_improve += 1
                if self._no_improve >= self.patience // 50:
                    self.stop_reason = f'early stop: no improvement for {self.patience} episodes'
                    print(f"\n  [Early Stop] No improvement for {self.patience} episodes.")
                    break

        self.minutes = (time.time() - t0) / 60
        print("\n  Training complete.")
        self.agent.restore_best()
        self.save_checkpoint()
        self.save_training_log()

    # ------------------------------------------------------------------
    def save_checkpoint(self):
        self.agent.save_models(os.path.join(self.output_dir, 'checkpoint_best.pt'))

    def save_training_log(self):
        os.makedirs(os.path.join('results', 'logs', 'training'), exist_ok=True)
        path = os.path.join('results', 'logs', 'training',
                            f"train_{self.dataset}_{slug(self.method)}_seed{self.seed}.csv")
        pd.DataFrame(self.rows).to_csv(path, index=False)
        print(f"  [Training log: {path}]")


def evaluate_agent(agent, dataset, criteo_path, seed, method, output_dir, arrival_prob, observe_all, extra=None):
    """Final evaluation under the common protocol; writes the per-episode CSV and a JSON report."""
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
        'routing_share_eval_pct': (100 * df[[f'route_n{i}' for i in range(5)]].sum()
                                   / df[[f'route_n{i}' for i in range(5)]].values.sum()).round(2).tolist(),
    }
    if extra:
        report.update(extra)
    dump_json(report, os.path.join(output_dir, 'evaluation_report.json'))
    return df


def main():
    parser = argparse.ArgumentParser(description='HBA-RL: train and evaluate')
    parser.add_argument('criteo_path', nargs='?', default=None, help='Criteo folder (omit for synthetic)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--output', type=str, default='businessAware')
    parser.add_argument('--tag', type=str, default='', help='Ablation tag (required with ablation flags)')
    parser.add_argument('--no-shaping', action='store_true', help='ABL-2')
    parser.add_argument('--zero-subgoal', action='store_true', help='ABL-3')
    parser.add_argument('--arrival-prob', type=float, default=0.3, help='ABL-4')
    parser.add_argument('--legacy-shaping', action='store_true')
    parser.add_argument('--legacy-observation', action='store_true')
    parser.add_argument('--force', action='store_true', help='retrain even if this run already finished (default: skip it)')
    parser.add_argument('--eval-only', action='store_true', help='Skip training; load checkpoint_best.pt and evaluate')
    args = parse_with_config(parser)

    ablation = args.no_shaping or args.zero_subgoal or args.arrival_prob != 0.3
    if ablation and not args.tag:
        parser.error('ablation flags need --tag so that results do not overwrite the main runs')
    observe_all = not args.legacy_observation
    method = f"HBA-RL [{args.tag}]" if args.tag else 'HBA-RL'

    set_global_seed(args.seed)
    env, dataset = make_env(args.criteo_path, split='train', arrival_prob=args.arrival_prob,
                            observe_all_nodes=observe_all)

    root = os.path.join(args.output, 'ablations', args.tag) if args.tag else args.output
    output_dir = os.path.join(root, dataset, f'seed_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)

    agent = HierarchicalRLAgent(
        state_dim=50, num_nodes=5, high_level_lr=3e-4, low_level_lr=3e-4, gamma=0.99, gae_lambda=0.95,
        eps_clip=0.1, k_epochs=4, entropy_coef=0.05, entropy_coef_min=0.001, value_loss_coef=0.5,
        max_grad_norm=0.5, target_kl=0.05, normalize_rewards=True, total_episodes=args.episodes,
        seed=args.seed, zero_subgoal=args.zero_subgoal)

    config = {'method': method, 'dataset': dataset, 'seed': args.seed, 'episodes_cap': args.episodes,
              'patience_episodes': 600, 'patience_check_every': 50, 'patience_min_improvement': 0.1,
              'checkpoint_rule': 'mean of the last 10 update rewards (training reward INCLUDING shaping; no validation split)',
              'shaping': not args.no_shaping, 'legacy_shaping': args.legacy_shaping,
              'arrival_prob': args.arrival_prob, 'observed_nodes': env.observe_nodes,
              'training_episode_seeds': 'seed_utils.get_env_seed(seed, episode)',
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
        agent.load_models(ckpt)
        evaluate_agent(agent, dataset, args.criteo_path, args.seed, method, output_dir,
                       args.arrival_prob, observe_all)
        return

    trainer = TrainingManager(env, agent, dataset, args.seed, output_dir, method, num_episodes=args.episodes,
                              update_frequency=10, patience=600, shaping=not args.no_shaping,
                              legacy_shaping=args.legacy_shaping)
    trainer.train()

    config['software'] = run_metadata()
    dump_json(config, os.path.join(output_dir, 'run_config.json'))
    summary = {'method': method, 'dataset': dataset, 'seed': args.seed, 'episodes_run': len(trainer.rows),
               'stop_reason': trainer.stop_reason, 'best_checkpoint_episode': trainer.best_episode,
               'wall_clock_minutes': round(trainer.minutes, 2), 'routing_counts_training': agent.routing_counts.tolist()}
    dump_json(summary, os.path.join(output_dir, 'training_summary.json'))

    print(f"\n  Evaluating the best checkpoint ({N_EVAL} episodes, seeds {EVAL_SEED_BASE}+e, deterministic) ...")
    evaluate_agent(agent, dataset, args.criteo_path, args.seed, method, output_dir, args.arrival_prob,
                   observe_all, extra={'training_episodes': len(trainer.rows)})
    print(f"\n  Results: {output_dir}\n  Per-episode logs: results/logs/")


if __name__ == '__main__':
    main()
