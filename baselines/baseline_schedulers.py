"""
Heuristic baselines (revision 1)
================================
Six non-learning schedulers, evaluated by the SAME protocol as the learned agents
(evaluation.py): same environment, same Criteo evaluation split, the same 100
episode seeds (42 + e), unified profit and SLA definitions.

Fixes for the supervisor's feedback
-----------------------------------
A1  Criteo defaults to --split eval (the last 40,000 rows). The old default
    (train) is still selectable for diagnostics but prints a warning: those results
    must never be put in a table with the learned agents.
A2/A3/D11  Per-episode logs contain penalties, completed / violated / expired
    counts, failed placements, CPU utilisation and completion time.
D6  Each scheduler documents job ordering, no-fit behaviour and tie-breaking below.

Common behaviour of all six (D6)
--------------------------------
* One decision per step: the scheduler picks ONE (job, node) pair, as the agents do.
* Candidate jobs are scanned in the ORDER STATED for each scheduler; the first
  (job, node) pair that fits (all of CPU, GPU, memory, bandwidth) is submitted.
* NO-FIT: if no queued job fits any node, the scheduler submits job 0 to node 0.
  This placement fails (reward -1, counted in failed_placements). With an empty
  queue the step is a no-op.
* Allocation fraction is always 1.0 (the environment ignores it).
* Heuristics are deterministic: seed is logged as 0, and the only variation across
  episodes comes from the episode seed (node capacities, job stream).

Usage
-----
    python baseline_schedulers.py --mode synth
    python baseline_schedulers.py --mode criteo /path/to/criteo               # eval split
    python baseline_schedulers.py --mode synth --arrival-prob 0.6 --tag ABL-4_p0.6
"""

import os as _os
import sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_R] + [_os.path.join(_R, _d) for _d in ('env', 'agents', 'baselines')]
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import parse_with_config, EVAL_SEED_BASE, N_EVAL, dump_json, make_env, run_metadata
from evaluation import evaluate, log_path
from hrl_scheduler_simulation import MLSchedulingEnvironment


# ==================== Scheduler base ====================

class BaselineScheduler:
    def __init__(self, name: str):
        self.name = name

    def get_action(self, env: MLSchedulingEnvironment) -> np.ndarray:
        raise NotImplementedError

    def _to_action_array(self, w_idx, n_idx, alloc, num_workloads, num_nodes) -> np.ndarray:
        if num_workloads == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        a0 = (w_idx + 0.5) / num_workloads     # decodes back to job index w_idx
        a1 = (n_idx + 0.5) / num_nodes         # decodes back to node index n_idx
        a2 = float(np.clip(alloc, 0.3, 1.0))
        return np.array([a0, a1, a2], dtype=np.float32)

    def _fallback(self, env) -> np.ndarray:
        """No-fit rule: submit job 0 to node 0 (the placement will fail)."""
        return self._to_action_array(0, 0, 1.0, max(len(env.pending_workloads), 1), env.num_nodes)

    @staticmethod
    def _fits(node, workload) -> bool:
        return all(node.available_resources[r] >= workload.resource_requirements[r]
                   for r in workload.resource_requirements)

    def reset(self):
        """Called at the start of every episode."""
        pass

    def act(self, state, env):
        """Adapter for evaluation.evaluate(): returns (env_action, node index)."""
        action = self.get_action(env)
        node = min(int(action[1] * env.num_nodes), env.num_nodes - 1)
        return action, node


# ==================== Schedulers ====================

class FirstFitScheduler(BaselineScheduler):
    """Job order: arrival order (queue head first). Node order: 0,1,2,3,4.
    Picks the first job that fits any node, on the first node that fits. Ties: lowest index."""
    def __init__(self):
        super().__init__("First-Fit")

    def get_action(self, env):
        if not env.pending_workloads:
            return self._fallback(env)
        for w_idx, workload in enumerate(env.pending_workloads):
            for n_idx, node in enumerate(env.nodes):
                if self._fits(node, workload):
                    return self._to_action_array(w_idx, n_idx, 1.0, len(env.pending_workloads), env.num_nodes)
        return self._fallback(env)


class BestFitScheduler(BaselineScheduler):
    """Job order: arrival order. For the first job that fits somewhere, picks the node
    that leaves the least total free capacity (sum over CPU, GPU, memory, bandwidth of
    available - demand; the sum mixes units, as in the original code). Ties: lowest node index."""
    def __init__(self):
        super().__init__("Best-Fit")

    def get_action(self, env):
        if not env.pending_workloads:
            return self._fallback(env)
        for w_idx, workload in enumerate(env.pending_workloads):
            best_node, best_waste = -1, float('inf')
            for n_idx, node in enumerate(env.nodes):
                if self._fits(node, workload):
                    waste = sum(node.available_resources[r] - workload.resource_requirements[r]
                                for r in workload.resource_requirements)
                    if waste < best_waste:
                        best_waste, best_node = waste, n_idx
            if best_node != -1:
                return self._to_action_array(w_idx, best_node, 1.0, len(env.pending_workloads), env.num_nodes)
        return self._fallback(env)


class PriorityBasedScheduler(BaselineScheduler):
    """Job order: priority, highest first (ties: earlier arrival, Python's stable sort).
    Node order: 0,1,...,4; for priority > 0.7 nodes 0 and 1 are tried before the rest
    (the same order, so this rule changes nothing in practice). First job/node that fits wins."""
    def __init__(self):
        super().__init__("Priority-Based")

    def get_action(self, env):
        if not env.pending_workloads:
            return self._fallback(env)
        order = sorted(range(len(env.pending_workloads)),
                       key=lambda i: env.pending_workloads[i].priority, reverse=True)
        for w_idx in order:
            workload = env.pending_workloads[w_idx]
            candidates = (list(range(min(2, len(env.nodes)))) + list(range(len(env.nodes)))
                          if workload.priority > 0.7 else list(range(len(env.nodes))))
            seen = []
            for n_idx in candidates:
                if n_idx in seen:
                    continue
                seen.append(n_idx)
                if self._fits(env.nodes[n_idx], workload):
                    return self._to_action_array(w_idx, n_idx, 1.0, len(env.pending_workloads), env.num_nodes)
        return self._fallback(env)


class CostAwareScheduler(BaselineScheduler):
    """Job order: arrival order. For the first job that fits somewhere, picks the node with the
    lowest cost_per_hour x duration / 60 (i.e. the cheapest node; tuple sort => ties go to the lowest index)."""
    def __init__(self):
        super().__init__("Cost-Aware")

    def get_action(self, env):
        if not env.pending_workloads:
            return self._fallback(env)
        for w_idx, workload in enumerate(env.pending_workloads):
            viable = [(node.cost_per_hour * workload.estimated_duration / 60, n_idx)
                      for n_idx, node in enumerate(env.nodes) if self._fits(node, workload)]
            if viable:
                viable.sort()
                return self._to_action_array(w_idx, viable[0][1], 1.0, len(env.pending_workloads), env.num_nodes)
        return self._fallback(env)


class SJFScheduler(BaselineScheduler):
    """Job order: estimated duration, shortest first (ties: earlier arrival). Node order 0..4; first fit."""
    def __init__(self):
        super().__init__("SJF")

    def get_action(self, env):
        if not env.pending_workloads:
            return self._fallback(env)
        order = sorted(range(len(env.pending_workloads)),
                       key=lambda i: env.pending_workloads[i].estimated_duration)
        for w_idx in order:
            workload = env.pending_workloads[w_idx]
            for n_idx, node in enumerate(env.nodes):
                if self._fits(node, workload):
                    return self._to_action_array(w_idx, n_idx, 1.0, len(env.pending_workloads), env.num_nodes)
        return self._fallback(env)


class RoundRobinScheduler(BaselineScheduler):
    """Job order: arrival order. A pointer cycles over nodes 0..4; for each job it advances the
    pointer up to 5 times and takes the first node that fits. The pointer keeps moving even when a
    node does not fit, and is reset to 0 at every episode start."""
    def __init__(self, num_nodes: int):
        super().__init__("Round-Robin")
        self.num_nodes    = num_nodes
        self.current_node = 0

    def reset(self):
        self.current_node = 0

    def get_action(self, env):
        if not env.pending_workloads:
            return self._fallback(env)
        for w_idx, workload in enumerate(env.pending_workloads):
            for _ in range(self.num_nodes):
                n_idx = self.current_node
                self.current_node = (self.current_node + 1) % self.num_nodes
                if self._fits(env.nodes[n_idx], workload):
                    return self._to_action_array(w_idx, n_idx, 1.0, len(env.pending_workloads), env.num_nodes)
        return self._fallback(env)


def all_schedulers(num_nodes: int):
    return [FirstFitScheduler(), BestFitScheduler(), PriorityBasedScheduler(),
            CostAwareScheduler(), SJFScheduler(), RoundRobinScheduler(num_nodes)]


# ==================== Evaluation ====================

def summarise(name, df, split, arrival_prob):
    viol = df.violated + df.expired
    return {
        'scheduler'            : name,
        'avg_profit'           : float(df.profit.mean()),
        'std_profit'           : float(df.profit.std(ddof=1)),       # per-episode SD
        'avg_revenue'          : float(df.revenue.mean()),
        'avg_penalties'        : float(df.penalties.mean()),
        'avg_cost'             : float(df.cost.mean()),              # operating cost only
        'avg_completed'        : float(df.completed.mean()),         # finished on time
        'avg_violated'         : float(df.violated.mean()),          # finished late
        'avg_expired'          : float(df.expired.mean()),
        'avg_sla_violations'   : float(viol.mean()),                 # late + expired
        'std_sla_violations'   : float(viol.std(ddof=1)),
        'sla_compliance_pct'   : float(df.sla_pct.mean()),           # unified formula
        'avg_failed_placements': float(df.failed_placements.mean()),
        'avg_cpu_util'         : float(df.mean_cpu_util.mean()),
        'avg_completion_time'  : float(df.mean_completion_time.mean()),
        'split'                : split,
        'arrival_prob'         : arrival_prob,
        'episodes'             : int(len(df)),
    }


def plot_results(df, out_dir, mode, tag):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(df))
    axes[0].bar(x, df['avg_profit'], yerr=df['std_profit'], capsize=3, color='#4C72B0')
    axes[0].set_ylabel('Profit per episode ($)')
    axes[1].bar(x, df['avg_penalties'], label='Penalties', color='#C44E52')
    axes[1].bar(x, df['avg_cost'], bottom=df['avg_penalties'], label='Operating cost', color='#8172B3')
    axes[1].set_ylabel('Penalties and cost per episode ($)'); axes[1].legend(fontsize=8)
    axes[2].bar(x, df['sla_compliance_pct'], color='#55A868')
    axes[2].set_ylabel('SLA compliance (%)'); axes[2].set_ylim(0, 105)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(df['scheduler'], rotation=30, ha='right'); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Heuristic baselines, error bars: SD over episodes', fontsize=10)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'baseline_comparison_{mode}{tag}.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  [Plot saved: {fname}]")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Heuristic baselines, common evaluation protocol')
    parser.add_argument('--mode', type=str, default='synth', choices=['synth', 'criteo'])
    parser.add_argument('criteo_path', nargs='?', default=None, help='Folder with the Criteo extract (train.txt)')
    parser.add_argument('--episodes', type=int, default=N_EVAL)
    parser.add_argument('--output', type=str, default='businessAware/baselines')
    parser.add_argument('--split', type=str, default='eval', choices=['train', 'eval'],
                        help="Criteo split. Default 'eval' (required for results reported with the learned agents).")
    parser.add_argument('--arrival-prob', type=float, default=0.3, help='Job arrival probability per step (ABL-4)')
    parser.add_argument('--tag', type=str, default='', help='Suffix for output files and method labels (ablations)')
    parser.add_argument('--plots', action='store_true', help='also write the legacy bar chart (not used in the paper)')
    parser.add_argument('--log-dir', type=str, default=os.path.join('results', 'logs'))
    args = parse_with_config(parser)

    tag_sfx = f'_{args.tag}' if args.tag else ''
    print(f"\n{'=' * 60}\n  Heuristic baselines | mode={args.mode} | episodes={args.episodes}"
          f" | episode seeds {EVAL_SEED_BASE}..{EVAL_SEED_BASE + args.episodes - 1}"
          f" | arrival_prob={args.arrival_prob}\n{'=' * 60}")

    if args.mode == 'criteo':
        if not args.criteo_path or not os.path.exists(args.criteo_path):
            raise SystemExit("--mode criteo requires the path to the Criteo folder.")
        if args.split != 'eval':
            print("\n  *** WARNING: --split train. These numbers are NOT comparable with the learned "
                  "agents and must not be used in the paper (feedback A1). ***\n")
        env, dataset = make_env(args.criteo_path, split=args.split, arrival_prob=args.arrival_prob)
        split = args.split
    else:
        env, dataset = make_env(None, arrival_prob=args.arrival_prob)
        split = 'n/a'

    summaries = []
    for sched in all_schedulers(env.num_nodes):
        label = f"{sched.name} ({args.tag})" if args.tag else sched.name
        print(f"\nEvaluating {label} ...")
        df = evaluate(env, sched.act, label, dataset, seed=0, split=split, episodes=args.episodes,
                      arrival_prob=args.arrival_prob, on_start=sched.reset,
                      out_csv=log_path(label, dataset, 0, root=args.log_dir))
        summaries.append(summarise(sched.name, df, split, args.arrival_prob))

    out = pd.DataFrame(summaries).set_index('scheduler')
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, f'baseline_results_{args.mode}{tag_sfx}.csv')
    out.to_csv(csv_path)
    dump_json({'results': summaries, 'meta': run_metadata({'mode': args.mode, 'split': split,
              'arrival_prob': args.arrival_prob, 'episode_seeds': [EVAL_SEED_BASE, EVAL_SEED_BASE + args.episodes - 1]})},
              os.path.join(args.output, f'baseline_results_{args.mode}{tag_sfx}.json'))
    if args.plots:   # old Fig. 2 duplicated Table R1 and is deleted from the paper (11.3); kept only on request
        plot_results(out.reset_index(), args.output, args.mode, tag_sfx)

    cols = ['avg_profit', 'std_profit', 'avg_revenue', 'avg_penalties', 'avg_cost',
            'sla_compliance_pct', 'avg_sla_violations', 'avg_failed_placements']
    print(f"\n{'=' * 60}\n  Baseline summary ({args.mode}, split={split})\n{'=' * 60}")
    print(out[cols].round(2).to_string())
    print(f"\n  Saved: {csv_path}\n  Per-episode logs: {args.log_dir}/")


if __name__ == '__main__':
    main()
