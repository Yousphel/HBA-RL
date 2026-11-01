"""
Shared helpers (revision 1): environment factory, routing-shaping term, run metadata.

Everything that must be IDENTICAL across HBA-RL, Flat PPO and the heuristics lives
here, so the comparison cannot drift between scripts.
"""
import os as _os
import sys as _sys
_R = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_R] + [_os.path.join(_R, _d) for _d in ('env', 'agents', 'baselines')]
import json
import os
import platform
import subprocess
import time
from importlib import metadata as _md

import numpy as np

from hrl_scheduler_simulation import MLSchedulingEnvironment

NUM_NODES      = 5
MAX_WORKLOADS  = 20
EPISODE_LENGTH = 100
EVAL_SEED_BASE = 42       # evaluation episode e uses seed 42 + e for EVERY method (A1, D8)
N_EVAL         = 100


def make_env(criteo_path=None, split='train', arrival_prob=0.3, observe_all_nodes=True):
    """Return (env, dataset_label). `split` only matters for Criteo ('train' or 'eval')."""
    if criteo_path and os.path.exists(criteo_path):
        from criteo import make_criteo_environment
        env, _, _ = make_criteo_environment(
            criteo_folder=criteo_path, split=split,
            num_nodes=NUM_NODES, max_workloads=MAX_WORKLOADS, episode_length=EPISODE_LENGTH,
            arrival_prob=arrival_prob, observe_all_nodes=observe_all_nodes)
        return env, 'criteo'
    if criteo_path:
        print(f"[WARN] Criteo path '{criteo_path}' not found. Using the synthetic generator.")
    env = MLSchedulingEnvironment(num_nodes=NUM_NODES, max_workloads=MAX_WORKLOADS,
                                  episode_length=EPISODE_LENGTH, arrival_prob=arrival_prob,
                                  observe_all_nodes=observe_all_nodes)
    return env, 'synth'


def routing_shaping(env, node_idx, legacy=False):
    """
    Routing-shaping bonus, TRAINING ONLY (used by HBA-RL, and by Flat PPO with --matched).

    Revision 1 (matches Algorithm 1 line 9, "if a job was admitted"): applied only on
    steps where a job was admitted, using that job and the node it was placed on.
        +0.5  priority > 0.6 on node 0 or 1
        -0.3  priority > 0.6 on nodes 2-4
        +0.2  priority <= 0.6 on nodes 2-4
         0    priority <= 0.6 on node 0 or 1
    legacy=True reproduces the earlier behaviour, which applied the rule on every
    step with a non-empty queue, to the most recently admitted job still running (even if
    it was admitted many steps earlier).
    """
    if legacy:
        if env.pending_workloads and env.active_workloads:
            job, _, _ = env.active_workloads[-1]
            premium, high = node_idx < 2, job.priority > 0.6
        else:
            return 0.0
    else:
        job = env.last_admitted
        if job is None:
            return 0.0
        premium, high = env.last_admitted_node < 2, job.priority > 0.6
    if high and premium:
        return 0.5
    if high and not premium:
        return -0.3
    if (not high) and (not premium):
        return 0.2
    return 0.0


def git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() or 'not a git repository'
    except Exception:
        return 'unknown'


def run_metadata(extra=None):
    """Software versions, hardware and git commit for run_config.json (D-HW)."""
    pk = {}
    for p in ('numpy', 'pandas', 'scipy', 'torch', 'gymnasium', 'matplotlib'):
        try:
            pk[p] = _md.version(p)
        except Exception:
            pk[p] = None
    cpu = platform.processor()
    try:
        cpu = next(l.split(':')[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        pass
    meta = {'python': platform.python_version(), 'platform': platform.platform(), 'cpu': cpu,
            'cpu_count': os.cpu_count(), 'packages': pk, 'git_commit': git_commit(),
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    if extra:
        meta.update(extra)
    return meta


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)


def parse_with_config(parser):
    """Add --config FILE.yaml: every key is an argparse default (command-line flags still win). Prints all seeds and
    settings that will be used, so the log shows them (repo rule: every seed set from the config and printed)."""
    parser.add_argument('--config', type=str, default=None, help='YAML file with defaults for these arguments')
    pre, _ = parser.parse_known_args()
    if pre.config:
        import yaml
        with open(pre.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        unknown = [k for k in cfg if not any(a.dest == k.replace('-', '_') for a in parser._actions)]
        if unknown:
            raise SystemExit(f'{pre.config}: unknown keys {unknown}')
        parser.set_defaults(**{k.replace('-', '_'): v for k, v in cfg.items()})
    args = parser.parse_args()
    print('Settings:', json.dumps({k: v for k, v in vars(args).items()}, default=str))
    print(f'Seeds: training seed {getattr(args, "seed", "n/a")}; training episode seeds '
          f'10_000_000 + seed*100_000 + episode; evaluation episode seeds {EVAL_SEED_BASE}+e (e=0..{N_EVAL - 1}); '
          'Criteo row sample: first 200,000 rows of train.txt (no random sample); per-episode row draw: '
          'default_rng(episode seed).')
    return args
