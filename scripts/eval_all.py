"""
eval_all.py - evaluate EVERY method on the SAME evaluation episodes (A1, D8), from saved checkpoints.

  * heuristics: re-run on the evaluation split with episode seeds 42 + e, once per dataset and arrival probability found
  * learned agents: every businessAware/**/run_config.json is replayed with --eval-only (loads checkpoint_best.pt,
    deterministic actions, episode seeds 42 + e) - no training happens here
  * then tables (stats, workload, make_tables) and figures are regenerated

    python scripts/eval_all.py [CRITEO_DIR] [--root businessAware] [--no-reports]

This is what `reproduce.sh --quick` runs.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, log):
    t0 = time.time()
    print('\n$', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    dt = time.time() - t0
    log.append((' '.join(cmd), dt, r.returncode))
    print(f'   [{dt:.1f} s, exit {r.returncode}]', flush=True)
    if r.returncode:
        sys.exit(f'FAILED: {" ".join(cmd)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('criteo_path', nargs='?', default=None)
    ap.add_argument('--root', default='businessAware')
    ap.add_argument('--no-reports', action='store_true', help='evaluate only; skip tables and figures')
    args = ap.parse_args()
    py, log = sys.executable, []
    configs = sorted(glob.glob(os.path.join(HERE, args.root, '**', 'run_config.json'), recursive=True))
    if not configs:
        sys.exit(f'No run_config.json under {args.root}: nothing to evaluate. Train first, or restore the checkpoints.')

    # 1. heuristics, one run per (dataset, arrival probability, tag) that a learned run used
    combos = {('synth', 0.3, '')}
    for f in configs:
        c = json.load(open(f))
        cli = c['cli']
        ds = c['dataset']
        p = cli.get('arrival_prob', 0.3)
        tag = cli.get('tag', '') if p != 0.3 else ''
        tag = tag if tag.startswith('abl4') else ''
        combos.add((ds, p, tag))
    for ds, p, tag in sorted(combos):
        if ds == 'criteo' and not args.criteo_path:
            continue
        cmd = [py, 'baselines/baseline_schedulers.py', '--mode', ds] + ([args.criteo_path] if ds == 'criteo' else [])
        if p != 0.3:
            cmd += ['--arrival-prob', str(p)] + (['--tag', tag] if tag else [])
        run(cmd, log)

    # 2. every trained agent, evaluation only
    for f in configs:
        c = json.load(open(f))
        cli = c['cli']
        if c['dataset'] == 'criteo' and not args.criteo_path:
            print('skip (no Criteo path):', f)
            continue
        flat = c['method'].startswith('Flat PPO')
        cmd = [py, 'scripts/train_flat_ppo.py' if flat else 'scripts/train_hrl_scheduler.py']
        if c['dataset'] == 'criteo':
            cmd.append(args.criteo_path)
        cmd += ['--seed', str(cli['seed']), '--eval-only', '--output', args.root]
        if cli.get('tag'):
            cmd += ['--tag', cli['tag']]
        if cli.get('arrival_prob', 0.3) != 0.3:
            cmd += ['--arrival-prob', str(cli['arrival_prob'])]
        for flag, key in (('--matched', 'matched'), ('--no-shaping', 'no_shaping'), ('--zero-subgoal', 'zero_subgoal'),
                          ('--legacy-shaping', 'legacy_shaping'), ('--legacy-observation', 'legacy_observation')):
            if cli.get(key):
                cmd.append(flag)
        run(cmd, log)

    # 3. tables and figures from the per-episode logs
    if not args.no_reports:
        crit = [args.criteo_path] if args.criteo_path else []
        run([py, 'scripts/workload_stats.py', *crit, '--arrival-probs', '0.3', '0.6', '0.9'], log)
        for s in ('stats.py', 'make_tables.py', 'make_figures.py', 'make_fig1.py'):
            run([py, f'scripts/{s}'], log)

    print('\n=== run times ===')
    for cmd, dt, rc in log:
        print(f'{dt:8.1f} s  {cmd}')
    print(f'{sum(d for _, d, _ in log):8.1f} s  TOTAL')


if __name__ == '__main__':
    main()
