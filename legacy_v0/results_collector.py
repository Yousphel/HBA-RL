"""
Results Collector — businessAware folder
==========================================
Aggregates evaluation_report.json from all 4 seeds into:
  - Summary table (mean ± std across seeds)
  - Per-seed breakdown
  - Comparison plot against baselines
  - CSV export for thesis tables

Usage (after all 4 seeds finish training):

    # Synthetic results:
    python results_collector.py --mode synth

    # Criteo results:
    python results_collector.py --mode criteo

    # Both:
    python results_collector.py --mode both

    # Custom root:
    python results_collector.py --mode synth --root /path/to/businessAware

    # Include baseline CSV for comparison plot:
    python results_collector.py --mode synth --baseline baseline_results_fair.csv

Output files (written to businessAware/<mode>/):
    summary_<mode>.csv          — per-seed table
    aggregate_<mode>.json       — mean ± std JSON
    comparison_<mode>.png       — bar chart vs baselines
    thesis_table_<mode>.txt     — LaTeX-ready table row
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==================== Loader ====================

SEEDS = [7, 42, 123, 456]

def load_seed_results(root: str, mode: str, seed: int) -> dict:
    """Load evaluation_report.json for one seed."""
    path = os.path.join(root, mode, f'seed_{seed}', 'evaluation_report.json')
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_training_metrics(root: str, mode: str, seed: int) -> dict:
    """Load metrics_best.json for training curve data."""
    path = os.path.join(root, mode, f'seed_{seed}', 'metrics_best.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ==================== Aggregator ====================

def aggregate_seeds(root: str, mode: str) -> dict:
    """Collect all 4 seeds and compute mean ± std."""
    records = []
    missing = []

    for seed in SEEDS:
        result = load_seed_results(root, mode, seed)
        if result is None:
            missing.append(seed)
            continue
        m = result['final_metrics']
        records.append({
            'seed'              : seed,
            'avg_profit'        : m['avg_profit'],
            'std_profit'        : m['std_profit'],
            'avg_revenue'       : m['avg_revenue'],
            'avg_cost'          : m['avg_cost'],
            'avg_sla_violations': m['avg_sla_violations'],
            'sla_compliance_pct': m['sla_compliance_rate'] * 100,
            'avg_completions'   : m['avg_completions'],
            'avg_reward'        : m['avg_reward'],
            'training_episodes' : result.get('training_episodes', 'N/A'),
        })

    if not records:
        print(f"  [ERROR] No results found for mode='{mode}' in {root}/")
        return None

    if missing:
        print(f"  [WARN] Missing seeds: {missing} — aggregating with {len(records)} seeds")

    df = pd.DataFrame(records).set_index('seed')

    aggregate = {
        'mode'         : mode,
        'seeds_found'  : [r['seed'] for r in records],
        'seeds_missing': missing,
        'per_seed'     : records,
        'mean': {
            'avg_profit'        : float(df['avg_profit'].mean()),
            'avg_revenue'       : float(df['avg_revenue'].mean()),
            'avg_cost'          : float(df['avg_cost'].mean()),
            'avg_sla_violations': float(df['avg_sla_violations'].mean()),
            'sla_compliance_pct': float(df['sla_compliance_pct'].mean()),
            'avg_completions'   : float(df['avg_completions'].mean()),
            'avg_reward'        : float(df['avg_reward'].mean()),
        },
        'std': {
            'avg_profit'        : float(df['avg_profit'].std()),
            'avg_revenue'       : float(df['avg_revenue'].std()),
            'avg_cost'          : float(df['avg_cost'].std()),
            'avg_sla_violations': float(df['avg_sla_violations'].std()),
            'sla_compliance_pct': float(df['sla_compliance_pct'].std()),
            'avg_completions'   : float(df['avg_completions'].std()),
            'avg_reward'        : float(df['avg_reward'].std()),
        },
    }
    return aggregate, df


# ==================== Printer ====================

def print_summary(aggregate: dict, df: pd.DataFrame, mode: str):
    mean = aggregate['mean']
    std  = aggregate['std']

    print(f"\n{'='*65}")
    print(f"  HBA-RL RESULTS — {mode.upper()} — {len(aggregate['seeds_found'])} seeds")
    print(f"{'='*65}")

    print(f"\n  Per-seed breakdown:")
    print(df[['avg_profit','std_profit','sla_compliance_pct',
              'avg_sla_violations','avg_completions']].round(2).to_string())

    print(f"\n  Aggregate (mean ± std across seeds):")
    print(f"    Avg Profit      : ${mean['avg_profit']:>10,.2f}  ±  {std['avg_profit']:,.2f}")
    print(f"    Avg Revenue     : ${mean['avg_revenue']:>10,.2f}  ±  {std['avg_revenue']:,.2f}")
    print(f"    Avg Cost        : ${mean['avg_cost']:>10,.2f}  ±  {std['avg_cost']:,.2f}")
    print(f"    SLA Compliance  :  {mean['sla_compliance_pct']:>6.1f}%  ±  {std['sla_compliance_pct']:.1f}%")
    print(f"    SLA Violations  :  {mean['avg_sla_violations']:>6.1f}  ±  {std['avg_sla_violations']:.1f}")
    print(f"    Avg Completions :  {mean['avg_completions']:>6.1f}  ±  {std['avg_completions']:.1f}")


# ==================== LaTeX Table Row ====================

def generate_latex_row(aggregate: dict, mode: str) -> str:
    m = aggregate['mean']
    s = aggregate['std']
    row = (
        f"HBA-RL (Ours) & "
        f"\\${m['avg_profit']:,.0f} $\\pm$ {s['avg_profit']:,.0f} & "
        f"\\${m['avg_revenue']:,.0f} & "
        f"\\${m['avg_cost']:,.0f} & "
        f"{m['sla_compliance_pct']:.1f}\\% & "
        f"{m['avg_sla_violations']:.1f} $\\pm$ {s['avg_sla_violations']:.1f} & "
        f"{m['avg_completions']:.1f} \\\\"
    )
    return row


# ==================== Comparison Plot ====================

def make_comparison_plot(aggregate: dict, df: pd.DataFrame,
                          mode: str, out_dir: str,
                          baseline_csv: str = None):
    """Bar chart: HBA-RL vs baselines (if baseline CSV provided)."""

    hba_profit = aggregate['mean']['avg_profit']
    hba_std    = aggregate['std']['avg_profit']
    hba_sla    = aggregate['mean']['sla_compliance_pct']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'HBA-RL vs Baselines — {mode.upper()}', fontsize=13)

    # ── Profit bar chart ─────────────────────────────────────────────
    ax = axes[0]
    labels  = [f'HBA-RL\n(seed {s})' for s in df.index] + ['HBA-RL\nMean']
    profits = list(df['avg_profit']) + [hba_profit]
    colors  = ['steelblue'] * len(df) + ['navy']
    errs    = [0] * len(df) + [hba_std]

    if baseline_csv and os.path.exists(baseline_csv):
        bdf = pd.read_csv(baseline_csv, index_col=0)
        if 'avg_profit' in bdf.columns:
            for bname in bdf.index:
                labels.insert(0, bname)
                profits.insert(0, bdf.loc[bname, 'avg_profit'])
                colors.insert(0, 'lightcoral')
                errs.insert(0, bdf.loc[bname, 'std_profit']
                            if 'std_profit' in bdf.columns else 0)

    x = np.arange(len(labels))
    bars = ax.bar(x, profits, color=colors, alpha=0.85, edgecolor='black')
    ax.errorbar(x, profits, yerr=errs, fmt='none', color='black',
                capsize=4, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_title('Average Profit ($)')
    ax.set_ylabel('Profit ($)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightcoral', label='Baselines'),
                       Patch(facecolor='steelblue',  label='HBA-RL (per seed)'),
                       Patch(facecolor='navy',       label='HBA-RL (mean)')]
    ax.legend(handles=legend_elements, fontsize=8)

    # ── SLA compliance bar chart ─────────────────────────────────────
    ax2 = axes[1]
    sla_labels  = [f'HBA-RL\n(seed {s})' for s in df.index] + ['HBA-RL\nMean']
    sla_vals    = list(df['sla_compliance_pct']) + [hba_sla]
    sla_colors  = ['steelblue'] * len(df) + ['navy']

    if baseline_csv and os.path.exists(baseline_csv):
        bdf = pd.read_csv(baseline_csv, index_col=0)
        if 'sla_compliance_pct' in bdf.columns:
            for bname in bdf.index:
                sla_labels.insert(0, bname)
                sla_vals.insert(0, bdf.loc[bname, 'sla_compliance_pct'])
                sla_colors.insert(0, 'lightcoral')

    x2 = np.arange(len(sla_labels))
    ax2.bar(x2, sla_vals, color=sla_colors, alpha=0.85, edgecolor='black')
    ax2.axhline(100, color='green',  linestyle='--', alpha=0.5, label='100%')
    ax2.axhline(80,  color='orange', linestyle='--', alpha=0.5, label='80% target')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(sla_labels, rotation=30, ha='right', fontsize=9)
    ax2.set_title('SLA Compliance (%)')
    ax2.set_ylabel('SLA Compliance (%)')
    ax2.set_ylim([0, 110])
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(out_dir, f'comparison_{mode}.png')
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  [Plot saved: {fname}]")


# ==================== Training Curve Plot ====================

def make_training_curve_plot(root: str, mode: str, out_dir: str):
    """Overlay training curves for all 4 seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Training Curves — HBA-RL — {mode.upper()} — all seeds', fontsize=12)

    colors = ['steelblue', 'green', 'orange', 'red']
    w = 50

    for seed, color in zip(SEEDS, colors):
        metrics = load_training_metrics(root, mode, seed)
        if metrics is None:
            continue
        rewards  = metrics.get('rewards', [])
        profits  = metrics.get('profits', [])
        sla      = metrics.get('sla_compliance', [])

        if len(rewards) >= w:
            ma_r = pd.Series(rewards).rolling(w).mean()
            ma_p = pd.Series(profits).rolling(w).mean()
            ma_s = pd.Series(sla).rolling(w).mean()
            axes[0].plot(ma_r, color=color, alpha=0.85, label=f'seed={seed}')
            axes[1].plot(ma_p, color=color, alpha=0.85, label=f'seed={seed}')
            axes[2].plot(ma_s, color=color, alpha=0.85, label=f'seed={seed}')

    for ax, title, ylabel in zip(
            axes,
            ['Reward (50-ep MA)', 'Profit (50-ep MA)', 'SLA Compliance (50-ep MA)'],
            ['Reward', 'Profit ($)', 'Compliance']):
        ax.set_title(title); ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, f'training_curves_{mode}.png')
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  [Plot saved: {fname}]")


# ==================== Main ====================

def run_mode(mode: str, root: str, baseline_csv: str = None):
    print(f"\n{'='*65}")
    print(f"  Collecting results: mode={mode}  root={root}")
    print(f"{'='*65}")

    result = aggregate_seeds(root, mode)
    if result is None:
        return
    aggregate, df = result

    out_dir = os.path.join(root, mode)
    os.makedirs(out_dir, exist_ok=True)

    # Print summary
    print_summary(aggregate, df, mode)

    # LaTeX row
    latex = generate_latex_row(aggregate, mode)
    print(f"\n  LaTeX table row:\n  {latex}")

    # Save CSV
    csv_path = os.path.join(out_dir, f'summary_{mode}.csv')
    df.to_csv(csv_path)
    print(f"\n  [CSV saved: {csv_path}]")

    # Save aggregate JSON
    agg_path = os.path.join(out_dir, f'aggregate_{mode}.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"  [JSON saved: {agg_path}]")

    # Save LaTeX row
    tex_path = os.path.join(out_dir, f'thesis_table_{mode}.txt')
    with open(tex_path, 'w') as f:
        f.write("% HBA-RL row for thesis results table\n")
        f.write("% Columns: Method | Avg Profit±std | Avg Revenue | Avg Cost | SLA% | Violations±std | Completions\n")
        f.write(latex + "\n")
        f.write(f"\n% Raw numbers:\n")
        m = aggregate['mean']; s = aggregate['std']
        f.write(f"% Avg Profit:      ${m['avg_profit']:,.2f} ± {s['avg_profit']:,.2f}\n")
        f.write(f"% Avg Revenue:     ${m['avg_revenue']:,.2f}\n")
        f.write(f"% Avg Cost:        ${m['avg_cost']:,.2f}\n")
        f.write(f"% SLA Compliance:   {m['sla_compliance_pct']:.2f}% ± {s['sla_compliance_pct']:.2f}%\n")
        f.write(f"% SLA Violations:   {m['avg_sla_violations']:.2f} ± {s['avg_sla_violations']:.2f}\n")
        f.write(f"% Avg Completions:  {m['avg_completions']:.2f} ± {s['avg_completions']:.2f}\n")
    print(f"  [LaTeX saved: {tex_path}]")

    # Plots
    make_comparison_plot(aggregate, df, mode, out_dir, baseline_csv)
    make_training_curve_plot(root, mode, out_dir)


def main():
    parser = argparse.ArgumentParser(description='HBA-RL Results Collector')
    parser.add_argument('--mode', type=str, default='synth',
                        choices=['synth', 'criteo', 'both'],
                        help='Which dataset results to collect (default: synth)')
    parser.add_argument('--root', type=str, default='businessAware',
                        help='Root output folder (default: businessAware)')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Path to baseline CSV for comparison plot '
                             '(e.g. baseline_results_fair.csv)')
    args = parser.parse_args()

    modes = ['synth', 'criteo'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        run_mode(mode, args.root, args.baseline)

    print(f"\n  Done. Check {args.root}/ for all outputs.")


if __name__ == '__main__':
    main()
