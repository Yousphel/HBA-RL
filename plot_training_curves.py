#!/usr/bin/env python3
"""
plot_training_curves.py  -  exemplar for Fig. 3 (item m9, work order W4)

Look-and-do-same code. Your make_figures.py reads your own training logs; this
script shows the four fixes Fig. 3 needs, using one simple long-format table:

    results/tables/training_rewards_long.csv
    columns: agent, dataset, seed, episode, reward
      agent   = HBA-RL | Flat PPO | Flat PPO (matched)
      dataset = synth | criteo
      reward  = training reward of that episode (the logged reward incl. shaping)

Export that table from your training logs once, then run:

    python plot_training_curves.py

The four fixes:
  1. all three agents drawn, one colour each, with a legend
  2. min-max band only where at least one seed is still training (no All-NaN warning)
  3. full y-axis label, not clipped
  4. one panel per dataset, so the three agents share the same axes
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENTS = ["HBA-RL", "Flat PPO", "Flat PPO (matched)"]
COLOURS = {"HBA-RL": "#1f3b73", "Flat PPO": "#c0504d", "Flat PPO (matched)": "#d4a017"}
TITLES = {"synth": "(a) Synthetic training data", "criteo": "(b) Criteo training split"}


def curve(sub, window):
    """Return episode axis, mean, min, max of the 50-episode moving mean across seeds."""
    grid = sub.pivot_table(index="episode", columns="seed", values="reward").sort_index()
    smooth = grid.rolling(window, min_periods=window).mean()
    alive = smooth.notna().sum(axis=1) > 0          # fix 2: keep only episodes with data
    smooth = smooth[alive]
    return smooth.index.values, smooth.mean(axis=1).values, smooth.min(axis=1).values, smooth.max(axis=1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/tables/training_rewards_long.csv")
    ap.add_argument("--out", default="results/figures/fig3_training_curves.pdf")
    ap.add_argument("--window", type=int, default=50)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, ds in zip(axes, ["synth", "criteo"]):
        for agent in AGENTS:                       # fix 1: all three agents
            sub = df[(df.dataset == ds) & (df.agent == agent)]
            if sub.empty:
                print(f"WARNING: no rows for {agent} on {ds}")
                continue
            x, mean, lo, hi = curve(sub, a.window)
            ax.fill_between(x, lo, hi, color=COLOURS[agent], alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=COLOURS[agent], lw=1.5, label=f"{agent} (n = {sub.seed.nunique()} seeds)")
        ax.set_title(TITLES[ds], loc="left", fontsize=10)
        ax.set_xlabel("Training episode")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Training reward (50-episode mean)")   # fix 3: full label
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
