#!/usr/bin/env python
"""
stats.py - statistics and main tables (revision 1). Produces Tables R1, R2, R3 (+ IQM CIs).

  A2  profit is recomputed as revenue - penalties - cost from the logged components
  A3  one SLA formula for every method:  1 - (late + expired) / (on-time + late + expired)
  A4  two SD columns per method: SD across seed means, and mean per-episode SD
  D2  Welch t on seed means; paired Wilcoxon on per-episode profit (same episode seeds);
      95% stratified-bootstrap CI of the interquartile mean (Agarwal et al., 2021)
  D13 statistical comparison table (R3)

Input: per-episode logs in results/logs/eval_*.csv written by evaluation.py. Required columns:
    method, dataset, seed, episode_seed, revenue, penalties, cost, completed, violated
  (completed = on time, violated = late, expired = dropped from the queue)

Usage
    python scripts/stats.py                         # HBA-RL vs Flat PPO
    python scripts/stats.py --flat "Flat PPO (matched)"
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED = ["method", "dataset", "seed", "episode_seed", "revenue", "penalties", "cost", "completed", "violated"]
OPTIONAL = ["expired", "failed_placements", "mean_cpu_util", "mean_completion_time"]
N_BOOT = 10_000
BOOT_SEED = 20261027          # printed, so the bootstrap is reproducible


def load_episodes(logs_dir):
    frames = []
    for f in sorted(glob.glob(os.path.join(logs_dir, "eval_*.csv"))):
        d = pd.read_csv(f)
        if {"method", "dataset", "seed", "episode_seed"}.issubset(d.columns):
            frames.append(d)
    if not frames:
        sys.exit(f"No per-episode evaluation logs (eval_*.csv) in {logs_dir}. Run the evaluation first.")
    ep = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED if c not in ep.columns]
    if missing:
        sys.exit(f"Logs are missing required columns: {missing}")
    for c in OPTIONAL:
        if c not in ep.columns:
            print(f"WARNING: optional column '{c}' not in logs; NaN in the tables (D11).")
            ep[c] = np.nan
    ep["expired"] = ep["expired"].fillna(0)
    ep["violations_total"] = ep["violated"] + ep["expired"]
    calc = ep["revenue"] - ep["penalties"] - ep["cost"]
    if "profit" in ep.columns and ((ep["profit"] - calc).abs() > 0.5).any():
        print("WARNING: stored profit differs from revenue - penalties - cost; the recomputed value is used (A2).")
    ep["profit"] = calc
    finished = (ep["completed"] + ep["violations_total"]).replace(0, np.nan)
    ep["sla_pct"] = 100 * (1 - ep["violations_total"] / finished)
    return ep


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    diff = a.mean() - b.mean()
    t = diff / se if se > 0 else np.nan
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df) if se > 0 else np.nan
    half = stats.t.ppf(0.975, df) * se if se > 0 else np.nan
    d = diff / np.sqrt((va + vb) / 2) if (va + vb) > 0 else np.nan
    return dict(diff=diff, t=t, df=df, p=p, ci_low=diff - half, ci_high=diff + half, cohens_d=d)


def iqm(x):
    return stats.trim_mean(np.asarray(x, float), 0.25)


def _iqm_rows(m):
    """Interquartile mean of each row of a matrix (same trimming as scipy's trim_mean(x, 0.25))."""
    n = m.shape[1]
    lo = int(0.25 * n)
    return np.sort(m, axis=1)[:, lo:n - lo].mean(axis=1)


def boot_iqm(groups, rng, batch=500):
    """Stratified bootstrap of the IQM (vectorised). groups = one array of episode values per seed.
    Seeds are resampled with replacement and all their episodes kept; a single group (the deterministic
    heuristics) resamples episodes instead."""
    k = len(groups)
    vals = []
    for start in range(0, N_BOOT, batch):
        b = min(batch, N_BOOT - start)
        if k == 1:
            g = np.asarray(groups[0], float)
            vals.append(_iqm_rows(g[rng.integers(0, len(g), (b, len(g)))]))
        else:
            arr = np.stack([np.asarray(g, float) for g in groups])        # (k, episodes)
            idx = rng.integers(0, k, (b, k))
            vals.append(_iqm_rows(arr[idx].reshape(b, -1)))
    vals = np.concatenate(vals)
    return iqm(np.concatenate(groups)), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def boot_gap_pct(a, b, rng):
    """Bootstrap of (mean(a) - mean(b)) / mean(b) in %, resampling the seed means of each agent independently."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma = a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), (N_BOOT, len(b)))].mean(axis=1)
    return 100 * (ma - mb) / mb


def seed_table(ep):
    g = ep.groupby(["dataset", "method", "seed"])
    return g.agg(profit=("profit", "mean"), profit_ep_sd=("profit", lambda x: x.std(ddof=1)),
                 revenue=("revenue", "mean"), penalties=("penalties", "mean"), cost=("cost", "mean"),
                 completed=("completed", "mean"), violations=("violations_total", "mean"),
                 violated_late=("violated", "mean"), expired=("expired", "mean"),
                 failed_placements=("failed_placements", "mean"), cpu_util=("mean_cpu_util", "mean"),
                 completion_time=("mean_completion_time", "mean"), sla_pct=("sla_pct", "mean"),
                 n_episodes=("profit", "size")).reset_index()


def table_r1(st):
    rows = []
    for (ds, m), d in st.groupby(["dataset", "method"]):
        learned = d["seed"].nunique() > 1
        rows.append({
            "dataset": ds, "method": m, "n_seeds": d["seed"].nunique(),
            "profit_mean": d["profit"].mean(),
            "sd_seed_means": d["profit"].std(ddof=1) if learned else np.nan,      # n/a for heuristics (A4)
            "mean_per_episode_sd": d["profit_ep_sd"].mean(),
            "revenue": d["revenue"].mean(), "penalties": d["penalties"].mean(), "cost": d["cost"].mean(),
            "completed": d["completed"].mean(), "violated_late": d["violated_late"].mean(),
            "expired": d["expired"].mean(), "violations_total": d["violations"].mean(),
            "failed_placements": d["failed_placements"].mean(), "sla_pct": d["sla_pct"].mean(),
            "cpu_util": d["cpu_util"].mean(), "completion_time": d["completion_time"].mean()})
    return pd.DataFrame(rows)


def table_iqm(ep, rng):
    rows = []
    for (ds, m), d in ep.groupby(["dataset", "method"]):
        groups = [x["profit"].to_numpy() for _, x in d.groupby("seed")]
        v, lo, hi = boot_iqm(groups, rng)
        rows.append({"dataset": ds, "method": m, "iqm_profit": v, "ci95_low": lo, "ci95_high": hi,
                     "n_seeds": len(groups), "n_boot": N_BOOT})
    return pd.DataFrame(rows)


def per_episode_profit(ep, ds, m):
    d = ep[(ep.dataset == ds) & (ep.method == m)]
    return d.groupby("episode_seed")["profit"].mean()


def table_r3(ep, st, hba, flat, rng):
    rows, boots = [], {}
    for ds in sorted(ep.dataset.unique()):
        sh, sf = st[(st.dataset == ds) & (st.method == hba)], st[(st.dataset == ds) & (st.method == flat)]
        if len(sh) < 2 or len(sf) < 2:
            print(f"WARNING: need >= 2 seeds for {hba!r} and {flat!r} on {ds}; skipping R3 there.")
            continue
        for metric in ["profit", "violations", "sla_pct"]:
            w = welch(sh[metric], sf[metric])
            rows.append({"comparison": f"{hba} vs {flat}", "dataset": ds, "metric": metric, "gap": w["diff"],
                         "gap_pct": 100 * w["diff"] / sf[metric].mean(), "test": "Welch t (seed means)",
                         "statistic": w["t"], "df": w["df"], "p": w["p"], "ci95_low": w["ci_low"],
                         "ci95_high": w["ci_high"], "effect_size_d": w["cohens_d"], "n_a": len(sh), "n_b": len(sf)})
        boots[ds] = boot_gap_pct(sh["profit"], sf["profit"], rng)
        a, b = per_episode_profit(ep, ds, hba).align(per_episode_profit(ep, ds, flat), join="inner")
        try:
            wp = stats.wilcoxon(a, b).pvalue
        except ValueError:
            wp = np.nan
        rows.append({"comparison": f"{hba} vs {flat}", "dataset": ds, "metric": "profit (per episode)",
                     "gap": (a - b).mean(), "gap_pct": 100 * (a - b).mean() / b.mean(),
                     "test": "Wilcoxon signed-rank (paired by episode seed)", "statistic": np.nan, "df": np.nan,
                     "p": wp, "ci95_low": np.nan, "ci95_high": np.nan, "effect_size_d": np.nan,
                     "n_a": len(a), "n_b": len(b)})
        lo, hi = np.percentile(boots[ds], [2.5, 97.5])
        print(f"  {ds}: profit gap {100 * (sh.profit.mean() - sf.profit.mean()) / sf.profit.mean():.2f}% "
              f"(bootstrap 95% CI {lo:.2f}% to {hi:.2f}%)")
    if {"synth", "criteo"} <= set(boots):
        diff = boots["criteo"] - boots["synth"]            # hypothesis P2: gap larger on Criteo
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"  Interaction (criteo gap minus synth gap, % points): mean {diff.mean():.2f}, 95% CI {lo:.2f} to {hi:.2f}")
        rows.append({"comparison": f"{hba} vs {flat}", "dataset": "criteo - synth", "metric": "gap_pct interaction",
                     "gap": np.nan, "gap_pct": diff.mean(), "test": "independent-seed bootstrap", "statistic": np.nan,
                     "df": np.nan, "p": np.nan, "ci95_low": lo, "ci95_high": hi, "effect_size_d": np.nan,
                     "n_a": np.nan, "n_b": np.nan})
    return pd.DataFrame(rows)


def table_vs_heuristics(ep, st, hba):
    """HBA-RL vs every heuristic at the SAME arrival probability: the nominal-load HBA-RL is paired with the
    nominal-load heuristics, and HBA-RL [abl4_pX] with the heuristics '(abl4_pX)' (same episode seeds, same load)."""
    rows = []
    for ds in sorted(ep.dataset.unique()):
        methods = set(ep[ep.dataset == ds].method)
        for m in sorted(methods):
            if st[(st.dataset == ds) & (st.method == m)].seed.nunique() > 1:
                continue                                       # learned agents (several seeds) are not heuristics
            tag = m[m.index("(abl4_") + 1:-1] if "(abl4_" in m else ""
            ref = f"{hba} [{tag}]" if tag else hba
            if ref not in methods:
                continue
            a = per_episode_profit(ep, ds, ref)
            x, y = a.align(per_episode_profit(ep, ds, m), join="inner")
            if len(x) == 0:
                print(f"WARNING: {m} on {ds} shares no episode seeds with {ref}; paired test impossible (A1).")
                continue
            try:
                p = stats.wilcoxon(x, y).pvalue
            except ValueError:
                p = np.nan
            rows.append({"dataset": ds, "comparison": f"{ref} vs {m}", "gap": (x - y).mean(),
                         "gap_pct": 100 * (x - y).mean() / y.mean(), "test": "Wilcoxon (paired by episode seed)",
                         "p": p, "n_episodes": len(x)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join("results", "logs"))
    ap.add_argument("--out", default=os.path.join("results", "tables"))
    ap.add_argument("--hba", default="HBA-RL")
    ap.add_argument("--flat", default="Flat PPO", help='"Flat PPO" (original) or "Flat PPO (matched)" (ABL-1)')
    args = ap.parse_args()
    suffix = "_matched" if "matched" in args.flat.lower() else ""

    ep = load_episodes(args.logs)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    print(f"Bootstrap seed {BOOT_SEED}, {N_BOOT} resamples. Methods: {sorted(ep.method.unique())}")
    for need in (args.hba, args.flat):
        if need not in set(ep.method):
            sys.exit(f"Method label {need!r} not in the logs.")

    st = seed_table(ep)
    r1 = table_r1(st)
    r1.to_csv(os.path.join(args.out, "R1_main_results.csv"), index=False)
    st.to_csv(os.path.join(args.out, "R2_per_seed.csv"), index=False)
    table_iqm(ep, rng).to_csv(os.path.join(args.out, "IQM_profit.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n== R1 main results (profit = revenue - penalties - cost; one SLA formula) ==")
    print(r1.round(2).to_string(index=False))

    print(f"\n== R3 {args.hba} vs {args.flat} ==")
    r3 = table_r3(ep, st, args.hba, args.flat, rng)
    r3.to_csv(os.path.join(args.out, f"R3_statistical_comparison{suffix}.csv"), index=False)
    print(r3.round(4).to_string(index=False))

    print(f"\n== {args.hba} vs heuristics (paired by episode seed) ==")
    rh = table_vs_heuristics(ep, st, args.hba)
    rh.to_csv(os.path.join(args.out, "R3b_vs_heuristics.csv"), index=False)
    print(rh.round(4).to_string(index=False))

    matched = "Flat PPO (matched)"
    if matched in set(ep.method):
        print(f"\n== {matched} vs heuristics (paired by episode seed, same arrival probability) ==")
        rc = table_vs_heuristics(ep, st, matched)
        rc.to_csv(os.path.join(args.out, "R3c_matched_vs_heuristics.csv"), index=False)
        print(rc.round(4).to_string(index=False))
    print(f"\nTables written to {args.out}/")


if __name__ == "__main__":
    main()
