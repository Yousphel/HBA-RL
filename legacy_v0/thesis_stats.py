"""Statistics for Chapter Four (Section 4.8.2) of the HBA-RL thesis.

Run from the project folder:
    python thesis_stats.py                     # HBA-RL vs original Flat PPO
    python thesis_stats.py --flat-tag matched  # HBA-RL vs matched Flat PPO
After tidy_folder.py, results are read from businessAware/flat_ppo/<synth|criteo|matched_synth|matched_criteo>/.
In the untidied layout, --synth-month / --criteo-month give the start of the folder timestamps
(e.g. --synth-month 20261015_20 --criteo-month 20261015_19 for the matched runs).
Writes per_seed_results.csv and prints the tests reported in Table 4.7.
"""
import argparse, csv, glob, json, os
import numpy as np
from scipy import stats

SEEDS = [7, 42, 123, 456]
METRICS = {"profit": "avg_profit", "violations": "sla_violations", "completions": "avg_completions"}


def hba(ds, seeds):
    out = {}
    for s in seeds:
        m = json.load(open(f"businessAware/{ds}/seed_{s}/evaluation_report.json"))["final_metrics"]
        out[s] = {"profit": m["avg_profit"], "profit_sd": m["std_profit"],
                  "violations": m["avg_sla_violations"], "completions": m["avg_completions"],
                  "sla_pct": 100 * m["sla_compliance_rate"]}
    return out


def flat(pattern, seeds):
    out = {}
    for s in seeds:
        files = sorted(glob.glob(pattern.format(seed=s)))
        if not files:
            raise SystemExit(f"No Flat PPO result for seed {s}: {pattern.format(seed=s)}")
        d = json.load(open(files[-1]))   # latest run for that seed
        out[s] = {"profit": d["avg_profit"], "violations": d["sla_violations"],
                  "completions": d["avg_completions"], "sla_pct": d["sla_compliance"]}
    return out


def paired(h, f, seeds, key):
    a = np.array([h[s][key] for s in seeds]); b = np.array([f[s][key] for s in seeds]); d = a - b
    n = len(d); t = stats.ttest_rel(a, b)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(n))
    w = stats.wilcoxon(a, b).pvalue if np.any(d) else float("nan")
    return d, dict(mean_diff=d.mean(), ci_low=ci[0], ci_high=ci[1], t=t.statistic, df=n - 1,
                   p_t=t.pvalue, p_wilcoxon=w, d_z=d.mean() / d.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--flat-tag", default="", help="'' for original runs, 'matched' for matched runs")
    ap.add_argument("--synth-month", default="202604", help="date prefix of synthetic Flat PPO folders")
    ap.add_argument("--criteo-month", default="202609", help="date prefix of Criteo Flat PPO folders")
    a = ap.parse_args()
    tag = f"_{a.flat_tag}" if a.flat_tag else ""
    rows, gaps, pct = [], {}, {}
    for ds, month in [("synth", a.synth_month), ("criteo", a.criteo_month)]:
        H = hba(ds, a.seeds)
        if os.path.isdir(os.path.join("businessAware", "flat_ppo")):   # tidied layout
            sub = f"matched_{ds}" if a.flat_tag else ds
            pattern = f"businessAware/flat_ppo/{sub}/flat_ppo{tag}_results_seed{{seed}}_*/flat_ppo_eval_results.json"
        else:                                                          # original layout
            pattern = f"flat_ppo{tag}_results_seed{{seed}}_{month}*/flat_ppo_eval_results.json"
        F = flat(pattern, a.seeds)
        for s in a.seeds:
            rows.append({"dataset": ds, "seed": s, **{f"hba_{k}": v for k, v in H[s].items()},
                         **{f"flat_{k}": v for k, v in F[s].items()}})
        print(f"\n=== {ds} (n = {len(a.seeds)} seeds) ===")
        for name, key in METRICS.items():
            d, r = paired(H, F, a.seeds, name)
            if name == "profit":
                gaps[ds] = d
                pct[ds] = 100 * d / np.array([F[s]["profit"] for s in a.seeds])
            print(f"{name:12s} diff {r['mean_diff']:9.2f}  95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]  "
                  f"t({r['df']}) = {r['t']:.2f}  p = {r['p_t']:.4f}  Wilcoxon p = {r['p_wilcoxon']:.3f}  d_z = {r['d_z']:.2f}")
        # HBA-RL (pooled episodes) vs each heuristic, Welch t on summary statistics
        n = 100; m = np.array([H[s]["profit"] for s in a.seeds]); sd = np.array([H[s]["profit_sd"] for s in a.seeds])
        N = n * len(m); mu = m.mean()
        var = (((n - 1) * sd ** 2).sum() + (n * (m - mu) ** 2).sum()) / (N - 1)
        for r in csv.DictReader(open(f"businessAware/baselines/baseline_results_{ds}.csv")):
            bm, bs = float(r["avg_profit"]), float(r["std_profit"])
            t = stats.ttest_ind_from_stats(mu, np.sqrt(var), N, bm, bs, n, equal_var=False)
            print(f"  vs {r['scheduler']:14s} diff {mu - bm:8.1f}  Welch p = {t.pvalue:.3f}  "
                  f"d = {(mu - bm) / np.sqrt((var + bs ** 2) / 2):.3f}")
    t = stats.ttest_ind(gaps["criteo"], gaps["synth"], equal_var=False)
    print(f"\nP2 interaction, dollar gap (Criteo vs synthetic): {gaps['criteo'].mean() - gaps['synth'].mean():.2f}  "
          f"Welch t({t.df:.1f}) = {t.statistic:.2f}  p = {t.pvalue:.4f}")
    t = stats.ttest_ind(pct["criteo"], pct["synth"], equal_var=False)
    print(f"P2 interaction, percentage gap (Criteo vs synthetic): {pct['criteo'].mean():.2f}% vs {pct['synth'].mean():.2f}%  "
          f"Welch t({t.df:.1f}) = {t.statistic:.2f}  p = {t.pvalue:.4f}")
    out_dir = os.path.join("businessAware", "stats") if os.path.isdir(os.path.join("businessAware", "flat_ppo")) else "."
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"per_seed_results{tag}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
