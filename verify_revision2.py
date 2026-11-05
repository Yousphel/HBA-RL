#!/usr/bin/env python3
"""
verify_revision2.py  -  HBA-RL revision 2 check script (items M1, M1a, W1, W3)

Run from the repository root, AFTER all evaluation logs exist:

    python verify_revision2.py --extract <path to the 200,000-row Criteo extract>

What it does (it never trains anything, and it never edits your files):
  1. Prints provenance: commit hash, tag, SHA-256 of the Criteo extract, Python
     and package versions.  Paste this at the top of PART 4 of the log.
  2. Reads every per-episode evaluation log in results/logs and lists what it found.
  3. Builds LOAD-MATCHED comparisons:
       - matched Flat PPO vs HBA-RL at the same load (Welch t on seed means)
       - HBA-RL and matched Flat PPO vs each heuristic at the same load
         (paired Wilcoxon over the 100 episode seeds)
  4. Checks the results against the values printed in the manuscript.
  5. Builds routing share per node under heavy load (W3), if the logs hold
     routing columns.
  6. Writes CSV files to results/tables/verify_revision2/.

If a check prints FAIL, stop and phone your supervisor before you change the paper.
"""
import argparse, glob, hashlib, os, re, subprocess, sys
import numpy as np
import pandas as pd
from scipy import stats

HEURISTICS = {"firstfit": "First-Fit", "bestfit": "Best-Fit", "prioritybased": "Priority-Based",
              "costaware": "Cost-Aware", "sjf": "SJF", "roundrobin": "Round-Robin"}
SIMPLE4 = ["First-Fit", "Best-Fit", "Cost-Aware", "Round-Robin"]

# Gap = matched Flat PPO minus HBA-RL (seed means), as in the manuscript and Supplement R1/R6.
EXPECTED = {
    ("criteo", "nominal"): (-3.71, 0.86),
    ("synth", "nominal"): (125.32, 0.20),
    ("criteo", "p0.6"): (357.31, 0.001),
    ("criteo", "p0.9"): (211.96, 0.008),
    ("synth", "p0.6"): (210.79, 0.53),
    ("synth", "p0.9"): (9.01, 0.99),
}
TOL_DOLLARS = 1.0

PROFIT_COLS = ["profit", "net_profit", "episode_profit", "profit_net"]
EPISODE_COLS = ["episode_seed", "eval_seed", "seed_episode", "episode", "ep", "episode_idx"]


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip() or "(none)"
    except Exception as e:  # noqa
        return f"(could not run: {e})"


def provenance(extract):
    print("=" * 70)
    print("PROVENANCE")
    print("=" * 70)
    ok_all = True
    if os.path.isdir(".git"):
        print("commit :", run("git rev-parse HEAD"))
        print("tag    :", run("git describe --tags --always"))
        dirty = run("git status --porcelain | wc -l")
        print("dirty  :", dirty, "uncommitted file(s)  (must be 0)")
        if dirty.isdigit() and int(dirty) > 0:
            print("         FAIL: commit your changes and tag the commit before you run this")
            ok_all = False
    else:   # run from an unpacked copy of the tagged files (no .git folder): commit and tag come from the author
        c, t = os.environ.get("HBA_COMMIT"), os.environ.get("HBA_TAG")
        print("commit :", c or "FAIL: no .git folder and HBA_COMMIT not set")
        print("tag    :", t or "FAIL: no .git folder and HBA_TAG not set")
        print("note   : no .git folder here; commit and tag were supplied by the author for the files of that tag")
        ok_all = bool(c and t)
    if extract and os.path.isfile(extract):
        h = hashlib.sha256()
        with open(extract, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        d = h.hexdigest()
        ok = d.startswith("69824b15") and d.endswith("18d9bd0")
        print("sha256 :", d, "PASS" if ok else "FAIL (differs from the Data availability statement)")
        ok_all &= ok
    else:
        print("sha256 : not checked (give --extract <path>)")
    print("python :", sys.version.split()[0])
    for mod in ["numpy", "pandas", "scipy", "torch", "gymnasium", "matplotlib"]:
        try:
            m = __import__(mod)
            print(f"{mod:10s}: {m.__version__}")
        except Exception:
            print(f"{mod:10s}: not installed")
    print()
    return ok_all


def parse_name(path):
    """eval_<dataset>_<method>[_ or __<tag>]_seed<N>.csv -> dict, or None."""
    base = os.path.basename(path)
    m = re.match(r"eval_(criteo|synth)_(.+?)_seed(\d+)\.csv$", base)
    if not m:
        return None
    dataset, rest, seed = m.group(1), m.group(2), int(m.group(3))
    low = rest.lower()
    tag = re.search(r"abl\d+[a-z0-9._]*", low)
    tag = tag.group(0) if tag else ""
    stem = re.sub(r"_+$", "", low.replace(tag, "")) if tag else low
    key = re.sub(r"[^a-z]", "", stem)
    if "hba" in key:
        method = "HBA-RL"
    elif "flat" in key:
        method = "Flat PPO (matched)" if "matched" in key or "matched" in tag else "Flat PPO"
    else:
        method = next((v for k, v in HEURISTICS.items() if k in key), None)
        if method is None:
            return None
    lm = re.search(r"p(0\.\d+)", tag)
    if lm and tag.startswith("abl4"):
        variant = "p" + lm.group(1)
    elif lm:  # e.g. abl5_gpu2_p0.6 -> "abl5_gpu2_p0.6", compared only with the same tag
        variant = tag
    elif tag.startswith("abl2"):
        variant = "abl2"
    elif tag.startswith("abl3"):
        variant = "abl3"
    elif tag:
        variant = tag
    else:
        variant = "nominal"
    return dict(dataset=dataset, method=method, variant=variant, seed=seed, file=path)


def pick(cols, wanted, contains=None):
    low = {c.lower(): c for c in cols}
    for w in wanted:
        if w in low:
            return low[w]
    if contains:
        for c in cols:
            if contains in c.lower():
                return c
    return None


def load_logs(log_dir):
    rows, skipped = [], []
    for path in sorted(glob.glob(os.path.join(log_dir, "eval_*_seed*.csv"))):
        info = parse_name(path)
        if info is None:
            skipped.append(os.path.basename(path))
            continue
        df = pd.read_csv(path)
        pcol = pick(df.columns, PROFIT_COLS, contains="profit")
        if pcol is None:
            skipped.append(os.path.basename(path) + " (no profit column)")
            continue
        ecol = pick(df.columns, EPISODE_COLS)
        ep = df[ecol].values if ecol else np.arange(len(df))
        route = [c for c in df.columns if c.lower().startswith(("route", "routed", "placed"))]
        out = pd.DataFrame({"episode": ep, "profit": df[pcol].astype(float).values})
        for c in route:
            out[c] = df[c].values
        for k, v in info.items():
            if k != "file":
                out[k] = v
        rows.append(out)
    if not rows:
        sys.exit(f"No evaluation logs matched {log_dir}/eval_*_seed*.csv. Give --logs <folder>.")
    return pd.concat(rows, ignore_index=True), skipped


def welch(a, b):
    r = stats.ttest_ind(a, b, equal_var=False)
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return r.statistic, df, r.pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="results/logs")
    ap.add_argument("--extract", default=None, help="path to the 200,000-row Criteo extract")
    ap.add_argument("--out", default="results/tables/verify_revision2")
    a = ap.parse_args()

    prov_ok = provenance(a.extract)
    data, skipped = load_logs(a.logs)
    os.makedirs(a.out, exist_ok=True)

    print("=" * 70)
    print("INVENTORY (files per dataset / method / load)")
    print("=" * 70)
    inv = data.groupby(["dataset", "method", "variant"]).seed.nunique().rename("n_files").reset_index()
    print(inv.to_string(index=False))
    if skipped:
        print("\nSkipped (name or columns not recognised):", *skipped[:20], sep="\n  ")
    print()

    seed_means = data.groupby(["dataset", "method", "variant", "seed"]).profit.mean().reset_index()
    seed_means.to_csv(os.path.join(a.out, "seed_means.csv"), index=False)

    # ---- R6, load-matched: matched Flat PPO vs HBA-RL at the SAME load ----
    print("=" * 70)
    print("R6 LOAD-MATCHED: matched Flat PPO vs HBA-RL (Welch t on seed means)")
    print("=" * 70)
    r6, all_ok = [], prov_ok
    for (ds, var), _ in seed_means.groupby(["dataset", "variant"]):
        hb = seed_means.query("dataset==@ds and variant==@var and method=='HBA-RL'").profit.values
        mf = seed_means.query("dataset==@ds and variant==@var and method=='Flat PPO (matched)'").profit.values
        if len(hb) < 2 or len(mf) < 2:
            continue
        t, df, p = welch(mf, hb)
        gap = mf.mean() - hb.mean()
        exp = EXPECTED.get((ds, var))
        verdict = ""
        if exp:
            ok = abs(gap - exp[0]) <= TOL_DOLLARS
            all_ok &= ok
            verdict = "PASS" if ok else f"FAIL (expected {exp[0]:+.2f})"
        r6.append(dict(dataset=ds, load=var, n_hba=len(hb), n_matched=len(mf), hba_mean=hb.mean(),
                       matched_mean=mf.mean(), gap=gap, gap_pct=100 * gap / hb.mean(),
                       welch_t=t, df=df, p=p, check=verdict))
    r6 = pd.DataFrame(r6)
    if len(r6):
        print(r6.round(4).to_string(index=False))
        r6.to_csv(os.path.join(a.out, "R6_load_matched.csv"), index=False)
    print()

    # ---- R3b / R3c, load-matched: learned agent vs heuristic, paired by episode ----
    print("=" * 70)
    print("R3b / R3c LOAD-MATCHED: learned agent vs heuristic (paired Wilcoxon by episode)")
    print("=" * 70)
    ep_mean = data.groupby(["dataset", "method", "variant", "episode"]).profit.mean().reset_index()
    r3 = []
    for agent in ["HBA-RL", "Flat PPO (matched)"]:
        for (ds, var), _ in ep_mean.query("method==@agent").groupby(["dataset", "variant"]):
            if var in ("abl2", "abl3"):
                continue
            ag = ep_mean.query("dataset==@ds and variant==@var and method==@agent").set_index("episode").profit
            for h in HEURISTICS.values():
                he = ep_mean.query("dataset==@ds and variant==@var and method==@h").set_index("episode").profit
                if he.empty:
                    continue
                common = ag.index.intersection(he.index)
                if len(common) < 10:
                    print(f"  WARNING {ds} {var} {agent} vs {h}: only {len(common)} shared episodes")
                    continue
                x, y = ag.loc[common].values, he.loc[common].values
                p = stats.wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
                r3.append(dict(dataset=ds, load=var, agent=agent, heuristic=h, n_episodes=len(common),
                               gap=x.mean() - y.mean(), gap_pct=100 * (x.mean() - y.mean()) / y.mean(),
                               wilcoxon_p=p))
    r3 = pd.DataFrame(r3)
    if len(r3):
        print(r3.round(4).to_string(index=False))
        r3.to_csv(os.path.join(a.out, "R3bc_load_matched.csv"), index=False)
        # spot-check against the manuscript (5.4): HBA-RL vs the four simple heuristics on Criteo
        for var, lo, hi in [("p0.6", 35.0, 37.5), ("p0.9", 104.5, 109.5)]:
            s = r3.query("dataset=='criteo' and load==@var and agent=='HBA-RL' and heuristic in @SIMPLE4")
            if len(s):
                ok = s.gap_pct.between(lo, hi).all()
                all_ok &= ok
                print(f"  Criteo {var}, HBA-RL vs four simple heuristics: "
                      f"{s.gap_pct.min():.1f}% to {s.gap_pct.max():.1f}%  "
                      f"{'PASS' if ok else 'FAIL (manuscript says 36% and 105-109%)'}")
        for var in ["p0.6", "p0.9"]:
            s = r3.query("dataset=='synth' and load==@var and agent=='HBA-RL' and heuristic in @SIMPLE4")
            if len(s):
                print(f"  Synthetic {var}, HBA-RL vs four simple heuristics: "
                      f"{s.gap_pct.min():.1f}% to {s.gap_pct.max():.1f}%  (report this in the Abstract, T1)")
    print()

    # ---- W3: routing share under heavy load ----
    print("=" * 70)
    print("W3 ROUTING SHARE PER NODE (all loads, learned agents)")
    print("=" * 70)
    rcols = [c for c in data.columns if c.lower().startswith(("route", "routed"))]
    if not rcols:
        print("  No routing columns found in the evaluation logs.")
        print("  Use the routine that builds R4_routing_share in make_tables.py and point it at the")
        print("  abl4_p0.6 and abl4_p0.9 logs instead.")
    else:
        learned = data[data.method.isin(["HBA-RL", "Flat PPO", "Flat PPO (matched)"])]
        g = learned.groupby(["dataset", "method", "variant"])[rcols].sum()
        share = (100 * g.div(g.sum(axis=1), axis=0)).round(2).reset_index()
        print(share.to_string(index=False))
        share.to_csv(os.path.join(a.out, "R4b_routing_share_by_load.csv"), index=False)
    print()
    print("OVERALL:", "PASS - paste this output into PART 4 of the log" if all_ok
          else "FAIL - stop and phone your supervisor before changing the paper")
    print("CSV files written to", a.out)


if __name__ == "__main__":
    main()
