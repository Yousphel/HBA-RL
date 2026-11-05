#!/usr/bin/env python3
"""
make_main_tables.py  -  rebuild Tables 2 and 3 with a Penalties column (item m11, fix T18)

Run from the repository root after scripts/stats.py has written R1:

    python make_main_tables.py

Reads  results/tables/R1_main_results.csv
Writes results/tables/Table2_synthetic.tsv and results/tables/Table3_criteo.tsv

Each row is checked: Revenue - Penalties - Cost must equal Profit (within $1).
To paste into Word: open the .tsv in Notepad, copy all, paste into Word,
select the pasted lines, then Insert > Table > Convert Text to Table (separate at: Tabs).
"""
import argparse, os, sys
import pandas as pd

ORDER = ["First-Fit", "Best-Fit", "Priority-Based", "Cost-Aware", "SJF", "Round-Robin",
         "Flat PPO", "Flat PPO (matched)", "HBA-RL"]
LABEL = {"Flat PPO": "Flat PPO (original)", "HBA-RL": "HBA-RL (Ours)"}
HEADER = ["Method", "Profit = Revenue − Penalties − Cost ($)", "Avg Revenue ($)",
          "Penalties ($)", "Avg Cost ($)", "SLA %", "Violations"]


def money(x):
    return f"${x:,.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", default="results/tables/R1_main_results.csv")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    if not os.path.isfile(a.r1):
        sys.exit(f"Cannot find {a.r1}. Run scripts/stats.py first.")
    r1 = pd.read_csv(a.r1)
    r1["method"] = r1["method"].str.strip()
    bad = 0
    for ds, name in [("synth", "Table2_synthetic.tsv"), ("criteo", "Table3_criteo.tsv")]:
        sub = r1[(r1.dataset.str.strip() == ds) & (r1.method.isin(ORDER))].set_index("method")
        lines = ["\t".join(HEADER)]
        print(f"\n== {name} ==")
        for m in ORDER:
            if m not in sub.index:
                print(f"  missing row: {m}")
                bad += 1
                continue
            r = sub.loc[m]
            check = r.revenue - r.penalties - r.cost - r.profit_mean
            flag = "" if abs(check) <= 1 else f"   <-- CHECK: revenue - penalties - cost differs from profit by {check:.2f}"
            bad += bool(flag)
            profit = money(r.profit_mean)
            if r.n_seeds > 1:
                profit += f" ± {r.sd_seed_means:,.0f}"
            row = [LABEL.get(m, m), profit, money(r.revenue), money(r.penalties), money(r.cost),
                   f"{r.sla_pct:.1f}%", f"{r.violations_total:.1f}"]
            lines.append("\t".join(row))
            print("  " + " | ".join(row) + flag)
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print("\nRESULT:", "PASS" if bad == 0 else f"FAIL ({bad} problem(s)) - phone your supervisor")


if __name__ == "__main__":
    main()
