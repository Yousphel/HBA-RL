"""Fills Table 3.10 and the missing-value rates in Section 3.7.4 of the thesis.

Run in Colab from your project folder (the one with criteo.py):
    !python table_3_10_stats.py /content/drive/MyDrive/businessAware/criteo
It uses the same loader and feature mapping as the experiments, so the numbers
describe exactly the workloads the agents saw. Takes a few minutes.
"""
import random
import sys

import numpy as np

from criteo import CriteoLoader, CriteoWorkloadMapper

folder = sys.argv[1]
FEATURES = ["I1", "I2", "I3", "I5", "I6", "I7"]

for split in ["train", "eval"]:
    random.seed(42)
    np.random.seed(42)
    loader = CriteoLoader(folder, split=split, seed=42)
    stats = loader.get_statistics()
    mapper = CriteoWorkloadMapper(loader)
    n = stats["total_rows"]
    value, prio, dur = [], [], []
    for _ in range(n):
        w = mapper.next_workload_params()
        value.append(w["revenue_value"])
        prio.append(w["priority"])
        dur.append(w["estimated_duration"])
    value, prio, dur = map(np.array, (value, prio, dur))
    print(f"\n===== {split.upper()} split =====")
    print(f"Records                           : {n:,}")
    print(f"Click rate                        : {100 * stats['click_rate']:.1f}%")
    print(f"Business value mean / median ($)  : {value.mean():,.0f} / {np.median(value):,.0f}")
    print(f"Share of high-priority jobs (>0.6): {100 * (prio > 0.6).mean():.1f}%")
    print(f"Duration mean (steps)             : {dur.mean():.1f}")
    print("Missing values per feature        : "
          + ", ".join(f"{c} {100 * stats['missing_rates'][c]:.1f}%" for c in FEATURES))
