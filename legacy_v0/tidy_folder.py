"""Tidy the HBA-RL project folder on Google Drive.

Usage in Colab (from the project folder):
    !python tidy_folder.py            # dry run: only prints what it would do
    !python tidy_folder.py --apply    # backs everything up, then moves files

Nothing is deleted except __pycache__ folders (Python recreates them).
Code files stay at the top level because the scripts import each other.
"""
import argparse
import datetime
import glob
import os
import re
import shutil
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true", help="actually move files (default: dry run)")
args = ap.parse_args()
ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "train_flat_ppo.py")):
    sys.exit("Run this from the project folder (the one containing train_flat_ppo.py).")

FLAT = os.path.join("businessAware", "flat_ppo")
moves = []  # (source, destination folder)


def plan(src, dest_dir):
    moves.append((src, dest_dir))


# 1. Flat PPO result folders -> businessAware/flat_ppo/<group>/
def group_of(name):
    m = re.match(r"flat_ppo_(matched_)?results_seed(\d+)_(\d{8})_(\d{6})$", name)
    if not m:
        return None, None
    matched, seed, date, time = m.groups()
    if matched:
        if date == "20261015" and time.startswith("19"):
            return "matched_criteo", seed
        if date == "20261015" and time.startswith("20"):
            return "matched_synth", seed
        return "superseded", seed
    if date.startswith("202604"):
        return "synth", seed
    if date.startswith("202609"):
        return "criteo", seed
    return "superseded", seed


groups = {}
for d in sorted(glob.glob("flat_ppo*_results_seed*_*")):
    if not os.path.isdir(d):
        continue
    g, seed = group_of(os.path.basename(d))
    if g is None:
        continue
    has_result = os.path.exists(os.path.join(d, "flat_ppo_eval_results.json"))
    if g != "superseded" and not has_result:
        g = "superseded"  # an unfinished run
    groups.setdefault((g, seed), []).append(d)

for (g, seed), dirs in sorted(groups.items()):
    dirs = sorted(dirs)  # timestamps sort chronologically
    if g == "superseded":
        for d in dirs:
            plan(d, os.path.join(FLAT, "superseded"))
    else:
        for d in dirs[:-1]:  # older duplicates of the same seed
            plan(d, os.path.join(FLAT, "superseded"))
        plan(dirs[-1], os.path.join(FLAT, g))

# 2. Notebooks -> notebooks/
for nb in sorted(glob.glob("*.ipynb")):
    plan(nb, "notebooks")

# 3. Old copies of the Flat PPO files -> backup/
for old in sorted(glob.glob("*_original.py")):
    plan(old, "backup")

# 4. Loose per-seed CSVs from thesis_stats.py -> businessAware/stats/
for c in sorted(glob.glob("per_seed_results*.csv")):
    plan(c, os.path.join("businessAware", "stats"))

pycache = [p for p in glob.glob("**/__pycache__", recursive=True)]

print("Planned moves:")
for src, dest in moves:
    print(f"  {src:<55} -> {dest}/")
for p in pycache:
    print(f"  delete {p}/ (recreated automatically)")
if not moves and not pycache:
    print("  nothing to do — the folder is already tidy.")

if not args.apply:
    print("\nDry run only. Re-run with --apply to back up and move.")
    sys.exit(0)

# Backup first (everything except the Criteo data, which is not modified)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup = os.path.join(os.path.dirname(ROOT), f"{os.path.basename(ROOT)}_backup_{stamp}")
print(f"\nBacking up to {backup} (excluding criteo/) ...")
shutil.copytree(ROOT, backup, ignore=shutil.ignore_patterns("criteo", "__pycache__"))
print("Backup done.")

for src, dest in moves:
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, os.path.basename(src))
    if os.path.exists(target):
        print(f"  skip {src}: {target} already exists")
        continue
    shutil.move(src, target)
for p in pycache:
    shutil.rmtree(p, ignore_errors=True)
print(f"\nDone. {len(moves)} items moved. Backup kept at {backup}.")
