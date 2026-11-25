#!/usr/bin/env bash
# run_elevation.sh - elevation additions: tests, the eight heuristics at three loads, then every table.
#
#   bash scripts/run_elevation.sh [CRITEO_DIR]
#
# Without CRITEO_DIR only the synthetic runs are done. Learned agents are NOT retrained: their
# per-episode logs must already be in results/logs (from reproduce.sh --quick or the release bundle).
set -euo pipefail
cd "$(dirname "$0")/.."
CRITEO="${1:-}"
PY="${PYTHON:-python}"

echo "== 1. unit tests"
"$PY" -m unittest tests/test_elevation.py -v

echo "== 2. heuristics (all eight) at p = 0.3, 0.6, 0.9"
for DS in synth criteo; do
  if [ "$DS" = criteo ] && [ -z "$CRITEO" ]; then echo "   skip criteo (no CRITEO_DIR given)"; continue; fi
  ARGS=(--mode "$DS"); [ "$DS" = criteo ] && ARGS+=("$CRITEO")
  "$PY" baselines/baseline_schedulers.py "${ARGS[@]}"
  for P in 0.6 0.9; do
    "$PY" baselines/baseline_schedulers.py "${ARGS[@]}" --arrival-prob "$P" --tag "abl4_p$P"
  done
done

echo "== 3. tables (R1-R3, R3b/R3c with p_holm, R6 with the equivalence columns)"
"$PY" scripts/stats.py
"$PY" scripts/stats.py --flat "Flat PPO (matched)"
"$PY" scripts/make_tables.py
echo "Done. Compare results/tables with the expected/ folder of the elevation pack."
