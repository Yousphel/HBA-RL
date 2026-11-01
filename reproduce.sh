#!/usr/bin/env bash
# Reproduce every table and figure.
#   bash reproduce.sh --quick [CRITEO_DIR]   EVAL ONLY, from the saved checkpoints in businessAware/ (minutes)
#   bash reproduce.sh --full  [CRITEO_DIR]   full retraining, 10 seeds x 2000 episodes, then evaluation (hours)
#   bash reproduce.sh --smoke [CRITEO_DIR]   2 seeds x 100 episodes: a plumbing test, NOT for reporting
# CRITEO_DIR: folder with the 200,000-row train.txt made by data/make_criteo_extract.py (omit = synthetic only).
# The console output is also written to results/reproduction_log.txt (with run times).
set -euo pipefail
MODE="${1:---quick}"; CRITEO="${2:-}"
mkdir -p results
exec > >(tee results/reproduction_log.txt) 2>&1
START=$(date +%s)
echo "reproduce.sh $MODE $CRITEO  started $(date -u +%FT%TZ)"
python --version; pip freeze 2>/dev/null | grep -iE "^(torch|numpy|gymnasium|scipy|pandas|matplotlib)==" || true

if [ -n "$CRITEO" ]; then python data/make_criteo_extract.py --check "$CRITEO"; fi

timed () {   # timed "label" command ...  (prints the elapsed seconds; does not need GNU time)
  local msg="$1"; shift; local t0=$(date +%s)
  "$@"; local rc=$?
  echo "  [$(( $(date +%s) - t0 )) s] $msg"
  return $rc
}

train_all () {
  local SEEDS="$1" EPS="$2" DATASETS="synth"; [ -n "$CRITEO" ] && DATASETS="synth criteo"
  for ds in $DATASETS; do
    C=""; [ "$ds" = criteo ] && C="$CRITEO"
    for s in $SEEDS; do
      timed "hba-rl $ds seed $s" python scripts/train_hrl_scheduler.py $C --seed "$s" --episodes "$EPS"
      timed "flat $ds seed $s"   python scripts/train_flat_ppo.py      $C --seed "$s" --episodes "$EPS"
      timed "ABL-1 $ds seed $s"  python scripts/train_flat_ppo.py      $C --seed "$s" --episodes "$EPS" --matched
      timed "ABL-2 $ds seed $s"  python scripts/train_hrl_scheduler.py $C --seed "$s" --episodes "$EPS" --no-shaping   --tag abl2_noshaping
      timed "ABL-3 $ds seed $s"  python scripts/train_hrl_scheduler.py $C --seed "$s" --episodes "$EPS" --zero-subgoal --tag abl3_zerosubgoal
      for p in 0.6 0.9; do   # ABL-4: all three method families at the same arrival probability, both datasets
        timed "ABL-4 hba p=$p $ds seed $s"  python scripts/train_hrl_scheduler.py $C --seed "$s" --episodes "$EPS" --arrival-prob $p --tag abl4_p$p
        timed "ABL-4 flat p=$p $ds seed $s" python scripts/train_flat_ppo.py      $C --seed "$s" --episodes "$EPS" --arrival-prob $p --tag abl4_p$p
      done
    done
  done
}

case "$MODE" in
  --quick) ;;                                                     # evaluation only
  --smoke) train_all "7 42" 100 ;;
  --full)  train_all "7 42 123 456 11 22 33 44 55 66" 2000 ;;
  *) echo "usage: $0 --quick|--full|--smoke [CRITEO_DIR]"; exit 1 ;;
esac

python scripts/eval_all.py ${CRITEO:+"$CRITEO"}                   # heuristics + all checkpoints on seeds 42+e, then stats/tables/figures
echo "Done in $(( $(date +%s) - START )) s. Tables: results/tables  Figures: results/figures  Logs: results/logs"
