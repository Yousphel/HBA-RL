# Change log (tied to supervisor issue IDs)

| ID | Change |
|---|---|
| A1 | One `evaluation.evaluate()` for all methods; same split, seeds 42+e, deterministic actions |
| A2 | Penalties separated from cost; profit = revenue - penalties - cost; `penalties` column |
| A3 | Single SLA formula |
| A4 | Both SD columns in R1 |
| A12 | Reward components logged per episode (`r_*`) |
| A13 | Routing share counted on evaluation only (deterministic) |
| A15 | State observes all 5 nodes (`--legacy-observation` for the old 3-node state) |
| A16 | Offered load per resource in `workload_stats.py` |
| ABL-1..4 | `--matched`, `--no-shaping`, `--zero-subgoal`, `--arrival-prob` (with `--tag`) |
| Bugs found | Criteo episodes replayed ~30 identical rows; learned agents and heuristics saw different seeds/shuffles; shaping applied to non-admitted routes; penalties folded into cost; stochastic evaluation; training/eval seed collisions. **All earlier numbers are void; re-run everything.** |
| D3 | M1 now has HBA-RL macro / micro / Flat / Flat matched columns; unmatched rows flagged |
| D7 | Flat PPO patience (300, checked each episode) documented; `--matched` uses the 50-episode grid |
| D17 | R6 ablation table (variant vs matching HBA-RL reference, Welch t on seed means) |
| Repo | env/, agents/, baselines/, configs/ (YAML, `--config`), scripts/eval_all.py, `reproduce.sh --quick` = evaluation from checkpoints, reproduction log with run times |
| Figures | Fig. 1 redrawn from the code; Fig. 2 and Fig. 5 dropped; Fig. 3/4 per spec; panel letters, captions generated |


## Revision 1 (final run)
- Full retraining and evaluation executed on Kaggle CPU in three parts (see README, `notebooks/`); 10 training seeds per agent.
- `.reshape(-1)` replaces `.squeeze()` in `agents/hrl_agent.py` and `agents/flat_ppo.py` (identical numerics; removes a single-job warning).
- `scripts/make_tables.py::table_r6`: fixed the reference logic for ablation comparisons.
- `scripts/stats.py::table_vs_heuristics`: heuristics are now compared with HBA-RL at the same arrival probability and the same episode seeds (the earlier version mixed nominal-load HBA-RL with heavier-load heuristics).
- requirements.txt pins the versions used on Kaggle.


## Revision 1, ABL-4 matched Flat PPO
- Added matched Flat PPO at arrival probability 0.6 and 0.9 (Part 3 log: docs/reproduction_log_part3_abl4.txt).
- Fixed table_vs_heuristics so a method is paired with heuristics at the same load; added R3c (matched vs heuristics).
- Result: matched Flat PPO >= HBA-RL under load on Criteo (+4.0%, +2.2%), no difference on synthetic.
