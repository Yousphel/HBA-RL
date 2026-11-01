# configs/
One YAML per experiment. Use `python train_hrl_scheduler.py [CRITEO_DIR] --config configs/hba_rl.yaml --seed 7`.
Keys are the command-line arguments; flags on the command line override the file. `baselines.yaml` has no `seed`:
heuristics are deterministic per episode seed (42 + e). Every run prints all settings and seeds at the start of its log.
The ablation configs set `tag` and the flag; `abl4_*` apply to HBA-RL, Flat PPO and the heuristics alike.
