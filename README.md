# HBA-RL: Hierarchical Business-Aware Reinforcement Learning for E-commerce Cloud ML Workload Scheduling

This repository contains the experimental code used to train, ablate, and benchmark **HBA-RL**, a two-level hierarchical PPO scheduler for machine-learning workloads on shared cloud infrastructure. A single macro-level PPO policy issues a 12-dimensional business subgoal (priority, resource, and SLA components) and a discrete routing decision assigning each incoming job to one of the five server nodes; the micro-level PPO policy at that node consumes the subgoal and decides a 2-dimensional action: which queued job to admit, and a resource-allocation fraction. The framework is evaluated against six heuristic baselines and a single-level ("Flat") PPO ablation, on both a synthetic workload generator and the real-world Criteo Display Advertising dataset.

## Question being tested

Does a hierarchical decomposition improve business-aware scheduling, once capacity, reward shaping and training settings are matched? HBA-RL is compared with six heuristics and with Flat PPO (original and matched, ABL-1) on a synthetic generator and on Criteo-derived workloads. The Criteo mapping is an arbitrary fixed assignment, not real cloud workloads. The two agents differ in several respects, listed in [Known differences](#known-differences-between-hba-rl-and-flat-ppo).

## Architecture

HBA-RL has a global job queue, one macro-controller and five per-node micro-controllers. Every controller is a PPO actor-critic.

```
                 Global job queue  +  cluster state (50-dim)
                                  |
                 +----------------v-----------------+
                 |  Macro-controller (PPO)          |
                 |  trunk 50 -> 256 -> 256 -> 128   |
                 |  LayerNorm, orthogonal init      |
                 +-------+------------------+-------+
                         |                  |
          12-dim business subgoal      routing head
          (4 priority, 4 resource,     (softmax over
           4 SLA, softmax)              5 nodes)
                         |                  |
        +--------+-------+--+--------+------+--+
        v        v          v        v         v
     Micro 0  Micro 1    Micro 2  Micro 3   Micro 4      one PPO policy per node
     input: state (50) + subgoal (12) = 62 -> 128 -> 128 -> 64
     output: [workload index, allocation fraction]
        |        |          |        |         |
        v        v          v        v         v
     Node 0   Node 1     Node 2   Node 3    Node 4       heterogeneous GPU/CPU/memory
                                  |
               reward, SLA status, utilisation -> both levels
```

| Component | Input | Network | Output |
|---|---|---|---|
| Macro-controller (`HighLevelPolicy`) | 50-dim cluster state | 50 → 256 → 256 → 128 shared trunk | 12-dim subgoal and a 5-way routing decision |
| Micro-controller × 5 (`LocalResourceAllocator`) | 50-dim state + 12-dim subgoal (62) | 62 → 128 → 128 → 64 | 2-dim action: workload index, allocation fraction |
| Flat PPO ablation (`FlatPolicy`) | 50-dim cluster state | 50 → 256 → 256 → 128 (default); 50 → 416 → 416 → 208 with `--matched` | 3-dim action: workload, node, allocation fraction |

Points to know when reading the code:

- The 50-dim state has 19 informative features, each scaled to roughly [0, 1], followed by 31 zeros. It carries CPU, GPU and memory utilisation for all five nodes (A15; `--legacy-observation` restores the old three-node state).
- All five micro-controllers see the same cluster state. Each has its own optimizer and rollout buffer.
- The environment does not consume the allocation fraction. Allocation is fixed at 1.0 for every admitted job.
- HBA-RL has 282,795 trainable parameters in total (macro 115,090; each micro 33,541). Flat PPO has 113,287 by default and 283,927 with `--matched`.
- The routing shaping term (+0.5 / -0.3 / +0.2) is added during HBA-RL training only (and Flat PPO training with `--matched`). It is applied only to jobs that were actually admitted, based on the job's priority and the node it was routed to (`--legacy-shaping` restores the old rule). It is never part of evaluation or of the reported reward.

## Repository layout

```
env/         hrl_scheduler_simulation.py (Gymnasium environment), criteo.py (Criteo -> job-attribute mapping)
agents/      hrl_agent.py (HBA-RL; ABL-3 zero subgoal), flat_ppo.py (Flat PPO; ABL-1 matched variant)
baselines/   baseline_schedulers.py (six heuristics, one class each; also the CLI that evaluates them)
configs/     one YAML per experiment (hba_rl, flat_ppo, flat_ppo_matched, abl2, abl3, abl4_p0.6, abl4_p0.9, baselines)
data/        make_criteo_extract.py, criteo_extract.sha256 (written by the script)
scripts/     train_hrl_scheduler.py, train_flat_ppo.py, eval_all.py, stats.py, make_tables.py, make_figures.py,
             make_fig1.py, workload_stats.py
businessAware/**/checkpoint_best.pt, run_config.json, training_summary.json   (checkpoints: commit or release asset)
results/     logs/ (per-episode CSV, every method), tables/, figures/, reproduction_log.txt
common.py, evaluation.py, seed_utils.py   shared modules (constants, env factory, evaluation routine, seeding)
reproduce.sh --quick | --full | --smoke
legacy_v0/   superseded scripts and results. Do NOT cite these numbers.
```

## Table and figure map

| Item | Produced by | File |
|---|---|---|
| R1a/b main results (both SD columns) | scripts/stats.py | results/tables/R1_main_results.csv |
| R2 per-seed results (with stop episode) | stats.py + make_tables.py | R2_per_seed.csv |
| R3 statistical comparison; IQM CIs | scripts/stats.py | R3_statistical_comparison.csv, R3b_vs_heuristics.csv, IQM_profit.csv |
| R4 routing share (evaluation only) | scripts/make_tables.py | R4_routing_share.csv |
| R5 training behaviour | scripts/make_tables.py | R5_training_behaviour.csv |
| R6 ablations ABL-1..4 | scripts/make_tables.py | R6_ablations.csv |
| M1 hyperparameters | scripts/make_tables.py | M1_hyperparameters.csv |
| M2 environment parameters | scripts/make_tables.py | M2_environment.csv |
| M3 attribute statistics, load level | scripts/workload_stats.py | M3_workload_attributes.csv, M3_load_level.csv |
| Fig. 1 architecture | scripts/make_fig1.py | results/figures/fig1_architecture.pdf |
| Fig. 3 training curves | scripts/make_figures.py | fig3_training_curves.pdf (+ _env) |
| Fig. 4 profit, all methods | scripts/make_figures.py | fig4_profit.pdf |
| Routing share figure, M3 histograms | make_figures.py, workload_stats.py | fig_routing_share.pdf, M3_criteo_attribute_histograms.pdf |
| Figure captions (error types, panels) | scripts/make_figures.py | results/figures/figure_captions.md |

## Evaluation protocol (same for all methods)

* 100 episodes on the held-out evaluation split (Criteo: last 40,000 of 200,000 rows; train: first 160,000).
* Episode seeds 42 + e, e = 0..99, for every method, so each method sees the same job stream (generated at `reset()`).
* Deterministic actions (argmax / mean), unshaped environment reward.
* **Profit = revenue - SLA penalties - operating cost.** Penalties are reported in their own column.
* **SLA compliance = 1 - (late + expired) / (on-time + late + expired).**
* Per-episode logs: `results/logs/eval_<dataset>_<method>_seed<N>.csv` (revenue, penalties, cost, profit, counts, reward components, routing counts, offered load). Every table is computed from these files only.
* Reported: both SDs (across episodes within a seed, and across seed means), Welch t on seed means, paired Wilcoxon on per-episode profit, stratified-bootstrap IQM CIs.
* Training seeds are `10_000_000 + seed*100_000 + episode`, so they cannot collide with the evaluation seeds.

## Running

```
git clone <repo-url> && cd hba-rl && git checkout <release-tag>
pip install -r requirements.txt
python data/make_criteo_extract.py --src /path/to/train.txt --out criteo     # checks the SHA-256 on later runs
bash reproduce.sh --quick criteo     # evaluation only, from the saved checkpoints: all tables and figures
bash reproduce.sh --full  criteo     # full retraining (10 seeds x 2000 episodes), then evaluation
bash reproduce.sh --smoke criteo     # 2 seeds x 100 episodes: plumbing test only, never report these numbers
```

The console output is saved to `results/reproduction_log.txt` with the run time of each step.
Individual runs: `python scripts/train_hrl_scheduler.py criteo --config configs/hba_rl.yaml --seed 7`,
`python scripts/train_flat_ppo.py criteo --config configs/flat_ppo_matched.yaml --seed 7`,
`python baselines/baseline_schedulers.py --mode criteo criteo`. Ablations need `--tag` (the configs set it):
`--no-shaping` (ABL-2), `--zero-subgoal` (ABL-3), `--arrival-prob 0.6|0.9` (ABL-4; same value and tag for heuristics,
HBA-RL and Flat PPO), `--matched` (ABL-1). `--eval-only` re-evaluates a saved `checkpoint_best.pt`.
`--legacy-shaping` / `--legacy-observation` restore the pre-revision behaviour.

Hardware and run times: filled from `run_config.json` / `training_summary.json` (R5) after the full run.

## Known differences between HBA-RL and Flat PPO

Unmatched Flat PPO (original): hidden width 256 (~fewer parameters), continuous node output, no routing shaping, patience 300 on single-episode reward.
Matched Flat PPO (ABL-1, `--matched`): width 416 (~283k parameters), categorical 5-way node head, same shaping, KL 0.05, patience 600 on the 50-episode grid, checkpoint on the mean of the last 10 episodes.
Both are reported. Table M1 (generated by `make_tables.py` from each run's `run_config.json`) lists every hyperparameter side by side.

## Limitations (stated honestly)

* Criteo is click-labelled; the mapping to job attributes is arbitrary and fixed. The Criteo rows are real click-log values turned into simulated jobs, so the results do not show behaviour on real cloud traces.
* At the nominal arrival probability (0.3) the cluster is lightly loaded (offered GPU load about 0.28), so simple heuristics tie with the learned agents. ABL-4 raises the arrival probability to 0.6 and 0.9. The capacity-matched Flat PPO, retrained at each load, equals HBA-RL on synthetic workloads and earns 4.0% (p < 0.001) and 2.2% (p = 0.008) more on Criteo at p = 0.6 and 0.9, so the gain over simple heuristics under load is not specific to the hierarchy.
* HBA-RL and the matched Flat PPO route only to nodes 0 and 1; nodes 2 to 4 are not used.
* Checkpoint selection uses training-split episodes; a separate validation split is not implemented.
* No YAML configs are executed; hyperparameters are CLI flags and defaults, recorded in `run_config.json`.

## Main findings (10 training seeds, 100 shared evaluation episodes)

| Setting | HBA-RL | Original Flat PPO | Matched Flat PPO (ABL-1) |
|---|---|---|---|
| Synthetic, p = 0.3 | $17,132 | $12,552 | $17,257 |
| Criteo, p = 0.3 | $6,059 | $5,483 | $6,055 |

HBA-RL beats the original Flat PPO, but a capacity-matched Flat PPO equals it (ABL-1). Removing the subgoal (ABL-3) or the routing shaping (ABL-2) changes profit without significance. The hierarchy claim is therefore not supported; see the manuscript, Sections 5 and 6. All numbers regenerate with `reproduce.sh`.

## How the reported runs were produced

The full run (10 seeds per agent, 3 agents, 2 datasets) was executed on Kaggle CPU notebooks with a 4-process parallel runner, in three parts, which are in `notebooks/`:

1. `kaggle_run1_synthetic.ipynb`: all synthetic-data training runs (about 211 minutes).
2. `kaggle_run2_criteo_and_evaluation.ipynb`: all Criteo training runs, evaluation of every method and all tables and figures.
3. `kaggle_abl4_load_sensitivity.ipynb`: ABL-4 (arrival probability 0.6 and 0.9) retrained at each load.

Software: Python 3.13, the pinned versions in `requirements.txt`. The notebooks only wrap the same scripts that `reproduce.sh` calls.

## Changes relative to the original code
See `docs/CHANGELOG.md`.

## Data and code availability
Code: this repository, private until the paper is accepted and open to the editor and reviewers on request. It will be made public and archived on Zenodo on acceptance. Data: Criteo Display Advertising Challenge (https://kaggle.com/competitions/criteo-display-ad-challenge; not redistributed). `data/make_criteo_extract.py` rebuilds the 200,000-row extract (the first 200,000 rows of `train.txt`) and verifies its SHA-256 (69824b15277d6dd1cfc1bba0ab9e2874f11556942e86fbaaa04a0514c18d9bd0).
