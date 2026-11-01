# Commit plan for the revision (one commit per group, issue IDs in the message)

git checkout -b revision-1
(copy the revised files over the repository, then commit in groups)

1. git add env agents seed_utils.py common.py && git commit -m "A2,A3,A15,A18,D5,D10: profit with penalties, one SLA formula, five-node state, action order"
2. git add baselines && git commit -m "D6: heuristic ordering, no-fit rule and tie-breaking documented"
3. git add scripts/train_flat_ppo.py agents/flat_ppo.py && git commit -m "C7,D7,ABL-1: Flat PPO specification and matched variant"
4. git add scripts/train_hrl_scheduler.py && git commit -m "A17,ABL-2,ABL-3: zero-subgoal and no-shaping switches"
5. git add data && git commit -m "A10,D4: Criteo extract script and SHA-256"
6. git add evaluation.py scripts/eval_all.py scripts/stats.py scripts/make_tables.py scripts/make_figures.py scripts/make_fig1.py scripts/workload_stats.py && git commit -m "A1,A4,A11-A13,A16,D11-D17,ABL-4: shared evaluation, statistics, tables and figures"
7. git add notebooks reproduce.sh requirements.txt configs && git commit -m "D-HW,D17: Kaggle notebooks, pinned requirements, reproduce.sh"
8. git add README.md CITATION.cff LICENSE docs && git commit -m "D9: README, changelog, availability statements"
9. git mv the old flat files into legacy_v0 (already done in this layout) 
10. git tag -a v1.0-revision1 -m "HBA-RL revision 1" && git push origin revision-1 --tags
Then: git rev-parse HEAD   (send me the full hash for the manuscript)

Each commit message must end with the attribution lines your tooling adds, if you use it.
