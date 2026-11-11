## Revision 3.1 (v1.1-revision3.1)
- `verify_revision2.py`: restored to the supplied version; load labels p0_6 and p0_9 in the log file names are now read as p0.6 and p0.9, so the heavy-load checks run.
- `scripts/make_tables.py`: the Welch t of Table R6 is variant minus reference, so t has the sign of the gap (p-values unchanged); p-values below 0.001 print as "< 0.001".
- `scripts/stats.py`: p-values below 0.001 print as "< 0.001" in R3, R3b and R3c.
- `scripts/export_training_rewards.py` and `results/tables/training_rewards_long.csv`: the table that plot_training_curves.py uses for Fig. 3.
- Supplement: Table R4b has a "Load" column (p = 0.3, 0.6, 0.9, and ABL-2 / ABL-3 at p = 0.3); R6 and the p-value cells follow the changes above.
- `.gitignore`: run outputs (businessAware/, results/logs, results/figures, other results/tables files) are not tracked.
