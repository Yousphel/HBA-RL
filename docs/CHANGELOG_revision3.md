## Revision 3 (v1.1-revision3)
- `scripts/stats.py` now also prints and writes R3c (matched Flat PPO against the heuristics at the same arrival probability), so R3b, R3c and the load-matched R6 all appear in the console output of the log.
- `scripts/make_figures.py`: Fig. 3 label mapping fixed (the matched Flat PPO was dropped because of a label mismatch); Fig. 3 now draws all three agents.
- Added `verify_revision2.py` (provenance, SHA-256, load-matched checks, routing share by load, Table R4b), `make_main_tables.py` (Tables 2 and 3 with a Penalties column), `plot_training_curves.py` (Fig. 3) and `check_text_fixes.py`.
- Reproduction log: header says revision 3; Part 4 holds the final statistics and tables at this tag.
- Supplement: added Table R4b (routing share per node by arrival probability); R6 regenerated with load-matched references.
