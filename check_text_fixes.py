#!/usr/bin/env python3
"""
check_text_fixes.py  -  checks that text fixes T1 to T18 are in your clean manuscript

    pip install python-docx
    python check_text_fixes.py "HBA-RL_Manuscript_CLEAN_revision3.docx"

It only READS your Word file. For each fix it checks two things:
  OLD  - the old wording is gone
  NEW  - a key phrase of the new wording is present
A fix shows DONE only when both hold. Fix every TODO in Word, save, and run again.
If you chose your own wording for a fix and it reads correctly, a TODO on the NEW
check is acceptable - write the reason in the response matrix.
"""
import re, sys, unicodedata

try:
    import docx
except ImportError:
    sys.exit("Install python-docx first:  pip install python-docx")

# (fix ID, old text that must be GONE, key phrase of the new text that must be PRESENT)
CHECKS = [
    ("T1",  "−0.1% on Criteo", "−0.06% on Criteo"),
    ("T1",  "as the result that holds", "11% to 16% less on synthetic workloads"),
    ("T2",  "and at p = 0.9 more than Priority-Based and SJF.", "7–8% below Priority-Based and SJF at p = 0.6"),
    ("T2",  "a training procedure that is stable across seeds", "shared by HBA-RL and the matched Flat PPO"),
    ("T3",  "and the difference is not significant (p = 0.20; Table R6)", "Wilcoxon test on per-episode profit does"),
    ("T3",  "$4 below HBA-RL (p = 0.86)", "Wilcoxon p = 0.87"),
    ("T4",  None, "the Welch test is primary"),
    ("T5",  "exceeds the simple heuristics clearly", "falls 11–16% below them on synthetic workloads"),
    ("T6",  "A learned scheduler is worth considering", "16–17 jobs per episode expire"),
    ("T7",  None, "83–86% of Criteo jobs sit at that cap"),
    ("T8",  "heavy-tailed", "skewed"),
    ("T8",  None, "61–65% of job values at the $50 floor"),
    ("T9",  "is enough to test whether hierarchy adds value", "exercises the routing decision only in part"),
    ("T9",  None, "On Criteo this is forced by the mapping"),
    ("T10", "Business-Aware Hierarchical Reinforcement Learning for Scheduling", None),
    ("T11", "first version of this paper", "this gap reflects the weaker original baseline"),
    ("T11", "−26.8 percentage points", "−26.6 percentage points"),
    ("T11", "more heterogeneous workload", "more dispersed"),
    ("T12", "Real-world dataset evaluation", "Click-log-derived workloads"),
    ("T12", "Performance validation:", "Matched comparison protocol"),
    ("T12", "whether the benefit of hierarchical decomposition differs", "whether the result differs"),
    ("T13", "The structural deficit:", "The structural question"),
    ("T14", "A flat architecture also creates a credit-assignment problem", "from one reward signal"),
    ("T15", "about 283,000 parameters, a categorical", "about 285,000 parameters"),
    ("T15", "about 284,000", None),
    ("T16", "at least 96% of its placements", "at least 96% of its routing decisions"),
    ("T17", None, "reached the 2,000-episode cap"),
    ("T18", None, "Penalties ($)"),
]


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace(" ", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s)


def text_of(path):
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.extend(c.text for c in row.cells)
    return norm(" ".join(parts))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    body = text_of(sys.argv[1])
    todo = 0
    print(f"{'Fix':5s} {'Old gone':9s} {'New in':7s} Status")
    for fid, old, new in CHECKS:
        old_ok = True if old is None else norm(old) not in body
        new_ok = True if new is None else norm(new) in body
        status = "DONE" if old_ok and new_ok else "TODO"
        todo += status == "TODO"
        detail = ""
        if not old_ok:
            detail += f'  still has: "{old}"'
        if not new_ok:
            detail += f'  missing: "{new}"'
        print(f"{fid:5s} {'yes' if old_ok else 'NO':9s} {'yes' if new_ok else 'NO':7s} {status}{detail}")
    print(f"\n{len(CHECKS) - todo} of {len(CHECKS)} checks DONE.")
    print("All text fixes are in." if todo == 0 else "Fix the TODO lines in Word, save, and run again.")


if __name__ == "__main__":
    main()
