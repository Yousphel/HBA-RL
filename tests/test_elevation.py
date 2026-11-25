"""
Unit tests for the elevation additions: EDF and Value-Density heuristics, TOST, Holm.

    python -m unittest tests/test_elevation.py -v        (from the repository root)

The TOST check values come from the R1 seed-mean summaries in the manuscript (n = 10 seeds per arm).
"""
import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

import numpy as np

from baselines.baseline_schedulers import (EDFScheduler, ValueDensityScheduler, all_schedulers)
from equivalence import EQ_MARGIN_PCT, holm, tost, tost_from_summary

RES = {"cpu": 1.0, "memory": 1.0}


def job(i, deadline, revenue, duration, cpu=0.2):
    return SimpleNamespace(workload_id=i, deadline=deadline, revenue_value=revenue,
                           estimated_duration=duration, resource_requirements={"cpu": cpu, "memory": 0.1})


def toy_env(jobs, free=(1.0, 1.0, 1.0, 1.0, 1.0)):
    nodes = [SimpleNamespace(available_resources={"cpu": f, "memory": 1.0}) for f in free]
    return SimpleNamespace(pending_workloads=list(jobs), nodes=nodes, num_nodes=len(nodes))


def decode(action, env):
    """Same decoding as the environment: index = floor(a * count)."""
    return int(action[0] * len(env.pending_workloads)), int(action[1] * env.num_nodes)


class TestHeuristics(unittest.TestCase):
    def test_registered(self):
        names = [s.name for s in all_schedulers(5)]
        self.assertEqual(len(names), 8)
        self.assertIn("EDF", names)
        self.assertIn("Value-Density", names)

    def test_edf_picks_nearest_deadline(self):
        env = toy_env([job(0, 30, 100, 5), job(1, 8, 10, 20), job(2, 15, 500, 2)])
        self.assertEqual(decode(EDFScheduler().get_action(env), env), (1, 0))

    def test_edf_tie_keeps_arrival_order(self):
        env = toy_env([job(0, 9, 1, 1), job(1, 9, 999, 1)])
        self.assertEqual(decode(EDFScheduler().get_action(env), env)[0], 0)

    def test_edf_skips_job_that_does_not_fit(self):
        env = toy_env([job(0, 5, 1, 1, cpu=0.9), job(1, 7, 1, 1, cpu=0.2)], free=(0.5,) * 5)
        self.assertEqual(decode(EDFScheduler().get_action(env), env), (1, 0))

    def test_edf_uses_first_node_that_fits(self):
        env = toy_env([job(0, 5, 1, 1, cpu=0.6)], free=(0.1, 0.5, 0.7, 1.0, 1.0))
        self.assertEqual(decode(EDFScheduler().get_action(env), env), (0, 2))

    def test_value_density_picks_highest_value_per_step(self):
        # densities: 100/5 = 20, 10/20 = 0.5, 500/2 = 250
        env = toy_env([job(0, 30, 100, 5), job(1, 8, 10, 20), job(2, 15, 500, 2)])
        self.assertEqual(decode(ValueDensityScheduler().get_action(env), env), (2, 0))

    def test_value_density_is_not_highest_revenue(self):
        # job 0 has the higher revenue, job 1 the higher density (60/2 = 30 > 200/10 = 20)
        env = toy_env([job(0, 9, 200, 10), job(1, 9, 60, 2)])
        self.assertEqual(decode(ValueDensityScheduler().get_action(env), env)[0], 1)

    def test_empty_queue_falls_back(self):
        env = toy_env([])
        for s in (EDFScheduler(), ValueDensityScheduler()):
            self.assertEqual(len(s.get_action(env)), 3)

    def test_nothing_fits_falls_back_to_job0_node0(self):
        env = toy_env([job(0, 5, 1, 1, cpu=0.9), job(1, 3, 9, 1, cpu=0.9)], free=(0.1,) * 5)
        for s in (EDFScheduler(), ValueDensityScheduler()):
            self.assertEqual(decode(s.get_action(env), env), (0, 0))


# R1 seed-mean summaries: (mean, SD across 10 seed means)
HBA = {"synth": (17132.157, 270.099), "criteo": (6058.657, 37.672)}
MATCHED = {"synth": (17257.481, 104.993), "criteo": (6054.942, 51.546)}
ABL3 = {"synth": (16712.853, 1281.308), "criteo": (6031.11, 34.061)}


def tost_summary(variant, ds):
    (mv, sv), (mr, sr) = variant[ds], HBA[ds]
    return tost_from_summary(mv, sv, 10, mr, sr, 10, abs(mr) * EQ_MARGIN_PCT / 100)


class TestTOST(unittest.TestCase):
    def test_margin_is_two_percent(self):
        self.assertEqual(EQ_MARGIN_PCT, 2.0)

    def test_matched_synth_check_values(self):
        r = tost_summary(MATCHED, "synth")
        self.assertAlmostEqual(r["ci90_low"], -38.4, delta=0.1)
        self.assertAlmostEqual(r["ci90_high"], 289.0, delta=0.1)
        self.assertTrue(r["equivalent"])

    def test_matched_criteo_check_values(self):
        r = tost_summary(MATCHED, "criteo")
        self.assertAlmostEqual(r["ci90_low"], -38.9, delta=0.1)
        self.assertAlmostEqual(r["ci90_high"], 31.5, delta=0.1)
        self.assertTrue(r["equivalent"])

    def test_zero_subgoal_synth_not_equivalent(self):
        r = tost_summary(ABL3, "synth")
        self.assertAlmostEqual(r["ci90_low"], -1171.4, delta=0.1)
        self.assertAlmostEqual(r["ci90_high"], 332.8, delta=0.1)
        self.assertFalse(r["equivalent"])

    def test_p_tost_agrees_with_ci(self):
        for v in (MATCHED, ABL3):
            for ds in ("synth", "criteo"):
                r = tost_summary(v, ds)
                self.assertEqual(r["equivalent"], r["p_tost"] < 0.05)

    def test_array_version_matches_summary_version(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(100, 3, 10), rng.normal(101, 5, 10)
        r1 = tost(a, b)
        r2 = tost_from_summary(a.mean(), a.std(ddof=1), 10, b.mean(), b.std(ddof=1), 10, abs(b.mean()) * 0.02)
        self.assertAlmostEqual(r1["ci90_low"], r2["ci90_low"])
        self.assertAlmostEqual(r1["p_tost"], r2["p_tost"])


class TestHolm(unittest.TestCase):
    def test_textbook_example(self):
        # m = 4: sorted p 0.01, 0.02, 0.03, 0.04 -> 0.04, 0.06, 0.06, 0.06
        self.assertTrue(np.allclose(holm([0.03, 0.01, 0.04, 0.02]), [0.06, 0.04, 0.06, 0.06]))

    def test_capped_at_one_and_nan_kept(self):
        out = holm([0.5, np.nan, 0.9])
        self.assertTrue(np.isnan(out[1]))
        self.assertEqual(out[2], 1.0)

    def test_matches_statsmodels_if_present(self):
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            self.skipTest("statsmodels not installed")
        p = np.array([0.001, 0.2, 0.03, 0.04, 0.5, 0.012])
        self.assertTrue(np.allclose(holm(p), multipletests(p, method="holm")[1]))


if __name__ == "__main__":
    unittest.main()
