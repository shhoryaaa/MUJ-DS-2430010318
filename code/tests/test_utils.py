"""Unit tests (no TensorFlow needed):  python -m unittest discover -s code/tests -v   (run from the repository root)"""
import os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import LABELS
from src.data import patient_split
from src.evaluate import tune_thresholds, hanley_ci, macro_auc, per_class_report


class TestPipeline(unittest.TestCase):
    def test_patient_split_has_no_leakage_and_is_deterministic(self):
        df = pd.DataFrame({"pid": np.repeat(np.arange(200), 3), "x": range(600)})
        a = patient_split(df, "pid"); b = patient_split(df, "pid")
        self.assertEqual(sum(len(p) for p in a), 600)
        self.assertFalse(set(a[0].pid) & set(a[1].pid) | set(a[0].pid) & set(a[2].pid) | set(a[1].pid) & set(a[2].pid))
        self.assertTrue(a[2].equals(b[2]))
        self.assertAlmostEqual(len(set(a[0].pid)) / 200, 0.70, delta=0.01)

    def test_threshold_search_prefers_low_threshold_for_shifted_scores(self):
        rng = np.random.RandomState(0); y = (rng.rand(400, 14) < 0.1).astype(float)
        p = np.clip(0.02 + 0.2 * y + 0.05 * rng.rand(400, 14), 0, 1)       # positives score ~0.2, never reach 0.5
        thr = tune_thresholds(y, p)
        self.assertTrue(all(0.05 <= t <= 0.30 for t in thr.values()))

    def test_hanley_ci_contains_auc_and_widens_with_few_positives(self):
        lo1, hi1 = hanley_ci(0.8, 200, 600); lo2, hi2 = hanley_ci(0.8, 5, 800)
        self.assertTrue(lo1 < 0.8 < hi1 and lo2 < 0.8 < hi2)
        self.assertGreater(hi2 - lo2, hi1 - lo1)

    def test_report_counts_add_up(self):
        rng = np.random.RandomState(1); y = (rng.rand(300, 14) < 0.15).astype(float); p = rng.rand(300, 14)
        for r in per_class_report(y, p, {d: 0.5 for d in LABELS}):
            self.assertEqual(r["tp"] + r["fp"] + r["fn"] + r["tn"], 300)
            self.assertEqual(r["tp"] + r["fn"], r["positives"])
        self.assertTrue(0.4 < macro_auc(y, p) < 0.6)


if __name__ == "__main__":
    unittest.main()
