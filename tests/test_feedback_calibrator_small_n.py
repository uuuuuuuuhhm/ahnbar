import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_model import _fit_feedback_calibrator


class FeedbackCalibratorSmallNTests(unittest.TestCase):
    def test_default_min_rows_collects_but_apply_threshold_blocks_30_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "prediction_feedback_training.csv"
            rows = []
            for i in range(30):
                y = int(i % 2 == 0)
                rows.append(
                    {
                        "model_home_win_probability": 0.58 if y == 1 else 0.42,
                        "actual_home_win": y,
                        "run_timestamp": f"2026-04-{(i % 28) + 1:02d}T12:00:00",
                    }
                )
            pd.DataFrame(rows).to_csv(p, index=False)

            calibrator, meta = _fit_feedback_calibrator(feedback_path=str(p))
            self.assertIsNone(calibrator)
            self.assertEqual(meta["feedback_rows_used"], 30)
            self.assertEqual(meta["feedback_min_rows"], 20)
            self.assertTrue(meta["feedback_calibrator_low_sample_warning"])
            self.assertFalse(meta["feedback_calibrator_selected"])
            self.assertEqual(meta["feedback_calibrator_reason"], "below_apply_rows")

    def test_apply_threshold_allows_40_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "prediction_feedback_training.csv"
            rows = []
            for i in range(40):
                y = int(i % 2 == 0)
                rows.append(
                    {
                        "model_home_win_probability": 0.60 if y == 1 else 0.40,
                        "actual_home_win": y,
                        "run_timestamp": f"2026-04-{(i % 28) + 1:02d}T12:00:00",
                    }
                )
            pd.DataFrame(rows).to_csv(p, index=False)
            calibrator, meta = _fit_feedback_calibrator(feedback_path=str(p))
            self.assertIsNotNone(calibrator)
            self.assertFalse(meta["feedback_calibrator_low_sample_warning"])
            self.assertTrue(meta["feedback_calibrator_selected"])

    def test_validation_gate_can_reject_feedback_calibrator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "prediction_feedback_training.csv"
            rows = []
            # Training slice (first 21 rows): p=0.6 means home win.
            for i in range(21):
                y = int(i % 2 == 0)
                rows.append(
                    {
                        "model_home_win_probability": 0.60 if y == 1 else 0.40,
                        "actual_home_win": y,
                        "run_timestamp": f"2026-03-{(i % 28) + 1:02d}T12:00:00",
                    }
                )
            # Validation slice (last 9 rows): relation flips.
            for i in range(9):
                y = int(i % 2 == 0)
                rows.append(
                    {
                        "model_home_win_probability": 0.40 if y == 1 else 0.60,
                        "actual_home_win": y,
                        "run_timestamp": f"2026-04-{(i % 28) + 1:02d}T12:00:00",
                    }
                )
            pd.DataFrame(rows).to_csv(p, index=False)
            calibrator, meta = _fit_feedback_calibrator(feedback_path=str(p))
            self.assertIsNone(calibrator)
            self.assertFalse(meta["feedback_calibrator_selected"])
            self.assertIn(
                meta["feedback_calibrator_reason"],
                {"below_apply_rows", "validation_not_improved"},
            )

    def test_explicit_min_rows_gate_still_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "prediction_feedback_training.csv"
            pd.DataFrame(
                {
                    "model_home_win_probability": [0.55] * 12,
                    "actual_home_win": [1, 0] * 6,
                }
            ).to_csv(p, index=False)
            calibrator, meta = _fit_feedback_calibrator(feedback_path=str(p), min_rows=30)
            self.assertIsNone(calibrator)
            self.assertEqual(meta["feedback_dataset_rows"], 12)


if __name__ == "__main__":
    unittest.main()
