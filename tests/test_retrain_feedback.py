import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrain_from_feedback import build_feedback_dataset


class RetrainFeedbackTests(unittest.TestCase):
    def test_build_feedback_dataset_resolved_rows_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scored_path = Path(tmpdir) / "scored_predictions.csv"
            out_path = Path(tmpdir) / "prediction_feedback_training.csv"

            pd.DataFrame(
                [
                    {
                        "game_date": "04/30/2026",
                        "home_team": "NYK",
                        "away_team": "ATL",
                        "home_win_probability": 0.59,
                        "win_home_actual": 1,
                        "pred_home_win": 1,
                        "run_timestamp": "2026-04-30T20:00:00",
                    },
                    {
                        "game_date": "05/01/2026",
                        "home_team": "ORL",
                        "away_team": "DET",
                        "home_win_probability": 0.56,
                        "win_home_actual": None,
                        "pred_home_win": 1,
                        "run_timestamp": "2026-05-01T12:00:00",
                    },
                ]
            ).to_csv(scored_path, index=False)

            out = build_feedback_dataset(str(scored_path), str(out_path))
            self.assertEqual(len(out), 1)
            self.assertEqual(out.iloc[0]["home_team"], "NYK")
            self.assertEqual(int(out.iloc[0]["was_correct"]), 1)
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
