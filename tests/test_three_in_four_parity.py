import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_model import _prepare_team_rows, three_in_four_flag_from_schedule


class ThreeInFourTests(unittest.TestCase):
    def test_inference_matches_team_row_flag(self):
        """Dense schedule so last completed game fires THREE_IN_FOUR; next-night inference agrees."""
        base = pd.Timestamp("2025-02-01")
        team = 303
        rows = []
        for i in range(4):
            rows.append(
                {
                    "SEASON_ID": "22024",
                    "TEAM_ID": team,
                    "GAME_ID": f"00224{i:05d}",
                    "GAME_DATE": base + pd.Timedelta(days=i),
                    "MATCHUP": "TST vs. OPP",
                    "WL": "W",
                    "PTS": 110 + i,
                }
            )
        raw = pd.DataFrame(rows).sort_values(["TEAM_ID", "GAME_DATE"])
        prepared = _prepare_team_rows(raw)
        last = prepared[prepared["TEAM_ID"] == team].iloc[-1]
        self.assertEqual(int(last["THREE_IN_FOUR_FLAG"]), 1)

        upcoming = pd.Timestamp(base + pd.Timedelta(days=5))
        lag1 = last["GAME_DATE"]
        lag2 = prepared[prepared["TEAM_ID"] == team].iloc[-2]["GAME_DATE"]
        n_done = len(prepared[prepared["TEAM_ID"] == team])
        inf = three_in_four_flag_from_schedule(upcoming, lag1, lag2, n_done)
        self.assertEqual(1, int(inf))


if __name__ == "__main__":
    unittest.main()
