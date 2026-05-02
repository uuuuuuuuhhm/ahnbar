import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_model import build_game_level_frame, feature_columns


def _tiny_history_with_box() -> pd.DataFrame:
    """Two teams, four completed games — enough for rolling(3) after shift."""
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    gids = ["0022400123", "0022400124", "0022400125", "0022400126"]
    for gi, gid in enumerate(gids):
        d = base_date + pd.Timedelta(days=gi * 2)
        rows.append(
            {
                "SEASON_ID": "22023",
                "TEAM_ID": 100,
                "TEAM_ABBREVIATION": "AAA",
                "TEAM_NAME": "Team A",
                "GAME_ID": gid,
                "GAME_DATE": d,
                "MATCHUP": "AAA vs. BBB",
                "WL": "W",
                "PTS": 110 + gi,
                "FGM": 40,
                "FGA": 85,
                "FG3M": 12,
                "FG3A": 35,
                "FTM": 18,
                "FTA": 22,
                "OREB": 10,
                "DREB": 30,
                "TOV": 12,
            }
        )
        rows.append(
            {
                "SEASON_ID": "22023",
                "TEAM_ID": 200,
                "TEAM_ABBREVIATION": "BBB",
                "TEAM_NAME": "Team B",
                "GAME_ID": gid,
                "GAME_DATE": d,
                "MATCHUP": "BBB @ AAA",
                "WL": "L",
                "PTS": 100 - gi,
                "FGM": 36,
                "FGA": 88,
                "FG3M": 10,
                "FG3A": 32,
                "FTM": 18,
                "FTA": 20,
                "OREB": 9,
                "DREB": 28,
                "TOV": 14,
            }
        )
    return pd.DataFrame(rows)


class TrainFeaturesTests(unittest.TestCase):
    def test_build_game_level_frame_has_four_factor_rollings(self):
        raw = _tiny_history_with_box()
        games = build_game_level_frame(raw)
        cols = feature_columns()
        missing = [c for c in cols if c not in games.columns]
        self.assertEqual(missing, [], msg=f"Missing feature columns: {missing[:5]}")
        x = games.iloc[-1][cols].to_numpy(dtype=float)
        self.assertFalse(np.isnan(x).any())
        self.assertFalse(np.isinf(x).any())

    def test_build_game_level_frame_without_box_columns_uses_fallbacks(self):
        raw = _tiny_history_with_box().drop(
            columns=[
                "FGM",
                "FGA",
                "FG3M",
                "FG3A",
                "FTM",
                "FTA",
                "OREB",
                "DREB",
                "TOV",
            ],
            errors="ignore",
        )
        games = build_game_level_frame(raw)
        cols = feature_columns()
        x = games.iloc[-1][cols].to_numpy(dtype=float)
        self.assertFalse(np.isnan(x).any())


if __name__ == "__main__":
    unittest.main()
