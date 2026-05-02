import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fetch_box_scores import (
    _finder_date_bounds_for_season,
    build_box_score_lookup_chunked,
    iter_date_windows,
    merge_box_scores_onto_finder_rows,
    normalize_nba_game_id,
    season_start_year_from_game_date,
)


class NormalizeGameIdTests(unittest.TestCase):
    def test_pad_stripped_leading_zeros(self):
        self.assertEqual(normalize_nba_game_id(22100001), "0022100001")
        self.assertEqual(normalize_nba_game_id("22100002"), "0022100002")

    def test_already_ten_digit(self):
        self.assertEqual(normalize_nba_game_id("0022100001"), "0022100001")

    def test_merge_finder_and_log_formats(self):
        finder = pd.DataFrame(
            {
                "SEASON_ID": [22021],
                "TEAM_ID": [1610612749],
                "GAME_ID": ["22100001"],
                "GAME_DATE": [pd.Timestamp("2021-10-19")],
                "MATCHUP": ["x"],
                "WL": ["W"],
                "PTS": [127],
                # Stale placeholders (like after a failed merge) must not block real FGA.
                "FGA": [pd.NA],
            }
        )
        box = pd.DataFrame(
            {
                "GAME_ID": ["0022100001"],
                "TEAM_ID": [1610612749],
                "FGA": [105.0],
                "FGM": [48.0],
                "FG3M": [0],
                "FG3A": [0],
                "FTM": [0],
                "FTA": [0],
                "OREB": [0],
                "DREB": [0],
                "REB": [0],
                "AST": [0],
                "TOV": [0],
                "STL": [0],
                "BLK": [0],
                "PF": [0],
                "MIN": [240],
                "PLUS_MINUS": [0],
            }
        )
        out = merge_box_scores_onto_finder_rows(finder, box)
        self.assertGreaterEqual(float(out.iloc[0]["FGA"]), 105.0 - 1e-6)


class DateWindowTests(unittest.TestCase):
    def test_iter_date_windows_two_chunks(self):
        lo = pd.Timestamp("2024-10-01")
        hi = pd.Timestamp("2024-10-20")
        w = iter_date_windows(lo, hi, 14)
        self.assertEqual(w[0], ("2024-10-01", "2024-10-14"))
        self.assertEqual(w[1], ("2024-10-15", "2024-10-20"))

    def test_iter_date_windows_single_day(self):
        lo = hi = pd.Timestamp("2025-04-01")
        w = iter_date_windows(lo, hi, 7)
        self.assertEqual(w, [("2025-04-01", "2025-04-01")])

    def test_iter_date_windows_rejects_inverted_range(self):
        self.assertEqual(iter_date_windows(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-05-01"), 7), [])

    def test_season_start_year_from_game_date(self):
        self.assertEqual(season_start_year_from_game_date(pd.Timestamp("2024-11-01")), 2024)
        self.assertEqual(season_start_year_from_game_date(pd.Timestamp("2025-03-15")), 2024)

    def test_finder_date_bounds_narrows_to_games(self):
        df = pd.DataFrame(
            {
                "GAME_DATE": pd.to_datetime(["2025-01-05", "2025-01-28"]),
            }
        )
        lo, hi = _finder_date_bounds_for_season(df, 2024)
        self.assertEqual(lo.strftime("%Y-%m-%d"), "2025-01-05")
        self.assertEqual(hi.strftime("%Y-%m-%d"), "2025-01-28")


class ChunkedLookupTests(unittest.TestCase):
    @patch("fetch_box_scores.time.sleep", lambda *a, **k: None)
    @patch("fetch_box_scores.fetch_league_team_game_logs_date_range")
    def test_chunked_dedupes_across_windows(self, mock_fetch):
        """Same GAME_ID/TEAM_ID from RS and PO windows collapses to one row (keep last)."""
        finder = pd.DataFrame({"GAME_DATE": pd.to_datetime(["2025-01-02", "2025-01-03"])})
        ctr = {"n": 0}

        def _row(fga: float) -> dict:
            return {
                "GAME_ID": "0022400999",
                "TEAM_ID": 1610612738,
                "FGM": 40.0,
                "FGA": fga,
                "FG3M": 12.0,
                "FG3A": 35.0,
                "FTM": 18.0,
                "FTA": 22.0,
                "OREB": 10.0,
                "DREB": 30.0,
                "REB": 40.0,
                "AST": 22.0,
                "TOV": 12.0,
                "STL": 7.0,
                "BLK": 4.0,
                "PF": 18.0,
                "MIN": "240:00",
                "PLUS_MINUS": 5.0,
            }

        def side_effect(*args, **kwargs):
            ctr["n"] += 1
            return pd.DataFrame([_row(80.0 + float(ctr["n"]))])

        mock_fetch.side_effect = side_effect

        out = build_box_score_lookup_chunked(
            [2024],
            chunk_days=14,
            sleep_between_calls_s=0.0,
            sleep_jitter_s=0.0,
            timeout_s=30,
            max_retries=1,
            finder_df=finder,
            cache_dir=None,
            use_cache=False,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out.iloc[0]["FGA"]), 82.0)


if __name__ == "__main__":
    unittest.main()
