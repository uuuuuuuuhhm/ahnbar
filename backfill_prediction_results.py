"""
Score every row in predictions_log against historical_games, rebuild feedback CSV.

Run after fetch_data / historical refresh so past predicted games can resolve to real W/L.
Optional --retrain refreshes model artifacts (including feedback calibrator).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(
        description="Merge predictions_log with historical results and rebuild prediction_feedback_training.csv."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="After feedback build, run train_model.main() to refresh artifacts/model.joblib.",
    )
    args = parser.parse_args()

    pred_path = PROJECT_ROOT / "data" / "predictions_log.csv"
    hist_path = PROJECT_ROOT / "data" / "historical_games.csv"
    if not pred_path.exists():
        print("Missing data/predictions_log.csv — nothing to backfill.")
        sys.exit(1)
    if not hist_path.exists():
        print("Missing data/historical_games.csv — run fetch_data.py first.")
        sys.exit(1)

    import pandas as pd

    raw_log = pd.read_csv(pred_path)
    log_rows = len(raw_log)
    unique_games = (
        raw_log.drop_duplicates(subset=["game_date", "home_team", "away_team"]).shape[0]
        if {"game_date", "home_team", "away_team"}.issubset(raw_log.columns)
        else 0
    )
    print(f"Predictions log: {log_rows} rows, ~{unique_games} unique games (raw, before team normalize).")

    import score_predictions

    print("\nScoring logged predictions vs historical results...")
    score_predictions.main()

    from retrain_from_feedback import build_feedback_dataset

    print("\nBuilding prediction_feedback_training.csv...")
    fb = build_feedback_dataset()
    if fb.empty:
        print(
            "Feedback dataset is still empty.\n"
            "This is NOT because predictions_log is empty — it means zero rows matched a finished game in "
            "historical_games.csv (same game_date + home_team + away_team as in the NBA schedule export).\n"
            "Check the 'Merge summary' and date-range lines printed by score_predictions above.\n"
            "Most common fix: logged dates are past the last GAME_DATE in historical_games.csv — refresh:\n"
            "  python fetch_data.py\n"
            "  # optional cap: python fetch_data.py --season-to 2025 --recent-days 180"
        )
    else:
        print(f"Feedback training rows: {len(fb)}")

    if args.retrain:
        import train_model

        print("\nRetraining model artifacts...")
        train_model.main()


if __name__ == "__main__":
    main()
