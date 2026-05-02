from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

import score_predictions
import train_model
from fetch_data import build_historical_games

PROJECT_ROOT = Path(__file__).resolve().parent


def refresh_historical_games(
    season_from: int = 2021,
    season_to: int | None = None,
    *,
    recent_days: int = 120,
    include_recent_date_fetch: bool = True,
) -> str:
    os.makedirs("data", exist_ok=True)
    out_path = "data/historical_games.csv"
    games = build_historical_games(
        season_from=season_from,
        season_to=season_to,
        recent_days=recent_days,
        include_recent_date_fetch=include_recent_date_fetch,
    )
    games.to_csv(out_path, index=False)
    print(f"Saved {len(games)} rows to {out_path}")
    return out_path


def build_feedback_dataset(
    scored_path: str = "data/scored_predictions.csv",
    out_path: str = "data/prediction_feedback_training.csv",
) -> pd.DataFrame:
    if not os.path.exists(scored_path):
        raise FileNotFoundError(f"Missing {scored_path}. Run score_predictions.py first.")

    scored = pd.read_csv(scored_path)
    required_cols = {
        "game_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "win_home_actual",
        "pred_home_win",
        "run_timestamp",
    }
    missing = required_cols - set(scored.columns)
    if missing:
        raise ValueError(f"scored predictions missing columns: {sorted(missing)}")

    resolved = scored[scored["win_home_actual"].notna()].copy()
    if resolved.empty:
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(
            columns=[
                "game_date",
                "home_team",
                "away_team",
                "model_home_win_probability",
                "model_pick_home",
                "actual_home_win",
                "was_correct",
                "absolute_error",
                "brier_component",
                "run_timestamp",
            ]
        ).to_csv(out_path, index=False)
        return pd.DataFrame()

    resolved["actual_home_win"] = resolved["win_home_actual"].astype(int)
    resolved["model_pick_home"] = resolved["pred_home_win"].astype(int)
    resolved["was_correct"] = (resolved["model_pick_home"] == resolved["actual_home_win"]).astype(int)
    resolved["model_home_win_probability"] = pd.to_numeric(
        resolved["home_win_probability"], errors="coerce"
    )
    resolved["absolute_error"] = (
        resolved["model_home_win_probability"] - resolved["actual_home_win"]
    ).abs()
    resolved["brier_component"] = (
        resolved["model_home_win_probability"] - resolved["actual_home_win"]
    ) ** 2

    feedback = resolved[
        [
            "game_date",
            "home_team",
            "away_team",
            "model_home_win_probability",
            "model_pick_home",
            "actual_home_win",
            "was_correct",
            "absolute_error",
            "brier_component",
            "run_timestamp",
        ]
    ].copy()
    feedback = feedback.sort_values(["game_date", "home_team", "away_team", "run_timestamp"])
    feedback = feedback.drop_duplicates(["game_date", "home_team", "away_team"], keep="last")

    os.makedirs("data", exist_ok=True)
    feedback.to_csv(out_path, index=False)
    print(f"Saved {len(feedback)} resolved feedback rows to {out_path}")
    return feedback


def ensure_predictions_log(path: str = "data/predictions_log.csv") -> None:
    if os.path.exists(path):
        return

    print("No predictions log found. Running predict_next_games.py to create it...")
    subprocess.run(
        [sys.executable, "predict_next_games.py"],
        check=False,
        cwd=str(PROJECT_ROOT),
    )

    if os.path.exists(path):
        return

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(
        columns=[
            "matchup",
            "game_date",
            "game_start_time_utc",
            "home_team",
            "away_team",
            "home_win_probability",
            "run_timestamp",
            "confidence_pct",
            "confidence_level",
            "pred_home_pts",
            "pred_away_pts",
        ]
    ).to_csv(path, index=False)
    print(f"Created empty {path} because no predictable games were available.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh results for past predictions and retrain model artifacts."
    )
    parser.add_argument("--season-from", type=int, default=2021)
    parser.add_argument(
        "--season-to",
        type=int,
        default=None,
        help="Last season start year to pull by season id; default is inferred from today.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=120,
        help="Days for the date-only LeagueGameFinder merge (see fetch_data.build_historical_games).",
    )
    parser.add_argument(
        "--no-recent-date-boost",
        action="store_true",
        help="Skip the date-only recent-game fetch in build_historical_games.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip historical data refresh and use existing data/historical_games.csv.",
    )
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)

    if not args.skip_fetch:
        print("Step 1/4: Refreshing historical game results...")
        refresh_historical_games(
            season_from=args.season_from,
            season_to=args.season_to,
            recent_days=args.recent_days,
            include_recent_date_fetch=not args.no_recent_date_boost,
        )
    else:
        print("Step 1/4: Skipping historical data refresh (--skip-fetch).")

    print("Step 2/4: Resolving previous predictions against actual outcomes...")
    ensure_predictions_log()
    score_predictions.main()

    print("Step 3/4: Building prediction feedback training dataset...")
    build_feedback_dataset()

    hist_df = pd.read_csv("data/historical_games.csv")
    quality = train_model._data_quality_snapshot(hist_df)  # internal helper used by train_model main
    print(
        "Data quality check: "
        f"nan_rate={quality.get('nan_rate', 0.0):.4f}, "
        f"stale_days={quality.get('stale_days', 0)}, "
        f"recent_games_30d={quality.get('recent_games_30d', 0)}"
    )
    if quality.get("quality_warning"):
        print(
            "Warning: data quality guard tripped: "
            + ",".join(quality.get("quality_warning_reasons", []))
        )

    print("Step 4/4: Retraining model artifacts with refreshed historical data...")
    train_model.main()
    if os.path.exists("artifacts/model_metrics.json"):
        with open("artifacts/model_metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        os.makedirs("artifacts", exist_ok=True)
        registry_path = "artifacts/model_registry.csv"
        row = pd.DataFrame(
            [
                {
                    "run_timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
                    "champion_model": metrics.get("champion_model", "unknown"),
                    "holdout_accuracy": metrics.get("holdout_metrics", {}).get("accuracy"),
                    "holdout_log_loss": metrics.get("holdout_metrics", {}).get("log_loss"),
                    "holdout_brier": metrics.get("holdout_metrics", {}).get("brier"),
                    "calibrator_method": metrics.get("calibrator_method", "unknown"),
                }
            ]
        )
        if os.path.exists(registry_path):
            old = pd.read_csv(registry_path)
            pd.concat([old, row], ignore_index=True).to_csv(registry_path, index=False)
        else:
            row.to_csv(registry_path, index=False)
        print(f"Updated artifact registry: {registry_path}")
        if bool(metrics.get("drift_warning", False)):
            print(
                "Warning: post-train drift warning: "
                + ",".join(metrics.get("drift_warning_reasons", []))
            )
    print("Done. Retraining loop completed.")


if __name__ == "__main__":
    main()
