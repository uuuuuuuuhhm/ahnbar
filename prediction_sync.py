"""
Resolve finished games in the predictions log against historical results and refresh
feedback data used by the feedback calibrator in model artifacts.

Intended to run once when the Streamlit app starts (see app.py). Does not fetch new
NBA history; extend data/historical_games.csv (e.g. fetch_data.py) if games stay pending.
"""

from __future__ import annotations

import io
import os
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import pandas as pd

import score_predictions
import train_model
from retrain_from_feedback import build_feedback_dataset

PROJECT_ROOT = Path(__file__).resolve().parent


def sync_completed_predictions(
    *,
    patch_feedback: bool = True,
    silent: bool = True,
) -> dict[str, Any]:
    """
    Merge predictions_log with historical_games, write scored_predictions + weekly_summary,
    rebuild prediction_feedback_training.csv, and optionally refit only the feedback
    calibrator into artifacts/model.joblib (champion estimator unchanged).
    """
    summary: dict[str, Any] = {
        "scored": False,
        "skipped": False,
        "reason": "",
        "resolved": 0,
        "pending": 0,
        "feedback_rows": 0,
        "feedback_patched": False,
        "patch_detail": {},
    }

    pred_path = PROJECT_ROOT / "data" / "predictions_log.csv"
    hist_path = PROJECT_ROOT / "data" / "historical_games.csv"
    if not pred_path.exists() or not hist_path.exists():
        summary["skipped"] = True
        summary["reason"] = "missing predictions_log or historical_games"
        return summary

    prev_cwd = os.getcwd()
    buf = io.StringIO()
    ctx = redirect_stdout(buf) if silent else nullcontext()
    try:
        os.chdir(PROJECT_ROOT)
        with ctx:
            score_predictions.main()
            fb_df = build_feedback_dataset()
        summary["scored"] = True
        summary["feedback_rows"] = int(len(fb_df))

        scored_path = PROJECT_ROOT / "data" / "scored_predictions.csv"
        if scored_path.exists():
            sp = pd.read_csv(scored_path)
            if "result_status" in sp.columns:
                summary["resolved"] = int((sp["result_status"] == "resolved").sum())
                summary["pending"] = int((sp["result_status"] == "pending").sum())

        if patch_feedback:
            summary["patch_detail"] = train_model.patch_feedback_calibrator_into_artifacts()
            summary["feedback_patched"] = bool(summary["patch_detail"].get("patched"))
    finally:
        os.chdir(prev_cwd)

    return summary
