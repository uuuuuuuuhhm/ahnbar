"""
Train HistGradientBoostingRegressor models for PTS_HOME and PTS_AWAY using the same
feature_columns() contract as the win model (see train_model.py).
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_model import build_game_level_frame, feature_columns


def main() -> None:
    input_path = "data/historical_games.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError("Missing data/historical_games.csv. Run fetch_data.py first.")

    raw = pd.read_csv(input_path, parse_dates=["GAME_DATE"])
    games = build_game_level_frame(raw)
    cols = feature_columns()
    missing = [c for c in cols if c not in games.columns]
    if missing:
        raise ValueError(f"Game frame missing feature columns: {missing[:10]}")

    X = games[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_home = pd.to_numeric(games["PTS_HOME"], errors="coerce").fillna(110.0)
    y_away = pd.to_numeric(games["PTS_AWAY"], errors="coerce").fillna(110.0)

    n = len(games)
    i_train = int(n * 0.75)
    train_idx = slice(0, i_train)
    test_idx = slice(i_train, n)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yh_train, yh_test = y_home.iloc[train_idx], y_home.iloc[test_idx]
    ya_train, ya_test = y_away.iloc[train_idx], y_away.iloc[test_idx]

    est_home = HistGradientBoostingRegressor(
        max_depth=7,
        learning_rate=0.06,
        max_iter=450,
        random_state=42,
    )
    est_away = HistGradientBoostingRegressor(
        max_depth=7,
        learning_rate=0.06,
        max_iter=450,
        random_state=43,
    )
    est_home.fit(X_train, yh_train)
    est_away.fit(X_train, ya_train)

    ph = est_home.predict(X_test)
    pa = est_away.predict(X_test)

    mae_h = float(mean_absolute_error(yh_test, ph))
    mae_a = float(mean_absolute_error(ya_test, pa))
    rmse_h = float(mean_squared_error(yh_test, ph) ** 0.5)
    rmse_a = float(mean_squared_error(ya_test, pa) ** 0.5)

    # Refit on full history for deployment (holdout metrics above are from pre-refit split).
    est_home.fit(X, y_home)
    est_away.fit(X, y_away)

    bundle = {
        "estimator_home": est_home,
        "estimator_away": est_away,
        "feature_columns": cols,
        "train_rows": int(i_train),
        "test_rows": int(n - i_train),
    }

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(bundle, "artifacts/score_models.joblib")
    metrics = {
        "mae_home_holdout": mae_h,
        "mae_away_holdout": mae_a,
        "rmse_home_holdout": rmse_h,
        "rmse_away_holdout": rmse_a,
        "model": "HistGradientBoostingRegressor",
        "hyperparams": {"max_depth": 7, "learning_rate": 0.06, "max_iter": 450},
    }
    with open("artifacts/score_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Score models trained.")
    print(f"Holdout MAE home/away: {mae_h:.2f} / {mae_a:.2f}")
    print(f"Holdout RMSE home/away: {rmse_h:.2f} / {rmse_a:.2f}")
    print("Saved artifacts/score_models.joblib, artifacts/score_model_metrics.json")


if __name__ == "__main__":
    main()
