import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from train_model import (
    _fit_estimator,
    _raw_probabilities,
    _walk_forward_splits,
    apply_calibrator,
    build_game_level_frame,
    feature_columns,
)


def _print_probability_deciles_report(y_true: np.ndarray, y_prob: np.ndarray, *, label: str) -> None:
    """Rough calibration table: empirical win rate vs mean predicted probability by score decile."""
    d = pd.DataFrame({"y": y_true.astype(int), "p": np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)})
    if len(d) < 50:
        return
    try:
        d["bin"] = pd.qcut(d["p"], q=10, duplicates="drop")
    except ValueError:
        return
    g = (
        d.groupby("bin", observed=False)
        .agg(mean_predicted=("p", "mean"), win_rate=("y", "mean"), n=("y", "size"))
        .reset_index(drop=False)
    )
    print(f"\nHoldout probability deciles ({label})")
    print(g.to_string(index=False))


def main() -> None:
    input_path = "data/historical_games.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "Missing data/historical_games.csv. Run: python fetch_data.py"
        )

    raw = pd.read_csv(input_path, parse_dates=["GAME_DATE"])
    games = build_game_level_frame(raw).sort_values("GAME_DATE_HOME").reset_index(drop=True)
    feature_cols = feature_columns()

    eval_rows: list[dict] = []
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(
        _walk_forward_splits(len(games), n_folds=5),
        start=1,
    ):
        train_df = games.iloc[train_start:train_end].copy()
        test_df = games.iloc[test_start:test_end].copy()
        y_test = test_df["WIN_HOME"].astype(int).to_numpy()

        for model_name in ("logistic", "hist_gb", "elo_baseline"):
            estimator = _fit_estimator(model_name)
            if estimator is not None:
                estimator.fit(train_df[feature_cols], train_df["WIN_HOME"])

            y_prob = _raw_probabilities(model_name, estimator, test_df, feature_cols)
            y_pred = (y_prob >= 0.5).astype(int)
            eval_rows.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "log_loss": log_loss(y_test, y_prob, labels=[0, 1]),
                    "brier": brier_score_loss(y_test, y_prob),
                }
            )

    eval_df = pd.DataFrame(eval_rows)
    summary = (
        eval_df.groupby("model", as_index=False)[["accuracy", "log_loss", "brier"]]
        .mean()
        .sort_values("log_loss")
    )
    print("Walk-forward evaluation summary")
    print(summary.to_string(index=False))

    wf_ll_champion = str(summary.iloc[0]["model"])
    wf_acc_champion = str(summary.sort_values("accuracy", ascending=False).iloc[0]["model"])
    print("\nChampion selection tradeoff")
    print(
        f"  Lowest mean walk-forward log_loss: {wf_ll_champion} "
        "(used when training picks the deployed head model)."
    )
    print(
        f"  Highest mean walk-forward accuracy: {wf_acc_champion}"
        + (" (same as LL champion)" if wf_acc_champion == wf_ll_champion else "")
    )

    champion = summary.iloc[0]["model"]
    split_fit = int(len(games) * 0.7)
    split_calib = int(len(games) * 0.8)
    fit_df = games.iloc[:split_fit].copy()
    calib_df = games.iloc[split_fit:split_calib].copy()
    test_df = games.iloc[split_calib:].copy()

    estimator = _fit_estimator(champion)
    if estimator is not None:
        estimator.fit(fit_df[feature_cols], fit_df["WIN_HOME"])
    calib_prob = _raw_probabilities(champion, estimator, calib_df, feature_cols)
    from train_model import _fit_calibrator  # local import keeps script clear
    calibrator = _fit_calibrator(calib_df["WIN_HOME"], calib_prob)

    test_prob_raw = _raw_probabilities(champion, estimator, test_df, feature_cols)
    test_prob = apply_calibrator(test_prob_raw, calibrator)
    test_pred = (test_prob >= 0.5).astype(int)

    print("\nChampion out-of-time holdout")
    print(f"Model: {champion}")
    print(f"Accuracy: {accuracy_score(test_df['WIN_HOME'], test_pred):.3f}")
    print(f"Log loss: {log_loss(test_df['WIN_HOME'], test_prob, labels=[0, 1]):.3f}")
    print(f"Brier score: {brier_score_loss(test_df['WIN_HOME'], test_prob):.3f}")

    y_test_hold = test_df["WIN_HOME"].astype(int).to_numpy()
    _print_probability_deciles_report(y_test_hold, test_prob, label=f"{champion} + Platt")

    if "PLAYOFF_GAME" in test_df.columns:
        tg = pd.DataFrame({"y": y_test_hold, "p": test_prob, "po": test_df["PLAYOFF_GAME"].to_numpy()})
        print("\nHoldout segments (GAME_ID playoff prefix ~004)")
        for name, subset in tg.groupby(tg["po"].map({0: "regular_season_estimate", 1: "playoff"})):
            if subset.empty:
                continue
            print(
                f"  {name}: n={len(subset)}  "
                f"acc={accuracy_score(subset['y'], (subset['p'] >= 0.5).astype(int)):.3f}  "
                f"log_loss={log_loss(subset['y'], subset['p'], labels=[0, 1]):.3f}"
            )


if __name__ == "__main__":
    main()
