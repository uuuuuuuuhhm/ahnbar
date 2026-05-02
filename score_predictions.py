import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from team_aliases import normalize_team_abbr
from value_betting import simulate_bankroll_curve


def _build_actual_pts_by_game(hist_path: str) -> pd.DataFrame:
    """Game-level home/away points keyed like predictions (game_date + abbr teams)."""
    raw = pd.read_csv(hist_path, parse_dates=["GAME_DATE"])
    raw["IS_HOME"] = raw["MATCHUP"].str.contains(" vs. ").astype(int)
    home = raw[raw["IS_HOME"] == 1][["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "home_team", "PTS": "home_pts_actual"}
    )
    away = raw[raw["IS_HOME"] == 0][["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "away_team", "PTS": "away_pts_actual"}
    )
    games = home.merge(away, on="GAME_ID", how="inner")
    games["home_team"] = games["home_team"].map(normalize_team_abbr)
    games["away_team"] = games["away_team"].map(normalize_team_abbr)
    games["game_date"] = pd.to_datetime(games["GAME_DATE"]).dt.strftime("%m/%d/%Y")
    return games[["game_date", "home_team", "away_team", "home_pts_actual", "away_pts_actual"]].drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="last"
    )


def write_pending_predictions_diagnostic(
    pending_df: pd.DataFrame,
    hist_path: str,
    out_path: str = "data/pending_predictions_diagnostic.csv",
) -> None:
    if pending_df.empty:
        return
    hist = pd.read_csv(hist_path, parse_dates=["GAME_DATE"])
    hmax = hist["GAME_DATE"].max()
    out = pending_df.copy()
    out["game_date_dt"] = pd.to_datetime(out["game_date"], format="%m/%d/%Y", errors="coerce")
    out["historical_game_date_max"] = (
        pd.Timestamp(hmax).isoformat() if hmax is not None and pd.notna(hmax) else ""
    )
    out["after_history_max"] = out["game_date_dt"].notna() & (out["game_date_dt"] > hmax)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote pending diagnostics ({len(out)} rows) to {out_path}")


def write_reliability_bins_csv(
    resolved_df: pd.DataFrame,
    out_path: str = "data/scoring_reliability_bins.csv",
    bins: int = 10,
) -> pd.DataFrame:
    """Bin mean predicted p vs empirical win rate; per-bin ECE contribution."""
    if resolved_df.empty or bins < 2:
        return pd.DataFrame()
    y = pd.to_numeric(resolved_df["win_home_actual"], errors="coerce").fillna(0).astype(int).to_numpy()
    p = np.clip(
        pd.to_numeric(resolved_df["home_win_probability"], errors="coerce").fillna(0.5).to_numpy(),
        1e-6,
        1 - 1e-6,
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    n = len(p)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n": 0,
                    "mean_predicted_p": float("nan"),
                    "empirical_win_rate": float("nan"),
                    "ece_contribution": 0.0,
                }
            )
            continue
        ph = p[mask]
        yh = y[mask]
        mean_p = float(ph.mean())
        emp = float(yh.mean())
        ece_c = (cnt / n) * abs(mean_p - emp)
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": cnt,
                "mean_predicted_p": mean_p,
                "empirical_win_rate": emp,
                "ece_contribution": float(ece_c),
            }
        )
    df = pd.DataFrame(rows)
    df["ece_total"] = float(df["ece_contribution"].sum())
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Reliability bins (ECE sum={df['ece_contribution'].sum():.4f}) saved to {out_path}")
    return df


def write_calibration_trend_csv(
    weekly_summary: pd.DataFrame, out_path: str = "data/calibration_trend.csv"
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if weekly_summary.empty:
        out = pd.DataFrame(
            columns=[
                "season_week",
                "resolved_predictions",
                "weekly_accuracy",
                "weekly_log_loss",
                "weekly_brier",
                "weekly_ece",
            ]
        )
        out.to_csv(out_path, index=False)
        return out
    cols = [
        c
        for c in (
            "season_week",
            "resolved_predictions",
            "weekly_accuracy",
            "weekly_log_loss",
            "weekly_brier",
            "weekly_ece",
        )
        if c in weekly_summary.columns
    ]
    if not cols:
        out = pd.DataFrame()
        out.to_csv(out_path, index=False)
        return out
    trend = weekly_summary[cols].copy().sort_values("season_week")
    trend.to_csv(out_path, index=False)
    return trend


def _parse_matchup_to_abbr(matchup: str) -> tuple[str, str]:
    text = str(matchup)
    if "@" not in text:
        return "", ""
    away, home = [s.strip() for s in text.split("@", 1)]
    return normalize_team_abbr(away), normalize_team_abbr(home)


def write_value_backtest_snapshot(
    resolved_export: pd.DataFrame,
    value_path: str = "data/value_recommendations.csv",
    out_summary_path: str = "data/value_backtest_summary.json",
    out_curve_path: str = "data/value_backtest_curve.csv",
) -> dict:
    summary = {
        "bets_resolved": 0,
        "hit_rate": 0.0,
        "avg_ev_per_unit": 0.0,
        "realized_roi_per_bet": 0.0,
        "start_bankroll": 1000.0,
        "ending_bankroll": 1000.0,
        "max_drawdown": 0.0,
    }
    if resolved_export.empty or not os.path.exists(value_path):
        with open(out_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    value_df = pd.read_csv(value_path)
    if value_df.empty:
        with open(out_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    value_df = value_df.copy()
    value_df["away_team"], value_df["home_team"] = zip(
        *value_df["matchup"].map(_parse_matchup_to_abbr)
    )
    value_df = value_df[(value_df["home_team"] != "") & (value_df["away_team"] != "")]
    if "run_timestamp" in value_df.columns:
        value_df["run_timestamp"] = pd.to_datetime(value_df["run_timestamp"], errors="coerce")
        value_df = value_df.sort_values("run_timestamp")

    value_df = value_df.drop_duplicates(
        subset=["game_date", "home_team", "away_team", "side"], keep="last"
    )
    value_df["model_win_probability"] = pd.to_numeric(
        value_df.get("model_win_probability"), errors="coerce"
    ).fillna(0.5)
    value_df["market_decimal_odds"] = pd.to_numeric(
        value_df.get("market_decimal_odds"), errors="coerce"
    ).fillna(2.0)
    value_df["ev_per_unit_stake"] = pd.to_numeric(
        value_df.get("ev_per_unit_stake"), errors="coerce"
    ).fillna(0.0)

    resolved = resolved_export.copy()
    resolved["home_team"] = resolved["home_team"].map(normalize_team_abbr)
    resolved["away_team"] = resolved["away_team"].map(normalize_team_abbr)
    resolved["win_home_actual"] = pd.to_numeric(resolved["win_home_actual"], errors="coerce")
    merged = value_df.merge(
        resolved[["game_date", "home_team", "away_team", "win_home_actual"]],
        on=["game_date", "home_team", "away_team"],
        how="inner",
    ).dropna(subset=["win_home_actual"])
    if merged.empty:
        with open(out_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    merged["win_home_actual"] = merged["win_home_actual"].astype(int)
    merged["is_win"] = (
        ((merged["side"] == "home") & (merged["win_home_actual"] == 1))
        | ((merged["side"] == "away") & (merged["win_home_actual"] == 0))
    ).astype(int)
    merged["pnl_unit"] = np.where(
        merged["is_win"] == 1,
        merged["market_decimal_odds"] - 1.0,
        -1.0,
    )
    curve = simulate_bankroll_curve(merged["pnl_unit"].to_list(), start_bankroll=1000.0)
    os.makedirs(os.path.dirname(out_curve_path) or ".", exist_ok=True)
    curve.to_csv(out_curve_path, index=False)

    summary = {
        "bets_resolved": int(len(merged)),
        "hit_rate": float(merged["is_win"].mean()),
        "avg_ev_per_unit": float(merged["ev_per_unit_stake"].mean()),
        "realized_roi_per_bet": float(merged["pnl_unit"].mean()),
        "start_bankroll": 1000.0,
        "ending_bankroll": float(curve["bankroll"].iloc[-1]),
        "max_drawdown": float(curve["drawdown"].min()),
    }
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved value backtest snapshot to {out_summary_path}")
    return summary


def _build_actual_results(hist_path: str) -> pd.DataFrame:
    raw = pd.read_csv(hist_path, parse_dates=["GAME_DATE"])
    raw["IS_HOME"] = raw["MATCHUP"].str.contains(" vs. ").astype(int)
    raw["RESULT"] = (raw["WL"] == "W").astype(int)

    home = raw[raw["IS_HOME"] == 1][
        ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "RESULT"]
    ].rename(
        columns={
            "TEAM_ABBREVIATION": "home_team",
            "RESULT": "win_home_actual",
        }
    )
    away = raw[raw["IS_HOME"] == 0][["GAME_ID", "TEAM_ABBREVIATION"]].rename(
        columns={"TEAM_ABBREVIATION": "away_team"}
    )

    games = home.merge(away, on="GAME_ID", how="inner")
    games["game_date"] = pd.to_datetime(games["GAME_DATE"]).dt.strftime("%m/%d/%Y")
    return games[["game_date", "home_team", "away_team", "win_home_actual"]].drop_duplicates()


def _label_confidence(prob_home: float) -> str:
    conf = max(prob_home, 1.0 - prob_home)
    if conf >= 0.65:
        return "high"
    if conf >= 0.55:
        return "medium"
    return "low"


def _expected_calibration_error(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> float:
    p = np.clip(pd.to_numeric(y_prob, errors="coerce").fillna(0.5).to_numpy(), 1e-6, 1 - 1e-6)
    y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).to_numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def main() -> None:
    pred_path = "data/predictions_log.csv"
    hist_path = "data/historical_games.csv"
    if not os.path.exists(pred_path):
        raise FileNotFoundError("Missing data/predictions_log.csv. Run predict_next_games.py first.")
    if not os.path.exists(hist_path):
        raise FileNotFoundError("Missing data/historical_games.csv. Run fetch_data.py first.")

    preds = pd.read_csv(pred_path)
    required_cols = {
        "game_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "run_timestamp",
    }
    missing = required_cols - set(preds.columns)
    if missing:
        raise ValueError(f"predictions_log.csv missing columns: {sorted(missing)}")

    preds = preds.copy()
    preds["home_team"] = preds["home_team"].map(normalize_team_abbr)
    preds["away_team"] = preds["away_team"].map(normalize_team_abbr)

    # Keep latest prediction per game.
    preds["run_timestamp"] = pd.to_datetime(preds["run_timestamp"], errors="coerce")
    preds = preds.sort_values("run_timestamp").drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="last"
    )

    actuals = _build_actual_results(hist_path)
    scored = preds.merge(actuals, on=["game_date", "home_team", "away_team"], how="left")
    try:
        pts_actual = _build_actual_pts_by_game(hist_path)
        scored = scored.merge(pts_actual, on=["game_date", "home_team", "away_team"], how="left")
    except Exception:
        pass

    # Stable de-duplication key even if log is appended many times.
    scored = scored.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="last")

    pending = scored["win_home_actual"].isna().sum()
    resolved = scored.dropna(subset=["win_home_actual"]).copy()
    if resolved.empty:
        print("No completed games found yet for logged predictions.")
        print(f"Pending predictions: {pending}")
    else:
        resolved["win_home_actual"] = resolved["win_home_actual"].astype(int)
        resolved["pred_home_win"] = (resolved["home_win_probability"] >= 0.5).astype(int)
        resolved["correct"] = (resolved["pred_home_win"] == resolved["win_home_actual"]).astype(int)
        resolved["confidence_bucket"] = resolved["home_win_probability"].apply(_label_confidence)

        overall_acc = resolved["correct"].mean()
        resolved_prob = pd.to_numeric(resolved["home_win_probability"], errors="coerce").fillna(0.5)
        print("Prediction scoring summary")
        print(f"Resolved predictions: {len(resolved)}")
        print(f"Pending predictions: {pending}")
        print(f"Overall accuracy: {overall_acc:.3f}")
        print(f"Overall log loss: {log_loss(resolved['win_home_actual'], resolved_prob, labels=[0, 1]):.3f}")
        print(f"Overall Brier: {brier_score_loss(resolved['win_home_actual'], resolved_prob):.3f}")
        print(f"Overall ECE: {_expected_calibration_error(resolved['win_home_actual'], resolved_prob):.3f}")

        by_bucket = (
            resolved.groupby("confidence_bucket", as_index=False)
            .agg(predictions=("correct", "size"), accuracy=("correct", "mean"))
            .sort_values("confidence_bucket")
        )
        print("\nAccuracy by confidence bucket")
        print(by_bucket.to_string(index=False))
        write_reliability_bins_csv(resolved, "data/scoring_reliability_bins.csv")

    # Export scored rows sorted by playing week.
    export_df = scored.copy()
    export_df["win_home_actual"] = pd.to_numeric(export_df["win_home_actual"], errors="coerce")
    export_df["pred_home_win"] = (export_df["home_win_probability"] >= 0.5).astype(int)
    export_df["correct"] = (
        export_df["win_home_actual"].notna()
        & (export_df["pred_home_win"] == export_df["win_home_actual"].astype("Int64"))
    ).astype("Int64")
    export_df["confidence_bucket"] = export_df["home_win_probability"].apply(_label_confidence)
    export_df["result_status"] = export_df["win_home_actual"].apply(
        lambda x: "resolved" if pd.notna(x) else "pending"
    )

    if (
        "pred_home_pts" in export_df.columns
        and "pred_away_pts" in export_df.columns
        and "home_pts_actual" in export_df.columns
        and "away_pts_actual" in export_df.columns
    ):
        export_df["pts_abs_err_home"] = (
            pd.to_numeric(export_df["pred_home_pts"], errors="coerce")
            - pd.to_numeric(export_df["home_pts_actual"], errors="coerce")
        ).abs()
        export_df["pts_abs_err_away"] = (
            pd.to_numeric(export_df["pred_away_pts"], errors="coerce")
            - pd.to_numeric(export_df["away_pts_actual"], errors="coerce")
        ).abs()

    export_df["game_date_dt"] = pd.to_datetime(export_df["game_date"], format="%m/%d/%Y", errors="coerce")
    iso = export_df["game_date_dt"].dt.isocalendar()
    export_df["season_week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)

    export_df = export_df.sort_values(
        by=["season_week", "game_date_dt", "home_team", "away_team"],
        ascending=[True, True, True, True],
    )

    _base_export_cols = [
        "season_week",
        "game_date",
        "matchup",
        "home_team",
        "away_team",
        "home_win_probability",
        "pred_home_win",
        "win_home_actual",
        "correct",
        "confidence_bucket",
        "result_status",
        "run_timestamp",
    ]
    _extra_score_cols = [
        c
        for c in (
            "pred_home_pts",
            "pred_away_pts",
            "home_pts_actual",
            "away_pts_actual",
            "pts_abs_err_home",
            "pts_abs_err_away",
        )
        if c in export_df.columns
    ]
    export_df = export_df[_base_export_cols + _extra_score_cols]
    os.makedirs("data", exist_ok=True)
    out_path = "data/scored_predictions.csv"
    export_df.to_csv(out_path, index=False)
    print(f"\nSaved scored rows to {out_path}")

    n_resolved = int((export_df["result_status"] == "resolved").sum())
    n_pending = int((export_df["result_status"] == "pending").sum())
    print(
        f"Merge summary: {n_resolved} resolved (matched a finished game in historical_games.csv), "
        f"{n_pending} pending."
    )
    if n_pending > 0:
        pend_rows = export_df[export_df["result_status"] == "pending"].copy()
        write_pending_predictions_diagnostic(pend_rows, hist_path, "data/pending_predictions_diagnostic.csv")
    if n_pending > 0 and n_resolved == 0:
        try:
            hist = pd.read_csv(hist_path, parse_dates=["GAME_DATE"])
            pred_dt = pd.to_datetime(export_df["game_date"], format="%m/%d/%Y", errors="coerce")
            print(
                f"Predictions game_date range: {pred_dt.min().date()} .. {pred_dt.max().date()} "
                f"(unique games after dedupe: {export_df.shape[0]})."
            )
            print(
                f"historical_games.csv GAME_DATE range: {hist['GAME_DATE'].min().date()} .. "
                f"{hist['GAME_DATE'].max().date()}."
            )
            if pred_dt.max() > hist["GAME_DATE"].max():
                print(
                    "Hint: your latest logged game dates are AFTER the newest game in historical_games.csv. "
                    "Re-fetch history (current season is inferred automatically; a recent date-only merge "
                    "pulls the last ~120d from the NBA stats API). If still pending, the stats feed may lag:\n"
                    "  python fetch_data.py\n"
                    "  # optional: python fetch_data.py --recent-days 180"
                )
        except Exception as exc:
            print(f"(Could not print date-range hint: {exc})")

    # Weekly summary on resolved games only.
    resolved_export = export_df[export_df["result_status"] == "resolved"].copy()
    if not resolved_export.empty:
        weekly_base = (
            resolved_export.groupby("season_week", as_index=False)
            .agg(
                resolved_predictions=("correct", "size"),
                weekly_accuracy=("correct", "mean"),
            )
            .sort_values("season_week")
        )
        weekly_prob_metrics = (
            resolved_export.groupby("season_week", as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "weekly_log_loss": log_loss(
                            pd.to_numeric(g["win_home_actual"], errors="coerce").astype(int),
                            np.clip(pd.to_numeric(g["home_win_probability"], errors="coerce").fillna(0.5), 1e-6, 1 - 1e-6),
                            labels=[0, 1],
                        ),
                        "weekly_brier": brier_score_loss(
                            pd.to_numeric(g["win_home_actual"], errors="coerce").astype(int),
                            np.clip(pd.to_numeric(g["home_win_probability"], errors="coerce").fillna(0.5), 1e-6, 1 - 1e-6),
                        ),
                        "weekly_ece": _expected_calibration_error(
                            pd.to_numeric(g["win_home_actual"], errors="coerce").astype(int),
                            pd.to_numeric(g["home_win_probability"], errors="coerce").fillna(0.5),
                        ),
                    }
                )
            )
            .reset_index(drop=True)
        )

        weekly_bucket = (
            resolved_export.groupby(["season_week", "confidence_bucket"], as_index=False)
            .agg(
                bucket_predictions=("correct", "size"),
                bucket_accuracy=("correct", "mean"),
            )
            .sort_values(["season_week", "confidence_bucket"])
        )
        weekly_bucket["bucket_accuracy"] = weekly_bucket["bucket_accuracy"].round(3)
        weekly_bucket["bucket_predictions"] = weekly_bucket["bucket_predictions"].astype(int)

        # Pivot bucket accuracies into columns for quick weekly scanning.
        pivot_acc = weekly_bucket.pivot(
            index="season_week", columns="confidence_bucket", values="bucket_accuracy"
        ).reset_index()
        pivot_cnt = weekly_bucket.pivot(
            index="season_week", columns="confidence_bucket", values="bucket_predictions"
        ).reset_index()
        pivot_acc = pivot_acc.rename(
            columns={
                "high": "high_accuracy",
                "medium": "medium_accuracy",
                "low": "low_accuracy",
            }
        )
        pivot_cnt = pivot_cnt.rename(
            columns={
                "high": "high_predictions",
                "medium": "medium_predictions",
                "low": "low_predictions",
            }
        )

        weekly_summary = weekly_base.merge(weekly_prob_metrics, on="season_week", how="left").merge(pivot_acc, on="season_week", how="left").merge(
            pivot_cnt, on="season_week", how="left"
        )
        weekly_summary["weekly_accuracy"] = weekly_summary["weekly_accuracy"].round(3)
        weekly_summary["weekly_log_loss"] = weekly_summary["weekly_log_loss"].round(3)
        weekly_summary["weekly_brier"] = weekly_summary["weekly_brier"].round(3)
        weekly_summary["weekly_ece"] = weekly_summary["weekly_ece"].round(3)
        weekly_summary = weekly_summary.fillna(0)

        weekly_out_path = "data/weekly_summary.csv"
        weekly_summary.to_csv(weekly_out_path, index=False)
        print(f"Saved weekly summary to {weekly_out_path}")
        write_calibration_trend_csv(weekly_summary, "data/calibration_trend.csv")
    else:
        # Keep the file present for downstream tooling, even before results resolve.
        weekly_out_path = "data/weekly_summary.csv"
        pd.DataFrame(
            columns=[
                "season_week",
                "resolved_predictions",
                "weekly_accuracy",
                "weekly_log_loss",
                "weekly_brier",
                "weekly_ece",
                "high_accuracy",
                "medium_accuracy",
                "low_accuracy",
                "high_predictions",
                "medium_predictions",
                "low_predictions",
            ]
        ).to_csv(weekly_out_path, index=False)
        print(f"Saved empty weekly summary to {weekly_out_path}")
        write_calibration_trend_csv(pd.DataFrame(), "data/calibration_trend.csv")

    # Sidecar file for point predictions vs actuals (does not affect feedback CSV schema).
    _pts_related = [c for c in export_df.columns if "pts" in c.lower()]
    if _pts_related:
        _ctx = ["season_week", "game_date", "matchup", "home_team", "away_team", "result_status"]
        score_side = export_df[[c for c in _ctx + _pts_related if c in export_df.columns]].copy()
        score_side_path = "data/scored_score_predictions.csv"
        score_side.to_csv(score_side_path, index=False)
        print(f"Saved score prediction vs actual columns to {score_side_path}")

    _resolved_for_backtest = export_df[export_df["result_status"] == "resolved"].copy()
    write_value_backtest_snapshot(
        _resolved_for_backtest,
        value_path="data/value_recommendations.csv",
        out_summary_path="data/value_backtest_summary.json",
        out_curve_path="data/value_backtest_curve.csv",
    )


if __name__ == "__main__":
    main()
