from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd

from odds_provider import fetch_nba_market_odds
from predict_next_games import main as run_predictions_main
from team_aliases import normalize_team_abbr
from value_betting import (
    edge_percent,
    expected_value_per_unit_stake,
    fair_odds_from_probability,
    kelly_fraction,
    kelly_stake_amount,
)

def _to_date_string(iso_or_date: str) -> str:
    if not iso_or_date:
        return ""
    try:
        if "T" in iso_or_date:
            dt = pd.to_datetime(iso_or_date, utc=True).tz_convert(None)
        else:
            dt = pd.to_datetime(iso_or_date)
    except Exception:
        return ""
    return dt.strftime("%m/%d/%Y")


def _to_team_abbr(team: str) -> str:
    return normalize_team_abbr(team)


def _best_book_line(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.copy()
    ranked["home_rank"] = ranked.groupby(["game_id"])["home_decimal_odds"].rank(
        method="first", ascending=False
    )
    ranked["away_rank"] = ranked.groupby(["game_id"])["away_decimal_odds"].rank(
        method="first", ascending=False
    )
    home_best = ranked[ranked["home_rank"] == 1].copy()
    away_best = ranked[ranked["away_rank"] == 1].copy()

    merged = home_best[
        ["game_id", "home_team", "away_team", "commence_time", "bookmaker", "home_decimal_odds"]
    ].rename(columns={"bookmaker": "home_best_bookmaker"})
    merged = merged.merge(
        away_best[["game_id", "bookmaker", "away_decimal_odds"]].rename(
            columns={"bookmaker": "away_best_bookmaker"}
        ),
        on="game_id",
        how="inner",
    )
    return merged


def build_value_plays(
    bankroll: float = 1000.0,
    edge_threshold_pct: float = 3.0,
    kelly_multiplier: float = 0.25,
    max_stake_pct: float = 5.0,
    diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    stats: dict[str, int | float] = {
        "pred_rows": 0,
        "odds_raw_rows": 0,
        "odds_best_rows": 0,
        "merged_rows": 0,
        "analysis_rows": 0,
        "positive_ev_rows": 0,
        "team_overlap_home": 0,
        "team_overlap_away": 0,
    }
    pred_path = "data/predictions_log.csv"
    if not os.path.exists(pred_path):
        run_predictions_main()
    if not os.path.exists(pred_path):
        raise FileNotFoundError("Missing data/predictions_log.csv. Run predict_next_games.py first.")

    preds = pd.read_csv(pred_path)
    stats["pred_rows"] = int(len(preds))
    required_cols = ["game_date", "home_team", "away_team", "matchup", "home_win_probability"]
    for col in required_cols:
        if col not in preds.columns:
            raise ValueError(f"predictions_log.csv is missing required column: {col}")

    if "run_timestamp" in preds.columns:
        preds["run_timestamp"] = pd.to_datetime(preds["run_timestamp"], errors="coerce")
        preds = preds.sort_values("run_timestamp").drop_duplicates(
            subset=["game_date", "home_team", "away_team"], keep="last"
        )

    odds_raw = fetch_nba_market_odds()
    if odds_raw.empty:
        empty = pd.DataFrame()
        return (empty, stats) if diagnostics else empty
    stats["odds_raw_rows"] = int(len(odds_raw))
    odds_best = _best_book_line(odds_raw)
    if odds_best.empty:
        empty = pd.DataFrame()
        return (empty, stats) if diagnostics else empty
    stats["odds_best_rows"] = int(len(odds_best))
    odds_best["game_date"] = odds_best["commence_time"].apply(_to_date_string)

    preds = preds.copy()
    preds["home_team_abbr"] = preds["home_team"].apply(_to_team_abbr)
    preds["away_team_abbr"] = preds["away_team"].apply(_to_team_abbr)

    odds_best["home_team_abbr"] = odds_best["home_team"].apply(_to_team_abbr)
    odds_best["away_team_abbr"] = odds_best["away_team"].apply(_to_team_abbr)

    merged = preds.merge(
        odds_best,
        on=["game_date", "home_team_abbr", "away_team_abbr"],
        how="inner",
        suffixes=("_pred", "_odds"),
    )
    if merged.empty:
        empty = pd.DataFrame()
        stats["team_overlap_home"] = int(
            len(set(preds["home_team_abbr"]).intersection(set(odds_best["home_team_abbr"])))
        )
        stats["team_overlap_away"] = int(
            len(set(preds["away_team_abbr"]).intersection(set(odds_best["away_team_abbr"])))
        )
        return (empty, stats) if diagnostics else empty
    stats["merged_rows"] = int(len(merged))

    rows = []
    for _, row in merged.iterrows():
        home_prob = float(row["home_win_probability"])
        away_prob = 1.0 - home_prob

        home_market_odds = float(row["home_decimal_odds"])
        away_market_odds = float(row["away_decimal_odds"])

        for side, model_prob, market_odds, best_book in [
            ("home", home_prob, home_market_odds, row["home_best_bookmaker"]),
            ("away", away_prob, away_market_odds, row["away_best_bookmaker"]),
        ]:
            fair = fair_odds_from_probability(model_prob)
            edge_pct = edge_percent(model_prob, market_odds)
            ev = expected_value_per_unit_stake(model_prob, market_odds)
            k_frac = kelly_fraction(
                model_win_prob=model_prob,
                market_decimal_odds=market_odds,
                safety_multiplier=kelly_multiplier,
                max_fraction=max_stake_pct / 100.0,
            )
            stake_amt = kelly_stake_amount(
                current_bankroll=bankroll,
                model_win_prob=model_prob,
                market_decimal_odds=market_odds,
                safety_multiplier=kelly_multiplier,
                max_fraction=max_stake_pct / 100.0,
            )
            rows.append(
                {
                    "game_date": row["game_date"],
                    "matchup": row["matchup"],
                    "side": side,
                    "model_win_probability": model_prob,
                    "model_win_pct": model_prob * 100.0,
                    "fair_decimal_odds": fair.decimal,
                    "fair_american_odds": fair.american,
                    "fair_fractional_odds": fair.fractional,
                    "market_decimal_odds": market_odds,
                    "best_bookmaker": best_book,
                    "edge_pct": edge_pct,
                    "ev_per_unit_stake": ev,
                    "kelly_stake_pct": k_frac * 100.0,
                    "suggested_stake_amount": stake_amt,
                }
            )

    analysis_df = pd.DataFrame(rows)
    if analysis_df.empty:
        return (analysis_df, stats) if diagnostics else analysis_df
    stats["analysis_rows"] = int(len(analysis_df))

    analysis_df = analysis_df.sort_values("edge_pct", ascending=False).reset_index(drop=True)
    # Temporarily disable edge threshold filtering and select by EV only.
    # We still compute and output edge_pct for visibility/debugging.
    recommended = analysis_df[
        (analysis_df["ev_per_unit_stake"] > 0)
    ].copy()
    stats["positive_ev_rows"] = int(len(recommended))
    recommended = recommended.sort_values(["edge_pct", "ev_per_unit_stake"], ascending=False)
    recommended["run_timestamp"] = datetime.now().isoformat(timespec="seconds")

    os.makedirs("data", exist_ok=True)
    rec_path = "data/value_recommendations.csv"
    if os.path.exists(rec_path):
        prev = pd.read_csv(rec_path)
        pd.concat([prev, recommended], ignore_index=True).to_csv(rec_path, index=False)
    else:
        recommended.to_csv(rec_path, index=False)
    return (recommended, stats) if diagnostics else recommended


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NBA value betting recommendations.")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--edge-threshold", type=float, default=3.0)
    parser.add_argument("--kelly-multiplier", type=float, default=0.25)
    parser.add_argument("--max-stake-pct", type=float, default=5.0)
    args = parser.parse_args()

    recommendations, stats = build_value_plays(
        bankroll=args.bankroll,
        edge_threshold_pct=args.edge_threshold,
        kelly_multiplier=args.kelly_multiplier,
        max_stake_pct=args.max_stake_pct,
        diagnostics=True,
    )

    if recommendations.empty:
        print("No recommended plays found.")
        print(
            "Diagnostics:",
            {
                "pred_rows": stats["pred_rows"],
                "odds_raw_rows": stats["odds_raw_rows"],
                "odds_best_rows": stats["odds_best_rows"],
                "merged_rows": stats["merged_rows"],
                "analysis_rows": stats["analysis_rows"],
                "positive_ev_rows": stats["positive_ev_rows"],
                "team_overlap_home": stats["team_overlap_home"],
                "team_overlap_away": stats["team_overlap_away"],
            },
        )
        return

    print("Recommended Plays")
    print(recommendations.to_string(index=False))
    print("\nSaved/updated data/value_recommendations.csv")


if __name__ == "__main__":
    main()
