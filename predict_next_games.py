import argparse
import env_bootstrap  # noqa: F401 — load `.env` (HTTP_PROXY / HTTPS_PROXY / NO_PROXY)
import json
import os
import re
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.endpoints import scoreboardv3
from requests.exceptions import RequestException

from game_time_display import display_timezone_for_cli, format_local_from_utc_iso, utc_iso_from_game_row
from prediction_explain import explain_matchup
from train_model import (
    DEFAULT_LATEST_TEAM_CACHE,
    apply_calibrator,
    build_game_level_frame,
    latest_team_snapshot_from_frames,
    load_model_bundle,
    three_in_four_flag_from_schedule,
)


def _normalize_col(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def _pick_col(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize_col(c): c for c in columns}
    for cand in candidates:
        key = _normalize_col(cand)
        if key in normalized:
            return normalized[key]
    return None


def load_latest_team_state(
    hist_path: str = "data/historical_games.csv",
    prefer_cache: bool = True,
    cache_path: str = DEFAULT_LATEST_TEAM_CACHE,
) -> pd.DataFrame:
    if not os.path.exists(hist_path):
        raise FileNotFoundError(
            "Missing data/historical_games.csv. Run fetch_data.py first."
        )

    cache_ok = (
        prefer_cache
        and os.path.exists(cache_path)
        and os.path.getmtime(cache_path) >= os.path.getmtime(hist_path)
    )
    if cache_ok:
        latest = pd.read_csv(
            cache_path,
            parse_dates=["GAME_DATE", "SECOND_LAST_GAME_DATE"],
        )
        if "N_COMPLETED_GAMES" in latest.columns:
            latest["N_COMPLETED_GAMES"] = pd.to_numeric(
                latest["N_COMPLETED_GAMES"], errors="coerce"
            ).fillna(0).astype(int)
        return latest.copy()

    raw = pd.read_csv(hist_path, parse_dates=["GAME_DATE"])
    games = build_game_level_frame(raw)
    latest = latest_team_snapshot_from_frames(raw, games=games)

    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        latest.to_csv(cache_path, index=False)
    except OSError:
        pass

    return latest.copy()


def load_score_model_bundle(path: str = "artifacts/score_models.joblib"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_point_totals(score_bundle: dict | None, feature_df: pd.DataFrame) -> tuple[float | None, float | None]:
    if score_bundle is None or not isinstance(score_bundle, dict):
        return None, None
    cols = score_bundle.get("feature_columns") or []
    if not cols:
        return None, None
    X = feature_df.reindex(columns=cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    try:
        ph = float(score_bundle["estimator_home"].predict(X)[0])
        pa = float(score_bundle["estimator_away"].predict(X)[0])
        return round(ph, 1), round(pa, 1)
    except Exception:
        return None, None


def _raw_predict_probability(model_bundle, feature_df: pd.DataFrame) -> np.ndarray:
    if isinstance(model_bundle, dict):
        model_type = model_bundle.get("model_type", "logistic")
        estimator = model_bundle.get("estimator")
        if model_type == "elo_baseline":
            return feature_df["ELO_HOME_WIN_PROB"].to_numpy(dtype=float)
        return estimator.predict_proba(feature_df)[:, 1]
    return model_bundle.predict_proba(feature_df)[:, 1]


_FEATURE_SKIP = frozenset(
    {
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "IS_HOME",
        "RESULT",
        "SEASON_ID",
        "TEAM_NAME",
        "GAME_ID",
        "SECOND_LAST_GAME_DATE",
        "N_COMPLETED_GAMES",
        "SOS_OPP_NET_10_INCL_CURRENT",
        "OPP_SNAP_NET_10",
        "_sos_in",
        "OPP_TEAM_ID",
    }
)


def _build_feature_row(
    home_row: pd.Series,
    away_row: pd.Series,
    game_date: str,
    *,
    playoff_game: bool = False,
) -> dict:
    game_dt = pd.to_datetime(game_date, format="%m/%d/%Y", errors="coerce")
    home_days = (
        int((game_dt - pd.to_datetime(home_row["GAME_DATE"])).days)
        if pd.notna(game_dt)
        else int(home_row.get("DAYS_SINCE_LAST_GAME", 2))
    )
    away_days = (
        int((game_dt - pd.to_datetime(away_row["GAME_DATE"])).days)
        if pd.notna(game_dt)
        else int(away_row.get("DAYS_SINCE_LAST_GAME", 2))
    )
    home_days = max(0, min(10, home_days))
    away_days = max(0, min(10, away_days))

    row = {}
    for col in home_row.index:
        if col in _FEATURE_SKIP or col == "ELO_PRE":
            continue
        row[f"{col}_HOME"] = float(home_row[col])
    for col in away_row.index:
        if col in _FEATURE_SKIP or col == "ELO_PRE":
            continue
        row[f"{col}_AWAY"] = float(away_row[col])

    if "ELO_PRE" in home_row.index:
        row["ELO_HOME_PRE"] = float(home_row["ELO_PRE"])
    if "ELO_PRE" in away_row.index:
        row["ELO_AWAY_PRE"] = float(away_row["ELO_PRE"])

    row["DAYS_SINCE_LAST_GAME_HOME"] = home_days
    row["DAYS_SINCE_LAST_GAME_AWAY"] = away_days
    row["REST_DIFF"] = home_days - away_days
    row["B2B_FLAG_HOME"] = int(home_days <= 1)
    row["B2B_FLAG_AWAY"] = int(away_days <= 1)
    row["B2B_DIFF"] = row["B2B_FLAG_HOME"] - row["B2B_FLAG_AWAY"]

    def _n_completed(x: object) -> int:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return 0
        return int(v)

    n_h = _n_completed(home_row.get("N_COMPLETED_GAMES"))
    n_a = _n_completed(away_row.get("N_COMPLETED_GAMES"))
    row["THREE_IN_FOUR_FLAG_HOME"] = three_in_four_flag_from_schedule(
        game_dt,
        home_row["GAME_DATE"],
        home_row.get("SECOND_LAST_GAME_DATE"),
        n_h,
    )
    row["THREE_IN_FOUR_FLAG_AWAY"] = three_in_four_flag_from_schedule(
        game_dt,
        away_row["GAME_DATE"],
        away_row.get("SECOND_LAST_GAME_DATE"),
        n_a,
    )
    row["THREE_IN_FOUR_DIFF"] = row["THREE_IN_FOUR_FLAG_HOME"] - row["THREE_IN_FOUR_FLAG_AWAY"]

    sos_h = float(
        home_row.get("SOS_OPP_NET_10_INCL_CURRENT", home_row.get("SOS_OPP_NET_10", 0.0)) or 0.0
    )
    sos_a = float(
        away_row.get("SOS_OPP_NET_10_INCL_CURRENT", away_row.get("SOS_OPP_NET_10", 0.0)) or 0.0
    )
    row["SOS_OPP_NET_10_HOME"] = sos_h
    row["SOS_OPP_NET_10_AWAY"] = sos_a

    row["PLAYOFF_GAME"] = float(playoff_game)
    row["ELO_DIFF_PRE"] = float(row.get("ELO_HOME_PRE", 1500.0) - row.get("ELO_AWAY_PRE", 1500.0) + 70.0)
    row["ELO_HOME_WIN_PROB"] = 1.0 / (1.0 + 10.0 ** (-row["ELO_DIFF_PRE"] / 400.0))
    return row


def _games_from_scoreboard_v3(sb: scoreboardv3.ScoreboardV3, game_date_mmdd: str) -> pd.DataFrame:
    """
    ScoreboardV3 game_header has tip times but (as of 2025–26) no home/away team IDs.
    Merge team_leaders (leaderType home/away + teamId) onto game_header by gameId.
    """
    empty = pd.DataFrame(columns=["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_DATE"])
    hdr = sb.game_header.get_data_frame()
    if hdr is None or hdr.empty:
        return empty

    hdr_cols = list(hdr.columns)
    gid_hdr = _pick_col(hdr_cols, ["gameId", "GAME_ID"])
    if not gid_hdr:
        return empty

    tl_ds = sb.team_leaders
    if tl_ds is None:
        return empty
    tldf = tl_ds.get_data_frame()
    if tldf is None or tldf.empty:
        return empty

    tl_cols = list(tldf.columns)
    lt_col = _pick_col(tl_cols, ["leaderType", "LEADERTYPE"])
    gid_tl = _pick_col(tl_cols, ["gameId", "GAME_ID"])
    tid_col = _pick_col(tl_cols, ["teamId", "TEAM_ID"])
    if not (lt_col and gid_tl and tid_col):
        return empty

    home = tldf[tldf[lt_col].astype(str).str.lower() == "home"][[gid_tl, tid_col]].drop_duplicates(subset=[gid_tl])
    home = home.rename(columns={gid_tl: "gameId", tid_col: "HOME_TEAM_ID"})
    away = tldf[tldf[lt_col].astype(str).str.lower() == "away"][[gid_tl, tid_col]].drop_duplicates(subset=[gid_tl])
    away = away.rename(columns={gid_tl: "gameId", tid_col: "VISITOR_TEAM_ID"})
    teams = home.merge(away, on="gameId", how="inner")
    if teams.empty:
        return empty

    merged = hdr.merge(teams, left_on=gid_hdr, right_on="gameId", how="inner", suffixes=("", "_tl"))
    if merged.empty:
        return empty

    out = pd.DataFrame()
    out["GAME_ID"] = merged[gid_hdr].astype(str)
    out["HOME_TEAM_ID"] = pd.to_numeric(merged["HOME_TEAM_ID"], errors="coerce").fillna(0).astype(np.int64)
    out["VISITOR_TEAM_ID"] = pd.to_numeric(merged["VISITOR_TEAM_ID"], errors="coerce").fillna(0).astype(np.int64)

    mcols = list(merged.columns)
    st = _pick_col(mcols, ["gameStatusText", "GAME_STATUS_TEXT"])
    if st:
        out["GAME_STATUS_TEXT"] = merged[st].to_numpy()
        out["GAME_STATUS_TEXT_V3"] = merged[st].to_numpy()
    et = _pick_col(mcols, ["gameEt", "GAME_ET"])
    if et:
        out["GAME_ET"] = merged[et].to_numpy()
    utc = _pick_col(mcols, ["gameTimeUTC", "GAME_TIME_UTC", "gameDateTimeUTC"])
    if utc:
        out["GAME_TIME_UTC"] = merged[utc].to_numpy()
    out["GAME_DATE"] = game_date_mmdd

    return out.drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)


def fetch_games_for_date(game_date: str) -> pd.DataFrame:
    """Prefer ScoreboardV3 (recommended for 2025–26+); fall back to V2 with warnings suppressed."""
    empty = pd.DataFrame(columns=["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_DATE"])
    parsed = pd.to_datetime(game_date, format="%m/%d/%Y", errors="coerce")
    if pd.notna(parsed):
        iso = parsed.strftime("%Y-%m-%d")
        try:
            sb_v3 = scoreboardv3.ScoreboardV3(game_date=iso, league_id="00", timeout=8)
            games = _games_from_scoreboard_v3(sb_v3, game_date)
            if not games.empty:
                return games
        except Exception:
            pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sb = scoreboardv2.ScoreboardV2(
                game_date=game_date, day_offset=0, league_id="00", timeout=8
            )
    except RequestException:
        return empty
    except Exception:
        return empty

    frame = sb.game_header.get_data_frame()
    if frame.empty:
        return empty

    cols = list(frame.columns)
    game_col = _pick_col(cols, ["GAME_ID"])
    home_col = _pick_col(cols, ["HOME_TEAM_ID"])
    away_col = _pick_col(cols, ["VISITOR_TEAM_ID"])
    status_text_col = _pick_col(cols, ["GAME_STATUS_TEXT"])

    if not (game_col and home_col and away_col):
        return empty

    rename_map = {
        game_col: "GAME_ID",
        home_col: "HOME_TEAM_ID",
        away_col: "VISITOR_TEAM_ID",
    }
    keep_cols = [game_col, home_col, away_col]
    if status_text_col:
        keep_cols.append(status_text_col)
        rename_map[status_text_col] = "GAME_STATUS_TEXT"

    games = frame[keep_cols].copy().rename(columns=rename_map)
    games["GAME_DATE"] = game_date
    games = games.drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)
    return games


def fetch_upcoming_games(limit: int = 5, max_days_ahead: int = 14) -> pd.DataFrame:
    now = datetime.now()
    all_days = []

    for day_offset in range(0, max_days_ahead + 1):
        date_str = (now + timedelta(days=day_offset)).strftime("%m/%d/%Y")
        print(f"Checking date {date_str}...")
        day_games = fetch_games_for_date(date_str)
        if not day_games.empty:
            all_days.append(day_games)

        total_rows = sum(len(df) for df in all_days)
        if total_rows >= limit:
            break

    if not all_days:
        return pd.DataFrame(columns=["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_DATE"])

    out = pd.concat(all_days, ignore_index=True)
    out = out.drop_duplicates(subset=["GAME_ID"]).head(limit).reset_index(drop=True)
    return out


def _extract_start_time_24h(game_status_text: str | None, default_tz: str = "ET") -> str:
    if game_status_text is None:
        return "unknown"
    if isinstance(game_status_text, float) and pd.isna(game_status_text):
        return "unknown"
    raw = str(game_status_text).strip()
    if not raw or raw.lower() == "nan":
        return "unknown"

    # ISO-8601 fragment anywhere in string (ScoreboardV3 gameDateTimeUTC / gameTimeUTC).
    iso_match = re.search(r"T(\d{2}):(\d{2})(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", raw)
    if iso_match:
        hour = int(iso_match.group(1))
        minute = int(iso_match.group(2))
        return f"{hour:02d}:{minute:02d} {default_tz.upper()}"

    match = re.search(
        r"(\d{1,2}):(\d{2})\s*([AP]M)(?:\s*([A-Z]{2,4}))?",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return "unknown"

    hour = int(match.group(1))
    minute = int(match.group(2))
    ampm = match.group(3).upper()
    if ampm == "PM" and hour != 12:
        hour += 12
    if ampm == "AM" and hour == 12:
        hour = 0

    tz = (match.group(4) or default_tz).upper()
    return f"{hour:02d}:{minute:02d} {tz}"


def _start_time_24h_from_game_row(g: pd.Series) -> str:
    """Prefer V3 tip text (gameStatusText, e.g. '7:00 pm ET'), then UTC ISO, then gameEt."""
    for key, tz in (
        ("GAME_STATUS_TEXT_V3", "ET"),
        ("GAME_STATUS_TEXT", "ET"),
        ("GAME_TIME_UTC", "UTC"),
        ("GAME_ET", "ET"),
    ):
        raw = g.get(key)
        t = _extract_start_time_24h(raw, default_tz=tz)
        if t != "unknown":
            return t
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--days-ahead", type=int, default=14)
    parser.add_argument(
        "--no-team-cache",
        action="store_true",
        help="Rebuild latest team rows from scratch (ignore data/latest_team_state_cache.csv).",
    )
    args = parser.parse_args()

    if not os.path.exists("artifacts/model.joblib"):
        raise FileNotFoundError("Missing artifacts/model.joblib. Run train_model.py first.")
    if not os.path.exists("artifacts/features.json"):
        raise FileNotFoundError("Missing artifacts/features.json. Run train_model.py first.")

    model_bundle = load_model_bundle("artifacts/model.joblib")
    if isinstance(model_bundle, dict) and "feature_columns" in model_bundle:
        feature_cols = model_bundle["feature_columns"]
    else:
        with open("artifacts/features.json", "r", encoding="utf-8") as f:
            feature_cols = json.load(f)["feature_columns"]

    team_state = load_latest_team_state(prefer_cache=not args.no_team_cache)
    score_bundle = load_score_model_bundle()
    upcoming_games = fetch_upcoming_games(limit=args.count, max_days_ahead=args.days_ahead)
    if upcoming_games.empty:
        print(f"No games found in the next {args.days_ahead} days.")
        return

    rows = []
    for _, g in upcoming_games.iterrows():
        home = team_state[team_state["TEAM_ID"] == g["HOME_TEAM_ID"]]
        away = team_state[team_state["TEAM_ID"] == g["VISITOR_TEAM_ID"]]
        if home.empty or away.empty:
            continue

        home_row = home.iloc[0]
        away_row = away.iloc[0]
        playoff = bool(str(g.get("GAME_ID", "")).startswith("004"))
        feat = _build_feature_row(home_row, away_row, g["GAME_DATE"], playoff_game=playoff)
        feature_df = pd.DataFrame([feat]).reindex(columns=feature_cols, fill_value=0.0)
        raw_prob = _raw_predict_probability(model_bundle, feature_df)
        calibrator = model_bundle.get("calibrator") if isinstance(model_bundle, dict) else None
        feedback_calibrator = (
            model_bundle.get("feedback_calibrator") if isinstance(model_bundle, dict) else None
        )
        win_home_prob = float(apply_calibrator(raw_prob, calibrator)[0])
        win_home_prob = float(apply_calibrator(np.array([win_home_prob]), feedback_calibrator)[0])
        ph, pa = predict_point_totals(score_bundle, feature_df)
        utc_tip = utc_iso_from_game_row(g)
        rationale_bullets = explain_matchup(
            feat,
            str(home_row["TEAM_ABBREVIATION"]),
            str(away_row["TEAM_ABBREVIATION"]),
        )
        row = {
            "matchup": f"{away_row['TEAM_ABBREVIATION']} @ {home_row['TEAM_ABBREVIATION']}",
            "game_date": g["GAME_DATE"],
            "game_start_time_utc": utc_tip if utc_tip else pd.NA,
            "home_team": home_row["TEAM_ABBREVIATION"],
            "away_team": away_row["TEAM_ABBREVIATION"],
            "home_win_probability": win_home_prob,
            "rationale": " / ".join(rationale_bullets),
        }
        if ph is not None and pa is not None:
            row["pred_home_pts"] = ph
            row["pred_away_pts"] = pa
        rows.append(row)

    if not rows:
        print("No predictable games found (team state missing).")
        return

    out = pd.DataFrame(rows).sort_values("home_win_probability", ascending=False)
    _tz_cli = display_timezone_for_cli()
    _print = out.copy()
    _print["start_local"] = _print["game_start_time_utc"].map(
        lambda u: format_local_from_utc_iso(u if pd.notna(u) and str(u).strip() else None, _tz_cli)
    )
    _cols = [
        "matchup",
        "game_date",
        "start_local",
        "home_team",
        "away_team",
        "home_win_probability",
    ]
    if "pred_home_pts" in _print.columns:
        _cols += ["pred_home_pts", "pred_away_pts"]
    if "rationale" in _print.columns:
        _cols.append("rationale")
    print(_print[[c for c in _cols if c in _print.columns]].to_string(index=False))

    os.makedirs("data", exist_ok=True)
    log_path = "data/predictions_log.csv"
    run_ts = datetime.now().isoformat(timespec="seconds")
    log_df = out.copy()
    log_df["run_timestamp"] = run_ts

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        for c in ("pred_home_pts", "pred_away_pts", "game_start_time_utc", "rationale"):
            if c not in existing.columns:
                existing[c] = pd.NA
        combined = pd.concat([existing, log_df], ignore_index=True)
        combined.to_csv(log_path, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    print(f"\nSaved predictions to {log_path}")


if __name__ == "__main__":
    main()
