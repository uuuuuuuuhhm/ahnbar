import io
import html
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def _apply_proxy_secrets_from_streamlit() -> None:
    """Copy proxy keys from st.secrets into os.environ before env_bootstrap / requests."""
    try:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
            if key not in st.secrets:
                continue
            val = str(st.secrets[key]).strip().strip('"').strip("'")
            if val:
                os.environ.setdefault(key, val)
    except Exception:
        pass


_apply_proxy_secrets_from_streamlit()

import env_bootstrap  # noqa: E402, F401 — after Streamlit secrets so .env does not override secrets

import joblib
import pandas as pd
from nba_api.stats.static import teams as nba_static_teams

from game_time_display import (
    COMMON_DISPLAY_TIMEZONES,
    DEFAULT_DISPLAY_TZ,
    format_start_display,
    utc_iso_from_game_row,
)
from predict_value_plays import build_value_plays
from predict_next_games import (
    _build_feature_row,
    _raw_predict_probability,
    fetch_games_for_date,
    fetch_upcoming_games,
    load_latest_team_state,
    load_score_model_bundle,
    predict_point_totals,
)
from prediction_explain import explain_matchup
from zoneinfo import ZoneInfo


def _confidence_level(prob_home: float) -> str:
    """Bucket by distance from 50% (lean strength), not statistical calibration."""
    conf = max(prob_home, 1.0 - prob_home)
    if conf >= 0.65:
        return "high"
    if conf >= 0.55:
        return "medium"
    return "low"


def _confidence_badge(level: str) -> str:
    if level == "high":
        return "🟢 high"
    if level == "medium":
        return "🟡 medium"
    return "🔴 low"


def _traffic_light_color(value: float) -> str:
    # 0% red -> 50% yellow/orange -> 100% green
    v = max(0.0, min(100.0, float(value)))
    red = (239, 68, 68)
    yellow = (234, 179, 8)
    green = (34, 197, 94)

    if v <= 50.0:
        t = v / 50.0
        rgb = tuple(int(red[i] + t * (yellow[i] - red[i])) for i in range(3))
    else:
        t = (v - 50.0) / 50.0
        rgb = tuple(int(yellow[i] + t * (green[i] - yellow[i])) for i in range(3))
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def _style_percentage_bars(df: pd.DataFrame, bar_cols: list[str]):
    def style_row(row):
        styles = [""] * len(row)
        col_map = {c: i for i, c in enumerate(df.columns)}
        for col in bar_cols:
            pct = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(pct):
                continue
            pct = float(pct)
            color = _traffic_light_color(pct)
            styles[col_map[col]] = (
                "background: linear-gradient(90deg, "
                f"{color} {pct:.1f}%, transparent {pct:.1f}%);"
            )
        return styles

    return df.style.apply(style_row, axis=1)


def run_predictions(count: int, days_ahead: int) -> pd.DataFrame:
    from train_model import apply_calibrator, load_model_bundle

    _rp_t0 = pd.Timestamp.now()
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

    team_state = load_latest_team_state()
    score_bundle = load_score_model_bundle()
    upcoming_games = fetch_upcoming_games(limit=count, max_days_ahead=days_ahead)
    if upcoming_games.empty:
        return pd.DataFrame()

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
        win_home_prob = float(apply_calibrator(pd.Series([win_home_prob]).to_numpy(), feedback_calibrator)[0])
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
            "home_team": home_row["TEAM_NAME"],
            "away_team": away_row["TEAM_NAME"],
            "home_win_probability": round(win_home_prob, 4),
            "confidence_pct": round(max(win_home_prob, 1.0 - win_home_prob), 4),
            "confidence_level": _confidence_level(win_home_prob),
            "rationale": " / ".join(rationale_bullets),
        }
        if ph is not None and pa is not None:
            row["pred_home_pts"] = ph
            row["pred_away_pts"] = pa
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("home_win_probability", ascending=False)
    if out.empty:
        return out

    os.makedirs("data", exist_ok=True)
    log_path = "data/predictions_log.csv"
    log_df = out.copy()
    log_df["run_timestamp"] = pd.Timestamp.now().isoformat(timespec="seconds")
    if os.path.exists(log_path):
        old = pd.read_csv(log_path)
        for c in ("pred_home_pts", "pred_away_pts", "game_start_time_utc", "rationale"):
            if c not in old.columns:
                old[c] = pd.NA
        pd.concat([old, log_df], ignore_index=True).to_csv(log_path, index=False)
    else:
        log_df.to_csv(log_path, index=False)
    return out


def _prediction_cache_key(count: int, days_ahead: int) -> str:
    return f"count={int(count)}|days_ahead={int(days_ahead)}"


def _ensure_prediction_session_cache() -> dict[str, dict]:
    if "prediction_session_cache" not in st.session_state:
        st.session_state["prediction_session_cache"] = {}
    cache = st.session_state.get("prediction_session_cache")
    return cache if isinstance(cache, dict) else {}


def clear_prediction_session_cache() -> None:
    st.session_state["prediction_session_cache"] = {}


def get_predictions_cached(
    count: int,
    days_ahead: int,
    *,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cache = _ensure_prediction_session_cache()
    cache_key = _prediction_cache_key(int(count), int(days_ahead))
    entry = cache.get(cache_key)
    if not force_refresh and isinstance(entry, dict):
        payload = entry.get("predictions")
        if isinstance(payload, str):
            try:
                cached_df = pd.read_json(payload, orient="split")
                meta = {
                    "source": "cache",
                    "cached_at": entry.get("cached_at", ""),
                    "row_count": int(entry.get("row_count", len(cached_df))),
                    "cache_key": cache_key,
                }
                st.session_state["latest_prediction_meta"] = meta
                return cached_df, meta
            except Exception:
                pass

    preds = run_predictions(int(count), int(days_ahead))
    cached_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    cache[cache_key] = {
        "predictions": preds.to_json(orient="split", date_format="iso"),
        "cached_at": cached_at,
        "row_count": int(len(preds)),
    }
    st.session_state["prediction_session_cache"] = cache
    fresh_meta = {
        "source": "fresh",
        "cached_at": cached_at,
        "row_count": int(len(preds)),
        "cache_key": cache_key,
    }
    st.session_state["latest_prediction_meta"] = fresh_meta
    return preds, fresh_meta


@st.cache_data(ttl=300, show_spinner=False)
def _load_playoff_actual_games() -> pd.DataFrame:
    _t0 = pd.Timestamp.now()
    path = "data/historical_games.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        raw = pd.read_csv(
            path,
            parse_dates=["GAME_DATE"],
            dtype={"GAME_ID": str, "TEAM_ABBREVIATION": str, "TEAM_NAME": str, "MATCHUP": str},
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()
    if raw.empty or "GAME_ID" not in raw.columns:
        return pd.DataFrame()
    raw["GAME_ID"] = raw["GAME_ID"].astype(str).str.strip()
    raw = raw[raw["GAME_ID"].str.startswith("004")].copy()
    if raw.empty:
        return pd.DataFrame()
    raw["IS_HOME"] = raw["MATCHUP"].astype(str).str.contains(" vs. ")
    home = raw[raw["IS_HOME"]][["GAME_ID", "GAME_DATE", "TEAM_NAME", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={
            "TEAM_NAME": "home_team",
            "TEAM_ABBREVIATION": "home_abbr",
            "PTS": "home_pts_actual",
        }
    )
    away = raw[~raw["IS_HOME"]][["GAME_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={
            "TEAM_NAME": "away_team",
            "TEAM_ABBREVIATION": "away_abbr",
            "PTS": "away_pts_actual",
        }
    )
    games = home.merge(away, on="GAME_ID", how="inner")
    if games.empty:
        return pd.DataFrame()
    games["game_date"] = pd.to_datetime(games["GAME_DATE"], errors="coerce").dt.strftime("%m/%d/%Y")
    games["win_home_actual"] = (games["home_pts_actual"] > games["away_pts_actual"]).astype("Int64")
    games["series_round"] = games["GAME_ID"].astype(str).str[7].fillna("")
    games["series_slot"] = games["GAME_ID"].astype(str).str[8].fillna("")
    games["series_game_no"] = pd.to_numeric(games["GAME_ID"].astype(str).str[9], errors="coerce")
    out = games[
        [
            "GAME_ID",
            "game_date",
            "home_team",
            "away_team",
            "home_abbr",
            "away_abbr",
            "home_pts_actual",
            "away_pts_actual",
            "win_home_actual",
            "series_round",
            "series_slot",
            "series_game_no",
        ]
    ].drop_duplicates(subset=["GAME_ID"], keep="last")
    return out


def _build_playoff_predictions_view(pred_rows: int, pred_days_ahead: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    _bp_t0 = pd.Timestamp.now()
    upcoming, _pred_meta = get_predictions_cached(pred_rows, pred_days_ahead)
    st.session_state["playoffs_predictions_meta"] = _pred_meta
    if upcoming.empty:
        upcoming = pd.DataFrame(
            columns=[
                "matchup",
                "game_date",
                "home_team",
                "away_team",
                "home_win_probability",
                "confidence_pct",
                "confidence_level",
                "pred_home_pts",
                "pred_away_pts",
                "game_start_time_utc",
            ]
        )
    playoff_upcoming = upcoming.copy()
    if not playoff_upcoming.empty:
        playoff_upcoming["game_id"] = pd.NA
        playoff_upcoming["status"] = "upcoming"

    completed = pd.DataFrame()
    scored_path = "data/scored_predictions.csv"
    if os.path.exists(scored_path):
        try:
            completed = pd.read_csv(scored_path)
        except Exception:
            completed = pd.DataFrame()

    actual_games = _load_playoff_actual_games()
    if not completed.empty and not actual_games.empty:
        completed["home_team"] = completed["home_team"].astype(str).str.strip()
        completed["away_team"] = completed["away_team"].astype(str).str.strip()
        completed["home_abbr"] = completed["home_team"]
        completed["away_abbr"] = completed["away_team"]
        # Only attach GAME_ID + series metadata; scored_predictions already has
        # home_team / away_team / pts — overlapping names caused pandas _x/_y suffixes
        # and KeyError on column select (see debug log post_merge columns).
        completed = completed.merge(
            actual_games[["GAME_ID", "game_date", "home_abbr", "away_abbr", "series_round"]],
            on=["game_date", "home_abbr", "away_abbr"],
            how="inner",
        )
        completed = completed.rename(columns={"GAME_ID": "game_id"})
        completed["status"] = completed["result_status"].fillna("completed")
    else:
        completed = pd.DataFrame()

    playoff_completed = pd.DataFrame()
    if not completed.empty:
        needed_cols = [
            "matchup",
            "game_date",
            "home_team",
            "away_team",
            "home_win_probability",
            "confidence_bucket",
            "pred_home_pts",
            "pred_away_pts",
            "home_pts_actual",
            "away_pts_actual",
            "correct",
            "status",
            "game_id",
            "series_round",
        ]
        playoff_completed = completed[
            needed_cols
        ].copy()
        playoff_completed["confidence_pct"] = playoff_completed["home_win_probability"].apply(
            lambda p: max(float(p), 1.0 - float(p)) if pd.notna(p) else 0.5
        )
        playoff_completed["confidence_level"] = playoff_completed["confidence_bucket"].fillna("low")
        playoff_completed["result"] = playoff_completed["correct"].map(
            {1: "correct", 0: "incorrect"}
        ).fillna("pending")
    actual_only = pd.DataFrame()
    if not actual_games.empty:
        predicted_keys = set()
        if not playoff_completed.empty:
            predicted_keys = {
                (str(r["game_date"]), str(r["home_team"]), str(r["away_team"]))
                for _, r in playoff_completed[["game_date", "home_team", "away_team"]].dropna().iterrows()
            }
        actual_unmatched = actual_games.copy()
        actual_unmatched["key"] = list(
            zip(
                actual_unmatched["game_date"].astype(str),
                actual_unmatched["home_team"].astype(str),
                actual_unmatched["away_team"].astype(str),
            )
        )
        actual_unmatched = actual_unmatched[~actual_unmatched["key"].isin(predicted_keys)].copy()
        if not actual_unmatched.empty:
            actual_only = pd.DataFrame(
                {
                    "matchup": actual_unmatched["away_abbr"].astype(str)
                    + " @ "
                    + actual_unmatched["home_abbr"].astype(str),
                    "game_date": actual_unmatched["game_date"],
                    "home_team": actual_unmatched["home_team"],
                    "away_team": actual_unmatched["away_team"],
                    "home_win_probability": pd.NA,
                    "confidence_pct": pd.NA,
                    "confidence_level": pd.NA,
                    "pred_home_pts": pd.NA,
                    "pred_away_pts": pd.NA,
                    "home_pts_actual": actual_unmatched["home_pts_actual"],
                    "away_pts_actual": actual_unmatched["away_pts_actual"],
                    "result": "no_prediction",
                    "status": "resolved_no_prediction",
                    "game_id": actual_unmatched["GAME_ID"],
                    "series_round": actual_unmatched["series_round"],
                }
            )
    historical_resolved = pd.DataFrame()
    if not actual_games.empty:
        historical_resolved = pd.DataFrame(
            {
                "matchup": actual_games["away_abbr"].astype(str) + " @ " + actual_games["home_abbr"].astype(str),
                "game_date": actual_games["game_date"],
                "home_team": actual_games["home_team"],
                "away_team": actual_games["away_team"],
                "home_win_probability": pd.NA,
                "confidence_pct": pd.NA,
                "confidence_level": pd.NA,
                "pred_home_pts": pd.NA,
                "pred_away_pts": pd.NA,
                "home_pts_actual": actual_games["home_pts_actual"],
                "away_pts_actual": actual_games["away_pts_actual"],
                "result": "historical_result_only",
                "status": "resolved_historical",
                "game_id": actual_games["GAME_ID"],
                "series_round": actual_games["series_round"],
            }
        )
    all_cols = [
        "matchup",
        "game_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "confidence_pct",
        "confidence_level",
        "pred_home_pts",
        "pred_away_pts",
        "home_pts_actual",
        "away_pts_actual",
        "result",
        "status",
        "game_id",
        "series_round",
    ]
    playoff_upcoming = playoff_upcoming.reindex(columns=all_cols, fill_value=pd.NA)
    playoff_completed = playoff_completed.reindex(columns=all_cols, fill_value=pd.NA)
    actual_only = actual_only.reindex(columns=all_cols, fill_value=pd.NA)
    historical_resolved = historical_resolved.reindex(columns=all_cols, fill_value=pd.NA)
    all_games = pd.concat([playoff_completed, actual_only, historical_resolved, playoff_upcoming], ignore_index=True)
    if "game_id" in all_games.columns:
        with_id = all_games[all_games["game_id"].notna()].drop_duplicates(subset=["game_id"], keep="first")
        without_id = all_games[all_games["game_id"].isna()]
        all_games = pd.concat([with_id, without_id], ignore_index=True)
    if all_games.empty:
        return all_games, pd.DataFrame()
    all_games["game_date_dt"] = pd.to_datetime(all_games["game_date"], format="%m/%d/%Y", errors="coerce")
    all_games = all_games.sort_values(["game_date_dt", "matchup"], ascending=[True, True])
    return all_games, playoff_completed


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_live_playoff_games(days_back: int = 3, days_forward: int = 2) -> pd.DataFrame:
    _t0 = pd.Timestamp.now()
    frames: list[pd.DataFrame] = []
    start_day = date.today() - timedelta(days=days_back)
    end_day = date.today() + timedelta(days=days_forward)
    day = start_day
    while day <= end_day:
        date_label = day.strftime("%m/%d/%Y")
        try:
            one_day = fetch_games_for_date(date_label)
        except Exception:
            one_day = pd.DataFrame()
        if not one_day.empty:
            one_day = one_day[one_day["GAME_ID"].astype(str).str.startswith("004")].copy()
            if not one_day.empty:
                frames.append(one_day)
        day += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["GAME_ID"], keep="last")
    return out


def _build_playoff_series_state(live_games: pd.DataFrame, actual_games: pd.DataFrame) -> pd.DataFrame:
    if actual_games.empty and live_games.empty:
        return pd.DataFrame()
    actual_states = pd.DataFrame()
    if not actual_games.empty:
        actual_states = actual_games[
            [
                "GAME_ID",
                "home_abbr",
                "away_abbr",
                "home_pts_actual",
                "away_pts_actual",
                "series_round",
                "series_slot",
                "series_game_no",
            ]
        ].copy()
    live_states = pd.DataFrame()
    if not live_games.empty:
        live_states = live_games.merge(
            actual_games[
                [
                    "GAME_ID",
                    "home_abbr",
                    "away_abbr",
                    "home_pts_actual",
                    "away_pts_actual",
                    "series_round",
                    "series_slot",
                    "series_game_no",
                ]
            ]
            if not actual_games.empty
            else pd.DataFrame(
                columns=[
                    "GAME_ID",
                    "home_abbr",
                    "away_abbr",
                    "home_pts_actual",
                    "away_pts_actual",
                    "series_round",
                    "series_slot",
                    "series_game_no",
                ]
            ),
            on="GAME_ID",
            how="left",
        )
    states = pd.concat([actual_states, live_states], ignore_index=True)
    if states.empty:
        return pd.DataFrame()
    states = states.drop_duplicates(subset=["GAME_ID"], keep="first")
    states["series_id"] = states["GAME_ID"].astype(str).str[:-1]
    states["series_round"] = states["series_round"].fillna(states["GAME_ID"].astype(str).str[7])
    states["series_slot"] = states["series_slot"].fillna(states["GAME_ID"].astype(str).str[8])
    states["series_game_no"] = pd.to_numeric(
        states["series_game_no"].fillna(states["GAME_ID"].astype(str).str[9]),
        errors="coerce",
    )
    states = states.sort_values(["series_id", "series_game_no", "GAME_ID"])
    east_teams = {
        "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL",
        "NYK", "ORL", "PHI", "TOR", "WAS",
    }
    west_teams = {
        "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC",
        "PHX", "POR", "SAC", "SAS", "UTA",
    }
    rows = []
    for series_id, grp in states.groupby("series_id", dropna=False):
        if pd.isna(series_id) or str(series_id).strip() == "":
            continue
        wins: dict[str, int] = {}
        points: dict[str, int] = {}
        teams = set()
        for _, g in grp.iterrows():
            h = str(g.get("home_abbr", "")).strip()
            a = str(g.get("away_abbr", "")).strip()
            if h and h.lower() != "nan":
                teams.add(h)
            if a and a.lower() != "nan":
                teams.add(a)
            hp = g.get("home_pts_actual")
            ap = g.get("away_pts_actual")
            if pd.notna(hp) and h:
                points[h] = points.get(h, 0) + int(float(hp))
            if pd.notna(ap) and a:
                points[a] = points.get(a, 0) + int(float(ap))
            if pd.notna(hp) and pd.notna(ap):
                win_team = h if float(hp) > float(ap) else a
                wins[win_team] = wins.get(win_team, 0) + 1
        teams_sorted = sorted([t for t in teams if t])
        if not teams_sorted:
            continue
        t1 = teams_sorted[0]
        t2 = teams_sorted[1] if len(teams_sorted) > 1 else "TBD"
        w1, w2 = wins.get(t1, 0), wins.get(t2, 0)
        p1, p2 = points.get(t1, 0), points.get(t2, 0)
        lead = t1 if w1 > w2 else (t2 if w2 > w1 else "Tied")
        winner = t1 if w1 >= 4 else (t2 if w2 >= 4 else "")
        if t1 in east_teams and t2 in east_teams:
            conference = "East"
        elif t1 in west_teams and t2 in west_teams:
            conference = "West"
        elif str(grp["series_round"].dropna().astype(str).iloc[0]) == "4":
            conference = "NBA Finals"
        else:
            conference = "Mixed"
        rows.append(
            {
                "series_id": str(series_id),
                "series_round": str(grp["series_round"].dropna().astype(str).iloc[0])
                if grp["series_round"].notna().any()
                else "?",
                "series_slot": str(grp["series_slot"].dropna().astype(str).iloc[0])
                if grp["series_slot"].notna().any()
                else "?",
                "team_a": t1,
                "team_b": t2,
                "wins_a": int(w1),
                "wins_b": int(w2),
                "points_a": int(p1),
                "points_b": int(p2),
                "leader": lead,
                "winner": winner,
                "conference": conference,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["series_round_num"] = pd.to_numeric(out["series_round"], errors="coerce")
    out["series_slot_num"] = pd.to_numeric(out["series_slot"], errors="coerce")
    out = out.sort_values(["series_round_num", "conference", "series_slot_num", "series_slot"]).drop(
        columns=["series_round_num", "series_slot_num"]
    )
    return out


def _r1_slot_conference(series_slot: str) -> str:
    try:
        s = int(str(series_slot).strip())
    except (TypeError, ValueError):
        return "Mixed"
    if 0 <= s <= 3:
        return "East"
    if 4 <= s <= 7:
        return "West"
    return "Mixed"


def _index_playoff_series(series_state: pd.DataFrame) -> dict[tuple[str, str, str], dict]:
    """Key: (round, conference, series_slot) -> row dict."""
    out: dict[tuple[str, str, str], dict] = {}
    if series_state is None or series_state.empty:
        return out
    for _, row in series_state.iterrows():
        r = str(row.get("series_round", "")).strip()
        sl = str(row.get("series_slot", "")).strip()
        conf = str(row.get("conference", "")).strip()
        if r == "1" and conf not in ("East", "West"):
            conf = _r1_slot_conference(sl)
        out[(r, conf, sl)] = row.to_dict()
    return out


def _r1_maps_from_index(idx: dict[tuple[str, str, str], dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    east: dict[str, dict] = {}
    west: dict[str, dict] = {}
    for slot in ("0", "1", "2", "3"):
        k = ("1", "East", slot)
        if k in idx:
            east[slot] = idx[k]
    for slot in ("4", "5", "6", "7"):
        k = ("1", "West", slot)
        if k in idx:
            west[slot] = idx[k]
    return east, west


def _series_card_body(row: dict | None, *, fallback_title: str) -> tuple[str, str, str]:
    """Returns (title, body, footer)."""
    if not row:
        return fallback_title, "_Advancing team TBD · schedule TBD_", ""
    ta, tb = str(row.get("team_a", "")), str(row.get("team_b", ""))
    wa, wb = int(row.get("wins_a", 0) or 0), int(row.get("wins_b", 0) or 0)
    winner = str(row.get("winner", "") or "").strip()
    title = fallback_title
    status = ""
    lead = row.get("leader", "")
    if lead == ta:
        status = f"{ta} leads {wa}-{wb}"
    elif lead == tb:
        status = f"{tb} leads {wb}-{wa}"
    elif lead == "Tied":
        status = "Series tied"
    else:
        status = f"{wa}-{wb}"
    body = f"**{ta}** vs **{tb}**\n\nBest-of-7: {wa}-{wb}\n\n_{status}_"
    footer = f"**Advanced:** {winner}" if winner else "_Winner TBD_"
    return title, body, footer


def _winner_or_pair_label(
    side_rows: tuple[dict | None, dict | None], left_fallback: str, right_fallback: str
) -> str:
    """If both series decided, show winner vs winner; else placeholders per side."""

    def one(r: dict | None, fallback: str) -> str:
        if not r:
            return f"TBA · ({fallback})"
        w = str(r.get("winner", "") or "").strip()
        if w:
            return w
        ta, tb = str(r.get("team_a", "")), str(r.get("team_b", ""))
        wa, wb = int(r.get("wins_a", 0) or 0), int(r.get("wins_b", 0) or 0)
        return f"{ta} vs {tb} ({wa}-{wb})"

    a, b = side_rows
    if (
        a
        and str(a.get("winner", "") or "").strip()
        and b
        and str(b.get("winner", "") or "").strip()
    ):
        wa, wb = str(a["winner"]), str(b["winner"])
        return f"**{wa}** vs **{wb}**"
    left = one(a, left_fallback)
    right = one(b, right_fallback)
    return f"_{left}_\n\nvs\n\n_{right}_"


@st.cache_data(show_spinner=False)
def _team_abbr_to_id_map() -> dict[str, int]:
    _t0 = pd.Timestamp.now()
    out: dict[str, int] = {}
    try:
        for t in nba_static_teams.get_teams():
            abbr = str(t.get("abbreviation", "")).strip().upper()
            team_id = t.get("id")
            if abbr and team_id is not None:
                out[abbr] = int(team_id)
    except Exception:
        return {}
    return out


def _team_logo_url(team_abbr: str) -> str:
    team_id = _team_abbr_to_id_map().get(str(team_abbr or "").strip().upper())
    if not team_id:
        return ""
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"


def _team_state_class(row: dict | None, team_abbr: str) -> str:
    if not row:
        return "neutral"
    winner = str(row.get("winner", "") or "").strip()
    leader = str(row.get("leader", "") or "").strip()
    if winner:
        return "advanced" if winner == team_abbr else "eliminated"
    if leader == "Tied":
        return "tied"
    if leader == team_abbr:
        return "leading"
    if leader:
        return "trailing"
    return "neutral"


def _series_card_payload(row: dict | None, *, fallback_left: str, fallback_right: str) -> dict:
    if not row:
        return {
            "left": {"abbr": fallback_left, "wins": None, "points": None, "state_class": "neutral", "logo_url": ""},
            "right": {"abbr": fallback_right, "wins": None, "points": None, "state_class": "neutral", "logo_url": ""},
            "status": "Matchup TBD",
        }

    ta = str(row.get("team_a", "") or fallback_left).strip() or fallback_left
    tb = str(row.get("team_b", "") or fallback_right).strip() or fallback_right
    wa, wb = int(row.get("wins_a", 0) or 0), int(row.get("wins_b", 0) or 0)
    pa = int(row.get("points_a", 0) or 0)
    pb = int(row.get("points_b", 0) or 0)
    winner = str(row.get("winner", "") or "").strip()
    leader = str(row.get("leader", "") or "").strip()
    points_text = f"({pa}-{pb})" if (pa > 0 or pb > 0) else "(--)"

    if winner:
        status = f"{winner} advanced • {wa}-{wb} • {points_text}"
    elif leader == ta:
        status = f"{ta} leads {wa}-{wb} • {points_text}"
    elif leader == tb:
        status = f"{tb} leads {wb}-{wa} • {points_text}"
    elif leader == "Tied":
        status = f"Series tied {wa}-{wb} • {points_text}"
    else:
        status = f"Best-of-7 {wa}-{wb} • {points_text}"

    return {
        "left": {
            "abbr": ta,
            "wins": wa,
            "points": pa if (pa > 0 or pb > 0) else None,
            "state_class": _team_state_class(row, ta),
            "logo_url": _team_logo_url(ta),
        },
        "right": {
            "abbr": tb,
            "wins": wb,
            "points": pb if (pa > 0 or pb > 0) else None,
            "state_class": _team_state_class(row, tb),
            "logo_url": _team_logo_url(tb),
        },
        "status": status,
    }


def _render_bracket_team_row(side: dict) -> str:
    logo_url = str(side.get("logo_url", "") or "")
    logo_html = (
        f'<img class="team-logo" src="{html.escape(logo_url)}" alt="{html.escape(str(side.get("abbr", "TBD")))} logo" />'
        if logo_url
        else '<span class="team-logo team-logo-fallback"></span>'
    )
    wins = "--" if side.get("wins") is None else str(side.get("wins"))
    pts = "--" if side.get("points") is None else str(side.get("points"))
    return (
        f'<div class="team-row {html.escape(str(side.get("state_class", "neutral")))}">'
        f"{logo_html}"
        f'<div class="team-abbr">{html.escape(str(side.get("abbr", "TBD")))}</div>'
        f'<div class="team-chip">W {html.escape(wins)}</div>'
        f'<div class="team-chip">P {html.escape(pts)}</div>'
        "</div>"
    )


def _render_bracket_card_html(title: str, row: dict | None, fallback_left: str, fallback_right: str) -> str:
    payload = _series_card_payload(row, fallback_left=fallback_left, fallback_right=fallback_right)
    return (
        '<div class="bracket-card">'
        f'<div class="bracket-title">{html.escape(title)}</div>'
        f'{_render_bracket_team_row(payload["left"])}'
        f'{_render_bracket_team_row(payload["right"])}'
        f'<div class="bracket-status">{html.escape(payload["status"])}</div>'
        "</div>"
    )


def _bracket_ssr_theme_class() -> str:
    """First-paint bracket theme before client-side sync adjusts for instant toggles."""
    ctx = getattr(st, "context", None)
    th = getattr(ctx, "theme", None) if ctx else None

    tt = ""
    if isinstance(th, dict):
        tt = str(th.get("type") or "").strip().lower()
    elif th is not None:
        tv = getattr(th, "type", None)
        if tv is None and callable(getattr(th, "get", None)):
            try:
                tv = th.get("type")  # type: ignore[arg-type]
            except Exception:
                tv = None
        if callable(tv):
            tv = tv()
        tt = str(tv or "").strip().lower()

    if tt == "dark":
        return "theme-dark"
    if tt == "light":
        return "theme-light"

    try:
        base = str(st.get_option("theme.base") or "").strip().lower()
    except Exception:
        base = ""
    if base == "dark":
        return "theme-dark"
    return "theme-light"


def _connector_svg_html(kind: str, side: str) -> str:
    """SVG connector between bracket rounds with clear elbow geometry."""
    if kind == "r1_to_r2":
        paths = [
            "M0 12.5 H45",
            "M0 37.5 H45",
            "M45 12.5 V37.5",
            "M45 25 H100",
            "M0 62.5 H45",
            "M0 87.5 H45",
            "M45 62.5 V87.5",
            "M45 75 H100",
        ]
    elif kind == "r2_to_r3":
        paths = [
            "M0 25 H45",
            "M0 75 H45",
            "M45 25 V75",
            "M45 50 H100",
        ]
    else:  # r3_to_finals
        paths = ["M0 50 H100"]
    mirror_class = " mirror" if side == "right" else ""
    return (
        f'<div class="connector-col{mirror_class}">'
        '<svg class="connector-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">'
        + "".join(f'<path d="{p}"></path>' for p in paths)
        + "</svg></div>"
    )


def _render_playoff_bracket_scaffold(series_state: pd.DataFrame, round_label_fn) -> None:
    """Broadcast-style playoff bracket scaffold (West left, East right)."""
    if series_state is None or series_state.empty:
        st.caption("Bracket unavailable yet. Completed playoff results are needed to compute series records.")
        return

    idx = _index_playoff_series(series_state)
    east_r1, west_r1 = _r1_maps_from_index(idx)

    west_sf_0 = dict(idx[("2", "West", "0")]) if ("2", "West", "0") in idx else None
    west_sf_1 = dict(idx[("2", "West", "1")]) if ("2", "West", "1") in idx else None
    east_sf_0 = dict(idx[("2", "East", "0")]) if ("2", "East", "0") in idx else None
    east_sf_1 = dict(idx[("2", "East", "1")]) if ("2", "East", "1") in idx else None
    west_cf = dict(idx[("3", "West", "0")]) if ("3", "West", "0") in idx else None
    east_cf = dict(idx[("3", "East", "0")]) if ("3", "East", "0") in idx else None

    finals_row = None
    finals_keys = sorted([(k, v) for k, v in idx.items() if k[0] == "4"], key=lambda kv: kv[0][2])
    if finals_keys:
        finals_row = dict(finals_keys[0][1])

    west_r1_cards = "".join(
        _render_bracket_card_html(
            f"West · Series {slot}",
            west_r1.get(slot),
            f"West #{int(slot) + 1}",
            "TBD",
        )
        for slot in ("4", "5", "6", "7")
    )
    east_r1_cards = "".join(
        _render_bracket_card_html(
            f"East · Series {slot}",
            east_r1.get(slot),
            f"East #{int(slot) + 1}",
            "TBD",
        )
        for slot in ("0", "1", "2", "3")
    )
    west_sf_cards = "".join(
        (
            _render_bracket_card_html("West · Semifinal 1", west_sf_0, "Winner W4/5", "Winner W5/6")
            + _render_bracket_card_html("West · Semifinal 2", west_sf_1, "Winner W6/7", "Winner W7/8")
        )
    )
    east_sf_cards = "".join(
        (
            _render_bracket_card_html("East · Semifinal 1", east_sf_0, "Winner E0/1", "Winner E2/3")
            + _render_bracket_card_html("East · Semifinal 2", east_sf_1, "Winner E2/3", "Winner E4/5")
        )
    )
    west_cf_card = _render_bracket_card_html("West · Conference Final", west_cf, "West finalist A", "West finalist B")
    east_cf_card = _render_bracket_card_html("East · Conference Final", east_cf, "East finalist A", "East finalist B")
    finals_card = _render_bracket_card_html("NBA Finals", finals_row, "East Champion", "West Champion")
    west_r1_r2_connector = _connector_svg_html("r1_to_r2", "left")
    west_r2_r3_connector = _connector_svg_html("r2_to_r3", "left")
    west_r3_f_connector = _connector_svg_html("r3_to_finals", "left")
    east_f_r3_connector = _connector_svg_html("r3_to_finals", "right")
    east_r3_r2_connector = _connector_svg_html("r2_to_r3", "right")
    east_r2_r1_connector = _connector_svg_html("r1_to_r2", "right")

    theme_class = _bracket_ssr_theme_class()

    st.markdown(
        f"""
<style>
.playoff-bracket-wrap {{
    position: relative;
    overflow: hidden;
    isolation: isolate;
    /* Prefer Streamlit theme tokens (track viewer light/dark). Fallbacks mirror prior light styling. */
    --bracket-text: var(--st-text-color, #0f172a);
    --bracket-muted: color-mix(in srgb, var(--st-text-color, #334155) 72%, transparent);
    --bracket-title: color-mix(in srgb, var(--st-primary-color, #1d4ed8) 92%, var(--st-text-color, #0f172a) 8%);
    --bracket-border: var(--st-border-color, #cbd5e1);
    --bracket-chip: color-mix(in srgb, var(--st-secondary-background-color, #e2e8f0) 82%, transparent);
    --bracket-card-bg: color-mix(in srgb, var(--st-secondary-background-color, #ffffff) 88%, transparent);
    --line-color: color-mix(in srgb, var(--st-border-color, #64748b) 70%, transparent);
    --chip-bg: var(--bracket-chip);
    --leading: var(--st-green-color, #16a34a);
    --trailing: var(--st-red-color, #ef4444);
    --tied: var(--st-orange-color, #f59e0b);
    --advanced: var(--st-green-color, #15803d);
    --eliminated: var(--st-red-color, #b91c1c);
    --neutral: color-mix(in srgb, var(--st-border-color, #64748b) 75%, transparent);
    --bracket-bg: linear-gradient(
        180deg,
        color-mix(in srgb, var(--st-secondary-background-color, #f1f5f9) 94%, transparent) 0%,
        color-mix(in srgb, var(--st-background-color, #ffffff) 86%, var(--st-secondary-background-color, #f1f5f9)) 50%,
        color-mix(in srgb, var(--st-secondary-background-color, #f1f5f9) 94%, transparent) 100%
    );
    --bracket-card-shadow: color-mix(in srgb, var(--st-text-color, #000000) 12%, transparent);
    border-radius: 16px;
    padding: 18px;
    margin: 6px 0 12px 0;
    border: 1px solid var(--bracket-border);
    background: var(--bracket-bg);
    color: var(--bracket-text);
}}
.playoff-bracket-wrap.theme-dark {{
    --bracket-bg: linear-gradient(180deg, #0b1220 0%, #111827 50%, #0b1220 100%);
    --bracket-border: #334155;
    --bracket-text: #f8fafc;
    --bracket-muted: #cbd5e1;
    --bracket-title: #93c5fd;
    --bracket-card-bg: rgba(2, 6, 23, 0.72);
    --line-color: #64748b;
    --chip-bg: #1e293b;
    --leading: #16a34a;
    --trailing: #ef4444;
    --tied: #f59e0b;
    --advanced: #22c55e;
    --eliminated: #dc2626;
    --neutral: #64748b;
    --bracket-card-shadow: rgba(2, 6, 23, 0.5);
}}
.playoff-bracket-wrap.theme-light {{
    --bracket-bg: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
    --bracket-border: #cbd5e1;
    --bracket-text: #0f172a;
    --bracket-muted: #334155;
    --bracket-title: #1d4ed8;
    --bracket-card-bg: rgba(255, 255, 255, 0.92);
    --line-color: #64748b;
    --chip-bg: #e2e8f0;
    --leading: #15803d;
    --trailing: #b91c1c;
    --tied: #b45309;
    --advanced: #15803d;
    --eliminated: #b91c1c;
    --neutral: #64748b;
    --bracket-card-shadow: rgba(15, 23, 42, 0.08);
}}
.bracket-watermark {{
    position: absolute;
    inset: 0;
    z-index: 0;
    display: grid;
    place-items: center;
    pointer-events: none;
}}
.bracket-watermark-placeholder {{
    max-width: min(72%, 420px);
    aspect-ratio: 1;
    width: 100%;
    border-radius: 16px;
    border: 2px dashed color-mix(in srgb, var(--st-border-color, #94a3b8) 55%, transparent);
    background: color-mix(in srgb, var(--st-secondary-background-color, #f8fafc) 35%, transparent);
    display: grid;
    place-items: center;
    font-size: clamp(0.85rem, 2.4vw, 1.1rem);
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: color-mix(in srgb, var(--st-text-color, #0f172a) 22%, transparent);
    opacity: 0.55;
}}
.playoff-bracket-head {{
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 12px;
    align-items: center;
    margin-bottom: 14px;
}}
.playoff-bracket-head .west,
.playoff-bracket-head .east {{
    font-size: 1rem;
    font-weight: 800;
    color: var(--bracket-title);
    letter-spacing: 0.12em;
}}
.playoff-bracket-head .west {{ text-align: left; }}
.playoff-bracket-head .east {{ text-align: right; }}
.playoff-bracket-head .title {{
    font-size: 1.18rem;
    font-weight: 800;
    color: var(--bracket-text);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.playoff-bracket {{
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: 1.5fr 0.32fr 1.18fr 0.28fr 0.94fr 0.24fr 1.2fr 0.24fr 0.94fr 0.28fr 1.18fr 0.32fr 1.5fr;
    gap: 12px;
    align-items: stretch;
}}
.round-col {{
    display: grid;
    gap: 10px;
}}
.round-col.r1 {{ align-content: space-between; }}
.round-col.r2 {{ align-content: space-around; }}
.round-col.r3 {{ align-content: center; }}
.center-col {{ align-content: center; }}
.connector-col {{
    min-height: 100%;
    align-self: stretch;
}}
.connector-col.mirror {{
    transform: scaleX(-1);
}}
.connector-svg {{
    width: 100%;
    height: 100%;
    display: block;
}}
.connector-svg path {{
    stroke: var(--line-color);
    stroke-width: 3;
    fill: none;
    stroke-linecap: round;
}}
.bracket-card {{
    border: 1px solid var(--bracket-border);
    border-radius: 12px;
    background: var(--bracket-card-bg);
    padding: 10px 10px;
    min-height: 112px;
    box-shadow: 0 1px 0 var(--bracket-card-shadow);
}}
.bracket-title {{
    font-size: 0.75rem;
    color: var(--bracket-title);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}}
.team-row {{
    display: grid;
    grid-template-columns: 18px 1fr auto auto;
    gap: 7px;
    align-items: center;
    margin-bottom: 4px;
    border-left: 3px solid var(--neutral);
    padding-left: 6px;
}}
.team-row.leading {{ border-left-color: var(--leading); }}
.team-row.trailing {{ border-left-color: var(--trailing); }}
.team-row.tied {{ border-left-color: var(--tied); }}
.team-row.advanced {{ border-left-color: var(--advanced); }}
.team-row.eliminated {{ border-left-color: var(--eliminated); opacity: 0.84; }}
.team-logo {{
    width: 18px;
    height: 18px;
    object-fit: contain;
}}
.team-logo-fallback {{
    border-radius: 999px;
    background: var(--chip-bg);
    display: inline-block;
}}
.team-abbr {{
    font-size: 0.92rem;
    line-height: 1.1;
    font-weight: 700;
}}
.team-chip {{
    font-size: 0.66rem;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 999px;
    background: var(--chip-bg);
    color: var(--bracket-muted);
}}
.bracket-status {{
    color: var(--bracket-muted);
    font-size: 0.74rem;
    margin-top: 6px;
}}
@media (max-width: 1150px) {{
    .playoff-bracket {{
        display: flex;
        flex-direction: column;
    }}
    .connector-col {{
        display: none;
    }}
}}
</style>
<div id="playoff-bracket-root" class="playoff-bracket-wrap {theme_class}">
  <div class="bracket-watermark" aria-hidden="true">
    <div class="bracket-watermark-placeholder">Logo / watermark</div>
  </div>
  <div class="playoff-bracket-head">
    <div class="west">WEST</div>
    <div class="title">{html.escape(round_label_fn("4"))} Path</div>
    <div class="east">EAST</div>
  </div>
  <div class="playoff-bracket">
    <div class="round-col r1">{west_r1_cards}</div>
    {west_r1_r2_connector}
    <div class="round-col r2">{west_sf_cards}</div>
    {west_r2_r3_connector}
    <div class="round-col r3">{west_cf_card}</div>
    {west_r3_f_connector}
    <div class="round-col center-col">{finals_card}</div>
    {east_f_r3_connector}
    <div class="round-col r3">{east_cf_card}</div>
    {east_r3_r2_connector}
    <div class="round-col r2">{east_sf_cards}</div>
    {east_r2_r1_connector}
    <div class="round-col r1">{east_r1_cards}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    components.html(
        r"""
<script>
(function () {
  const parentWin = window.parent;
  const doc = parentWin.document;
  if (!doc || parentWin.__nbaBracketThemeSyncInstalled) return;
  parentWin.__nbaBracketThemeSyncInstalled = true;

  const parseRgba = (val) => {
    val = (val || "").trim();
    if (!val || val === "transparent") return null;
    if (val.startsWith("#")) {
      const h = val.slice(1);
      const hex =
        h.length === 3 ? h.split("").map((c) => c + c).join("") : h.slice(0, 6);
      const n = parseInt(hex, 16);
      if (Number.isNaN(n)) return null;
      return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255, a: 1 };
    }
    const m = val.match(
      /rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+)\s*)?\)/i,
    );
    if (m) {
      const a = m[4] !== undefined ? +m[4] : 1;
      return { r: +m[1], g: +m[2], b: +m[3], a };
    }
    return null;
  };

  const isDarkBg = (rgb) => {
    if (!rgb) return false;
    const a = rgb.a !== undefined ? rgb.a : 1;
    if (a < 0.08) return false;
    const lum = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
    return lum < 0.42;
  };

  const cssVar = (el, name) => {
    if (!el) return "";
    return parentWin.getComputedStyle(el).getPropertyValue(name).trim();
  };

  /** Theme tokens / effective paints often live on Streamlit shells, not <html>. */
  const probeThemeAnchors = () => {
    const q = (sel) => doc.querySelector(sel);
    const pairs = [
      ['[data-testid="stApp"]', q('[data-testid="stApp"]')],
      [
        '[data-testid="stAppViewContainer"]',
        q('[data-testid="stAppViewContainer"]'),
      ],
      ['section[data-testid="stMain"]', q('section[data-testid="stMain"]')],
      ["blockContainer", doc.querySelector("div.block-container")],
      ["html", doc.documentElement],
      ["body", doc.body],
    ];
    let chosenVar = "";
    let chosenComputed = "";

    for (const [, el] of pairs) {
      if (!el) continue;
      const gst = cssVar(el, "--st-background-color");
      const leg = cssVar(el, "--background-color");
      const comp = parentWin.getComputedStyle(el).backgroundColor || "";
      if (!chosenVar && gst) chosenVar = gst;
      else if (!chosenVar && leg) chosenVar = leg;
      const cr = parseRgba(comp);
      if (!chosenComputed && cr && (cr.a === undefined || cr.a >= 0.12))
        chosenComputed = comp.trim();
    }
    return {
      chosenVar,
      chosenComputed,
    };
  };

  function syncBracketThemeFromParentDom() {
    const root = doc.getElementById("playoff-bracket-root");
    if (!root) return;

    const { chosenVar, chosenComputed } = probeThemeAnchors();
    const bgStr = chosenVar || chosenComputed;
    let dark = isDarkBg(parseRgba(bgStr));

    const csHtml = parentWin.getComputedStyle(doc.documentElement);
    const scheme = (csHtml.colorScheme || "").toLowerCase();
    if (scheme === "dark") dark = true;
    if (scheme === "light") dark = false;

    root.classList.toggle("theme-dark", dark);
    root.classList.toggle("theme-light", !dark);
  }

  syncBracketThemeFromParentDom();

  const obs = new MutationObserver(() => syncBracketThemeFromParentDom());
  obs.observe(doc.documentElement, {
    attributes: true,
    attributeFilter: ["class", "style", "data-theme"],
  });
  if (doc.body) {
    obs.observe(doc.body, {
      attributes: true,
      attributeFilter: ["class", "style", "data-theme"],
    });
  }

  parentWin.setInterval(syncBracketThemeFromParentDom, 350);
})();
</script>
""",
        height=0,
        scrolling=False,
    )


def run_and_capture(func) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        func()
    return buf.getvalue()


class _TemporaryEnv:
    def __init__(self, overrides: dict[str, str]):
        self.overrides = {k: str(v) for k, v in overrides.items()}
        self._prev: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            self._prev[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, old in self._prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _latest_timestamp_from_csv(path: str, column: str) -> str:
    if not os.path.exists(path):
        return "not available"
    try:
        df = pd.read_csv(path)
        if column not in df.columns or df.empty:
            return "not available"
        ts = pd.to_datetime(df[column], errors="coerce").dropna()
        if ts.empty:
            return "not available"
        return ts.max().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "not available"


def _artifact_updated_at(path: str) -> str:
    if not os.path.exists(path):
        return "not available"
    ts = pd.Timestamp(os.path.getmtime(path), unit="s")
    return ts.strftime("%Y-%m-%d %H:%M")


def _render_feedback_calibration_panel() -> None:
    """Ops-oriented feedback calibrator status (lives under Model & data, not Predictions)."""
    st.markdown("##### Feedback-aware calibration")
    feedback_status = "OFF"
    feedback_rows = 0
    dataset_rows = 0
    min_rows = 30
    feedback_reason = ""
    if os.path.exists("artifacts/model_metrics.json"):
        try:
            with open("artifacts/model_metrics.json", "r", encoding="utf-8") as f:
                mm_fb = json.load(f)
            feedback_method = str(mm_fb.get("feedback_calibrator_method", "none")).lower()
            feedback_rows = int(mm_fb.get("feedback_calibrator_rows", 0) or 0)
            dataset_rows = int(mm_fb.get("feedback_dataset_rows", 0) or 0)
            min_rows = int(mm_fb.get("feedback_min_rows", 30) or 30)
            feedback_reason = str(mm_fb.get("feedback_calibrator_reason", "") or "").strip()
            if feedback_method != "none" and feedback_rows > 0:
                feedback_status = "ON"
        except Exception:
            pass
    hint = ""
    if feedback_status == "OFF":
        if dataset_rows == 0:
            hint = (
                " No resolved prediction history yet — run **Score Logged Predictions** after games finish, "
                "then retrain (see **Training** tab)."
            )
        elif dataset_rows < min_rows:
            hint = (
                f" Need at least **{min_rows}** resolved rows in `prediction_feedback_training.csv` "
                f"(have **{dataset_rows}**). Score more games, then retrain."
            )
        else:
            reason_text = feedback_reason if feedback_reason else "guardrails/validation checks"
            hint = (
                f" Row threshold met (**{dataset_rows}/{min_rows}**), but feedback calibrator is still OFF due to "
                f"**{reason_text}**. This is expected when safety guardrails keep historical calibration in control."
            )
    st.caption(
        f"Status: **{feedback_status}** | rows used in calibrator: **{feedback_rows}** | "
        f"resolved rows in dataset: **{dataset_rows}**"
        + hint
    )


def _historical_games_health() -> dict:
    path = "data/historical_games.csv"
    out: dict = {"exists": False, "rows": 0, "date_min": None, "date_max": None, "nan_rate": None}
    if not os.path.exists(path):
        return out
    out["exists"] = True
    try:
        df = pd.read_csv(path, parse_dates=["GAME_DATE"], low_memory=True)
        out["rows"] = len(df)
        if not df.empty and "GAME_DATE" in df.columns:
            ts = df["GAME_DATE"].dropna()
            if not ts.empty:
                out["date_min"] = ts.min()
                out["date_max"] = ts.max()
        if len(df):
            out["nan_rate"] = float(df.isna().mean().mean())
    except Exception:
        out["rows"] = 0
        out["exists"] = os.path.exists(path)
    return out


def _history_max_game_date_label() -> str:
    hh = _historical_games_health()
    if not hh["exists"] or hh["date_max"] is None:
        return "not available"
    return hh["date_max"].strftime("%Y-%m-%d")


def _pending_predictions_count() -> int:
    path = "data/scored_predictions.csv"
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path, usecols=["result_status"])
        s = df["result_status"].astype(str).str.strip().str.lower()
        return int((s == "pending").sum())
    except Exception:
        return 0


def _run_script_subprocess(rel_script: str, argv_extra: list[str]) -> tuple[int, str]:
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / rel_script)] + argv_extra
    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined


def _run_script_subprocess_with_env(
    rel_script: str, argv_extra: list[str], env_overrides: dict[str, str]
) -> tuple[int, str]:
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / rel_script)] + argv_extra
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined


st.set_page_config(page_title="NBA Win Predictor", layout="wide")

if "display_timezone" not in st.session_state:
    st.session_state["display_timezone"] = DEFAULT_DISPLAY_TZ

with st.sidebar:
    st.subheader("Tip-off display")
    _tz_idx = (
        COMMON_DISPLAY_TIMEZONES.index(st.session_state["display_timezone"])
        if st.session_state["display_timezone"] in COMMON_DISPLAY_TIMEZONES
        else 0
    )
    _picked = st.selectbox(
        "Timezone (IANA)",
        COMMON_DISPLAY_TIMEZONES,
        index=_tz_idx,
    )
    st.session_state["display_timezone"] = _picked

_APP_ROOT = Path(__file__).resolve().parent
# Two-pass startup: first rerun returns quickly so the browser paints layout before
# importing sklearn-backed scoring (prediction_sync pulls train_model).
if "prediction_sync_ready" not in st.session_state:
    st.session_state["prediction_sync_ready"] = False
if not st.session_state["prediction_sync_ready"]:
    if not st.session_state.get("_prediction_sync_second_pass"):
        st.session_state["_prediction_sync_second_pass"] = True
        st.session_state["startup_prediction_sync"] = {
            "skipped": True,
            "reason": "deferred first paint",
        }
        st.rerun()
    else:
        _sync_t0 = pd.Timestamp.now()
        try:
            import prediction_sync

            os.chdir(_APP_ROOT)
            st.session_state["startup_prediction_sync"] = prediction_sync.sync_completed_predictions(
                silent=True
            )
        except Exception as exc:
            st.session_state["startup_prediction_sync"] = {"error": str(exc)}
        finally:
            st.session_state["prediction_sync_ready"] = True

st.title("NBA Win Predictor")
_sync = st.session_state.get("startup_prediction_sync", {})
_caption = "Prediction, value betting, and model performance in one workspace."
if isinstance(_sync, dict) and _sync.get("error"):
    _caption += f" Startup sync: {_sync['error']}"
elif isinstance(_sync, dict) and _sync.get("skipped"):
    reason = str(_sync.get("reason", "") or "")
    if "deferred first paint" in reason:
        _caption += " Startup sync: loading on next pass…"
    else:
        _caption += f" Startup sync skipped ({reason})."
elif isinstance(_sync, dict) and _sync.get("scored"):
    _fb = "feedback calibrator refreshed" if _sync.get("feedback_patched") else "feedback calibrator unchanged"
    _caption += (
        f" Startup: {_sync.get('resolved', 0)} resolved, {_sync.get('pending', 0)} pending; "
        f"{_sync.get('feedback_rows', 0)} feedback rows; {_fb}."
    )
st.caption(_caption)

status1, status2, status3, status4, status5 = st.columns(5)
with status1:
    st.metric("Last Model Train", _artifact_updated_at("artifacts/model.joblib"))
with status2:
    st.metric("Latest Prediction Run", _latest_timestamp_from_csv("data/predictions_log.csv", "run_timestamp"))
with status3:
    st.metric(
        "Latest Value Run",
        _latest_timestamp_from_csv("data/value_recommendations.csv", "run_timestamp"),
    )
with status4:
    st.metric("History max GAME_DATE", _history_max_game_date_label())
with status5:
    st.metric("Pending predictions (scored log)", str(_pending_predictions_count()))

tab_games, tab_pipeline = st.tabs(["Games & markets", "Model & data"])

with tab_games:
    with st.expander("Quick start", expanded=False):
        st.markdown(
            "1. **Predictions** — run upcoming slate and log to `data/predictions_log.csv`.\n\n"
            "2. **Value Plays** — compare model probs to odds (needs Odds API key in `.env`).\n\n"
            "3. **Model & data** tab — evaluate, refresh history (`fetch_data`), train/retrain models."
        )

    nested_pred, nested_value, nested_playoffs = st.tabs(["Predictions", "Value Plays", "Playoffs"])

    with nested_pred:
        st.subheader("Predictions")
        st.caption(
            "**Confidence meter** is lean strength (distance from a 50/50 coin flip), not a "
            "statistical confidence interval. Probabilities use Platt (or isotonic) calibration on history "
            "and optional feedback tuning — see **Model & data → Performance**."
        )
        col1, col2 = st.columns(2)
        with col1:
            count = st.number_input("Number of games", min_value=1, max_value=20, value=5, step=1)
        with col2:
            days_ahead = st.number_input("Days ahead to scan", min_value=1, max_value=60, value=14, step=1)
        force_refresh_predictions = st.checkbox(
            "Force refresh (ignore session cache)",
            value=False,
            key="pred_force_refresh",
            help="Bypass session cache and recompute predictions for current inputs.",
        )

        if st.button("Predict Upcoming Games", type="primary", key="btn_predict_upcoming"):
            try:
                with st.spinner("Running predictions..."):
                    preds, pred_meta = get_predictions_cached(
                        int(count),
                        int(days_ahead),
                        force_refresh=bool(force_refresh_predictions),
                    )
                if preds.empty:
                    st.warning("No games found or no predictable rows.")
                else:
                    st.success("Predictions ready. Also appended to data/predictions_log.csv")
                    show_df = preds.copy()
                    _disp_tz = ZoneInfo(st.session_state["display_timezone"])
                    show_df["start_local"] = show_df.apply(
                        lambda r: format_start_display(
                            r.get("game_start_time_utc"),
                            _disp_tz,
                            legacy_24h=r.get("game_start_time_24h"),
                        ),
                        axis=1,
                    )
                    if "game_start_time_24h" in show_df.columns:
                        show_df = show_df.drop(columns=["game_start_time_24h"])
                    # Hide raw UTC from the UI; "Start (local)" is the only tip-off column users should see.
                    if "game_start_time_utc" in show_df.columns:
                        show_df = show_df.drop(columns=["game_start_time_utc"])
                    show_df["home_win_probability"] = (show_df["home_win_probability"] * 100).round(1)
                    show_df["away_win_probability"] = (100 - show_df["home_win_probability"]).round(1)
                    show_df["confidence_pct"] = (show_df["confidence_pct"] * 100).round(1)
                    show_df["confidence_level"] = show_df["confidence_level"].apply(_confidence_badge)

                    k1, k2, k3 = st.columns(3)
                    with k1:
                        st.metric("Predictions", f"{len(show_df)}")
                    with k2:
                        st.metric("Avg Confidence", f"{show_df['confidence_pct'].mean():.1f}%")
                    with k3:
                        st.metric("Avg Home Win %", f"{show_df['home_win_probability'].mean():.1f}%")

                    top_pick = show_df.sort_values("confidence_pct", ascending=False).iloc[0]
                    st.subheader("Highest Confidence Pick")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Matchup", str(top_pick["matchup"]))
                    with c2:
                        st.metric("Confidence", f"{top_pick['confidence_pct']:.1f}%")
                    with c3:
                        st.metric("Home Win Prob", f"{top_pick['home_win_probability']:.1f}%")

                    if pd.notna(top_pick.get("rationale")) and str(top_pick.get("rationale", "")).strip():
                        bullets = [s.strip() for s in str(top_pick["rationale"]).split(" / ") if s.strip()]
                        if bullets:
                            with st.expander("Why this pick (rolling form)", expanded=False):
                                st.markdown("\n".join(f"- {b}" for b in bullets))

                    _pred_table_order = [
                        "matchup",
                        "game_date",
                        "start_local",
                        "home_team",
                        "away_team",
                        "home_win_probability",
                        "away_win_probability",
                        "confidence_pct",
                        "confidence_level",
                        "rationale",
                        "pred_home_pts",
                        "pred_away_pts",
                    ]
                    _ordered = [c for c in _pred_table_order if c in show_df.columns]
                    _rest = [c for c in show_df.columns if c not in _ordered]
                    show_df = show_df[_ordered + _rest]

                    display_df = show_df.rename(
                        columns={
                            "matchup": "Matchup",
                            "game_date": "Game Date",
                            "start_local": "Start (local)",
                            "home_team": "Home Team",
                            "away_team": "Away Team",
                            "home_win_probability": "Home Win Probability (%)",
                            "away_win_probability": "Away Win Probability (%)",
                            "confidence_pct": "Confidence Meter (%)",
                            "confidence_level": "Confidence Level",
                            "rationale": "Why (rolling form)",
                            "pred_home_pts": "Projected Home Points",
                            "pred_away_pts": "Projected Away Points",
                        }
                    )
                    styled_df = _style_percentage_bars(
                        display_df,
                        ["Home Win Probability (%)", "Away Win Probability (%)", "Confidence Meter (%)"],
                    )
                    st.caption("Quick confidence snapshot")
                    conf_chart = display_df[["Matchup", "Confidence Meter (%)"]].copy()
                    conf_chart = conf_chart.sort_values("Confidence Meter (%)", ascending=False).set_index("Matchup")
                    st.bar_chart(conf_chart, height=240)

                    with st.expander("Detailed prediction table", expanded=True):
                        st.dataframe(styled_df, width="stretch")
            except Exception as exc:
                st.error(str(exc))

    with nested_value:
        st.subheader("Value Betting Analyzer")
        st.caption("Compare model probabilities vs market odds and find positive-EV plays.")

        vb1, vb2, vb3 = st.columns(3)
        with vb1:
            bankroll = st.number_input(
                "Bankroll",
                min_value=1.0,
                max_value=1_000_000.0,
                value=1000.0,
                step=100.0,
            )
        with vb2:
            kelly_multiplier = st.number_input(
                "Kelly multiplier",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
            )
        with vb3:
            max_stake_pct = st.number_input(
                "Max stake cap (%)",
                min_value=0.5,
                max_value=25.0,
                value=5.0,
                step=0.5,
            )

        if st.button("Find Value Plays", key="btn_find_value"):
            try:
                with st.spinner("Fetching odds and computing value..."):
                    recs, stats = build_value_plays(
                        bankroll=float(bankroll),
                        edge_threshold_pct=0.0,
                        kelly_multiplier=float(kelly_multiplier),
                        max_stake_pct=float(max_stake_pct),
                        diagnostics=True,
                    )
                if recs.empty:
                    st.warning("No qualifying value plays found.")
                    st.caption(
                        "Diagnostics: "
                        f"pred_rows={stats.get('pred_rows', 0)}, "
                        f"odds_raw_rows={stats.get('odds_raw_rows', 0)}, "
                        f"odds_best_rows={stats.get('odds_best_rows', 0)}, "
                        f"merged_rows={stats.get('merged_rows', 0)}, "
                        f"analysis_rows={stats.get('analysis_rows', 0)}, "
                        f"positive_ev_rows={stats.get('positive_ev_rows', 0)}, "
                        f"team_overlap_home={stats.get('team_overlap_home', 0)}, "
                        f"team_overlap_away={stats.get('team_overlap_away', 0)}"
                    )
                else:
                    display = recs.copy()
                    pct_cols = ["model_win_pct", "edge_pct", "kelly_stake_pct"]
                    for col in pct_cols:
                        display[col] = display[col].round(2)
                    display["ev_per_unit_stake"] = display["ev_per_unit_stake"].round(4)
                    display["suggested_stake_amount"] = display["suggested_stake_amount"].round(2)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Value Plays", f"{len(display)}")
                    with m2:
                        st.metric("Avg EV / Unit", f"{display['ev_per_unit_stake'].mean():.3f}")
                    with m3:
                        st.metric("Avg Stake %", f"{display['kelly_stake_pct'].mean():.2f}%")
                    with m4:
                        st.metric("Avg Edge %", f"{display['edge_pct'].mean():.2f}%")

                    st.success("Recommended plays generated and logged to data/value_recommendations.csv")
                    st.caption("Quick EV snapshot")
                    ev_chart = display[["matchup", "ev_per_unit_stake"]].copy()
                    ev_chart = ev_chart.sort_values("ev_per_unit_stake", ascending=False).set_index("matchup")
                    st.bar_chart(ev_chart, height=240)

                    f1, f2, f3 = st.columns(3)
                    with f1:
                        side_filter = st.selectbox("Filter side", ["all"] + sorted(display["side"].astype(str).unique().tolist()))
                    with f2:
                        book_filter = st.selectbox(
                            "Filter bookmaker",
                            ["all"] + sorted(display["best_bookmaker"].astype(str).unique().tolist()),
                        )
                    with f3:
                        date_filter = st.selectbox(
                            "Filter date",
                            ["all"] + sorted(display["game_date"].astype(str).unique().tolist()),
                        )

                    filtered = display.copy()
                    if side_filter != "all":
                        filtered = filtered[filtered["side"] == side_filter]
                    if book_filter != "all":
                        filtered = filtered[filtered["best_bookmaker"] == book_filter]
                    if date_filter != "all":
                        filtered = filtered[filtered["game_date"] == date_filter]

                    with st.expander("Detailed value plays table", expanded=True):
                        value_display = filtered[
                            [
                                "game_date",
                                "matchup",
                                "side",
                                "model_win_pct",
                                "fair_decimal_odds",
                                "market_decimal_odds",
                                "best_bookmaker",
                                "edge_pct",
                                "ev_per_unit_stake",
                                "kelly_stake_pct",
                                "suggested_stake_amount",
                            ]
                        ].rename(
                            columns={
                                "game_date": "Game Date",
                                "matchup": "Matchup",
                                "side": "Side",
                                "model_win_pct": "Model Win (%)",
                                "fair_decimal_odds": "Fair Decimal Odds",
                                "market_decimal_odds": "Market Decimal Odds",
                                "best_bookmaker": "Best Bookmaker",
                                "edge_pct": "Edge (%)",
                                "ev_per_unit_stake": "EV per Unit Stake",
                                "kelly_stake_pct": "Kelly Stake (%)",
                                "suggested_stake_amount": "Suggested Stake",
                            }
                        )
                        st.dataframe(value_display, width="stretch")
            except Exception as exc:
                st.error(str(exc))

    with nested_playoffs:
        st.subheader("Playoffs Predictions")
        st.caption(
            "Focused view for ongoing playoffs: upcoming picks, completed results, and live series bracket state."
        )
        round_names = {
            "0": "Play-In",
            "1": "First Round",
            "2": "Conference Semifinals",
            "3": "Conference Finals",
            "4": "NBA Finals",
        }

        def _round_label(v: object) -> str:
            s = str(v).strip()
            if not s or s.lower() == "nan":
                return "Unknown"
            return round_names.get(s, f"Round {s}")

        p1, p2 = st.columns(2)
        with p1:
            playoff_count = st.number_input(
                "Playoff games to predict",
                min_value=4,
                max_value=50,
                value=20,
                step=1,
            )
        with p2:
            playoff_days = st.number_input(
                "Days ahead to scan (playoffs)",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
            )
        if st.button("Refresh Playoffs View", type="primary", key="btn_playoffs_refresh"):
            st.session_state["playoffs_refresh"] = True
            _load_playoff_actual_games.clear()
            _fetch_live_playoff_games.clear()
            clear_prediction_session_cache()
            st.session_state.pop("playoffs_dashboard_cache", None)
        try:
            need_refresh = (
                st.session_state.get("playoffs_refresh", False)
                or "playoffs_dashboard_cache" not in st.session_state
                or st.session_state.get("playoffs_dashboard_params")
                != {"count": int(playoff_count), "days": int(playoff_days)}
            )
            if need_refresh:
                _refresh_t0 = pd.Timestamp.now()
                with st.spinner("Building playoffs dashboard..."):
                    _t_build = pd.Timestamp.now()
                    playoff_all, playoff_completed = _build_playoff_predictions_view(
                        int(playoff_count), int(playoff_days)
                    )
                    build_ms = int((pd.Timestamp.now() - _t_build).total_seconds() * 1000)
                    _t_actual = pd.Timestamp.now()
                    actual_games = _load_playoff_actual_games()
                    actual_ms = int((pd.Timestamp.now() - _t_actual).total_seconds() * 1000)
                    _t_live = pd.Timestamp.now()
                    live_playoff_games = _fetch_live_playoff_games()
                    live_ms = int((pd.Timestamp.now() - _t_live).total_seconds() * 1000)
                    series_state = _build_playoff_series_state(live_playoff_games, actual_games)
                    st.session_state["playoffs_dashboard_cache"] = {
                        "playoff_all": playoff_all,
                        "playoff_completed": playoff_completed,
                        "series_state": series_state,
                        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.session_state["playoffs_dashboard_params"] = {
                        "count": int(playoff_count),
                        "days": int(playoff_days),
                    }
                st.session_state["playoffs_refresh"] = False
            cache = st.session_state.get("playoffs_dashboard_cache", {})
            playoff_all = cache.get("playoff_all", pd.DataFrame())
            playoff_completed = cache.get("playoff_completed", pd.DataFrame())
            series_state = cache.get("series_state", pd.DataFrame())
            if cache.get("updated_at"):
                st.caption(f"Last refreshed: {cache['updated_at']}")
            playoff_meta = st.session_state.get("playoffs_predictions_meta", {})
            if isinstance(playoff_meta, dict) and playoff_meta.get("source"):
                source_label = (
                    "session cache" if playoff_meta.get("source") == "cache" else "fresh compute"
                )
                st.caption(
                    "Predictions source for this view: "
                    f"**{source_label}** | "
                    f"cached at: `{playoff_meta.get('cached_at', 'n/a')}`"
                )

            if playoff_all.empty:
                st.info(
                    "No playoff predictions available yet. Run predictions and ensure playoff games are present in schedule/history."
                )
            else:
                f1, f2, f3 = st.columns(3)
                with f1:
                    status_filter = st.selectbox(
                        "Filter status",
                        [
                            "all",
                            "upcoming",
                            "resolved",
                            "pending",
                            "resolved_no_prediction",
                            "resolved_historical",
                        ],
                        index=0,
                    )
                with f2:
                    rounds = sorted(
                        [r for r in playoff_all["series_round"].dropna().astype(str).unique().tolist() if r.strip()]
                    )
                    round_filter = st.selectbox("Filter round", ["all"] + rounds, index=0)
                with f3:
                    team_search = st.text_input("Team search", value="").strip().lower()

                filtered = playoff_all.copy()
                if status_filter != "all":
                    filtered = filtered[filtered["status"].astype(str) == status_filter]
                if round_filter != "all":
                    filtered = filtered[filtered["series_round"].astype(str) == str(round_filter)]
                if team_search:
                    filtered = filtered[
                        filtered["home_team"].astype(str).str.lower().str.contains(re.escape(team_search), regex=True)
                        | filtered["away_team"].astype(str).str.lower().str.contains(re.escape(team_search), regex=True)
                        | filtered["matchup"].astype(str).str.lower().str.contains(re.escape(team_search), regex=True)
                    ]

                resolved = playoff_completed[playoff_completed["status"].astype(str) == "resolved"].copy()
                if not resolved.empty:
                    resolved["correct"] = pd.to_numeric(resolved["result"], errors="coerce")
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Playoff games shown", f"{len(filtered)}")
                with k2:
                    st.metric("Completed picks", f"{len(resolved)}")
                with k3:
                    acc = (
                        pd.to_numeric(playoff_completed["result"].map({"correct": 1, "incorrect": 0}), errors="coerce")
                        .dropna()
                        .mean()
                    )
                    st.metric("Playoff accuracy", f"{(acc * 100):.1f}%" if pd.notna(acc) else "—")
                with k4:
                    mean_conf = pd.to_numeric(playoff_completed["confidence_pct"], errors="coerce").dropna().mean()
                    st.metric("Avg confidence", f"{(mean_conf * 100):.1f}%" if pd.notna(mean_conf) else "—")

                cbc = playoff_completed[playoff_completed["status"].astype(str) == "resolved"].copy()
                if not cbc.empty:
                    cbc["bucket"] = cbc["confidence_level"].astype(str).str.lower()
                    cbc["correct_num"] = cbc["result"].map({"correct": 1, "incorrect": 0})
                    cbc = cbc.dropna(subset=["correct_num"])
                    if not cbc.empty:
                        bucket_acc = (
                            cbc.groupby("bucket", as_index=False)
                            .agg(predictions=("correct_num", "size"), accuracy=("correct_num", "mean"))
                            .sort_values("bucket")
                        )
                        bucket_acc["accuracy"] = (bucket_acc["accuracy"] * 100).round(1)
                        with st.expander("Confidence bucket correctness", expanded=False):
                            st.dataframe(
                                bucket_acc.rename(
                                    columns={
                                        "bucket": "Confidence Bucket",
                                        "predictions": "Predictions",
                                        "accuracy": "Accuracy (%)",
                                    }
                                ),
                                width="stretch",
                            )

                show = filtered.copy()
                show["home_win_probability"] = (pd.to_numeric(show["home_win_probability"], errors="coerce") * 100).round(1)
                show["away_win_probability"] = (100 - show["home_win_probability"]).round(1)
                show["confidence_pct"] = (pd.to_numeric(show["confidence_pct"], errors="coerce") * 100).round(1)
                show["confidence_level"] = show["confidence_level"].astype(str).map(_confidence_badge)
                show["series_round"] = show["series_round"].map(_round_label)
                show["result"] = show["result"].replace({"no_prediction": "no prediction logged"})
                show["result"] = show["result"].replace({"historical_result_only": "historical result only"})
                show["status"] = show["status"].replace(
                    {
                        "resolved_no_prediction": "resolved (no prediction)",
                        "resolved_historical": "resolved (historical)",
                    }
                )
                display_cols = [
                    "game_date",
                    "matchup",
                    "status",
                    "series_round",
                    "home_win_probability",
                    "away_win_probability",
                    "confidence_pct",
                    "confidence_level",
                    "pred_home_pts",
                    "pred_away_pts",
                    "home_pts_actual",
                    "away_pts_actual",
                    "result",
                ]
                show = show[[c for c in display_cols if c in show.columns]].rename(
                    columns={
                        "game_date": "Game Date",
                        "matchup": "Matchup",
                        "status": "Status",
                        "series_round": "Round",
                        "home_win_probability": "Home Win Probability (%)",
                        "away_win_probability": "Away Win Probability (%)",
                        "confidence_pct": "Confidence Meter (%)",
                        "confidence_level": "Confidence Level",
                        "pred_home_pts": "Projected Home Points",
                        "pred_away_pts": "Projected Away Points",
                        "home_pts_actual": "Home Points (Actual)",
                        "away_pts_actual": "Away Points (Actual)",
                        "result": "Prediction Result",
                    }
                )
                show_upcoming = show[show["Status"] == "upcoming"] if "Status" in show.columns else pd.DataFrame()
                show_completed = show[show["Status"] != "upcoming"] if "Status" in show.columns else show
                with st.expander("Upcoming playoff games", expanded=True):
                    if show_upcoming.empty:
                        st.caption("No upcoming playoff games in the current filter.")
                    else:
                        st.dataframe(
                            _style_percentage_bars(
                                show_upcoming,
                                [
                                    c
                                    for c in [
                                        "Home Win Probability (%)",
                                        "Away Win Probability (%)",
                                        "Confidence Meter (%)",
                                    ]
                                    if c in show_upcoming.columns
                                ],
                            ),
                            width="stretch",
                        )
                with st.expander("Completed / tracked playoff games", expanded=True):
                    if show_completed.empty:
                        st.caption("No completed playoff games in the current filter.")
                    else:
                        st.dataframe(
                            _style_percentage_bars(
                                show_completed,
                                [
                                    c
                                    for c in [
                                        "Home Win Probability (%)",
                                        "Away Win Probability (%)",
                                        "Confidence Meter (%)",
                                    ]
                                    if c in show_completed.columns
                                ],
                            ),
                            width="stretch",
                        )

            st.markdown("##### Live playoff bracket")
            _render_playoff_bracket_scaffold(series_state, _round_label)
        except Exception as exc:
            st.warning(
                "Playoff view partially unavailable. Ensure NBA API access is healthy, then refresh."
            )
            st.error(str(exc))

with tab_pipeline:
    root = Path(__file__).resolve().parent
    latest_pred_meta = st.session_state.get("latest_prediction_meta", {})
    if isinstance(latest_pred_meta, dict) and latest_pred_meta.get("source"):
        _src = "session cache" if latest_pred_meta.get("source") == "cache" else "fresh compute"
        st.caption(
            "Latest prediction compute source this session: "
            f"**{_src}** | "
            f"cached at: `{latest_pred_meta.get('cached_at', 'n/a')}`"
        )
    sub_perf, sub_hist, sub_train = st.tabs(["Performance", "Historical data", "Training"])

    with sub_perf:
        st.subheader("Performance")
        _render_feedback_calibration_panel()

        mm: dict | None = None
        if os.path.exists("artifacts/model_metrics.json"):
            try:
                with open("artifacts/model_metrics.json", "r", encoding="utf-8") as f:
                    mm = json.load(f)
            except Exception:
                mm = None

        if mm:
            sel = mm.get("champion_selection_rule")
            if sel:
                st.caption(str(sel))
            if bool(mm.get("feedback_calibrator_low_sample_warning", False)):
                st.caption(
                    "Feedback calibrator is active but low-sample; treat probability adjustments as provisional "
                    "until more resolved games accumulate."
                )
            st.caption(
                f"Champion model: {mm.get('champion_model', 'unknown')} | "
                f"WF log-loss leader: {mm.get('walk_forward_log_loss_leader', '?')} | "
                f"WF accuracy leader: {mm.get('walk_forward_accuracy_leader', '?')} | "
                f"Calibrator: {mm.get('calibrator_method', 'unknown')} | "
                f"Feedback calibrator: {mm.get('feedback_calibrator_method', 'none')}"
            )
            fb_mode = str(mm.get("feedback_mode", "off"))
            shadow_db = mm.get("feedback_shadow_delta_brier")
            shadow_dll = mm.get("feedback_shadow_delta_log_loss")
            if shadow_db is not None and shadow_dll is not None:
                st.caption(
                    f"Feedback mode: **{fb_mode}** | shadow delta brier={float(shadow_db):+.4f}, "
                    f"shadow delta log_loss={float(shadow_dll):+.4f}"
                )
                if float(shadow_db) > 0 or float(shadow_dll) > 0:
                    st.caption(
                        "Confidence may feel lower because feedback calibration currently worsens holdout "
                        "quality in shadow checks, so deployment stays on historical calibration."
                    )
            fb_reason = str(mm.get("feedback_calibrator_reason", "")).strip()
            if fb_reason and fb_reason != "validation_improved":
                st.caption(f"Feedback calibrator status: `{fb_reason}`")
            fb_rows = int(mm.get("feedback_dataset_rows", 0) or 0)
            fb_apply_floor = int(mm.get("feedback_apply_min_rows", 40) or 40)
            if fb_rows < fb_apply_floor:
                st.caption(
                    f"Feedback calibrator auto-enables at **{fb_apply_floor}** resolved rows; "
                    f"currently **{fb_rows}/{fb_apply_floor}**."
                )
            st.caption(
                "Feedback thresholds | "
                f"fit_min={int(mm.get('feedback_min_rows', 20) or 20)}, "
                f"apply_min={fb_apply_floor}, "
                f"warn_rows={int(mm.get('feedback_low_sample_warning_rows', 40) or 40)}"
            )
            if bool(mm.get("drift_warning", False)):
                st.warning(
                    "Post-train drift warning: "
                    + ",".join(mm.get("drift_warning_reasons", []))
                )
            dq = mm.get("data_quality", {})
            if isinstance(dq, dict):
                if bool(dq.get("quality_warning", False)):
                    st.warning(
                        "Data quality warning: "
                        + ",".join(dq.get("quality_warning_reasons", []))
                    )
                else:
                    st.caption(
                        "Data quality: safe | "
                        f"nan_rate={float(dq.get('nan_rate', 0.0)):.4f}, "
                        f"stale_days={int(dq.get('stale_days', 0))}, "
                        f"recent_games_30d={int(dq.get('recent_games_30d', 0))}"
                    )
            hold = mm.get("holdout_metrics", {})
            hp1, hp2, hp3 = st.columns(3)
            with hp1:
                st.metric("Holdout Accuracy", f"{float(hold.get('accuracy', 0.0)):.3f}")
            with hp2:
                st.metric("Holdout Log Loss", f"{float(hold.get('log_loss', 0.0)):.3f}")
            with hp3:
                st.metric("Holdout Brier", f"{float(hold.get('brier', 0.0)):.3f}")
            wf_summary = mm.get("walk_forward_summary")
            if wf_summary:
                with st.expander("Walk-forward summary (last train)", expanded=False):
                    st.dataframe(pd.DataFrame(wf_summary), width="stretch")
            with st.expander("Full model_metrics.json", expanded=False):
                st.json(mm)
        else:
            st.caption("Model metrics unavailable — train a model first (Training tab).")

        if os.path.exists("data/scoring_reliability_bins.csv"):
            st.caption(
                "Probability reliability bins (ECE by decile): `data/scoring_reliability_bins.csv` "
                "(updated when you run Score Logged Predictions)."
            )
            try:
                rel = pd.read_csv("data/scoring_reliability_bins.csv")
                if not rel.empty:
                    rel_show = rel[["bin_lo", "bin_hi", "n", "mean_predicted_p", "empirical_win_rate", "ece_total"]].copy()
                    rel_show["bin"] = rel_show["bin_lo"].round(1).astype(str) + "-" + rel_show["bin_hi"].round(1).astype(str)
                    rel_chart = rel_show[["bin", "mean_predicted_p", "empirical_win_rate"]].set_index("bin")
                    st.line_chart(rel_chart, height=220)
                    with st.expander("Reliability decile table", expanded=False):
                        st.dataframe(rel_show, width="stretch")
            except Exception:
                pass
        if os.path.exists("data/calibration_trend.csv"):
            try:
                trend = pd.read_csv("data/calibration_trend.csv")
                if not trend.empty:
                    st.caption("Calibration trend (weekly)")
                    trend_chart = trend[["season_week", "weekly_log_loss", "weekly_brier", "weekly_ece"]].set_index("season_week")
                    st.line_chart(trend_chart, height=220)
            except Exception:
                pass
        if os.path.exists("data/value_backtest_summary.json"):
            try:
                with open("data/value_backtest_summary.json", "r", encoding="utf-8") as f:
                    vb = json.load(f)
                st.caption("Value backtest snapshot")
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Resolved value bets", f"{int(vb.get('bets_resolved', 0))}")
                with b2:
                    st.metric("Hit rate", f"{float(vb.get('hit_rate', 0.0)) * 100:.1f}%")
                with b3:
                    st.metric("Realized ROI / bet", f"{float(vb.get('realized_roi_per_bet', 0.0)):.3f}")
                with b4:
                    st.metric("Max drawdown", f"{float(vb.get('max_drawdown', 0.0)) * 100:.1f}%")
                if os.path.exists("data/value_backtest_curve.csv"):
                    curve = pd.read_csv("data/value_backtest_curve.csv")
                    if not curve.empty:
                        st.line_chart(curve.set_index("step")[["bankroll"]], height=180)
            except Exception:
                pass
        col3, col4 = st.columns(2)
        with col3:
            if st.button("Evaluate Model", key="btn_eval_pipeline"):
                try:
                    import evaluate_model

                    output = run_and_capture(evaluate_model.main)
                    st.code(output)
                except Exception as exc:
                    st.error(str(exc))
        with col4:
            if st.button("Score Logged Predictions", key="btn_score_pipeline"):
                try:
                    import score_predictions

                    output = run_and_capture(score_predictions.main)
                    st.code(output)
                    if os.path.exists("data/weekly_summary.csv"):
                        st.caption("Weekly Summary")
                        st.dataframe(pd.read_csv("data/weekly_summary.csv"), width="stretch")
                except Exception as exc:
                    st.error(str(exc))

    with sub_hist:
        st.subheader("Historical data")
        st.warning(
            "NBA Stats (`stats.nba.com`) may time out without a stable network path. Configure "
            "`.env` / Streamlit secrets: `HTTPS_PROXY`, `HTTP_PROXY`, and try `NO_PROXY=stats.nba.com,.nba.com` "
            "if your tunnel blocks the API."
        )
        hh = _historical_games_health()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "`historical_games.csv`",
                "present" if hh["exists"] else "missing",
            )
        with c2:
            st.metric("Rows", f"{hh['rows']:,}" if hh["exists"] else "—")
        with c3:
            st.metric(
                "GAME_DATE min",
                hh["date_min"].strftime("%Y-%m-%d") if hh["date_min"] is not None else "—",
            )
        with c4:
            st.metric(
                "GAME_DATE max",
                hh["date_max"].strftime("%Y-%m-%d") if hh["date_max"] is not None else "—",
            )
        if hh.get("nan_rate") is not None:
            nr = hh["nan_rate"]
            if nr > 0.05:
                st.error(f"Overall NaN rate in CSV ({nr:.4f}) is elevated — review merges before trusting training.")
            else:
                st.caption(f"Overall NaN rate (all columns): {nr:.4f}")

        merge_only = st.checkbox(
            "Merge box scores only (--merge-box-only, no Finder download)",
            value=False,
            help="Adds LeagueGameLog fields onto existing rows; resumes slow box merges.",
        )
        inp_csv = st.text_input(
            "Input CSV (--input-csv, merge-box-only)",
            value="data/historical_games.csv",
            disabled=not merge_only,
        )
        sf = st.number_input("Season from (--season-from)", min_value=2000, max_value=2100, value=2021, step=1)
        restrict_to = st.checkbox("Set explicit season ceiling (--season-to)")
        sto = (
            st.number_input("Season to (--season-to)", min_value=2000, max_value=2100, value=2025, step=1)
            if restrict_to
            else None
        )
        recent_days = st.number_input("Recent days merge (--recent-days)", min_value=1, max_value=366, value=120, step=1)
        skip_box_merge = st.checkbox("Skip LeagueGameLog (--skip-box-merge, PTS/WL only)", value=False)
        no_recent_boost = st.checkbox("Disable recent-date boost (--no-recent-date-boost)", value=False)

        bc_chunk = None
        box_cache_only = False
        no_box_cache = False
        with st.expander("Advanced fetch options"):
            bc_chunk = st.number_input(
                "--box-chunk-days",
                min_value=1,
                max_value=60,
                value=14,
                step=1,
            )
            box_cache_only = st.checkbox("--box-cache-only (box merge HTTP via cache only)")
            no_box_cache = st.checkbox("--no-box-cache")

        if st.button("Run fetch_data.py", type="primary", key="btn_fetch_data"):
            args: list[str] = []
            if merge_only:
                args.extend(["--merge-box-only", "--input-csv", inp_csv.strip() or "data/historical_games.csv"])
            else:
                args.extend(["--season-from", str(int(sf))])
                if restrict_to and sto is not None:
                    args.extend(["--season-to", str(int(sto))])
                args.extend(["--recent-days", str(int(recent_days))])
                if no_recent_boost:
                    args.append("--no-recent-date-boost")
                if skip_box_merge:
                    args.append("--skip-box-merge")
            if bc_chunk is not None and bc_chunk != 14:
                args.extend(["--box-chunk-days", str(int(bc_chunk))])
            if box_cache_only:
                args.append("--box-cache-only")
            if no_box_cache:
                args.append("--no-box-cache")
            try:
                os.chdir(root)
                with st.spinner("Fetching / merging — may take several minutes..."):
                    code_out, txt = _run_script_subprocess("fetch_data.py", args)
                st.code(txt if txt.strip() else "(no output)")
                if code_out != 0:
                    st.error(f"fetch_data.py exited with code {code_out}")
                else:
                    st.success(
                        "`data/historical_games.csv` updated. Refresh team-state cache ran from fetch when successful."
                    )
                    st.info("Train the win model in **Training**, then revisit **Evaluate Model**.")
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))

        with st.expander("Equivalent CLI snippets"):
            st.code(
                "python fetch_data.py --season-from 2021\n"
                "python fetch_data.py --merge-box-only --input-csv data/historical_games.csv\n",
                language="bash",
            )

    with sub_train:
        st.subheader("Training")
        st.warning(
            "Training and full retraining can take a long time. Close other heavy jobs; proxies apply to subprocesses."
        )
        st.markdown("##### Feedback threshold controls")
        mm_for_defaults = {}
        try:
            if os.path.exists("artifacts/model_metrics.json"):
                with open("artifacts/model_metrics.json", "r", encoding="utf-8") as f:
                    mm_for_defaults = json.load(f)
        except Exception:
            mm_for_defaults = {}
        preset_map = {
            "strict (safer activation)": {"min": 20, "apply": 60, "warn": 60},
            "balanced (recommended)": {"min": 20, "apply": 40, "warn": 40},
            "experimental (earlier activation)": {"min": 15, "apply": 30, "warn": 40},
        }
        st.caption(
            "Changing thresholds here does not affect live predictions until you run "
            "**Train win model** or **Run retrain_from_feedback.py**."
        )
        pc1, pc2 = st.columns([2, 1])
        with pc1:
            preset = st.selectbox(
                "Threshold preset",
                list(preset_map.keys()),
                index=1,
                key="fb_threshold_preset",
            )
        with pc2:
            if st.button("Apply preset", key="fb_apply_preset"):
                p = preset_map[preset]
                st.session_state["fb_min_rows_ui"] = int(p["min"])
                st.session_state["fb_apply_rows_ui"] = int(p["apply"])
                st.session_state["fb_warn_rows_ui"] = int(p["warn"])
                st.toast(f"Applied preset: {preset}")
            if st.button("Reset defaults", key="fb_reset_defaults"):
                p = preset_map["balanced (recommended)"]
                st.session_state["fb_min_rows_ui"] = int(p["min"])
                st.session_state["fb_apply_rows_ui"] = int(p["apply"])
                st.session_state["fb_warn_rows_ui"] = int(p["warn"])
                st.toast("Reset to recommended defaults (20 / 40 / 40)")
        d_min = int(mm_for_defaults.get("feedback_min_rows", 20) or 20)
        d_apply = int(mm_for_defaults.get("feedback_apply_min_rows", 40) or 40)
        d_warn = int(mm_for_defaults.get("feedback_low_sample_warning_rows", 40) or 40)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            fb_min_rows = st.number_input(
                "Feedback min rows (fit gate)",
                min_value=1,
                max_value=5000,
                value=d_min,
                step=1,
                key="fb_min_rows_ui",
            )
        with fc2:
            fb_apply_rows = st.number_input(
                "Feedback apply rows (live gate)",
                min_value=1,
                max_value=5000,
                value=max(d_apply, int(fb_min_rows)),
                step=1,
                key="fb_apply_rows_ui",
            )
        with fc3:
            fb_warn_rows = st.number_input(
                "Low-sample warning rows",
                min_value=1,
                max_value=5000,
                value=max(d_warn, int(fb_apply_rows)),
                step=1,
                key="fb_warn_rows_ui",
            )
        env_feedback = {
            "FEEDBACK_CAL_MIN_ROWS": str(int(fb_min_rows)),
            "FEEDBACK_CAL_APPLY_MIN_ROWS": str(max(int(fb_apply_rows), int(fb_min_rows))),
            "FEEDBACK_CAL_LOW_SAMPLE_WARNING_ROWS": str(
                max(int(fb_warn_rows), max(int(fb_apply_rows), int(fb_min_rows)))
            ),
        }
        st.caption(
            "Applied to train/retrain runs from this UI only via env vars; values are persisted in "
            "`artifacts/model_metrics.json` after training."
        )

        tc1, tc2 = st.columns(2)
        with tc1:
            if st.button("Train win model (train_model.py)", type="primary", key="btn_train_win"):
                try:
                    import train_model
                    from train_model import write_latest_team_state_cache

                    os.chdir(root)
                    with st.spinner("Training win model..."):
                        with _TemporaryEnv(env_feedback):
                            log = run_and_capture(train_model.main)
                    st.code(log)
                    try:
                        write_latest_team_state_cache()
                        st.caption("`data/latest_team_state_cache.csv` refreshed for predictions.")
                    except Exception:
                        pass
                    st.success("Win model artifacts written. Consider **Evaluate Model** in Performance.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        with tc2:
            if st.button("Train score models (train_score_models.py)", key="btn_train_score"):
                try:
                    import train_score_models

                    os.chdir(root)
                    with st.spinner("Training point-total regressors..."):
                        log = run_and_capture(train_score_models.main)
                    st.code(log)
                    st.success("Score models saved. Optionally **Evaluate Model**.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.markdown("##### Full pipeline (fetch + score + feedback + train)")
        r_sf = st.number_input("Retrain: season from", min_value=2000, max_value=2100, value=2021, step=1, key="r_sf")
        r_restrict = st.checkbox("Retrain: set season-to", key="r_restrict")
        r_st = (
            st.number_input(
                "Retrain: season to",
                min_value=2000,
                max_value=2100,
                value=2025,
                step=1,
                key="r_st",
            )
            if r_restrict
            else None
        )
        r_rd = st.number_input(
            "Retrain: recent-days",
            min_value=1,
            max_value=366,
            value=120,
            step=1,
            key="r_rd",
        )
        r_skip_fetch = st.checkbox("Retrain: --skip-fetch (use existing CSV)", value=False)
        r_no_recent = st.checkbox("Retrain: --no-recent-date-boost", value=False)

        if st.button("Run retrain_from_feedback.py", type="primary", key="btn_retrain_full"):
            r_args: list[str] = ["--season-from", str(int(r_sf)), "--recent-days", str(int(r_rd))]
            if r_restrict and r_st is not None:
                r_args.extend(["--season-to", str(int(r_st))])
            if r_skip_fetch:
                r_args.append("--skip-fetch")
            if r_no_recent:
                r_args.append("--no-recent-date-boost")
            try:
                from train_model import write_latest_team_state_cache

                os.chdir(root)
                with st.spinner(
                    "Full retrain pipeline — fetching (unless skipped), scoring log, feedback, train win model..."
                ):
                    code_out, txt = _run_script_subprocess_with_env(
                        "retrain_from_feedback.py", r_args, env_feedback
                    )
                st.code(txt if txt.strip() else "(no output)")
                if code_out != 0:
                    st.error(f"retrain_from_feedback.py exited with code {code_out}")
                else:
                    try:
                        write_latest_team_state_cache()
                    except Exception:
                        pass
                    st.success("Pipeline complete. Metrics header updated — open **Evaluate Model**.")
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))

        st.markdown("##### Quick ops")
        op1, op2 = st.columns(2)
        with op1:
            if st.button("Resolve log + feedback calibrator (no fetch)", key="btn_resolve_sync"):
                try:
                    import prediction_sync

                    with st.spinner("Scoring log, rebuilding feedback, patching calibrator..."):
                        os.chdir(root)
                        out = prediction_sync.sync_completed_predictions(silent=True)
                    if out.get("skipped"):
                        st.warning(out.get("reason", "skipped"))
                    else:
                        st.success(
                            f"Resolved {out.get('resolved', 0)}, pending {out.get('pending', 0)}; "
                            f"feedback rows {out.get('feedback_rows', 0)}; "
                            f"calibrator {'updated' if out.get('feedback_patched') else 'unchanged'}."
                        )
                except Exception as exc:
                    st.error(str(exc))
        with op2:
            if st.button("Backfill scores + feedback from log", key="btn_backfill"):
                try:
                    with st.spinner("Running backfill..."):
                        proc = subprocess.run(
                            [sys.executable, str(root / "backfill_prediction_results.py")],
                            cwd=str(root),
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                    st.code((proc.stdout or "") + (proc.stderr or ""))
                    if proc.returncode != 0:
                        st.error(f"Backfill exited with code {proc.returncode}")
                    else:
                        st.success("Backfill finished. Optionally run full **retrain** above.")
                        st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.caption("CLI equivalents:")
        st.code(
            "python retrain_from_feedback.py --season-from 2021\n"
            "python retrain_from_feedback.py --skip-fetch\n"
            "python backfill_prediction_results.py [--retrain]\n",
            language="bash",
        )
