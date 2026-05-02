import json
import os
from dataclasses import dataclass
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Calibrator:
    method: str
    model: object


def load_model_bundle(path: str = "artifacts/model.joblib"):
    # Provide compatibility for older artifacts pickled with __main__.Calibrator.
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "Calibrator"):
        setattr(main_mod, "Calibrator", Calibrator)
    return joblib.load(path)


_BOX_NUMERIC_COLS = (
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "TOV",
)

_FEEDBACK_CAL_MIN_ROWS_DEFAULT = 20
_FEEDBACK_CAL_LOW_SAMPLE_WARNING_ROWS = 40
_FEEDBACK_CAL_APPLY_MIN_ROWS = 40
_FEEDBACK_CAL_VAL_MIN_ROWS = 8
_FEEDBACK_CAL_TRAIN_MIN_ROWS = 12
_QUALITY_NAN_WARN = 0.05
_QUALITY_RECENT_DAYS = 30
_QUALITY_MIN_RECENT_GAMES = 200
_QUALITY_STALE_DAYS_WARN = 3
_DRIFT_CONFIDENCE_WARN = 0.03
_DRIFT_BRIER_WARN = 0.02


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def feedback_thresholds_from_env() -> dict[str, int]:
    min_rows = max(1, _env_int("FEEDBACK_CAL_MIN_ROWS", _FEEDBACK_CAL_MIN_ROWS_DEFAULT))
    apply_rows = max(min_rows, _env_int("FEEDBACK_CAL_APPLY_MIN_ROWS", _FEEDBACK_CAL_APPLY_MIN_ROWS))
    low_warn = max(apply_rows, _env_int("FEEDBACK_CAL_LOW_SAMPLE_WARNING_ROWS", _FEEDBACK_CAL_LOW_SAMPLE_WARNING_ROWS))
    return {
        "min_rows": int(min_rows),
        "apply_rows": int(apply_rows),
        "low_sample_warning_rows": int(low_warn),
    }


def _three_in_four_flag_series(
    game_dates: pd.Series,
    lag_1: pd.Series,
    lag_2: pd.Series,
    game_num: np.ndarray,
) -> pd.Series:
    """Vectorized THREE_IN_FOUR_FLAG; parity with inference helper."""
    okay = (
        (game_num >= 2)
        & lag_1.notna()
        & lag_2.notna()
        & ((game_dates - lag_2).dt.days <= 3)
        & ((game_dates - lag_1).dt.days <= 2)
    )
    return okay.astype(np.int32)


def three_in_four_flag_from_schedule(
    upcoming_game_date: pd.Timestamp | str | None,
    last_completed_game_date: pd.Timestamp | str | float | None,
    second_last_completed_game_date: pd.Timestamp | str | float | None,
    n_completed_games: int,
) -> int:
    """
    True when the upcoming game satisfies the same 3-games-in-4-nights heuristic as training
    (see _prepare_team_rows): cumcount>=2 plus date gaps vs the two prior games.
    """
    if n_completed_games < 2:
        return 0
    cur = pd.to_datetime(upcoming_game_date)
    lag1 = pd.to_datetime(last_completed_game_date)
    lag2 = pd.to_datetime(second_last_completed_game_date)
    if pd.isna(cur) or pd.isna(lag1) or pd.isna(lag2):
        return 0
    cur_day = pd.Timestamp(cur).normalize()
    d1 = (cur_day - pd.Timestamp(lag1).normalize()).days
    d2 = (cur_day - pd.Timestamp(lag2).normalize()).days
    if d2 <= 3 and d1 <= 2:
        return 1
    return 0


def _ensure_box_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in _BOX_NUMERIC_COLS:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _prepare_team_rows(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ").astype(int)
    df["RESULT"] = (df["WL"] == "W").astype(int)
    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)

    opp_pts = df.groupby("GAME_ID")["PTS"].transform("sum") - df["PTS"]
    df["OPP_PTS"] = opp_pts
    df["POINT_DIFF"] = df["PTS"] - df["OPP_PTS"]

    df["DAYS_SINCE_LAST_GAME"] = (
        df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(3).clip(lower=0, upper=10)
    )
    df["B2B_FLAG"] = (df["DAYS_SINCE_LAST_GAME"] <= 1).astype(int)

    game_num = df.groupby("TEAM_ID").cumcount()
    lag_1 = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    lag_2 = df.groupby("TEAM_ID")["GAME_DATE"].shift(2)
    df["THREE_IN_FOUR_FLAG"] = _three_in_four_flag_series(
        df["GAME_DATE"], lag_1, lag_2, game_num.to_numpy(dtype=int)
    )

    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    df = _ensure_box_numeric_columns(df)

    # Four-factor style game rates (rules enter via box-score measurements).
    fga_safe = df["FGA"].replace(0, np.nan)
    df["GAME_EFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / fga_safe
    denom_poss = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    denom_poss = denom_poss.replace(0, np.nan)
    df["GAME_TOV_PCT"] = df["TOV"] / denom_poss
    df["GAME_FTR"] = df["FTA"] / fga_safe
    # Team possession proxy: FGA - OREB + TOV + 0.44*FTA (common estimator).
    df["GAME_PACE_PROX"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    # Offensive rebound %: OREB / (OREB + opponent defensive rebounds); two-team game.
    team_dreb_sum = df.groupby("GAME_ID")["DREB"].transform("sum")
    df["OPP_DREB"] = team_dreb_sum - df["DREB"]
    df["GAME_ORB_PCT_OFF"] = df["OREB"] / (df["OREB"] + df["OPP_DREB"] + 1e-6)

    _efg_default = 0.535
    _tov_default = 0.135
    _orb_default = 0.22
    _ftr_default = 0.258
    _pace_default = 99.0
    df["GAME_EFG_PCT"] = df["GAME_EFG_PCT"].fillna(_efg_default)
    df["GAME_TOV_PCT"] = df["GAME_TOV_PCT"].fillna(_tov_default)
    df["GAME_ORB_PCT_OFF"] = df["GAME_ORB_PCT_OFF"].fillna(_orb_default)
    df["GAME_FTR"] = df["GAME_FTR"].fillna(_ftr_default)
    df["GAME_PACE_PROX"] = df["GAME_PACE_PROX"].fillna(_pace_default)

    grp = df.groupby("TEAM_ID", sort=False)
    tid = df["TEAM_ID"]
    pts_mean = float(df["PTS"].mean())
    opp_pts_mean = float(df["OPP_PTS"].mean())
    margin_std_default = df["POINT_DIFF"].std()
    if np.isnan(margin_std_default):
        margin_std_default = 8.0

    for window in (3, 5, 10, 20):
        min_periods = 2 if window <= 5 else 3

        sh_res = grp["RESULT"].shift(1)
        df[f"ROLL_WIN_PCT_{window}"] = (
            sh_res.groupby(tid)
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0.5)
        )
        sh_pts = grp["PTS"].shift(1)
        df[f"ROLL_PTS_FOR_{window}"] = (
            sh_pts.groupby(tid)
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(pts_mean)
        )
        sh_opp = grp["OPP_PTS"].shift(1)
        df[f"ROLL_PTS_AGAINST_{window}"] = (
            sh_opp.groupby(tid)
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(opp_pts_mean)
        )
        df[f"ROLL_NET_RATING_{window}"] = df[f"ROLL_PTS_FOR_{window}"] - df[f"ROLL_PTS_AGAINST_{window}"]
        sh_pd = grp["POINT_DIFF"].shift(1)
        df[f"ROLL_MARGIN_STD_{window}"] = (
            sh_pd.groupby(tid)
            .rolling(window, min_periods=min_periods)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(margin_std_default)
        )
        for base_col, default in (
            ("GAME_EFG_PCT", _efg_default),
            ("GAME_TOV_PCT", _tov_default),
            ("GAME_ORB_PCT_OFF", _orb_default),
            ("GAME_FTR", _ftr_default),
            ("GAME_PACE_PROX", _pace_default),
        ):
            short = base_col.replace("GAME_", "")
            sh_bc = grp[base_col].shift(1)
            df[f"ROLL_{short}_{window}"] = (
                sh_bc.groupby(tid)
                .rolling(window, min_periods=min_periods)
                .mean()
                .reset_index(level=0, drop=True)
                .fillna(default)
            )

    sides = df[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    pair = sides.merge(sides, on="GAME_ID")
    pair = pair[pair["TEAM_ID_x"] != pair["TEAM_ID_y"]]
    pair = pair.rename(columns={"TEAM_ID_x": "TEAM_ID", "TEAM_ID_y": "OPP_TEAM_ID"})
    df = df.merge(pair, on=["GAME_ID", "TEAM_ID"], how="left")

    opp_net_snap = df[["GAME_ID", "TEAM_ID", "ROLL_NET_RATING_10"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "ROLL_NET_RATING_10": "OPP_SNAP_NET_10"}
    )
    df = df.merge(opp_net_snap, on=["GAME_ID", "OPP_TEAM_ID"], how="left")
    df["OPP_SNAP_NET_10"] = pd.to_numeric(df["OPP_SNAP_NET_10"], errors="coerce").fillna(0.0)

    df["_sos_in"] = df.groupby("TEAM_ID", sort=False)["OPP_SNAP_NET_10"].shift(1)
    df["SOS_OPP_NET_10"] = (
        df["_sos_in"]
        .groupby(df["TEAM_ID"])
        .rolling(10, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["SOS_OPP_NET_10_INCL_CURRENT"] = (
        df["OPP_SNAP_NET_10"]
        .groupby(df["TEAM_ID"])
        .rolling(10, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    df = df.drop(columns=["_sos_in", "OPP_SNAP_NET_10", "OPP_TEAM_ID"], errors="ignore")

    return df


def _compute_elo_probabilities(games: pd.DataFrame) -> pd.DataFrame:
    df = games.sort_values("GAME_DATE_HOME").copy()
    ratings: dict[int, float] = {}
    base_rating = 1500.0
    home_adv = 70.0
    k_factor = 20.0

    elo_home_pre = []
    elo_away_pre = []
    elo_home_win_prob = []
    elo_diff = []

    for _, row in df.iterrows():
        home_id = int(row["TEAM_ID_HOME"])
        away_id = int(row["TEAM_ID_AWAY"])
        home_rating = ratings.get(home_id, base_rating)
        away_rating = ratings.get(away_id, base_rating)
        diff = home_rating - away_rating + home_adv
        p_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        actual_home = int(row["WIN_HOME"])

        mov = abs(float(row["PTS_HOME"]) - float(row["PTS_AWAY"]))
        mov_mult = np.log(max(1.0, mov) + 1.0)
        delta = k_factor * mov_mult * (actual_home - p_home)
        ratings[home_id] = home_rating + delta
        ratings[away_id] = away_rating - delta

        elo_home_pre.append(home_rating)
        elo_away_pre.append(away_rating)
        elo_home_win_prob.append(p_home)
        elo_diff.append(diff)

    df["ELO_HOME_PRE"] = elo_home_pre
    df["ELO_AWAY_PRE"] = elo_away_pre
    df["ELO_DIFF_PRE"] = elo_diff
    df["ELO_HOME_WIN_PROB"] = elo_home_win_prob
    return df


def build_game_level_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_team_rows(raw)

    home = df[df["IS_HOME"] == 1].copy()
    away = df[df["IS_HOME"] == 0].copy()

    merged = home.merge(
        away,
        on="GAME_ID",
        suffixes=("_HOME", "_AWAY"),
        how="inner",
    )

    merged["WIN_HOME"] = merged["RESULT_HOME"]
    merged = _compute_elo_probabilities(merged)
    merged["REST_DIFF"] = merged["DAYS_SINCE_LAST_GAME_HOME"] - merged["DAYS_SINCE_LAST_GAME_AWAY"]
    merged["B2B_DIFF"] = merged["B2B_FLAG_HOME"] - merged["B2B_FLAG_AWAY"]
    merged["THREE_IN_FOUR_DIFF"] = merged["THREE_IN_FOUR_FLAG_HOME"] - merged["THREE_IN_FOUR_FLAG_AWAY"]
    merged["PLAYOFF_GAME"] = merged["GAME_ID"].astype(str).str.startswith("004").astype(int)
    return merged.sort_values("GAME_DATE_HOME").reset_index(drop=True)


DEFAULT_LATEST_TEAM_CACHE = os.path.join("data", "latest_team_state_cache.csv")


def schedule_meta_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Per TEAM_ID: count of completed games plus date of second-most-recent game (for 3-in-4 parity)."""
    df = raw.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    counts = df.groupby("TEAM_ID").size().reset_index(name="N_COMPLETED_GAMES")
    df["SECOND_LAST_GAME_DATE"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    tail = df.groupby("TEAM_ID").tail(1)[["TEAM_ID", "SECOND_LAST_GAME_DATE"]]
    return tail.merge(counts, on="TEAM_ID", how="inner")


def latest_team_snapshot_from_frames(
    raw: pd.DataFrame, games: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per TEAM_ID — latest rolling features plus schedule metadata for inference."""
    if games is None:
        games = build_game_level_frame(raw)

    home_cols = [c for c in games.columns if c.endswith("_HOME")]
    away_cols = [c for c in games.columns if c.endswith("_AWAY")]

    h = games[home_cols].copy()
    h.columns = [c[:-5] for c in h.columns]
    a = games[away_cols].copy()
    a.columns = [c[:-5] for c in a.columns]

    if "ELO_HOME_PRE" in games.columns:
        h["ELO_PRE"] = games["ELO_HOME_PRE"].to_numpy()
    if "ELO_AWAY_PRE" in games.columns:
        a["ELO_PRE"] = games["ELO_AWAY_PRE"].to_numpy()

    all_team_rows = pd.concat([h, a], ignore_index=True)
    all_team_rows = all_team_rows.sort_values(["TEAM_ID", "GAME_DATE"])
    latest = all_team_rows.groupby("TEAM_ID").tail(1)
    meta = schedule_meta_from_raw(raw)
    latest = latest.merge(meta, on="TEAM_ID", how="left")
    latest["N_COMPLETED_GAMES"] = latest["N_COMPLETED_GAMES"].fillna(0).astype(int)
    return latest


def write_latest_team_state_cache(
    hist_path: str = "data/historical_games.csv",
    out_path: str = DEFAULT_LATEST_TEAM_CACHE,
) -> str:
    """
    Persist latest per-team feature rows derived from historical_games.csv.
    predict_next_games prefers this cache when newer than historical_games.csv.
    """
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"Missing {hist_path}. Run fetch_data.py first.")
    raw = pd.read_csv(hist_path, parse_dates=["GAME_DATE"])
    latest = latest_team_snapshot_from_frames(raw)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    latest.to_csv(out_path, index=False)
    return out_path


def feature_columns() -> list[str]:
    cols = []
    for window in (3, 5, 10, 20):
        cols.extend(
            [
                f"ROLL_WIN_PCT_{window}_HOME",
                f"ROLL_WIN_PCT_{window}_AWAY",
                f"ROLL_NET_RATING_{window}_HOME",
                f"ROLL_NET_RATING_{window}_AWAY",
                f"ROLL_MARGIN_STD_{window}_HOME",
                f"ROLL_MARGIN_STD_{window}_AWAY",
                f"ROLL_EFG_PCT_{window}_HOME",
                f"ROLL_EFG_PCT_{window}_AWAY",
                f"ROLL_TOV_PCT_{window}_HOME",
                f"ROLL_TOV_PCT_{window}_AWAY",
                f"ROLL_ORB_PCT_OFF_{window}_HOME",
                f"ROLL_ORB_PCT_OFF_{window}_AWAY",
                f"ROLL_FTR_{window}_HOME",
                f"ROLL_FTR_{window}_AWAY",
                f"ROLL_PACE_PROX_{window}_HOME",
                f"ROLL_PACE_PROX_{window}_AWAY",
            ]
        )
    cols.extend(
        [
            "DAYS_SINCE_LAST_GAME_HOME",
            "DAYS_SINCE_LAST_GAME_AWAY",
            "REST_DIFF",
            "B2B_FLAG_HOME",
            "B2B_FLAG_AWAY",
            "B2B_DIFF",
            "THREE_IN_FOUR_FLAG_HOME",
            "THREE_IN_FOUR_FLAG_AWAY",
            "THREE_IN_FOUR_DIFF",
            "ELO_HOME_PRE",
            "ELO_AWAY_PRE",
            "ELO_DIFF_PRE",
            "ELO_HOME_WIN_PROB",
            "SOS_OPP_NET_10_HOME",
            "SOS_OPP_NET_10_AWAY",
            "PLAYOFF_GAME",
        ]
    )
    return cols


def _walk_forward_splits(n_rows: int, n_folds: int = 5, initial_train_ratio: float = 0.5):
    initial_train_size = max(500, int(n_rows * initial_train_ratio))
    if n_rows <= initial_train_size + n_folds:
        return []
    fold_size = max(100, (n_rows - initial_train_size) // n_folds)
    splits = []
    for fold in range(n_folds):
        train_end = initial_train_size + fold * fold_size
        test_start = train_end
        test_end = min(n_rows, test_start + fold_size)
        if test_end - test_start < 50:
            continue
        splits.append((0, train_end, test_start, test_end))
    return splits


def _fit_calibrator(
    y_true: pd.Series,
    raw_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Calibrator:
    raw_prob = np.clip(raw_prob.astype(float), 1e-6, 1 - 1e-6)
    y = y_true.to_numpy().astype(int)

    logits = np.log(raw_prob / (1.0 - raw_prob)).reshape(-1, 1)
    platt_model = LogisticRegression(max_iter=1000)
    platt_model.fit(logits, y, sample_weight=sample_weight)
    platt_probs = platt_model.predict_proba(logits)[:, 1]
    platt_brier = brier_score_loss(y, platt_probs)

    if len(y) >= 1000:
        iso_model = IsotonicRegression(out_of_bounds="clip")
        iso_model.fit(raw_prob, y, sample_weight=sample_weight)
        iso_probs = iso_model.transform(raw_prob)
        iso_brier = brier_score_loss(y, iso_probs)
        if iso_brier < platt_brier:
            return Calibrator(method="isotonic", model=iso_model)

    return Calibrator(method="platt", model=platt_model)


def _fit_feedback_calibrator(
    feedback_path: str = "data/prediction_feedback_training.csv",
    min_rows: int | None = None,
    max_rows: int = 5000,
    apply_gate: bool = True,
    apply_min_rows: int | None = None,
) -> tuple[Calibrator | None, dict[str, int]]:
    """Fit calibrator from logged predictions vs outcomes. Returns (calibrator, meta)."""
    cfg = feedback_thresholds_from_env()
    min_rows = int(cfg["min_rows"] if min_rows is None else min_rows)
    apply_min_rows = int(cfg["apply_rows"] if apply_min_rows is None else apply_min_rows)
    low_sample_warning_rows = int(cfg["low_sample_warning_rows"])
    meta: dict[str, int] = {
        "feedback_dataset_rows": 0,
        "feedback_min_rows": int(min_rows),
        "feedback_apply_min_rows": int(apply_min_rows),
        "feedback_low_sample_warning_rows": int(low_sample_warning_rows),
        "feedback_rows_used": 0,
        "feedback_calibrator_low_sample_warning": False,
        "feedback_calibrator_selected": False,
    }
    if not os.path.exists(feedback_path):
        return None, meta
    feedback = pd.read_csv(feedback_path)
    required = {"model_home_win_probability", "actual_home_win"}
    if not required.issubset(set(feedback.columns)):
        return None, meta

    feedback = feedback.dropna(subset=["model_home_win_probability", "actual_home_win"]).copy()
    meta["feedback_dataset_rows"] = int(len(feedback))
    if feedback.empty:
        return None, meta

    if "run_timestamp" in feedback.columns:
        feedback["run_timestamp"] = pd.to_datetime(feedback["run_timestamp"], errors="coerce")
        feedback = feedback.sort_values("run_timestamp")
    if len(feedback) > max_rows:
        feedback = feedback.tail(max_rows).copy()
        meta["feedback_dataset_rows"] = int(len(feedback))
    if len(feedback) < min_rows:
        meta["feedback_calibrator_reason"] = "below_min_rows"
        return None, meta

    p = np.clip(
        pd.to_numeric(feedback["model_home_win_probability"], errors="coerce").fillna(0.5).to_numpy(),
        1e-6,
        1 - 1e-6,
    )
    y = pd.to_numeric(feedback["actual_home_win"], errors="coerce").fillna(0).astype(int).to_numpy()

    # Recency-weighted feedback: newer resolved predictions get higher influence.
    if "run_timestamp" in feedback.columns and feedback["run_timestamp"].notna().any():
        newest_ts = feedback["run_timestamp"].max()
        days_old = (newest_ts - feedback["run_timestamp"]).dt.total_seconds() / 86400.0
        # Half-life of 45 days => weight halves every 45 days.
        weights = np.power(0.5, np.clip(days_old.to_numpy(dtype=float), 0.0, None) / 45.0)
        weights = np.clip(weights, 0.1, 1.0)
    else:
        weights = np.ones(len(feedback), dtype=float)
    meta["feedback_rows_used"] = int(len(feedback))
    meta["feedback_calibrator_low_sample_warning"] = (
        int(meta["feedback_rows_used"]) < low_sample_warning_rows
    )
    apply_blocked = int(meta["feedback_rows_used"]) < apply_min_rows
    n = len(y)
    val_size = max(_FEEDBACK_CAL_VAL_MIN_ROWS, int(0.3 * n))
    train_size = n - val_size
    if train_size < _FEEDBACK_CAL_TRAIN_MIN_ROWS:
        meta["feedback_calibrator_reason"] = "insufficient_train_or_validation_rows"
        return None, meta

    p_train, p_val = p[:train_size], p[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    w_train = weights[:train_size]
    candidate = _fit_calibrator(pd.Series(y_train), p_train, sample_weight=w_train)
    p_val_cal = apply_calibrator(p_val, candidate)

    baseline_brier = float(brier_score_loss(y_val, p_val))
    candidate_brier = float(brier_score_loss(y_val, p_val_cal))
    baseline_ll = float(log_loss(y_val, p_val, labels=[0, 1]))
    candidate_ll = float(log_loss(y_val, p_val_cal, labels=[0, 1]))
    brier_gain = baseline_brier - candidate_brier
    ll_gain = baseline_ll - candidate_ll

    meta["feedback_validation_rows"] = int(len(y_val))
    meta["feedback_validation_brier_raw"] = baseline_brier
    meta["feedback_validation_brier_calibrated"] = candidate_brier
    meta["feedback_validation_log_loss_raw"] = baseline_ll
    meta["feedback_validation_log_loss_calibrated"] = candidate_ll
    meta["feedback_validation_brier_gain"] = brier_gain
    meta["feedback_validation_log_loss_gain"] = ll_gain

    improves_brier = candidate_brier <= baseline_brier
    improves_ll = candidate_ll <= baseline_ll
    if not (improves_brier and improves_ll):
        meta["feedback_calibrator_reason"] = "validation_not_improved"
        return None, meta

    meta["feedback_calibrator_selected"] = True
    meta["feedback_calibrator_reason"] = "validation_improved"
    if apply_gate and apply_blocked:
        meta["feedback_calibrator_selected"] = False
        meta["feedback_calibrator_reason"] = "below_apply_rows"
        return None, meta
    return _fit_calibrator(pd.Series(y), p, sample_weight=weights), meta


def _data_quality_snapshot(raw: pd.DataFrame) -> dict:
    game_dates = pd.to_datetime(raw.get("GAME_DATE"), errors="coerce")
    now = pd.Timestamp.now().normalize()
    max_date = game_dates.max() if not game_dates.empty else pd.NaT
    stale_days = int((now - max_date.normalize()).days) if pd.notna(max_date) else 999
    recent_cut = now - pd.Timedelta(days=_QUALITY_RECENT_DAYS)
    recent_games = int((game_dates >= recent_cut).sum())
    nan_rate = float(raw.isna().mean().mean()) if len(raw) else 1.0
    warnings: list[str] = []
    if nan_rate > _QUALITY_NAN_WARN:
        warnings.append(f"high_nan_rate>{_QUALITY_NAN_WARN:.2f}")
    if stale_days > _QUALITY_STALE_DAYS_WARN:
        warnings.append(f"stale_data_days>{_QUALITY_STALE_DAYS_WARN}")
    if recent_games < _QUALITY_MIN_RECENT_GAMES:
        warnings.append(f"recent_games<{_QUALITY_MIN_RECENT_GAMES}")
    return {
        "nan_rate": nan_rate,
        "stale_days": stale_days,
        "recent_games_30d": recent_games,
        "quality_warning": bool(warnings),
        "quality_warning_reasons": warnings,
    }


def apply_calibrator(raw_prob: np.ndarray, calibrator: Calibrator | None) -> np.ndarray:
    probs = np.clip(raw_prob.astype(float), 1e-6, 1 - 1e-6)
    if calibrator is None:
        return probs
    if calibrator.method == "isotonic":
        return np.clip(calibrator.model.transform(probs), 1e-6, 1 - 1e-6)
    logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)
    return np.clip(calibrator.model.predict_proba(logits)[:, 1], 1e-6, 1 - 1e-6)


def patch_feedback_calibrator_into_artifacts(
    model_path: str = "artifacts/model.joblib",
    metrics_path: str = "artifacts/model_metrics.json",
) -> dict:
    """Refit feedback calibrator from disk and write it into the existing saved model bundle."""
    out: dict = {"patched": False, "reason": ""}
    if not os.path.exists(model_path):
        out["reason"] = "missing_model_artifact"
        return out
    try:
        bundle = load_model_bundle(model_path)
    except Exception as exc:
        out["reason"] = f"load_failed:{exc}"
        return out
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        out["reason"] = "unexpected_bundle_shape"
        return out

    feedback_calibrator, feedback_meta = _fit_feedback_calibrator()
    bundle["feedback_calibrator"] = feedback_calibrator
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(bundle, model_path)
    out["patched"] = True
    out["feedback_meta"] = feedback_meta
    out["feedback_calibrator_method"] = (
        feedback_calibrator.method if feedback_calibrator is not None else "none"
    )

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            metrics["feedback_calibrator_method"] = out["feedback_calibrator_method"]
            metrics["feedback_calibrator_rows"] = int(feedback_meta.get("feedback_rows_used", 0))
            metrics["feedback_dataset_rows"] = int(feedback_meta.get("feedback_dataset_rows", 0))
            metrics["feedback_min_rows"] = int(
                feedback_meta.get("feedback_min_rows", _FEEDBACK_CAL_MIN_ROWS_DEFAULT)
            )
            metrics["feedback_apply_min_rows"] = int(
                feedback_meta.get("feedback_apply_min_rows", _FEEDBACK_CAL_APPLY_MIN_ROWS)
            )
            metrics["feedback_low_sample_warning_rows"] = int(
                feedback_meta.get("feedback_low_sample_warning_rows", _FEEDBACK_CAL_LOW_SAMPLE_WARNING_ROWS)
            )
            metrics["feedback_calibrator_low_sample_warning"] = bool(
                feedback_meta.get("feedback_calibrator_low_sample_warning", False)
            )
            metrics["feedback_calibrator_selected"] = bool(
                feedback_meta.get("feedback_calibrator_selected", False)
            )
            metrics["feedback_calibrator_reason"] = str(
                feedback_meta.get("feedback_calibrator_reason", "")
            )
            metrics["feedback_mode"] = (
                "active" if feedback_calibrator is not None else "off"
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass
    return out


def _fit_estimator(model_name: str):
    if model_name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42),
        )
    if model_name == "hist_gb":
        # Tuned vs prior (5 / 0.05 / 300): slightly deeper + slower lr + more trees for holdout log loss.
        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.04, max_iter=350, random_state=42
        )
    return None


def _raw_probabilities(model_name: str, estimator, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if model_name == "elo_baseline":
        return df["ELO_HOME_WIN_PROB"].to_numpy(dtype=float)
    return estimator.predict_proba(df[cols])[:, 1]


def main() -> None:
    input_path = "data/historical_games.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "Missing data/historical_games.csv. Run: python fetch_data.py"
        )

    prev_metrics = {}
    if os.path.exists("artifacts/model_metrics.json"):
        try:
            with open("artifacts/model_metrics.json", "r", encoding="utf-8") as f:
                prev_metrics = json.load(f)
        except Exception:
            prev_metrics = {}

    raw = pd.read_csv(input_path, parse_dates=["GAME_DATE"])
    quality = _data_quality_snapshot(raw)
    games = build_game_level_frame(raw)
    feature_cols = feature_columns()
    target_col = "WIN_HOME"

    model_candidates = ["logistic", "hist_gb", "elo_baseline"]
    wf_rows: list[dict] = []
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(
        _walk_forward_splits(len(games), n_folds=5),
        start=1,
    ):
        train_df = games.iloc[train_start:train_end]
        test_df = games.iloc[test_start:test_end]
        y_test = test_df[target_col].to_numpy().astype(int)
        for model_name in model_candidates:
            estimator = _fit_estimator(model_name)
            if estimator is not None:
                estimator.fit(train_df[feature_cols], train_df[target_col])
            raw_probs = _raw_probabilities(model_name, estimator, test_df, feature_cols)
            pred = (raw_probs >= 0.5).astype(int)
            wf_rows.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, pred),
                    "log_loss": log_loss(y_test, raw_probs, labels=[0, 1]),
                    "brier": brier_score_loss(y_test, raw_probs),
                }
            )

    wf_df = pd.DataFrame(wf_rows)
    by_model = (
        wf_df.groupby("model", as_index=False)[["accuracy", "log_loss", "brier"]]
        .mean()
        .sort_values("log_loss")
    )
    champion_model_name = str(by_model.iloc[0]["model"])
    wf_ll_champion = champion_model_name
    wf_acc_champion = str(by_model.sort_values("accuracy", ascending=False).iloc[0]["model"])

    fit_cut = int(len(games) * 0.7)
    calib_cut = int(len(games) * 0.8)
    fit_df = games.iloc[:fit_cut].copy()
    calib_df = games.iloc[fit_cut:calib_cut].copy()
    test_df = games.iloc[calib_cut:].copy()

    champion_estimator = _fit_estimator(champion_model_name)
    if champion_estimator is not None:
        champion_estimator.fit(fit_df[feature_cols], fit_df[target_col])
    calib_raw = _raw_probabilities(champion_model_name, champion_estimator, calib_df, feature_cols)
    calibrator = _fit_calibrator(calib_df[target_col], calib_raw)
    feedback_calibrator, feedback_meta = _fit_feedback_calibrator(apply_gate=True)
    feedback_shadow_calibrator, feedback_shadow_meta = _fit_feedback_calibrator(apply_gate=False)
    test_raw = _raw_probabilities(champion_model_name, champion_estimator, test_df, feature_cols)
    test_prob_hist = apply_calibrator(test_raw, calibrator)
    test_prob_shadow = (
        apply_calibrator(test_prob_hist, feedback_shadow_calibrator)
        if feedback_shadow_calibrator is not None
        else test_prob_hist.copy()
    )
    test_prob = (
        apply_calibrator(test_prob_hist, feedback_calibrator)
        if feedback_calibrator is not None
        else test_prob_hist.copy()
    )
    test_pred = (test_prob >= 0.5).astype(int)
    acc = accuracy_score(test_df[target_col], test_pred)
    ll = log_loss(test_df[target_col], test_prob, labels=[0, 1])
    brier = brier_score_loss(test_df[target_col], test_prob)
    ll_hist = log_loss(test_df[target_col], test_prob_hist, labels=[0, 1])
    brier_hist = brier_score_loss(test_df[target_col], test_prob_hist)
    ll_shadow = log_loss(test_df[target_col], test_prob_shadow, labels=[0, 1])
    brier_shadow = brier_score_loss(test_df[target_col], test_prob_shadow)

    if feedback_calibrator is not None:
        feedback_mode = "active"
    elif feedback_shadow_calibrator is not None:
        feedback_mode = "shadow"
    else:
        feedback_mode = "off"

    mean_conf = float(np.mean(np.maximum(test_prob, 1.0 - test_prob)))
    prev_hold = prev_metrics.get("holdout_metrics", {}) if isinstance(prev_metrics, dict) else {}
    prev_brier = float(prev_hold.get("brier")) if "brier" in prev_hold else None
    prev_conf = (
        float(prev_metrics.get("holdout_mean_confidence"))
        if isinstance(prev_metrics, dict) and "holdout_mean_confidence" in prev_metrics
        else None
    )
    brier_shift = (float(brier) - prev_brier) if prev_brier is not None else None
    conf_shift = (mean_conf - prev_conf) if prev_conf is not None else None
    drift_reasons: list[str] = []
    if brier_shift is not None and brier_shift > _DRIFT_BRIER_WARN:
        drift_reasons.append(f"brier_shift>{_DRIFT_BRIER_WARN:.3f}")
    if conf_shift is not None and abs(conf_shift) > _DRIFT_CONFIDENCE_WARN:
        drift_reasons.append(f"confidence_shift>{_DRIFT_CONFIDENCE_WARN:.3f}")
    drift_warning = bool(drift_reasons)

    model_bundle = {
        "model_type": champion_model_name,
        "estimator": champion_estimator,
        "feature_columns": feature_cols,
        "calibrator": calibrator,
        "feedback_calibrator": feedback_calibrator,
    }

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model_bundle, "artifacts/model.joblib")
    with open("artifacts/features.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_cols}, f, indent=2)
    with open("artifacts/model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "champion_model": champion_model_name,
                "walk_forward_summary": by_model.to_dict(orient="records"),
                "champion_selection_rule": (
                    "Minimum mean walk-forward log_loss among logistic, hist_gb, and elo_baseline "
                    "(probabilistic calibration). Highest walk-forward accuracy can differ "
                    "(see walk_forward_accuracy_leader)."
                ),
                "walk_forward_accuracy_leader": wf_acc_champion,
                "walk_forward_log_loss_leader": wf_ll_champion,
                "holdout_metrics": {
                    "accuracy": float(acc),
                    "log_loss": float(ll),
                    "brier": float(brier),
                },
                "calibrator_method": calibrator.method,
                "feedback_calibrator_method": feedback_calibrator.method if feedback_calibrator else "none",
                "feedback_calibrator_rows": int(feedback_meta.get("feedback_rows_used", 0)),
                "feedback_dataset_rows": int(feedback_meta.get("feedback_dataset_rows", 0)),
                "feedback_min_rows": int(
                    feedback_meta.get("feedback_min_rows", _FEEDBACK_CAL_MIN_ROWS_DEFAULT)
                ),
                "feedback_apply_min_rows": int(
                    feedback_meta.get("feedback_apply_min_rows", _FEEDBACK_CAL_APPLY_MIN_ROWS)
                ),
                "feedback_low_sample_warning_rows": int(
                    feedback_meta.get("feedback_low_sample_warning_rows", _FEEDBACK_CAL_LOW_SAMPLE_WARNING_ROWS)
                ),
                "feedback_calibrator_low_sample_warning": bool(
                    feedback_meta.get("feedback_calibrator_low_sample_warning", False)
                ),
                "feedback_calibrator_selected": bool(
                    feedback_meta.get("feedback_calibrator_selected", False)
                ),
                "feedback_calibrator_reason": str(
                    feedback_meta.get("feedback_calibrator_reason", "")
                ),
                "feedback_mode": feedback_mode,
                "feedback_shadow_delta_brier": float(brier_shadow - brier_hist),
                "feedback_shadow_delta_log_loss": float(ll_shadow - ll_hist),
                "feedback_shadow_available": bool(feedback_shadow_calibrator is not None),
                "feedback_shadow_reason": str(feedback_shadow_meta.get("feedback_calibrator_reason", "")),
                "holdout_mean_confidence": mean_conf,
                "data_quality": quality,
                "drift_warning": drift_warning,
                "drift_warning_reasons": drift_reasons,
                "drift_brier_shift_vs_prev": brier_shift,
                "drift_confidence_shift_vs_prev": conf_shift,
                "hist_gb_hyperparams": {"max_depth": 6, "learning_rate": 0.04, "max_iter": 350},
                "hist_gb_tuning_note": (
                    "HistGradientBoostingClassifier hparams adjusted from max_depth=5, "
                    "learning_rate=0.05, max_iter=300 to improve holdout log loss; compare holdout_metrics."
                ),
            },
            f,
            indent=2,
        )

    print("Training done.")
    print("Walk-forward model summary (mean by model):")
    print(by_model.to_string(index=False))
    print(f"Walk-forward accuracy leader (may differ): {wf_acc_champion}")
    print(f"Champion model (lowest WF log_loss): {champion_model_name}")
    print(f"Holdout accuracy: {acc:.3f}")
    print(f"Holdout log loss: {ll:.3f}")
    print(f"Holdout Brier: {brier:.3f}")
    print(f"Calibrator: {calibrator.method}")
    print(f"Feedback mode: {feedback_mode} | shadow_delta_brier={brier_shadow - brier_hist:+.4f}")
    if quality["quality_warning"]:
        print("Data quality warning: " + ",".join(quality["quality_warning_reasons"]))
    if drift_warning:
        print("Drift warning: " + ",".join(drift_reasons))
    print(
        "Feedback calibrator: "
        + (feedback_calibrator.method if feedback_calibrator is not None else "none")
        + f" | dataset_rows={feedback_meta.get('feedback_dataset_rows', 0)}"
        + f" | min_rows={feedback_meta.get('feedback_min_rows', _FEEDBACK_CAL_MIN_ROWS_DEFAULT)}"
    )
    print("Saved artifacts/model.joblib, artifacts/features.json, artifacts/model_metrics.json")


if __name__ == "__main__":
    main()
