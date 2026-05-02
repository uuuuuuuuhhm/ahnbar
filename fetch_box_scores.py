"""
Fetch team box-score columns via LeagueGameLog and merge onto LeagueGameFinder rows.

Uses GAME_ID + TEAM_ID as keys. Prefers small date-window requests (with Counter pagination
per window) to avoid huge full-season JSON payloads that often time out.
"""

from __future__ import annotations

import env_bootstrap  # noqa: F401 — load `.env` (HTTP_PROXY / HTTPS_PROXY / NO_PROXY)

import json
import os
import random
import re
import time
import warnings
from typing import Iterable

import pandas as pd
import requests

# Typical max rows per leaguegamelog response (NBA stats API).
NBA_STATS_PAGE_SIZE = 1000

def normalize_nba_game_id(raw: object) -> str:
    """
    Zero-pad stats.nba.com game ids to length 10.
    Finder CSV sometimes stores GAME_ID without leading zeros (e.g. 22100001) while
    LeagueGameLog uses 0022100001; merge fails without normalizing.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    try:
        n = int(s, 10)
    except ValueError:
        return s
    if abs(n) > 9999999999:
        return str(n)
    return f"{n:010d}"


_BOX_MERGE_COLS = [
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "PF",
    "MIN",
    "PLUS_MINUS",
]


def _season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_start_year_from_game_date(ts: pd.Timestamp) -> int:
    """NBA season label start year: Oct–Dec -> that calendar year; Jan–Sep -> prior year."""
    y, m = int(ts.year), int(ts.month)
    return y if m >= 10 else y - 1


def _nba_season_calendar_bounds(start_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Rough league calendar for season starting in start_year (e.g. 2024 -> 2024-10-01 .. 2025-06-30)."""
    lo = pd.Timestamp(year=start_year, month=10, day=1)
    hi = pd.Timestamp(year=start_year + 1, month=6, day=30)
    return lo, hi


def iter_date_windows(
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    chunk_days: int,
) -> list[tuple[str, str]]:
    """Non-overlapping inclusive windows [d0, d1] as YYYY-MM-DD strings."""
    if lo > hi:
        return []
    chunk_days = max(1, int(chunk_days))
    out: list[tuple[str, str]] = []
    cur = lo.normalize()
    end = hi.normalize()
    while cur <= end:
        nxt = cur + pd.Timedelta(days=chunk_days - 1)
        if nxt > end:
            nxt = end
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + pd.Timedelta(days=1)
    return out


def _finder_date_bounds_for_season(finder_df: pd.DataFrame | None, start_year: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Narrow Oct–Jun bounds to min/max GAME_DATE in finder for this season, if available."""
    if finder_df is None or finder_df.empty or "GAME_DATE" not in finder_df.columns:
        return None
    gd = pd.to_datetime(finder_df["GAME_DATE"], errors="coerce")
    mask = gd.notna() & (gd.map(season_start_year_from_game_date) == start_year)
    if not mask.any():
        return None
    sub = gd.loc[mask]
    lo = sub.min().normalize()
    hi = sub.max().normalize()
    cal_lo, cal_hi = _nba_season_calendar_bounds(start_year)
    lo = max(lo, cal_lo)
    hi = min(hi, cal_hi)
    if lo > hi:
        return None
    return lo, hi


def _sanitize_cache_stem(season_type: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "", season_type.replace(" ", ""))


def _cache_path(cache_dir: str, season: str, season_type: str, d0: str, d1: str) -> str:
    stem = _sanitize_cache_stem(season_type)
    name = f"{season}_{stem}_{d0}_{d1}.csv"
    return os.path.join(cache_dir, name)


def _load_cache_csv(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path) or os.path.getsize(path) < 10:
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def _save_cache_csv(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def _league_game_log_page(
    *,
    counter: int,
    season: str,
    season_type: str,
    timeout_s: int,
    max_retries: int,
    date_from: str | None = None,
    date_to: str | None = None,
):
    from nba_api.stats.endpoints import leaguegamelog

    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                kwargs = dict(
                    counter=counter,
                    season=season,
                    season_type_all_star=season_type,
                    player_or_team_abbreviation="T",
                    league_id="00",
                    timeout=timeout_s,
                )
                if date_from is not None and date_to is not None:
                    kwargs["date_from_nullable"] = date_from
                    kwargs["date_to_nullable"] = date_to
                endpoint = leaguegamelog.LeagueGameLog(**kwargs)
            return endpoint
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_exc = exc
            time.sleep(min(30.0, 3.0 * (2**attempt)))
    assert last_exc is not None
    raise last_exc


def fetch_league_team_game_logs_date_range(
    season: str,
    season_type: str,
    date_from: str,
    date_to: str,
    *,
    sleep_s: float = 0.65,
    timeout_s: int = 90,
    max_retries: int = 4,
) -> pd.DataFrame:
    """
    All team game log rows for season + season type intersecting [date_from, date_to] (paginated).
    Dates must be YYYY-MM-DD.
    """
    parts: list[pd.DataFrame] = []
    counter = 0
    while True:
        endpoint = _league_game_log_page(
            counter=counter,
            season=season,
            season_type=season_type,
            timeout_s=timeout_s,
            max_retries=max_retries,
            date_from=date_from,
            date_to=date_to,
        )
        frames = endpoint.get_data_frames()
        df = frames[0] if frames else None
        if df is None or df.empty:
            break
        parts.append(df.copy())
        if len(df) < NBA_STATS_PAGE_SIZE:
            break
        counter += len(df)
        time.sleep(sleep_s)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last")
    return out


def fetch_league_team_game_logs(
    season: str,
    season_type: str,
    *,
    sleep_s: float = 0.65,
    timeout_s: int = 90,
    max_retries: int = 4,
    chunk_days: int = 14,
) -> pd.DataFrame:
    """Full league year (Oct–Jun) via date windows; avoids one monolithic season request."""
    start_year = int(str(season).split("-")[0])
    lo, hi = _nba_season_calendar_bounds(start_year)
    parts: list[pd.DataFrame] = []
    for d0, d1 in iter_date_windows(lo, hi, chunk_days):
        chunk = fetch_league_team_game_logs_date_range(
            season,
            season_type,
            d0,
            d1,
            sleep_s=sleep_s,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        if not chunk.empty:
            parts.append(chunk)
        jitter = random.uniform(0.0, 0.35)
        time.sleep(sleep_s + jitter)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last")


def _standardize_log_for_merge(log: pd.DataFrame) -> pd.DataFrame:
    need = ["GAME_ID", "TEAM_ID"] + [c for c in _BOX_MERGE_COLS if c in log.columns]
    missing = [c for c in ["GAME_ID", "TEAM_ID"] if c not in log.columns]
    if missing:
        raise ValueError(f"LeagueGameLog missing required columns: {missing}")
    sub = log[need].copy()
    sub["GAME_ID"] = sub["GAME_ID"].map(normalize_nba_game_id)
    sub["TEAM_ID"] = pd.to_numeric(sub["TEAM_ID"], errors="coerce").astype("Int64")
    for c in _BOX_MERGE_COLS:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub


def build_box_score_lookup_chunked(
    season_years: Iterable[int],
    *,
    chunk_days: int = 14,
    sleep_between_calls_s: float = 0.65,
    sleep_jitter_s: float = 0.35,
    timeout_s: int = 90,
    max_retries: int = 4,
    finder_df: pd.DataFrame | None = None,
    cache_dir: str | None = "data/box_cache",
    use_cache: bool = True,
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    LeagueGameLog in small date windows per (season label, Regular Season | Playoffs).
    Optionally narrows windows to GAME_DATE range present in finder_df.
    Caches each successful window CSV under cache_dir when use_cache is True.
    If cache_only is True, skip network completely; missing windows are omitted.
    """
    frames: list[pd.DataFrame] = []
    chunk_days = max(1, int(chunk_days))

    for y in season_years:
        start_year = int(y)
        season = _season_label(start_year)
        bounds = _finder_date_bounds_for_season(finder_df, start_year)
        if bounds is None:
            bounds = _nba_season_calendar_bounds(start_year)
        lo, hi = bounds

        for st in ("Regular Season", "Playoffs"):
            windows = iter_date_windows(lo, hi, chunk_days)
            for d0, d1 in windows:
                cache_path = None
                if use_cache and cache_dir:
                    cache_path = _cache_path(cache_dir, season, st, d0, d1)
                    cached = _load_cache_csv(cache_path)
                    if cached is not None:
                        frames.append(cached)
                        continue

                if cache_only:
                    continue

                try:
                    chunk = fetch_league_team_game_logs_date_range(
                        season,
                        st,
                        d0,
                        d1,
                        sleep_s=sleep_between_calls_s,
                        timeout_s=timeout_s,
                        max_retries=max_retries,
                    )
                except Exception as exc:
                    print(
                        f"Warning: LeagueGameLog failed season={season} type={st} "
                        f"window={d0}..{d1}: {exc}"
                    )
                    time.sleep(sleep_between_calls_s)
                    continue

                if not chunk.empty and cache_path:
                    _save_cache_csv(cache_path, chunk)

                if not chunk.empty:
                    frames.append(chunk)

                jitter = random.uniform(0.0, sleep_jitter_s) if sleep_jitter_s > 0 else 0.0
                time.sleep(sleep_between_calls_s + jitter)

    if not frames:
        return pd.DataFrame(columns=["GAME_ID", "TEAM_ID"] + _BOX_MERGE_COLS)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last")
    return _standardize_log_for_merge(raw)


def build_box_score_lookup_for_seasons(
    season_years: Iterable[int],
    *,
    sleep_between_calls_s: float = 0.65,
    **kwargs,
) -> pd.DataFrame:
    """Backward-compatible name: delegates to chunked fetch with defaults."""
    return build_box_score_lookup_chunked(
        season_years,
        sleep_between_calls_s=sleep_between_calls_s,
        **kwargs,
    )


def merge_box_scores_onto_finder_rows(
    finder_df: pd.DataFrame,
    box_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge box columns onto gamefinder-style rows (GAME_ID, TEAM_ID)."""
    out = finder_df.copy()
    out["GAME_ID"] = out["GAME_ID"].map(normalize_nba_game_id)
    out["TEAM_ID"] = pd.to_numeric(out["TEAM_ID"], errors="coerce").astype("Int64")

    if box_lookup.empty:
        for c in _BOX_MERGE_COLS:
            out[c] = pd.NA
        return out

    # Finder CSV may carry empty placeholders from prior merges; pandas would suffix RHS (e.g. FGA_log).
    overlap = [c for c in _BOX_MERGE_COLS if c in out.columns]
    if overlap:
        out = out.drop(columns=overlap, errors="ignore")

    merge_cols = ["GAME_ID", "TEAM_ID"] + [c for c in _BOX_MERGE_COLS if c in box_lookup.columns]
    box = box_lookup[merge_cols].drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last").copy()
    box["GAME_ID"] = box["GAME_ID"].map(normalize_nba_game_id)
    box["TEAM_ID"] = pd.to_numeric(box["TEAM_ID"], errors="coerce").astype("Int64")
    out = out.merge(box, on=["GAME_ID", "TEAM_ID"], how="left", suffixes=("", "_log"))
    for c in list(out.columns):
        if c.endswith("_log") and c.replace("_log", "") in finder_df.columns:
            out = out.drop(columns=[c])
    return out
