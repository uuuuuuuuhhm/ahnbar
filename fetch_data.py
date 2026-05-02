import argparse
import os
import time
from typing import List

import env_bootstrap  # noqa: F401 — load `.env` (HTTP_PROXY / HTTPS_PROXY / NO_PROXY)

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

from fetch_box_scores import (
    build_box_score_lookup_for_seasons,
    merge_box_scores_onto_finder_rows,
    season_start_year_from_game_date,
)

_GAME_FINDER_COLS = [
    "SEASON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "PTS",
]


def season_string(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def infer_nba_season_start_year(reference: pd.Timestamp | None = None) -> int:
    """Calendar year that starts the NBA season label (e.g. April 2026 -> 2025 for 2025-26)."""
    t = reference if reference is not None else pd.Timestamp.now()
    return int(t.year) if t.month >= 10 else int(t.year) - 1


def _standardize_gamefinder_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _GAME_FINDER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"LeagueGameFinder result missing columns: {missing}")
    out = df[_GAME_FINDER_COLS].copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
    return out


def fetch_season_games(start_year: int) -> pd.DataFrame:
    season = season_string(start_year)
    endpoint = leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    df = endpoint.get_data_frames()[0]
    return _standardize_gamefinder_df(df)


def fetch_games_by_date_range(
    date_from: str,
    date_to: str,
    *,
    season_type_nullable: str,
) -> pd.DataFrame:
    """
    Pull team-game rows for a calendar window without season_nullable.
    Catches late-season / schedule edge cases where a fixed season_to year lagged the calendar.
    """
    endpoint = leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T",
        league_id_nullable="00",
        date_from_nullable=date_from,
        date_to_nullable=date_to,
        season_type_nullable=season_type_nullable,
    )
    df = endpoint.get_data_frames()[0]
    if df is None or df.empty:
        return pd.DataFrame(columns=_GAME_FINDER_COLS)
    return _standardize_gamefinder_df(df)


def fetch_recent_games_by_date(
    days: int = 120,
    season_types: tuple[str, ...] = ("Regular Season", "Playoffs"),
) -> pd.DataFrame:
    """Recent NBA games from the stats API using only date filters (no season string)."""
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=int(days))
    d0 = start.strftime("%Y-%m-%d")
    d1 = end.strftime("%Y-%m-%d")
    parts: List[pd.DataFrame] = []
    for st in season_types:
        try:
            chunk = fetch_games_by_date_range(d0, d1, season_type_nullable=st)
            if not chunk.empty:
                parts.append(chunk)
        except Exception:
            continue
        time.sleep(0.6)
    if not parts:
        return pd.DataFrame(columns=_GAME_FINDER_COLS)
    return pd.concat(parts, ignore_index=True)


def build_historical_games(
    season_from: int = 2021,
    season_to: int | None = None,
    *,
    recent_days: int = 120,
    include_recent_date_fetch: bool = True,
    merge_box_scores: bool = True,
    box_chunk_days: int = 14,
    box_timeout_s: int = 90,
    box_max_retries: int = 4,
    box_sleep_s: float = 0.65,
    box_sleep_jitter_s: float = 0.35,
    box_cache_dir: str | None = "data/box_cache",
    box_use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build team-game history: regular season by season id (sorted on the backend), plus a
    date-only slice of recent games (regular + playoffs) merged and de-duplicated.

    If season_to is None, it defaults to the current NBA season's start year so the file
    always reaches the ongoing season without manually bumping --season-to each year.
    """
    season_to_eff = infer_nba_season_start_year() if season_to is None else int(season_to)
    if season_to_eff < season_from:
        season_to_eff = season_from

    all_rows: List[pd.DataFrame] = []
    for year in range(season_from, season_to_eff + 1):
        print(f"Fetching season {season_string(year)}...")
        all_rows.append(fetch_season_games(year))
        time.sleep(0.7)

    if include_recent_date_fetch:
        print(
            f"Fetching recent games by date only ({recent_days}d window, no season filter) "
            "+ playoffs overlap..."
        )
        try:
            recent_df = fetch_recent_games_by_date(days=recent_days)
        except Exception as exc:
            print(f"Warning: date-only recent fetch failed ({exc}); continuing with season pulls only.")
            recent_df = pd.DataFrame(columns=_GAME_FINDER_COLS)
        if not recent_df.empty:
            all_rows.append(recent_df)
            time.sleep(0.5)

    raw = pd.concat(all_rows, ignore_index=True)
    raw = raw.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last")
    raw = raw.sort_values(
        ["SEASON_ID", "GAME_DATE", "GAME_ID", "TEAM_ID"],
        kind="mergesort",
    ).reset_index(drop=True)

    if merge_box_scores:
        print(
            "Merging LeagueGameLog box scores (date-chunked Regular Season + Playoffs; "
            "may take several minutes). Cache: "
            + ("off" if not box_use_cache or not box_cache_dir else str(box_cache_dir))
            + f" | chunk_days={box_chunk_days}"
        )
        try:
            lookup = build_box_score_lookup_for_seasons(
                range(season_from, season_to_eff + 1),
                chunk_days=box_chunk_days,
                sleep_between_calls_s=box_sleep_s,
                sleep_jitter_s=box_sleep_jitter_s,
                timeout_s=box_timeout_s,
                max_retries=box_max_retries,
                finder_df=raw,
                cache_dir=box_cache_dir,
                use_cache=box_use_cache,
            )
            raw = merge_box_scores_onto_finder_rows(raw, lookup)
            if "FGA" in raw.columns:
                n_miss = int(raw["FGA"].isna().sum())
                if n_miss:
                    print(f"Note: {n_miss} team-games missing FGA after box merge (training uses fallbacks).")
        except Exception as exc:
            print(f"Warning: box score merge failed ({exc}); continuing with finder columns only.")

    return raw


def merge_box_only_into_csv(
    csv_path: str = "data/historical_games.csv",
    *,
    box_chunk_days: int = 14,
    box_timeout_s: int = 90,
    box_max_retries: int = 4,
    box_sleep_s: float = 0.65,
    box_sleep_jitter_s: float = 0.35,
    box_cache_dir: str | None = "data/box_cache",
    box_use_cache: bool = True,
    box_cache_only: bool = False,
) -> pd.DataFrame:
    """
    Re-merge LeagueGameLog box columns into an existing Finder CSV (no LeagueGameFinder calls).
    Season years for API pulls are inferred from GAME_DATE in the file.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    raw = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])
    gd = pd.to_datetime(raw["GAME_DATE"], errors="coerce")
    years = gd.dropna().map(season_start_year_from_game_date)
    if years.empty:
        raise ValueError("No valid GAME_DATE rows to infer NBA seasons for box merge.")
    y0, y1 = int(years.min()), int(years.max())
    print(
        f"Box merge only: {csv_path} | seasons {y0}..{y1} (from GAME_DATE) | "
        f"chunk_days={box_chunk_days} | cache={box_cache_dir if box_use_cache and box_cache_dir else 'off'}"
        + (" | CACHE_ONLY (no HTTP)" if box_cache_only else "")
    )
    lookup = build_box_score_lookup_for_seasons(
        range(y0, y1 + 1),
        chunk_days=box_chunk_days,
        sleep_between_calls_s=box_sleep_s,
        sleep_jitter_s=box_sleep_jitter_s,
        timeout_s=box_timeout_s,
        max_retries=box_max_retries,
        finder_df=raw,
        cache_dir=box_cache_dir,
        use_cache=box_use_cache,
        cache_only=box_cache_only,
    )
    if lookup.empty:
        print(
            "Warning: LeagueGameLog produced no rows (timeouts, blocks, or empty responses). "
            "Box columns will be all-missing until a successful run; try again on a stable network "
            "or inspect data/box_cache/ for partial files."
        )
    else:
        print(f"LeagueGameLog lookup rows: {len(lookup)} (unique team-games)")
    out = merge_box_scores_onto_finder_rows(raw, lookup)
    if "FGA" in out.columns:
        n_miss = int(out["FGA"].isna().sum())
        if n_miss:
            print(f"Note: {n_miss} team-games still missing FGA after merge (training uses fallbacks).")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download NBA team-game rows into data/historical_games.csv. "
            "Season end defaults to the current NBA season; a recent date-range fetch "
            "(no season id) is merged so newest games resolve even if --season-to was stale."
        )
    )
    parser.add_argument("--season-from", type=int, default=2021)
    parser.add_argument(
        "--season-to",
        type=int,
        default=None,
        help="Last season start year to include (e.g. 2025 for 2025-26). "
        "Default: infer from today's date so the ongoing season is always included.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=120,
        help="Length of the date-only API window merged on top of season pulls (default 120).",
    )
    parser.add_argument(
        "--no-recent-date-boost",
        action="store_true",
        help="Skip the extra date-range fetch (season loops only).",
    )
    parser.add_argument(
        "--skip-box-merge",
        action="store_true",
        help="Do not call LeagueGameLog; CSV stays PTS/WL-only (faster, smaller).",
    )
    parser.add_argument(
        "--box-chunk-days",
        type=int,
        default=14,
        help="Calendar days per LeagueGameLog date window (smaller = smaller payloads, more calls).",
    )
    parser.add_argument(
        "--box-timeout",
        type=int,
        default=90,
        help="Read timeout in seconds for each LeagueGameLog HTTP request.",
    )
    parser.add_argument(
        "--box-max-retries",
        type=int,
        default=4,
        help="Retries per LeagueGameLog page on timeout or invalid JSON.",
    )
    parser.add_argument(
        "--box-cache-dir",
        type=str,
        default="data/box_cache",
        help="Directory for per-window CSV cache (empty string disables disk cache).",
    )
    parser.add_argument(
        "--no-box-cache",
        action="store_true",
        help="Do not read or write box-score chunk cache (always hit the API).",
    )
    parser.add_argument(
        "--merge-box-only",
        action="store_true",
        help="Skip LeagueGameFinder; only merge LeagueGameLog into existing --input-csv (Phase 1 resume).",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/historical_games.csv",
        help="Input path for --merge-box-only (default: data/historical_games.csv).",
    )
    parser.add_argument(
        "--box-cache-only",
        action="store_true",
        help="When merging box scores, only read data/box_cache (no NBA HTTP).",
    )
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    cache_dir = args.box_cache_dir.strip() or None

    if args.merge_box_only:
        out_path = args.input_csv.strip() or "data/historical_games.csv"
        df = merge_box_only_into_csv(
            out_path,
            box_chunk_days=max(1, int(args.box_chunk_days)),
            box_timeout_s=max(15, int(args.box_timeout)),
            box_max_retries=max(1, int(args.box_max_retries)),
            box_cache_dir=cache_dir,
            box_use_cache=not args.no_box_cache,
            box_cache_only=args.box_cache_only,
        )
        df.to_csv(out_path, index=False)
        gmax = df["GAME_DATE"].max()
        print(f"Saved {len(df)} rows to {out_path} (latest GAME_DATE in file: {gmax.date()})")
        try:
            from train_model import write_latest_team_state_cache

            cache_path = write_latest_team_state_cache(out_path)
            print(f"Refreshed prediction team-state cache ({cache_path}).")
        except Exception as exc:
            print(f"Note: team-state cache refresh skipped ({exc}).")
        return

    df = build_historical_games(
        season_from=args.season_from,
        season_to=args.season_to,
        recent_days=args.recent_days,
        include_recent_date_fetch=not args.no_recent_date_boost,
        merge_box_scores=not args.skip_box_merge,
        box_chunk_days=max(1, int(args.box_chunk_days)),
        box_timeout_s=max(15, int(args.box_timeout)),
        box_max_retries=max(1, int(args.box_max_retries)),
        box_cache_dir=cache_dir,
        box_use_cache=not args.no_box_cache,
    )
    out_path = "data/historical_games.csv"
    df.to_csv(out_path, index=False)
    gmax = df["GAME_DATE"].max()
    print(f"Saved {len(df)} rows to {out_path} (latest GAME_DATE in file: {gmax.date()})")

    try:
        from train_model import write_latest_team_state_cache

        cache_path = write_latest_team_state_cache(out_path)
        print(f"Refreshed prediction team-state cache ({cache_path}).")
    except Exception as exc:
        print(f"Note: team-state cache refresh skipped ({exc}).")


if __name__ == "__main__":
    main()
