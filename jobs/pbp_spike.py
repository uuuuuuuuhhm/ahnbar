#!/usr/bin/env python3
"""
Spike: fetch one game play-by-play via nba_api, save Parquet, print a few derived counts.

Not wired into train_model — validates storage + API access for a future tactics feature.

Usage:
    source .venv/bin/activate
    python jobs/pbp_spike.py
    python jobs/pbp_spike.py --game-id 0042500176

Requires: pyarrow (see requirements.txt) for Parquet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import env_bootstrap  # noqa: F401 — load `.env` before nba_api / requests

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PBP_DIR = PROJECT_ROOT / "data" / "pbp_spike"


def _latest_game_id_from_history(hist_csv: Path) -> str:
    df = pd.read_csv(hist_csv, usecols=["GAME_ID"], dtype={"GAME_ID": str})
    if df.empty:
        raise SystemExit("historical_games.csv has no GAME_ID")
    last = str(df["GAME_ID"].iloc[-1]).strip()
    if last.endswith(".0") and last[:-2].isdigit():
        last = last[:-2]
    return f"{int(last, 10):010d}" if last.isdigit() else last


def _derive_counts(pbp: pd.DataFrame) -> dict[str, float | int]:
    """Lightweight proxies from EVENTMSGTYPE (NBA stats convention)."""
    if pbp.empty or "EVENTMSGTYPE" not in pbp.columns:
        return {"n_events": len(pbp), "n_made_fg": 0, "n_turnovers": 0, "n_fta_events": 0}
    et = pd.to_numeric(pbp["EVENTMSGTYPE"], errors="coerce").fillna(-1).astype(int)
    # 1 = FIELD_GOAL_MADE, 2 = FIELD_GOAL_MISSED, 3 = FREE_THROW, 5 = TURNOVER, 6 = FOUL (varies)
    n_made_fg = int((et == 1).sum())
    n_tov = int((et == 5).sum())
    n_fta = int((et == 3).sum())
    return {
        "n_events": int(len(pbp)),
        "n_made_fg": n_made_fg,
        "n_turnovers": n_tov,
        "n_fta_events": n_fta,
    }


def main() -> int:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Fetch one NBA play-by-play and save Parquet + summary.")
    parser.add_argument(
        "--game-id",
        default="",
        help="10-digit NBA GAME_ID (default: last id in data/historical_games.csv).",
    )
    parser.add_argument(
        "--hist-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "historical_games.csv",
        help="CSV to pick default GAME_ID tail from.",
    )
    args = parser.parse_args()

    hist = Path(args.hist_csv)
    if not hist.exists():
        print(f"Missing {hist}", file=sys.stderr)
        return 1

    game_id = (args.game_id or "").strip() or _latest_game_id_from_history(hist)
    if not game_id.isdigit():
        print(f"Bad game id: {game_id!r}", file=sys.stderr)
        return 2
    game_id = f"{int(game_id, 10):010d}"

    from nba_api.stats.endpoints.playbyplayv2 import PlayByPlayV2

    print(f"Fetching PlayByPlayV2 for GAME_ID={game_id} ...")
    raw = PlayByPlayV2(game_id=game_id)
    frames = raw.get_data_frames()
    pbp = frames[0] if frames else pd.DataFrame()
    if pbp is None or pbp.empty:
        print("Empty play-by-play response.", file=sys.stderr)
        return 3

    PBP_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = PBP_DIR / f"pbp_{game_id}.parquet"
    pbp.to_parquet(out_parquet, index=False)

    derived = _derive_counts(pbp)
    summary_path = PBP_DIR / "pbp_derived_summary.parquet"
    row = pd.DataFrame([{"GAME_ID": game_id, **derived}])
    if summary_path.exists():
        old = pd.read_parquet(summary_path)
        row = pd.concat([old, row], ignore_index=True).drop_duplicates(subset=["GAME_ID"], keep="last")
    row.to_parquet(summary_path, index=False)

    print(f"Wrote {out_parquet} rows={len(pbp)}")
    print("Derived:", derived)
    print(f"Summary appended/merged: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
