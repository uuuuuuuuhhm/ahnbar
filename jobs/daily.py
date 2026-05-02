#!/usr/bin/env python3
"""
Background daily pipeline: refresh NBA history, score the predictions log, optional
full retrain / score models / CLI predictions.

Run from project root (or via scripts/daily_pipeline.sh):

    source .venv/bin/activate
    python jobs/daily.py
    python jobs/daily.py --skip-fetch --retrain
    python jobs/daily.py --fetch-extra "--season-from 2021 --recent-days 90"

Logs append to logs/daily_pipeline.log (UTC timestamps).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG = LOG_DIR / "daily_pipeline.log"


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run_script(rel_script: str, argv: list[str], log_path: Path) -> int:
    script_path = PROJECT_ROOT / rel_script
    cmd = [sys.executable, str(script_path)] + argv
    _append_log(log_path, f"{_utc_ts()} START {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    combined = (proc.stdout or "").strip() + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "").strip()
    if combined:
        for line in combined.splitlines():
            _append_log(log_path, f"{_utc_ts()} | {line}")
    _append_log(log_path, f"{_utc_ts()} END exit={proc.returncode}")
    if proc.returncode != 0:
        print(f"[daily] ERROR {rel_script} exit {proc.returncode}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
    return int(proc.returncode)


def main() -> int:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(
        description="Orchestrate fetch_data → backfill; optional --retrain, score models, predict_next_games."
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetch_data.py.")
    parser.add_argument(
        "--fetch-extra",
        default="",
        help="Extra arguments for fetch_data.py, shell-quoted (e.g. '\"--season-from\" \"2021\"').",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Pass --retrain to backfill_prediction_results.py (runs train_model after scoring).",
    )
    parser.add_argument(
        "--train-score-models",
        action="store_true",
        help="Run train_score_models.py after backfill.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run predict_next_games.py after prior steps.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=f"Log path (default: {DEFAULT_LOG}).",
    )
    args = parser.parse_args()
    log_path = Path(args.log_file) if args.log_file else DEFAULT_LOG

    _append_log(log_path, f"{_utc_ts()} === daily pipeline begin pid={os.getpid()} ===")

    if not args.skip_fetch:
        fetch_argv: list[str] = []
        if args.fetch_extra.strip():
            try:
                fetch_argv = shlex.split(args.fetch_extra)
            except ValueError as exc:
                print(f"[daily] Bad --fetch-extra: {exc}", file=sys.stderr)
                return 2
        rc = _run_script("fetch_data.py", fetch_argv, log_path)
        if rc != 0:
            _append_log(log_path, f"{_utc_ts()} === daily pipeline abort after fetch_data ===")
            return rc
    else:
        _append_log(log_path, f"{_utc_ts()} SKIP fetch_data.py")

    backfill_argv: list[str] = []
    if args.retrain:
        backfill_argv.append("--retrain")
    rc = _run_script("backfill_prediction_results.py", backfill_argv, log_path)
    if rc != 0:
        _append_log(log_path, f"{_utc_ts()} === daily pipeline abort after backfill ===")
        return rc

    if args.train_score_models:
        rc = _run_script("train_score_models.py", [], log_path)
        if rc != 0:
            _append_log(log_path, f"{_utc_ts()} === daily pipeline abort after train_score_models ===")
            return rc

    if args.predict:
        rc = _run_script("predict_next_games.py", [], log_path)
        if rc != 0:
            _append_log(log_path, f"{_utc_ts()} === daily pipeline abort after predict_next_games ===")
            return rc

    _append_log(log_path, f"{_utc_ts()} === daily pipeline complete ===")
    print(f"[daily] OK — log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
