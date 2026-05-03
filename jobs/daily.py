#!/usr/bin/env python3
"""
Background daily pipeline: refresh NBA history, score the predictions log, optional
full retrain / score models / CLI predictions / feedback patch / value plays.

Run from project root (or via scripts/daily_pipeline.sh):

    source .venv/bin/activate
    python jobs/daily.py
    python jobs/daily.py --skip-fetch --retrain
    python jobs/daily.py --profile nightly
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import env_bootstrap  # noqa: F401 — load `.env` before checking ODDS_API_KEY

PROJECT_ROOT = _PROJECT_ROOT
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


def apply_nightly_profile(args: argparse.Namespace) -> None:
    """Mutate args for --profile nightly (fetch + backfill + patch + predict + value)."""
    args.patch_feedback = True
    args.predict = True
    args.predict_value = True


def prepare_implicit_predict(args: argparse.Namespace, log_path: Path) -> None:
    """Value plays need fresh predictions; enable --predict when only value was requested."""
    if args.predict_value and not args.predict:
        args.predict = True
        _append_log(
            log_path,
            f"{_utc_ts()} NOTE implicit --predict enabled because --predict-value was set",
        )


def should_skip_patch_feedback(retrain: bool, patch_requested: bool) -> tuple[bool, str]:
    if not patch_requested:
        return True, "not_requested"
    if retrain:
        return True, "redundant_with_retrain"
    return False, ""


def run_feedback_patch_step() -> dict:
    """Refit feedback calibrator into saved bundle (for tests, patch this function)."""
    import train_model

    return train_model.patch_feedback_calibrator_into_artifacts()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate fetch_data → backfill; optional --retrain, score models, predict, value plays, feedback patch."
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
        "--predict-value",
        "--value",
        action="store_true",
        dest="predict_value",
        help="Run predict_value_plays.py after predictions (needs ODDS_API_KEY; skipped if unset).",
    )
    parser.add_argument(
        "--patch-feedback",
        action="store_true",
        help="After backfill, refit feedback calibrator into artifacts/model.joblib (skipped if --retrain).",
    )
    parser.add_argument(
        "--profile",
        choices=["nightly"],
        default=None,
        help="Preset: nightly = --patch-feedback --predict --predict-value (fetch unless --skip-fetch).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=f"Log path (default: {DEFAULT_LOG}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    os.chdir(PROJECT_ROOT)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    log_path = Path(args.log_file) if args.log_file else DEFAULT_LOG

    if args.profile == "nightly":
        apply_nightly_profile(args)

    _append_log(log_path, f"{_utc_ts()} === daily pipeline begin pid={os.getpid()} ===")
    prepare_implicit_predict(args, log_path)

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

    skip_patch, skip_reason = should_skip_patch_feedback(args.retrain, args.patch_feedback)
    if args.patch_feedback:
        if skip_patch:
            _append_log(
                log_path,
                f"{_utc_ts()} SKIP patch_feedback_calibrator ({skip_reason})",
            )
        else:
            _append_log(log_path, f"{_utc_ts()} START patch_feedback_calibrator (in-process)")
            try:
                detail = run_feedback_patch_step()
                patched = bool(detail.get("patched"))
                _append_log(
                    log_path,
                    f"{_utc_ts()} END patch_feedback_calibrator patched={patched} detail={detail!r}",
                )
            except Exception as exc:
                _append_log(log_path, f"{_utc_ts()} ERROR patch_feedback_calibrator {exc!r}")
                return 1

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

    if args.predict_value:
        if not os.environ.get("ODDS_API_KEY"):
            _append_log(
                log_path,
                f"{_utc_ts()} SKIP predict_value_plays.py (ODDS_API_KEY unset)",
            )
        else:
            rc = _run_script("predict_value_plays.py", [], log_path)
            if rc != 0:
                _append_log(log_path, f"{_utc_ts()} === daily pipeline abort after predict_value_plays ===")
                return rc

    _append_log(log_path, f"{_utc_ts()} === daily pipeline complete ===")
    print(f"[daily] OK — log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
