"""
Streamlit multipage entry: background jobs and CLI equivalents.

Main app remains `app.py`; this page documents how to run `jobs/daily.py` and where
to schedule it on macOS (see `scripts/LAUNCHD.md` and `scripts/launchd/*.example`).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.title("Background automation")
st.caption("Schedule data refresh and log scoring outside the main dashboard.")

st.markdown(
    "Full roadmap (completed sprint + backlog): `docs/PROJECT_PIPELINE_AND_ROADMAP.md`"
)

st.subheader("Daily pipeline (`jobs/daily.py`)")
st.markdown(
    "From the project root, with your virtualenv activated, run:\n\n"
    "- **Default** — `fetch_data.py` then `backfill_prediction_results.py`\n"
    "- **`--skip-fetch`** — only score/backfill using existing `historical_games.csv`\n"
    "- **`--retrain`** — forward `--retrain` to backfill (runs full `train_model` after)\n"
    "- **`--patch-feedback`** — after backfill, patch feedback calibrator into `artifacts/model.joblib` "
    "(skipped automatically if `--retrain` is set)\n"
    "- **`--train-score-models`** — run `train_score_models.py` after backfill\n"
    "- **`--predict`** — run `predict_next_games.py` at the end\n"
    "- **`--predict-value`** / **`--value`** — run `predict_value_plays.py` (needs `ODDS_API_KEY`; skipped if unset)\n"
    "- **`--profile nightly`** — same as `--patch-feedback --predict --predict-value` "
    "(fetch unless `--skip-fetch`)\n"
    "- **`--fetch-extra`** — shell-style extra args for `fetch_data.py` (quoted string)\n"
)

st.subheader("Recommended schedules")
st.markdown(
    "- **Typical night:** `python jobs/daily.py --profile nightly` "
    "or `./scripts/daily_pipeline.sh --profile nightly`\n"
    "- **After manual fetch:** `python jobs/daily.py --skip-fetch --profile nightly`\n"
    "- **Weekly full retrain:** add **`--retrain`** (e.g. `python jobs/daily.py --skip-fetch --retrain --predict`); "
    "`--patch-feedback` is redundant with `--retrain`.\n"
)

st.code(
    "cd path/to/nba-win-predictor\n"
    "source .venv/bin/activate\n"
    "python jobs/daily.py\n"
    "python jobs/daily.py --skip-fetch\n"
    "python jobs/daily.py --profile nightly\n"
    "python jobs/daily.py --retrain --train-score-models --predict\n"
    'python jobs/daily.py --fetch-extra "--season-from 2021 --recent-days 90"\n'
    "./scripts/daily_pipeline.sh --skip-fetch\n"
    "./scripts/daily_pipeline.sh --profile nightly\n",
    language="bash",
)

st.subheader("Logs")
st.markdown(
    f"Append-only log (UTC lines): `{Path('logs/daily_pipeline.log').as_posix()}` — "
    "created automatically on first run."
)

st.subheader("GitHub Actions & editor tasks")
st.markdown(
    "- **GitHub:** `.github/workflows/daily-pipeline.yml` — "
    "Actions tab → *Daily pipeline* → *Run workflow*; optional repo secret **`ODDS_API_KEY`**. "
    "Scheduled runs use **`--skip-fetch --profile nightly`** (see README §4b).\n"
    "- **Cursor / VS Code:** Command Palette → **Tasks: Run Task** → *Daily pipeline …* "
    "(`.vscode/tasks.json`); requires **`.venv`**."
)

st.subheader("macOS scheduler")
st.info(
    "Copy **`scripts/launchd/com.nba-win-predictor.daily.plist.example`** (or the template in "
    "`scripts/LAUNCHD.md`), set your username and venv Python path, "
    "then `launchctl load` it. Prefer your home network for `stats.nba.com` fetches. "
    "Set **`ODDS_API_KEY`** in the plist `EnvironmentVariables` dict if you use `--profile nightly`."
)

st.subheader("Play-by-play spike (optional)")
st.markdown(
    "One-off API + Parquet check (not used by the win model yet):\n\n"
    "`python jobs/pbp_spike.py [--game-id 0042500176]`"
)
