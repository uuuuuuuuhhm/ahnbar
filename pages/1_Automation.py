"""
Streamlit multipage entry: background jobs and CLI equivalents.

Main app remains `app.py`; this page documents how to run `jobs/daily.py` and where
to schedule it on macOS (see `scripts/LAUNCHD.md` in the repo).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.title("Background automation")
st.caption("Schedule data refresh and log scoring outside the main dashboard.")

st.subheader("Daily pipeline (`jobs/daily.py`)")
st.markdown(
    "From the project root, with your virtualenv activated, run:\n\n"
    "- **Default** — `fetch_data.py` then `backfill_prediction_results.py`\n"
    "- **`--skip-fetch`** — only score/backfill using existing `historical_games.csv`\n"
    "- **`--retrain`** — forward `--retrain` to backfill (runs full `train_model` after)\n"
    "- **`--train-score-models`** — run `train_score_models.py` after backfill\n"
    "- **`--predict`** — run `predict_next_games.py` at the end\n"
    "- **`--fetch-extra`** — shell-style extra args for `fetch_data.py` (quoted string)\n"
)

st.code(
    "cd path/to/nba-win-predictor\n"
    "source .venv/bin/activate\n"
    "python jobs/daily.py\n"
    "python jobs/daily.py --skip-fetch\n"
    "python jobs/daily.py --retrain --train-score-models --predict\n"
    'python jobs/daily.py --fetch-extra "--season-from 2021 --recent-days 90"\n'
    "./scripts/daily_pipeline.sh --skip-fetch\n",
    language="bash",
)

st.subheader("Logs")
st.markdown(
    f"Append-only log (UTC lines): `{Path('logs/daily_pipeline.log').as_posix()}` — "
    "created automatically on first run."
)

st.subheader("macOS scheduler")
st.info(
    "Copy the plist template from **`scripts/LAUNCHD.md`**, set your username and venv Python path, "
    "then `launchctl load` it. Prefer your home network for `stats.nba.com` fetches."
)

st.subheader("Play-by-play spike (optional)")
st.markdown(
    "One-off API + Parquet check (not used by the win model yet):\n\n"
    "`python jobs/pbp_spike.py [--game-id 0042500176]`"
)
