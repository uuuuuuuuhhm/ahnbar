# NBA Win Predictor (Beginner Friendly)

This project predicts **who wins an NBA game** and gives a probability (for example: `Home team win chance: 63%`). After training optional score models, it can also predict **home and away point totals** for upcoming games.

## Why this setup

- Core scope is win/loss plus probability; point totals are optional (`train_score_models.py`).
- We use `nba_api`, which is free to start with.
- We begin with a basic model (logistic regression), then improve later.

## Project files

- `fetch_data.py` -> downloads historical games into `data/historical_games.csv`
- `train_model.py` -> creates features and trains the win-probability model
- `train_score_models.py` -> trains regressors for home/away points (`artifacts/score_models.joblib`)
- `evaluate_model.py` -> evaluates model with a time-based split (better realism)
- `predict_next_games.py` -> predicts the next upcoming games
- `predict_today.py` -> compatibility wrapper (calls `predict_next_games.py`)
- `predict_value_plays.py` -> compares model probabilities vs live odds to find value bets
- `retrain_from_feedback.py` -> refreshes results, builds prediction feedback dataset, retrains model
- `score_predictions.py` -> compares logged predictions to actual game results
- `app.py` -> simple Streamlit GUI prototype
- `value_betting.py` -> fair odds, edge/EV, and Kelly stake sizing utilities
- `odds_provider.py` -> The Odds API fetch + normalized moneyline odds schema
- `env_bootstrap.py` -> loads `.env` (Odds API key + proxy vars) for network scripts
- `requirements.txt` -> Python packages

## 1) Setup

```bash
cd "/Users/prada4k/Documents/nba-win-predictor"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Environment / proxy (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`):** Copy [`.env.example`](.env.example) to `.env` and uncomment or add values. [`env_bootstrap.py`](env_bootstrap.py) loads `.env` at import time into `process` environment (via `python-dotenv`, **`override=False`** so exports in your shell still win). It is imported by [`fetch_data.py`](fetch_data.py), [`predict_next_games.py`](predict_next_games.py), [`fetch_box_scores.py`](fetch_box_scores.py), and [`odds_provider.py`](odds_provider.py), so **`streamlit run app.py`** picks up proxies when predictions or value bets hit the network. For **`streamlit run`**, you can alternatively copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) to `.streamlit/secrets.toml`; [`app.py`](app.py) copies those proxy keys into the process environment **before** `env_bootstrap` runs (still no built-in proxy URL—you set the hostname). Prefer **`HTTPS_PROXY=http://host:port`** (same scheme as curl). If a corporate tunnel returns **403** for `stats.nba.com`, try **`NO_PROXY=stats.nba.com,.nba.com`** or temporarily unset proxies for that terminal.

## 2) Download historical data

```bash
python fetch_data.py
```

`fetch_data.py` walks regular seasons from `--season-from` (default 2021) through the **current NBA season** inferred from the clock, sorts rows by `SEASON_ID` / `GAME_DATE`, and merges an extra **date-only** LeagueGameFinder window (`--recent-days`, default 120) so very new games still appear even if a manual `--season-to` used to lag. Use `--no-recent-date-boost` to disable that merge.

```bash
python fetch_data.py --season-from 2021 --season-to 2024
```

This creates `data/historical_games.csv`.

**Box scores (LeagueGameLog):** By default, `fetch_data.py` merges team box columns from `LeagueGameLog` (Regular Season and Playoffs) so training can use rolling **four-factor-style** signals. Fetches use **small calendar windows** (default **14 days** per request) instead of one full-season payload, which avoids many timeouts. Date ranges are tightened when possible to the **min/max `GAME_DATE`** already present in the Finder output for that NBA season. Successful windows are cached as CSV under **`data/box_cache/`** so interrupted runs can resume without re-downloading those slices (**`--no-box-cache`** disables read/write; **`--box-cache-dir ""`** disables cache path). Tunables: **`--box-chunk-days`**, **`--box-timeout`**, **`--box-max-retries`**. Run **`fetch_data.py` on your own machine** when possible; cloud and datacenter IPs often get empty or slow responses from `stats.nba.com`. For **`HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`**, configure **`.env`** (see Setup). Use **`--skip-box-merge`** for a faster PTS/WL-only CSV (training falls back to neutral placeholders for box-derived features). To **re-run only the box merge** on an existing `historical_games.csv` (no Finder re-download), use **`python fetch_data.py --merge-box-only`** (optional **`--input-csv path`**). Use **`python fetch_data.py --merge-box-only --box-cache-only`** to join from **`data/box_cache/`** only (no NBA HTTP), e.g. after fixing merge logic or avoiding slow calls.

## 3) Train the model

```bash
python train_model.py
```

The win model uses rolling win rate, a points-based “net rating” proxy, margin volatility, rest and schedule flags, Elo, and—when box columns are present—rolling **eFG%, turnover rate, offensive rebounding on missed shots, FTA/FGA, and a possession proxy** (same windows: 3 / 5 / 10 / 20 games).

This creates:
- `artifacts/model.joblib`
- `artifacts/features.json`
- `artifacts/model_metrics.json` (includes `hist_gb_hyperparams` when the champion uses HistGradientBoosting)

### 3b) Train point-total models (optional)

Uses the same feature columns as the win model and game-level `PTS_HOME` / `PTS_AWAY` from history:

```bash
python train_score_models.py
```

Writes `artifacts/score_models.joblib` and `artifacts/score_model_metrics.json`. If these files are missing, prediction scripts only output win probability.

## 4) Predict upcoming games

```bash
python predict_next_games.py
```

Optional flags:

```bash
python predict_next_games.py --count 5 --days-ahead 14
```

Output fields include:
- `matchup`
- `game_date`
- `start_local` (printed only): tip-off in your display timezone (see below)
- `home_win_probability`
- `rationale`: short template sentences from rolling form (eFG, turnovers, glass, pace, rest, Elo tilt)
- `pred_home_pts`, `pred_away_pts` (when `artifacts/score_models.joblib` exists)

**Tip-off times:** the log stores **`game_start_time_utc`** (canonical ISO-8601 UTC from the NBA scoreboard, e.g. `gameTimeUTC`). The CLI prints **`start_local`** derived from that field. Set the standard **`TZ`** environment variable to any valid IANA name (e.g. `TZ=America/Toronto`) to choose the print timezone; if `TZ` is unset or invalid, **`Europe/Berlin`** is used.

Each run also appends results to:
- `data/predictions_log.csv` (includes `game_start_time_utc`; older rows may still have `game_start_time_24h` for display fallback only)

### Streamlit app

In the sidebar, pick a **display timezone** from the IANA dropdown. Tip-off in tables is shown as **Start (local)** from `game_start_time_utc`; legacy **`game_start_time_24h`** is used only when UTC is missing. Default selection is **`Europe/Berlin`**.

The sidebar also lists a separate Streamlit page **`Automation`** (`pages/1_Automation.py`) with the same background-job commands documented below.

## 4b) Background automation (fetch, score log, optional train)

Use a single orchestrator so history and `scored_predictions` stay current without opening the UI:

```bash
cd "/Users/prada4k/Documents/nba-win-predictor"
source .venv/bin/activate
python jobs/daily.py
```

Flags:

- `--skip-fetch` — only run `backfill_prediction_results.py` on existing `data/historical_games.csv`
- `--retrain` — pass `--retrain` to backfill (runs full `train_model` after scoring)
- `--train-score-models` — run `train_score_models.py` after backfill
- `--predict` — run `predict_next_games.py` at the end
- `--fetch-extra "…"` — extra shell-quoted arguments forwarded to `fetch_data.py`

Shell wrapper (same flags):

```bash
./scripts/daily_pipeline.sh --skip-fetch
```

Logs append to **`logs/daily_pipeline.log`** (UTC lines).

**macOS schedule:** see **[`scripts/LAUNCHD.md`](scripts/LAUNCHD.md)** for a `launchd` plist template (`ProgramArguments` should use your **venv `python`** absolute path and `jobs/daily.py`).

**Play-by-play spike (optional, not used by the win model yet):**

```bash
python jobs/pbp_spike.py
python jobs/pbp_spike.py --game-id 0042500176
```

Writes Parquet under `data/pbp_spike/` (ignored by git except when you remove the ignore rule).

## 5) Evaluate model quality

```bash
python evaluate_model.py
```

This prints:
- Accuracy
- Log loss
- Brier score

using a time-based split (first 80% games for training, last 20% for testing).

## 6) Score logged predictions vs real outcomes

```bash
python score_predictions.py
```

This script:
- reads latest logged prediction per game from `data/predictions_log.csv`
- compares it against completed game outcomes in `data/historical_games.csv`
- prints overall accuracy and accuracy by confidence bucket (`low`, `medium`, `high`)
- writes `data/scored_predictions.csv` (deduplicated per game, sorted by `season_week`; may include optional point columns when the log has predictions and history has box scores)
- writes `data/weekly_summary.csv` (one row per week with overall + bucket accuracy)
- when there are pending merges: `data/pending_predictions_diagnostic.csv` (game dates vs max `GAME_DATE` in history)
- when there are resolved games: `data/scoring_reliability_bins.csv` (decile reliability and ECE contributions)
- when any `*pts*` columns exist: `data/scored_score_predictions.csv` (sidecar for point preds vs actuals)

Team names in the log are normalized to abbreviations before joining historical results (see `team_aliases.py`).

### Backfill all logged predictions vs history

Use this when you already have a large `predictions_log.csv` and want `scored_predictions.csv` plus `prediction_feedback_training.csv` filled from `historical_games.csv` (refresh history first so dates overlap).

```bash
python backfill_prediction_results.py
python backfill_prediction_results.py --retrain   # also runs train_model.main()
```

## 7) Run the GUI prototype

```bash
streamlit run app.py
```

The GUI lets you:
- run upcoming-game predictions
- evaluate model quality
- score logged predictions
- run Value Betting analysis with bankroll and edge/Kelly controls

## 8) Value Betting Analyzer (CLI)

Create a local env file:

```bash
cp .env.example .env
```

Open `.env` and set:

```bash
ODDS_API_KEY=your_rotated_real_api_key_here
```

`odds_provider.py` auto-loads `.env`, so you do not need to run `export` each time.

Generate recommended value plays:

```bash
python predict_value_plays.py --bankroll 1000 --edge-threshold 3 --kelly-multiplier 0.25 --max-stake-pct 5
```

Output includes:
- matchup + side (`home`/`away`)
- model win %
- fair odds (decimal/american/fractional)
- market decimal odds + bookmaker
- edge %
- EV per unit stake
- Kelly stake % and suggested stake amount

Recommendations are appended to:
- `data/value_recommendations.csv`

If no plays are found, CLI/UI prints diagnostics counters to show where rows dropped:
- predictions loaded
- raw odds fetched
- best-book odds rows
- merged matchup rows
- EV-positive rows

## 9) Refresh past prediction outcomes and retrain

Run the full loop:

```bash
python retrain_from_feedback.py --season-from 2021
```

This does:
- refresh `data/historical_games.csv`
- score previous predictions (`data/scored_predictions.csv`)
- build `data/prediction_feedback_training.csv`
- retrain model artifacts in `artifacts/`

Optional (if historical data is already up to date):

```bash
python retrain_from_feedback.py --skip-fetch
```

**Important: classifier training vs feedback calibration**

- The main win model (`logistic` / `hist_gb`) is trained from `data/historical_games.csv`.
- Resolved rows in `data/prediction_feedback_training.csv` are used for a **second-stage feedback calibrator** (probability adjustment), not to refit classifier weights directly.
- Feedback rows are tracked from **20+ resolved games** (previously 30), but the feedback calibrator is only applied once the sample is safer (currently **40+** rows). Metrics include `feedback_calibrator_low_sample_warning` plus calibrator selection reason in `artifacts/model_metrics.json`.

## Notes on API cost

- `nba_api` is free.
- It can sometimes be slow or fail temporarily because public endpoints can be rate-limited.
- If you later want more stable, real-time, commercial use, you can switch to a paid API.

## Next improvements (later)

- Add injuries and back-to-back games.
- Add rolling 10-game stats.
- Predict point spread after win model works well.
