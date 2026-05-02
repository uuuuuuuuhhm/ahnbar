# AGENTS.md

## Cursor Cloud specific instructions

### Overview

NBA Win Predictor — a Python ML system that predicts NBA game win/loss probabilities using scikit-learn, with a Streamlit web GUI. All data is stored as flat CSV/Parquet files (no database). See `README.md` for full documentation.

### Quick reference

| Action | Command |
|--------|---------|
| Activate venv | `source /workspace/.venv/bin/activate` |
| Run tests | `python -m pytest tests/ -v` |
| Evaluate model | `python evaluate_model.py` |
| Score predictions | `python score_predictions.py` |
| Start Streamlit GUI | `streamlit run app.py --server.port 8501 --server.headless true` |
| Train model | `python train_model.py` |

### Caveats

- **No linting tools configured**: The project has no flake8, ruff, mypy, or black configuration. Tests are the primary quality check.
- **External API dependency**: `predict_next_games.py` calls `stats.nba.com` via `nba_api`, which can be slow or rate-limited from datacenter IPs. Model evaluation (`evaluate_model.py`) and scoring (`score_predictions.py`) work entirely offline using existing CSV data.
- **The Odds API** is optional (value betting only). Requires `ODDS_API_KEY` in `.env`.
- **Streamlit tab rendering**: Some Streamlit tabs may not render content reliably in headless Chrome in the cloud environment. The underlying Python scripts (`evaluate_model.py`, `score_predictions.py`, etc.) always work and can be used to verify functionality directly.
- **pytest not in requirements.txt**: Tests use `unittest` under the hood but are easiest to run with pytest. The update script installs pytest separately.
- **python3.12-venv**: The system package `python3.12-venv` must be installed for venv creation. The update script handles this.
