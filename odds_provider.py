from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import env_bootstrap  # noqa: F401 — load `.env` (API key + proxy vars)

import pandas as pd
import requests


ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


@dataclass
class OddsConfig:
    api_key: str | None
    regions: str = "us"
    markets: str = "h2h"
    odds_format: str = "decimal"
    bookmakers: str | None = None
    timeout_seconds: int = 12


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_nba_market_odds(config: OddsConfig | None = None) -> pd.DataFrame:
    cfg = config or OddsConfig(api_key=os.getenv("ODDS_API_KEY"))
    if not cfg.api_key:
        return pd.DataFrame(
            columns=[
                "game_id",
                "commence_time",
                "home_team",
                "away_team",
                "bookmaker",
                "market",
                "home_decimal_odds",
                "away_decimal_odds",
                "source",
                "fetched_at",
            ]
        )

    params = {
        "apiKey": cfg.api_key,
        "regions": cfg.regions,
        "markets": cfg.markets,
        "oddsFormat": cfg.odds_format,
    }
    if cfg.bookmakers:
        params["bookmakers"] = cfg.bookmakers

    response = requests.get(ODDS_API_URL, params=params, timeout=cfg.timeout_seconds)
    response.raise_for_status()
    payload = response.json()

    rows = []
    fetched_at = datetime.utcnow().isoformat(timespec="seconds")
    for event in payload:
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        for bookmaker in event.get("bookmakers", []):
            bookmaker_name = bookmaker.get("title", "unknown")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                home_odds = None
                away_odds = None
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = _safe_float(outcome.get("price"))
                    if name == home_team:
                        home_odds = price
                    elif name == away_team:
                        away_odds = price

                if not home_odds or not away_odds:
                    continue

                rows.append(
                    {
                        "game_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker_name,
                        "market": "h2h",
                        "home_decimal_odds": home_odds,
                        "away_decimal_odds": away_odds,
                        "source": "the-odds-api",
                        "fetched_at": fetched_at,
                    }
                )

    return pd.DataFrame(rows)
