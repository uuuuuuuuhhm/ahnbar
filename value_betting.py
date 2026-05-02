from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import pandas as pd


def probability_to_decimal_odds(probability: float) -> float:
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1 (exclusive).")
    return 1.0 / probability


def probability_to_american_odds(probability: float) -> int:
    decimal_odds = probability_to_decimal_odds(probability)
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    return int(round(-100 / (decimal_odds - 1.0)))


def probability_to_fractional_odds(probability: float, max_denominator: int = 100) -> str:
    decimal_odds = probability_to_decimal_odds(probability)
    fractional = Fraction(decimal_odds - 1.0).limit_denominator(max_denominator)
    return f"{fractional.numerator}/{fractional.denominator}"


def implied_probability_from_decimal_odds(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0.")
    return 1.0 / decimal_odds


def edge_percent(model_win_probability: float, market_decimal_odds: float) -> float:
    market_implied = implied_probability_from_decimal_odds(market_decimal_odds)
    return (model_win_probability - market_implied) * 100.0


def expected_value_per_unit_stake(model_win_probability: float, market_decimal_odds: float) -> float:
    profit_if_win = market_decimal_odds - 1.0
    loss_probability = 1.0 - model_win_probability
    return (model_win_probability * profit_if_win) - (loss_probability * 1.0)


def kelly_fraction(
    model_win_prob: float,
    market_decimal_odds: float,
    safety_multiplier: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    if model_win_prob <= 0 or model_win_prob >= 1:
        raise ValueError("model_win_prob must be between 0 and 1 (exclusive).")
    if market_decimal_odds <= 1.0:
        raise ValueError("market_decimal_odds must be greater than 1.0.")
    if safety_multiplier < 0:
        raise ValueError("safety_multiplier must be >= 0.")
    if max_fraction <= 0:
        raise ValueError("max_fraction must be > 0.")

    break_even = implied_probability_from_decimal_odds(market_decimal_odds)
    if model_win_prob <= break_even:
        return 0.0

    b = market_decimal_odds - 1.0
    p = model_win_prob
    q = 1.0 - p
    raw_fraction = (p * b - q) / b
    adjusted = max(0.0, raw_fraction * safety_multiplier)
    return min(adjusted, max_fraction)


def kelly_stake_amount(
    current_bankroll: float,
    model_win_prob: float,
    market_decimal_odds: float,
    safety_multiplier: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    if current_bankroll < 0:
        raise ValueError("current_bankroll must be >= 0.")
    return current_bankroll * kelly_fraction(
        model_win_prob=model_win_prob,
        market_decimal_odds=market_decimal_odds,
        safety_multiplier=safety_multiplier,
        max_fraction=max_fraction,
    )


@dataclass
class FairOdds:
    decimal: float
    american: int
    fractional: str


def fair_odds_from_probability(probability: float) -> FairOdds:
    return FairOdds(
        decimal=probability_to_decimal_odds(probability),
        american=probability_to_american_odds(probability),
        fractional=probability_to_fractional_odds(probability),
    )


def simulate_bankroll_curve(
    returns: Iterable[float], start_bankroll: float = 1000.0
) -> pd.DataFrame:
    """
    Build bankroll curve from per-bet flat-unit returns (e.g., +0.2, -1.0).
    Each step is interpreted as PnL on 1 unit stake, not full-bankroll compounding.
    Returns DataFrame with step index, pnl_unit, and bankroll.
    """
    bankroll = float(start_bankroll)
    rows = [{"step": 0, "pnl_unit": 0.0, "bankroll": bankroll}]
    for i, r in enumerate(returns, start=1):
        bankroll += float(r)
        bankroll = max(bankroll, 0.0)
        rows.append({"step": i, "pnl_unit": float(r), "bankroll": bankroll})
    out = pd.DataFrame(rows)
    out["bankroll_peak"] = out["bankroll"].cummax()
    out["drawdown"] = (out["bankroll"] / out["bankroll_peak"]) - 1.0
    return out
