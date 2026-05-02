import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from value_betting import (
    edge_percent,
    expected_value_per_unit_stake,
    fair_odds_from_probability,
    implied_probability_from_decimal_odds,
    kelly_fraction,
)


class ValueBettingMathTests(unittest.TestCase):
    def test_fair_odds_conversion(self):
        fair = fair_odds_from_probability(0.65)
        self.assertAlmostEqual(fair.decimal, 1 / 0.65, places=6)
        self.assertIsInstance(fair.american, int)
        self.assertIn("/", fair.fractional)

    def test_ev_calculation_positive(self):
        ev = expected_value_per_unit_stake(model_win_probability=0.6, market_decimal_odds=2.1)
        self.assertGreater(ev, 0.0)

    def test_no_edge_gives_zero_kelly(self):
        market_odds = 2.0
        break_even = implied_probability_from_decimal_odds(market_odds)
        self.assertEqual(kelly_fraction(break_even, market_odds), 0.0)
        self.assertEqual(kelly_fraction(0.45, market_odds), 0.0)

    def test_quarter_kelly_with_cap(self):
        frac = kelly_fraction(
            model_win_prob=0.85,
            market_decimal_odds=3.2,
            safety_multiplier=0.25,
            max_fraction=0.05,
        )
        self.assertLessEqual(frac, 0.05)
        self.assertGreaterEqual(frac, 0.0)

    def test_edge_percent_sign(self):
        self.assertGreater(edge_percent(0.60, 2.10), 0.0)
        self.assertLess(edge_percent(0.40, 2.10), 0.0)


if __name__ == "__main__":
    unittest.main()
