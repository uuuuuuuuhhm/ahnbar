import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction_explain import explain_matchup, lean_strength_pct


class PredictionExplainTests(unittest.TestCase):
    def test_explain_returns_non_empty(self):
        feat = {
            "ROLL_EFG_PCT_5_HOME": 0.58,
            "ROLL_EFG_PCT_5_AWAY": 0.50,
            "ROLL_TOV_PCT_5_HOME": 0.11,
            "ROLL_TOV_PCT_5_AWAY": 0.15,
            "ROLL_ORB_PCT_OFF_5_HOME": 0.28,
            "ROLL_ORB_PCT_OFF_5_AWAY": 0.20,
            "ROLL_FTR_5_HOME": 0.30,
            "ROLL_FTR_5_AWAY": 0.22,
            "ROLL_PACE_PROX_5_HOME": 102.0,
            "ROLL_PACE_PROX_5_AWAY": 98.0,
            "REST_DIFF": 1.0,
            "B2B_DIFF": 0.0,
            "ELO_HOME_WIN_PROB": 0.62,
        }
        lines = explain_matchup(feat, "HOM", "AWY", max_bullets=4)
        self.assertTrue(len(lines) >= 1)
        self.assertTrue(all(isinstance(s, str) and len(s) > 5 for s in lines))

    def test_lean_strength_pct(self):
        self.assertEqual(lean_strength_pct(0.7), 70.0)


if __name__ == "__main__":
    unittest.main()
