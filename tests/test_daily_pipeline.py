"""Unit tests for jobs/daily.py orchestration helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import os
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import daily as daily_job


class DailyPipelineArgTests(unittest.TestCase):
    def test_value_alias_sets_predict_value(self) -> None:
        parser = daily_job.build_arg_parser()
        args = parser.parse_args(["--value"])
        self.assertTrue(args.predict_value)

    def test_predict_value_long_form(self) -> None:
        parser = daily_job.build_arg_parser()
        args = parser.parse_args(["--predict-value"])
        self.assertTrue(args.predict_value)

    def test_apply_nightly_profile(self) -> None:
        parser = daily_job.build_arg_parser()
        args = parser.parse_args([])
        daily_job.apply_nightly_profile(args)
        self.assertTrue(args.patch_feedback)
        self.assertTrue(args.predict)
        self.assertTrue(args.predict_value)

    def test_should_skip_patch_feedback(self) -> None:
        skip, reason = daily_job.should_skip_patch_feedback(retrain=False, patch_requested=False)
        self.assertTrue(skip)
        self.assertEqual(reason, "not_requested")

        skip, reason = daily_job.should_skip_patch_feedback(retrain=True, patch_requested=True)
        self.assertTrue(skip)
        self.assertEqual(reason, "redundant_with_retrain")

        skip, reason = daily_job.should_skip_patch_feedback(retrain=False, patch_requested=True)
        self.assertFalse(skip)
        self.assertEqual(reason, "")

    def test_prepare_implicit_predict(self) -> None:
        parser = daily_job.build_arg_parser()
        args = parser.parse_args(["--predict-value"])
        self.assertFalse(args.predict)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "log.txt"
            daily_job.prepare_implicit_predict(args, log_path)
            self.assertTrue(args.predict)
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("implicit --predict", text)


class DailyPipelineMainTests(unittest.TestCase):
    @patch.object(daily_job, "run_feedback_patch_step", return_value={"patched": True})
    @patch.object(daily_job, "_run_script", return_value=0)
    def test_patch_runs_after_backfill_when_requested(
        self, _mock_run: MagicMock, mock_patch: MagicMock
    ) -> None:
        rc = daily_job.main(["--skip-fetch", "--patch-feedback"])
        self.assertEqual(rc, 0)
        mock_patch.assert_called_once()

    @patch.object(daily_job, "run_feedback_patch_step", return_value={"patched": True})
    @patch.object(daily_job, "_run_script", return_value=0)
    def test_patch_skipped_when_retrain(self, _mock_run: MagicMock, mock_patch: MagicMock) -> None:
        rc = daily_job.main(["--skip-fetch", "--retrain", "--patch-feedback"])
        self.assertEqual(rc, 0)
        mock_patch.assert_not_called()

    @patch.object(daily_job, "_run_script", return_value=0)
    def test_predict_value_skipped_without_odds_key(self, mock_run: MagicMock) -> None:
        env_no_odds = {k: v for k, v in os.environ.items() if k != "ODDS_API_KEY"}
        with patch.dict(os.environ, env_no_odds, clear=True):
            rc = daily_job.main(["--skip-fetch", "--predict", "--predict-value"])
        self.assertEqual(rc, 0)
        rel_scripts = [c.args[0] for c in mock_run.call_args_list]
        self.assertIn("backfill_prediction_results.py", rel_scripts)
        self.assertIn("predict_next_games.py", rel_scripts)
        self.assertNotIn("predict_value_plays.py", rel_scripts)


if __name__ == "__main__":
    unittest.main()
