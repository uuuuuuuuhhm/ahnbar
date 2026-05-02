"""
Load project-root `.env` before any HTTP/NBA calls.

`requests` (used by `nba_api` and `odds_provider`) honors standard proxy variables:
`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY`.

Shell-set variables take precedence over `.env` (override=False).
"""

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent


def load_project_env(*, override: bool = False) -> None:
    """Populate os.environ from `.env`. Safe to call more than once."""
    load_dotenv(_PROJECT_ROOT / ".env", override=override)


load_project_env()
