"""Canonical UTC tip-off storage and local-time formatting for predictions UI/CLI."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_DISPLAY_TZ = "Europe/Berlin"

# Curated IANA zones for Streamlit selectbox (EU-first).
COMMON_DISPLAY_TIMEZONES: list[str] = [
    "Europe/Berlin",
    "Europe/London",
    "Europe/Paris",
    "Europe/Madrid",
    "Europe/Rome",
    "Europe/Amsterdam",
    "Europe/Warsaw",
    "Europe/Athens",
    "America/Toronto",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Phoenix",
    "Asia/Tokyo",
    "Australia/Sydney",
    "UTC",
]


def display_timezone_for_cli() -> ZoneInfo:
    """CLI / scripts: honor TZ if valid, else Europe/Berlin."""
    tz_name = os.environ.get("TZ")
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return ZoneInfo(DEFAULT_DISPLAY_TZ)


def utc_iso_from_game_row(g: pd.Series) -> str | None:
    """Prefer NBA Stats gameTimeUTC (ISO); optional gameEt ISO fallback."""
    for key in ("GAME_TIME_UTC", "gameTimeUTC"):
        if key not in g.index:
            continue
        out = _normalize_to_utc_iso(g.get(key))
        if out:
            return out
    for key in ("GAME_ET", "gameEt"):
        if key not in g.index:
            continue
        out = _normalize_to_utc_iso(g.get(key))
        if out:
            return out
    return None


def _normalize_to_utc_iso(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ZoneInfo("UTC")).isoformat()
    except Exception:
        return None


def format_local_from_utc_iso(iso_utc: str | None, tz: ZoneInfo) -> str:
    """Format UTC instant for display in tz (e.g. 2026-05-02 01:00 CEST)."""
    if not iso_utc or str(iso_utc).strip() == "" or str(iso_utc).lower() == "nan":
        return "TBD"
    s = str(iso_utc).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        local = dt.astimezone(tz)
        abbr = local.tzname() or str(tz)
        return f"{local.strftime('%Y-%m-%d %H:%M')} {abbr}"
    except Exception:
        return "TBD"


def format_start_display(
    iso_utc: str | None,
    tz: ZoneInfo,
    *,
    legacy_24h: str | None = None,
) -> str:
    """Prefer stored UTC ISO; fall back to legacy game_start_time_24h when UTC missing."""
    if iso_utc is not None and pd.notna(iso_utc) and str(iso_utc).strip() and str(iso_utc).lower() != "nan":
        s = format_local_from_utc_iso(str(iso_utc), tz)
        if s != "TBD":
            return s
    if legacy_24h is not None:
        leg = str(legacy_24h).strip()
        if leg and leg.lower() not in ("nan", "unknown", "tbd"):
            return leg
    return "TBD"


def safe_zoneinfo(name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(name.strip())
    except Exception:
        return None
