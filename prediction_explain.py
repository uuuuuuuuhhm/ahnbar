"""Template-based explanations from matchup feature rows (no LLM)."""

from __future__ import annotations

# Align with common rolling form window in train_model.
_DEFAULT_FORM_WINDOW = 5


def explain_matchup(
    feat: dict,
    home_abbr: str,
    away_abbr: str,
    *,
    form_window: int = _DEFAULT_FORM_WINDOW,
    max_bullets: int = 4,
) -> list[str]:
    """
    Return short bullet strings highlighting the largest home-vs-away gaps on
    basketball-grounded rolling stats plus rest and Elo context.
    """
    w = int(form_window)
    bullets: list[str] = []

    def g(name: str, default: float = 0.0) -> float:
        v = feat.get(name, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # Signed values: positive => advantage for home team (for basketball stats where higher is better).
    efg_h = g(f"ROLL_EFG_PCT_{w}_HOME", 0.535)
    efg_a = g(f"ROLL_EFG_PCT_{w}_AWAY", 0.535)
    efg_edge = efg_h - efg_a

    # Lower turnover rate is better => home advantage when home TOV% < away TOV%.
    tov_h = g(f"ROLL_TOV_PCT_{w}_HOME", 0.135)
    tov_a = g(f"ROLL_TOV_PCT_{w}_AWAY", 0.135)
    tov_edge_home = tov_a - tov_h

    orb_h = g(f"ROLL_ORB_PCT_OFF_{w}_HOME", 0.22)
    orb_a = g(f"ROLL_ORB_PCT_OFF_{w}_AWAY", 0.22)
    orb_edge = orb_h - orb_a

    ftr_h = g(f"ROLL_FTR_{w}_HOME", 0.25)
    ftr_a = g(f"ROLL_FTR_{w}_AWAY", 0.25)
    ftr_edge = ftr_h - ftr_a

    pace_h = g(f"ROLL_PACE_PROX_{w}_HOME", 99.0)
    pace_a = g(f"ROLL_PACE_PROX_{w}_AWAY", 99.0)
    pace_edge = pace_h - pace_a

    candidates: list[tuple[float, str]] = [
        (abs(efg_edge), ""),
        (abs(tov_edge_home), ""),
        (abs(orb_edge), ""),
        (abs(ftr_edge), ""),
        (abs(pace_edge), ""),
    ]
    # Fill messages after ranking by magnitude
    scored: list[tuple[float, str]] = []

    if abs(efg_edge) >= 0.008:
        if efg_edge > 0:
            msg = f"{home_abbr} has been the sharper shooting team recently (rolling eFG%)."
        else:
            msg = f"{away_abbr} has been the sharper shooting team recently (rolling eFG%)."
        scored.append((abs(efg_edge), msg))

    if abs(tov_edge_home) >= 0.006:
        if tov_edge_home > 0:
            msg = f"{home_abbr} has taken better care of the ball lately (lower turnover rate)."
        else:
            msg = f"{away_abbr} has taken better care of the ball lately (lower turnover rate)."
        scored.append((abs(tov_edge_home), msg))

    if abs(orb_edge) >= 0.02:
        if orb_edge > 0:
            msg = f"{home_abbr} has generated more second chances on the glass recently."
        else:
            msg = f"{away_abbr} has generated more second chances on the glass recently."
        scored.append((abs(orb_edge), msg))

    if abs(ftr_edge) >= 0.025:
        if ftr_edge > 0:
            msg = f"{home_abbr} has lived at the line more often (higher FTA/FGA)."
        else:
            msg = f"{away_abbr} has lived at the line more often (higher FTA/FGA)."
        scored.append((abs(ftr_edge), msg))

    if abs(pace_edge) >= 2.0:
        if pace_edge > 0:
            msg = f"{home_abbr}'s recent games have been played at a faster tempo (possession proxy)."
        else:
            msg = f"{away_abbr}'s recent games have been played at a faster tempo (possession proxy)."
        scored.append((abs(pace_edge), msg))

    rest_d = g("REST_DIFF", 0.0)
    if abs(rest_d) >= 1.0:
        if rest_d > 0:
            msg = f"{home_abbr} comes in with more rest than {away_abbr} (day gap)."
        else:
            msg = f"{away_abbr} comes in with more rest than {home_abbr} (day gap)."
        scored.append((abs(rest_d), msg))

    b2b_d = g("B2B_DIFF", 0.0)
    if abs(b2b_d) >= 1.0:
        if b2b_d < 0:
            msg = f"{home_abbr} is on a back-to-back while {away_abbr} is not."
        elif b2b_d > 0:
            msg = f"{away_abbr} is on a back-to-back while {home_abbr} is not."
        else:
            msg = ""
        if msg:
            scored.append((2.0 + abs(b2b_d), msg))

    elo = g("ELO_HOME_WIN_PROB", 0.5)
    if elo >= 0.58:
        scored.append((abs(elo - 0.5), f"Strength signal tilts toward {home_abbr} (pre-game Elo win proxy)."))
    elif elo <= 0.42:
        scored.append((abs(elo - 0.5), f"Strength signal tilts toward {away_abbr} (pre-game Elo win proxy)."))

    scored.sort(key=lambda x: x[0], reverse=True)
    for _, msg in scored[:max_bullets]:
        if msg and msg not in bullets:
            bullets.append(msg)

    if not bullets:
        bullets.append(
            "Recent form and rest are close; the pick is driven by the blended model rather than one clear edge."
        )

    return bullets[:max_bullets]


def lean_strength_pct(home_win_probability: float) -> float:
    """Distance from 50% — a 'strong lean', not a calibrated confidence interval."""
    p = float(home_win_probability)
    return round(max(p, 1.0 - p) * 100.0, 2)
