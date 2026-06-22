"""
notify.py — Discord notification layer for WNBA edge alerts.

Replaces Gmail SMTP entirely.  Uses the requests library for HTTP calls.

Webhook URLs from environment variables:
    DISCORD_WEBHOOK_WNBA     per-player edge alerts
    DISCORD_WEBHOOK_RESULTS  daily graded results
    DISCORD_WEBHOOK_WEEKLY   weekly summary

NOTE: on the VPS both predict_mlb.py and predict_wnba.py import from the
same /home/picks/notify.py.  The VPS file is models/mlb/notify.py, which
has a unified send_edge_sms() that detects MLB vs WNBA automatically.
This file is the WNBA-specific version used for local development; it deploys
to /home/picks/notify_wnba.py (unused directly on VPS).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False, encoding="utf-8-sig")

log = logging.getLogger(__name__)

_COLOR_WNBA = 15844367  # gold
_COLOR_WIN  = 3066993   # green
_COLOR_LOSS = 15158332  # red


def _discord_post(webhook_url: str, payload: dict) -> bool:
    """POST a JSON payload to a Discord webhook.  Returns True on 2xx."""
    if not webhook_url:
        log.warning("Discord webhook URL not set — notification skipped.")
        return False
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        ok   = resp.status_code in (200, 204)
        if not ok:
            log.warning("Discord returned HTTP %d: %s", resp.status_code, resp.text[:120])
        return ok
    except Exception as exc:
        log.error("Discord post failed: %s", exc)
        return False


def _short_name(full_name: str) -> str:
    """'Caitlin Clark' → 'C. Clark'"""
    parts = full_name.strip().split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts) >= 2 else full_name


def _utc_to_ct_str(game_time_utc) -> str:
    """
    Convert a UTC-aware datetime to Central Time string.
    CDT = UTC-5 during WNBA season (Apr-Oct).
    game_time in Postgres is stored as timezone-aware UTC.
    """
    try:
        ct_dt = game_time_utc - timedelta(hours=5)
        return ct_dt.strftime("%-I:%M %p CT")
    except Exception:
        return ""


def send_edge_sms(bet, config: dict) -> bool:
    """WNBA edge alert embed → DISCORD_WEBHOOK_WNBA."""
    webhook     = os.environ.get("DISCORD_WEBHOOK_WNBA", "").strip()
    player_name = getattr(bet, "player_name", "Player")
    home_team   = getattr(bet, "home_team", "")
    away_team   = getattr(bet, "away_team", "")
    game_time   = getattr(bet, "game_time", None)
    line_label  = getattr(bet, "threshold_label", "?")
    predicted   = getattr(bet, "predicted_points", 0.0)
    edge_pct    = getattr(bet, "edge", 0.0) * 100
    odds        = getattr(bet, "over_odds", 0)

    # Matchup header: "Caitlin Clark vs Indiana Fever @ New York Liberty  |  7:00 PM CT"
    matchup = f"{away_team} @ {home_team}" if away_team else home_team
    gtime_s = _utc_to_ct_str(game_time) if game_time else ""
    matchup_line = f"{matchup}  |  {gtime_s}" if gtime_s else matchup

    desc = (
        f"**{player_name}**\n"
        f"{matchup_line}\n\n"
        f"**{line_label} pts**\n"
        f"Predicted: {predicted:.1f} pts  |  Edge: +{edge_pct:.0f}%  |  Odds: {odds:+d}"
    )
    payload = {
        "embeds": [{
            "title":       "🏀 WNBA Edge Alert",
            "color":       _COLOR_WNBA,
            "description": desc,
            "footer":      {"text": "ObServatory WNBA Model"},
        }]
    }
    ok = _discord_post(webhook, payload)
    if ok:
        log.info("Discord WNBA edge sent: %s %s+ pts", player_name, line_label)
    return ok


def send_summary_sms(n_edges: int, config: dict) -> bool:
    """Count summary sent before individual alerts when 8+ edges fire."""
    webhook = os.environ.get("DISCORD_WEBHOOK_WNBA", "").strip()
    payload = {
        "embeds": [{
            "title":       "🏀 WNBA Edges Today",
            "description": f"**{n_edges}** qualifying edge{'s' if n_edges != 1 else ''} — alerts incoming.",
            "color":       _COLOR_WNBA,
            "footer":      {"text": "ObServatory WNBA Model"},
        }]
    }
    return _discord_post(webhook, payload)


def send_test_sms(config: dict) -> bool:
    """Send a test message to the WNBA webhook."""
    webhook = os.environ.get("DISCORD_WEBHOOK_WNBA", "").strip()
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    payload = {
        "embeds": [{
            "title":       "🏀 WNBA webhook test",
            "description": f"ObServatory WNBA test OK — {ts}",
            "color":       _COLOR_WNBA,
            "footer":      {"text": "ObServatory"},
        }]
    }
    ok = _discord_post(webhook, payload)
    if ok:
        log.info("Test Discord WNBA webhook: OK")
    else:
        log.error("Test Discord WNBA webhook: FAILED — check DISCORD_WEBHOOK_WNBA")
    return ok
