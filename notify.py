"""
notify.py -- Discord notification layer for MLB and WNBA edge alerts.

Replaces Gmail SMTP entirely.  Uses the requests library for HTTP calls.

Webhook URLs from environment variables:
    DISCORD_WEBHOOK_MLB      per-game edge alerts (predict_mlb.py)
    DISCORD_WEBHOOK_WNBA     per-player edge alerts (predict_wnba.py)
    DISCORD_WEBHOOK_RESULTS  daily graded results (daily_results.py, grade_wnba.py)
    DISCORD_WEBHOOK_WEEKLY   weekly summary (weekly_summary.py)

Public API (same names as the old Gmail version so callers need no changes):
    send_edge_sms(bet, shap_or_cfg, cfg=None)   MLB: (bet, shap, cfg)
                                                WNBA: (bet, config)
    send_summary_sms(n_edges, cfg)              MLB  count summary (8+ edges)
    send_test_sms(cfg)                          test all configured webhooks
    send_results_discord(results, perf_30d, run_date)  daily results embed
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

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
_COLOR_MLB   = 3447003    # blue
_COLOR_WNBA  = 15844367   # gold
_COLOR_WIN   = 3066993    # green
_COLOR_LOSS  = 15158332   # red
_COLOR_PUSH  = 9807270    # grey

# ---------------------------------------------------------------------------
# Core HTTP helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MLB edge alert
# ---------------------------------------------------------------------------

def _send_mlb_edge(bet, cfg: dict) -> bool:
    webhook  = os.environ.get("DISCORD_WEBHOOK_MLB", "").strip()
    away     = getattr(bet, "away_team", "")
    home     = getattr(bet, "home_team", "")
    game     = f"{away} @ {home}" if away else "MLB"
    gtime    = getattr(bet, "game_time_ct", "")
    extreme  = getattr(bet, "extreme_flag", False)
    edge_s   = f"+{bet.raw_edge_runs:.1f}" if bet.raw_edge_runs > 0 else f"{bet.raw_edge_runs:.1f}"

    game_line = f"{game}  |  {gtime}" if gtime else game
    desc = (
        f"**{game_line}**\n\n"
        f"**{bet.bet_direction} {bet.market_line}**\n"
        f"Predicted: {bet.predicted_total:.1f} runs  |  Edge: {edge_s}  |  Conf: {bet.p_win*100:.0f}%\n"
        f"Odds: {bet.juice}"
    )
    if extreme:
        desc = (
            f"⚠️ **Model may be extrapolating** — {abs(bet.raw_edge_runs):.1f}-run gap. Do not trust.\n\n"
            + desc
        )
    embed = {
        "title":       "⚾ MLB Edge Alert" + ("  ⚠️ EXTREME" if extreme else ""),
        "color":       _COLOR_PUSH if extreme else _COLOR_MLB,
        "description": desc,
        "footer":      {"text": "ObServatory MLB Model"},
    }
    ok = _discord_post(webhook, {"embeds": [embed]})
    if ok:
        log.info("Discord MLB edge sent: %s %s %s%s",
                 game, bet.bet_direction, bet.market_line, "  [EXTREME]" if extreme else "")
    return ok


# ---------------------------------------------------------------------------
# WNBA edge alert
# ---------------------------------------------------------------------------

def _short_name(full_name: str) -> str:
    """'Caitlin Clark' → 'C. Clark'"""
    parts = full_name.strip().split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts) >= 2 else full_name


def _utc_to_ct_str(game_time_utc) -> str:
    """Convert a UTC-aware datetime to CT string (CDT = UTC-5 in summer)."""
    try:
        ct_dt = game_time_utc - timedelta(hours=5)
        return ct_dt.strftime("%-I:%M %p CT")
    except Exception:
        return ""


def _send_wnba_edge(bet, cfg: dict) -> bool:
    webhook      = os.environ.get("DISCORD_WEBHOOK_WNBA", "").strip()
    player_name  = getattr(bet, "player_name", "Player")
    player_team  = getattr(bet, "player_team",  "")    # "Indiana Fever"
    opp_team     = getattr(bet, "opponent_team", "")   # "New York Liberty"
    game_time    = getattr(bet, "game_time", None)
    line_label   = getattr(bet, "threshold_label", "?")
    predicted    = getattr(bet, "predicted_points", 0.0)
    edge_pct     = getattr(bet, "edge", 0.0) * 100
    odds         = getattr(bet, "over_odds", 0)

    # Header line: "Caitlin Clark (Indiana Fever) vs New York Liberty"
    if player_team:
        player_line = f"{player_name} ({player_team})"
    else:
        player_line = player_name
    matchup_line = f"{player_line} vs {opp_team}" if opp_team else player_line

    # Game time
    gtime_str = _utc_to_ct_str(game_time) if game_time else ""

    desc = (
        f"**{matchup_line}**\n"
        + (f"{gtime_str}\n" if gtime_str else "")
        + f"\n**{line_label} pts**\n"
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
        log.info("Discord WNBA edge sent: %s %s pts", player_name, line_label)
    return ok


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_edge_sms(bet, shap_or_cfg, cfg=None) -> bool:
    """
    Unified entry point for both MLB and WNBA edge alerts.

    MLB callers (predict_mlb.py):   send_edge_sms(bet, shap, cfg)
    WNBA callers (predict_wnba.py): send_edge_sms(bet, config)

    Detection: if cfg is None the call is WNBA-style (two args).
    Falls back to duck-typing on bet.player_name if ambiguous.
    """
    if cfg is None:
        # WNBA two-arg call: send_edge_sms(bet, config)
        config = shap_or_cfg
        return _send_wnba_edge(bet, config)
    else:
        # MLB three-arg call: send_edge_sms(bet, shap, cfg)
        return _send_mlb_edge(bet, cfg)


def send_summary_sms(n_edges: int, cfg: dict) -> bool:
    """Count summary sent before individual alerts when 8+ MLB edges fire."""
    webhook = os.environ.get("DISCORD_WEBHOOK_MLB", "").strip()
    payload = {
        "embeds": [{
            "title":       "⚾ MLB Edges Today",
            "description": f"**{n_edges}** qualifying edge{'s' if n_edges != 1 else ''} — alerts incoming.",
            "color":       _COLOR_MLB,
            "footer":      {"text": "ObServatory MLB Model"},
        }]
    }
    return _discord_post(webhook, payload)


def send_test_sms(cfg: dict) -> bool:
    """
    Send a test message to every configured webhook.
    Called by predict_mlb.py --test-sms and predict_wnba.py --test-sms.
    Returns True if at least one webhook succeeds.
    """
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    results = []
    for env_var, label in [
        ("DISCORD_WEBHOOK_MLB",     "⚾ MLB"),
        ("DISCORD_WEBHOOK_WNBA",    "🏀 WNBA"),
        ("DISCORD_WEBHOOK_RESULTS", "📋 Results"),
        ("DISCORD_WEBHOOK_WEEKLY",  "📊 Weekly"),
    ]:
        url = os.environ.get(env_var, "").strip()
        if not url:
            log.info("Test: %s webhook not configured — skipped.", env_var)
            continue
        payload = {
            "embeds": [{
                "title":       f"{label} webhook test",
                "description": f"ObServatory test OK — {ts}",
                "color":       _COLOR_MLB,
                "footer":      {"text": "ObServatory"},
            }]
        }
        ok = _discord_post(url, payload)
        results.append(ok)
        log.info("Test %s (%s): %s", label, env_var, "OK" if ok else "FAILED")

    success = any(results)
    if success:
        log.info("Test complete — at least one webhook delivered.")
    else:
        log.error("All webhook tests failed — check DISCORD_WEBHOOK_* env vars.")
    return success


# ---------------------------------------------------------------------------
# Daily results embed  (called by daily_results.py)
# ---------------------------------------------------------------------------

def _build_mlb_result_line(r) -> str:
    """Single bet line for MLB results embed (one line per resolved/pending bet)."""
    emoji  = {"WIN": "✅", "LOSS": "❌", "PUSH": "➕"}.get(r.result, "⏳")
    label  = r.game_label          # "NYY @ BOS"
    bet    = f"{r.bet_direction} {r.market_line}"
    result = r.result

    if result == "PENDING":
        pred = getattr(r, "predicted_total", None)
        pred_s = f"Predicted: {pred:.1f}" if pred else ""
        return f"{emoji} {label} — {bet} — PENDING\n{pred_s}"

    profit  = f"{r.profit_units:+.2f}u"
    outcome = "WON" if result == "WIN" else ("LOST" if result == "LOSS" else "PUSH")
    odds_s  = f" @ {r.juice:+d}" if r.juice else ""
    line1   = f"{emoji} {label} — {bet} — {outcome} {profit}{odds_s}"

    actual  = int(r.actual_total) if r.actual_total else 0
    score   = f"{r.away_score}-{r.home_score}"
    pred    = getattr(r, "predicted_total", None)
    if pred is not None:
        off     = r.actual_total - pred
        sign    = "✓" if (result == "WIN") else "✗"
        line2   = f"Actual: {actual} runs ({score})  |  Predicted: {pred:.1f}  |  Off by: {off:+.1f} {sign}"
    else:
        line2   = f"Actual: {actual} runs ({score})"

    clv_s = ""
    if hasattr(r, "clv_beat") and r.clv_beat is not None and r.clv_beat >= 0:
        clv_s = f"  |  CLV: {r.clv:+.2f}pp"

    return f"{line1}\n{line2}{clv_s}"


def send_results_discord(results: list, perf_30d: dict, run_date: str) -> bool:
    """
    Post daily MLB results embed(s) to DISCORD_WEBHOOK_RESULTS.

    Each bet gets its own two-line block. Splits into multiple embeds when
    there are 5+ bets (Discord 6000-char safety margin).
    """
    webhook  = os.environ.get("DISCORD_WEBHOOK_RESULTS", "").strip()
    if not results:
        log.info("No bets for %s — results embed not sent.", run_date)
        return True

    resolved = [r for r in results if r.result != "PENDING"]
    pending  = [r for r in results if r.result == "PENDING"]

    net   = sum(r.profit_units for r in resolved) if resolved else 0.0
    color = _COLOR_WIN if net >= 0 else _COLOR_LOSS

    # Build per-bet lines (resolved first, then pending)
    lines = [_build_mlb_result_line(r) for r in resolved + pending]

    # Footer stats bar
    wins    = sum(1 for r in resolved if r.result == "WIN")
    losses  = sum(1 for r in resolved if r.result == "LOSS")
    record  = f"{wins}-{losses}"
    roi     = perf_30d.get("roi_pct")
    clv_r   = perf_30d.get("clv_rate")
    avg_clv = perf_30d.get("avg_clv")
    stats_parts = [f"📊 {record} | {net:+.2f}u"]
    if avg_clv  is not None: stats_parts.append(f"CLV: {avg_clv:+.2f}pp")
    if roi      is not None: stats_parts.append(f"30d ROI: {roi:+.1f}%")
    stats_bar = "  |  ".join(stats_parts)

    # Split into chunks of 4 bets to stay well under Discord 6000-char limit
    CHUNK = 4
    chunks = [lines[i:i+CHUNK] for i in range(0, len(lines), CHUNK)]
    n_chunks = len(chunks)

    ok = True
    for idx, chunk in enumerate(chunks):
        desc   = "\n\n".join(chunk) + f"\n\n{stats_bar}"
        suffix = f" ({idx+1}/{n_chunks})" if n_chunks > 1 else ""
        embed  = {
            "title":       f"⚾ MLB Results — {run_date}{suffix}",
            "color":       color,
            "description": desc,
            "footer":      {"text": "ObServatory MLB Model"},
        }
        ok = _discord_post(webhook, {"embeds": [embed]}) and ok

    if ok:
        log.info("Discord MLB results sent for %s (%d bet(s), %d embed(s)).",
                 run_date, len(results), n_chunks)
    return ok
