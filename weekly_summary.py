"""
weekly_summary.py -- Sport-agnostic weekly picks performance summary.

Reads bet_alerts for the current Mon-today window and sends one Discord
message with W/L/P record, profit, and CLV rate broken down by sport.

VPS cron (Sunday 10am UTC = 6am ET, but also usable any day):
    0 10 * * 0 cd /home/picks && ALERTS_DB_PATH=/home/picks/alerts.db \
        python3 weekly_summary.py >> /home/picks/logs/weekly_summary.log 2>&1

Usage:
    python weekly_summary.py               # current Mon-today window
    python weekly_summary.py --week-start 2026-05-25   # specific Monday
    python weekly_summary.py --dry-run     # print body, no Discord send
    python weekly_summary.py --db /home/picks/alerts.db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

_DIR = Path(__file__).resolve().parent

for _env in (_DIR / "wnba" / ".env", _DIR / "mlb" / ".env", _DIR / ".env"):
    if _env.exists():
        load_dotenv(dotenv_path=_env, override=False, encoding="utf-8-sig")
        break

_LOG_DIR = _DIR / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_DIR / "weekly_summary.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

_DEFAULT_DB = _DIR / "alerts.db"


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def current_week_window(today: date) -> tuple[date, date]:
    """
    Return (monday, today) for the in-progress current week.
    This ensures this week's alerts are always visible even before Sunday.
    Use --week-start to pull a completed historical week instead.
    """
    monday = today - timedelta(days=today.weekday())   # Mon=0, so this is always Mon
    return monday, today


def last_completed_week(today: date) -> tuple[date, date]:
    """Return (monday, sunday) of the most recently completed Mon-Sun week."""
    days_since_monday = today.weekday()
    last_sunday = today - timedelta(days=days_since_monday + 1)
    last_monday = last_sunday - timedelta(days=6)
    return last_monday, last_sunday


def week_from_monday(monday_str: str) -> tuple[date, date]:
    monday = date.fromisoformat(monday_str)
    return monday, monday + timedelta(days=6)


# ---------------------------------------------------------------------------
# Query
# NOTE: notified=1 filter intentionally removed.  Neither the WNBA nor Tennis
# pipelines update notified in alerts.db after Discord sends (they track it in
# their own sport DBs).  Filtering on notified hides every alert.
# ---------------------------------------------------------------------------

_QUERY = """
SELECT
    sport,
    result,
    profit_units,
    clv_beat
FROM bet_alerts
WHERE alert_date BETWEEN ? AND ?
  AND result IN ('WIN', 'LOSS', 'PUSH', 'PENDING')
ORDER BY sport, alert_date
"""


def load_bets(db_path: str, week_start: date, week_end: date) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(_QUERY, (week_start.isoformat(), week_end.isoformat())).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def sport_stats(bets: list[dict]) -> dict:
    """W/L/P record, profit, CLV beat rate for a list of bets."""
    graded = [b for b in bets if b["result"] in ("WIN", "LOSS", "PUSH")]
    wins   = sum(1 for b in graded if b["result"] == "WIN")
    losses = sum(1 for b in graded if b["result"] == "LOSS")
    pushes = sum(1 for b in graded if b["result"] == "PUSH")

    profit = sum(b["profit_units"] or 0.0 for b in graded)

    wagered = len([b for b in graded if b["result"] != "PUSH"])
    roi = (profit / wagered * 100) if wagered > 0 else None

    clv_bets  = [b for b in graded if b["clv_beat"] is not None]
    clv_wins  = sum(1 for b in clv_bets if b["clv_beat"] == 1)
    clv_rate  = (clv_wins, len(clv_bets)) if clv_bets else None

    pending = sum(1 for b in bets if b["result"] == "PENDING")

    return {
        "wins": wins, "losses": losses, "pushes": pushes,
        "profit": round(profit, 2),
        "roi": round(roi, 1) if roi is not None else None,
        "clv_rate": clv_rate,
        "pending": pending,
        "total_alerts": len(bets),
        "total_graded": len(graded),
    }


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

_COLOR_WIN  = 3066993   # green
_COLOR_LOSS = 15158332  # red
_COLOR_EVEN = 9807270   # grey


def _discord_post(webhook_url: str, payload: dict) -> bool:
    if not webhook_url:
        log.warning("DISCORD_WEBHOOK_WEEKLY not set -- weekly summary skipped.")
        return False
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            webhook_url,
            data    = data,
            headers = {"Content-Type": "application/json",
                       "User-Agent":   "ObServatory/1.0"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status in (200, 204)
            if not ok:
                log.warning("Discord returned HTTP %d", resp.status)
            return ok
    except Exception as exc:
        log.error("Discord post failed: %s", exc)
        return False


def _insight_line(totals: dict, by_sport: dict) -> str:
    """
    Generate a one-line qualitative insight based on CLV rate and P&L.

    CLV% here = percentage of bets that beat the closing line (clv_beat=1).
    - > 50%  → strong signal (more than half beat the close)
    - 30-50% → notable but not dominant
    - < 30%  → weak or noise

    Rules (first match wins):
      1. Any sport CLV > 50%  → "Strong CLV signal on X"
      2. All CLV < 30%        → "CLV weak across all models"
      3. ROI positive         → "Profitable week - verify sample size"
      4. Default              → "CLV positive on X (N%) despite losses..."
    """
    n = totals.get("total_graded", 0)
    if n == 0:
        return ""

    net = totals.get("profit", 0.0)

    # Build per-sport CLV% (pct of bets that beat the close)
    sport_clv: dict[str, float] = {}
    for sport, s in by_sport.items():
        if s.get("clv_rate") is not None:
            wins, total = s["clv_rate"]
            if total > 0:
                sport_clv[sport] = wins / total * 100.0

    # Rule 1: any sport CLV > 50%
    strong_sports = [sp for sp, clv in sport_clv.items() if clv > 50]
    if strong_sports:
        return f"Strong CLV signal on {' and '.join(strong_sports)}."

    # Rule 2: all sports CLV < 30%
    if sport_clv and all(clv < 30 for clv in sport_clv.values()):
        return "CLV weak across all models."

    # Rule 3: profitable week
    if net > 0:
        return "Profitable week — verify sample size."

    # Default: some sports have CLV >= 30% despite losses
    clv_pos_parts = [
        f"{sp} ({clv:.0f}%)"
        for sp, clv in sorted(sport_clv.items())
        if clv >= 30
    ]
    if clv_pos_parts:
        clv_str = " and ".join(clv_pos_parts)
        base = f"CLV positive on {clv_str} despite losses."
    else:
        base = "Losses this week."

    if n < 20:
        base += " Sample too small for conclusions. Keep observing."

    return base


def send_sms(
    week_start: date,
    week_end: date,
    by_sport: dict,
    totals: dict,
) -> bool:
    """Post weekly summary embed to DISCORD_WEBHOOK_WEEKLY using Discord fields."""
    webhook = os.environ.get("DISCORD_WEBHOOK_WEEKLY", "").strip()

    net   = totals.get("profit", 0.0)
    color = _COLOR_WIN if net > 0 else (_COLOR_LOSS if net < 0 else _COLOR_EVEN)

    sport_emojis = {"MLB": "⚾", "WNBA": "🏀", "TENNIS": "🎾"}

    # Per-sport fields
    fields: list[dict] = []
    for sport, s in sorted(by_sport.items()):
        if s["total_alerts"] == 0:
            continue
        emoji = sport_emojis.get(sport, "🎯")
        wlp   = f"{s['wins']}-{s['losses']}-{s['pushes']}"
        parts = [wlp]
        if s["total_graded"] > 0:
            parts.append(f"{s['profit']:+.2f}u")
            if s.get("roi") is not None:
                parts.append(f"ROI {s['roi']:+.1f}%")
            if s.get("clv_rate") is not None:
                wins, total = s["clv_rate"]
                rate = wins / total * 100.0 if total > 0 else 0.0
                parts.append(f"CLV: {rate:.0f}%")
        if s["pending"]:
            parts.append(f"{s['pending']} pending")
        fields.append({
            "name":   f"{emoji} {sport}",
            "value":  " | ".join(parts),
            "inline": False,
        })

    # Combined field
    if totals["total_graded"] > 0:
        wlp   = f"{totals['wins']}-{totals['losses']}-{totals['pushes']}"
        parts = [wlp, f"{net:+.2f}u"]
        if totals.get("roi") is not None:
            parts.append(f"ROI {totals['roi']:+.1f}%")
        fields.append({
            "name":   "📈 Combined",
            "value":  " | ".join(parts),
            "inline": False,
        })

    # Insight field
    insight = _insight_line(totals, by_sport)
    if insight:
        fields.append({
            "name":   "🔍 Insight",
            "value":  insight,
            "inline": False,
        })

    if not fields:
        fields.append({
            "name":  "No data",
            "value": "No alerts fired this week.",
            "inline": False,
        })

    # Title date range: "Jun 1-7, 2026" or "Jun 29-Jul 5, 2026"
    months = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    mo_s = months.get(week_start.month, str(week_start.month))
    if week_start.month == week_end.month:
        date_range = f"{mo_s} {week_start.day}-{week_end.day}, {week_start.year}"
    else:
        mo_e = months.get(week_end.month, str(week_end.month))
        date_range = f"{mo_s} {week_start.day}-{mo_e} {week_end.day}, {week_start.year}"

    payload = {
        "embeds": [{
            "title":       "📊 ObServatory Weekly Report",
            "description": f"Week of {date_range}",
            "color":       color,
            "fields":      fields,
            "footer":      {"text": "ObServatory"},
        }]
    }

    ok = _discord_post(webhook, payload)
    if ok:
        log.info("Discord weekly summary sent.")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Weekly picks performance Discord summary.")
    p.add_argument(
        "--week-start", default=None, metavar="YYYY-MM-DD",
        help="Monday of target week (default: current Mon-today window).",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print summary without sending Discord.")
    p.add_argument(
        "--db", default=None, metavar="PATH",
        help=f"Path to alerts.db (default: ALERTS_DB_PATH env var or {_DEFAULT_DB}).",
    )
    args = p.parse_args(argv)

    db_path = args.db or os.environ.get("ALERTS_DB_PATH") or str(_DEFAULT_DB)

    if not Path(db_path).exists():
        log.error("alerts.db not found: %s  (run setup_alerts_db.py first)", db_path)
        return 1

    today = date.today()

    if args.week_start:
        week_start, week_end = week_from_monday(args.week_start)
    else:
        # Default: current Mon-today so in-progress alerts are always visible.
        # Pass --week-start to pull a specific completed week.
        week_start, week_end = current_week_window(today)

    log.info("=== Weekly summary %s to %s ===", week_start, week_end)

    bets = load_bets(db_path, week_start, week_end)
    log.info("Loaded %d bets from alerts.db.", len(bets))

    webhook = os.environ.get("DISCORD_WEBHOOK_WEEKLY", "").strip()

    if not bets:
        log.info("No alerts in window %s-%s -- sending empty Discord notice.", week_start, week_end)
        if not args.dry_run:
            _discord_post(webhook, {"embeds": [{
                "title":       "📊 ObServatory Weekly Report",
                "description": "No alerts fired this week.",
                "color":       _COLOR_EVEN,
                "footer":      {"text": "ObServatory"},
            }]})
        return 0

    # Per-sport breakdown
    sports   = sorted({b["sport"] for b in bets})
    by_sport = {sp: sport_stats([b for b in bets if b["sport"] == sp]) for sp in sports}
    totals   = sport_stats(bets)

    for sp, s in by_sport.items():
        clv_str = (f"{s['clv_rate'][0]}/{s['clv_rate'][1]}"
                   if s["clv_rate"] else "n/a")
        log.info(
            "  %s: %d alerts, %d-%d-%d graded, profit=%+.2fu, roi=%s, CLV=%s, pending=%d",
            sp, s["total_alerts"], s["wins"], s["losses"], s["pushes"],
            s["profit"],
            f"{s['roi']:+.1f}%" if s["roi"] is not None else "n/a",
            clv_str, s["pending"],
        )

    if args.dry_run:
        log.info("DRY RUN -- Discord weekly summary not sent.")
        return 0

    ok = send_sms(week_start, week_end, by_sport, totals)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
