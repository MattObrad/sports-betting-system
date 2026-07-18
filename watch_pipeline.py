"""
watch_pipeline.py -- Daily health check for the ObServatory pipeline.

Checks all three sport pipelines and posts to Discord if anything looks dead.
Run after all morning jobs have completed:

  0 16 * * * cd /home/picks && python3 watch_pipeline.py >> /home/picks/logs/watchdog.log 2>&1
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_LOG_DIR = _DIR / "logs"

# Postgres (props_snapshots_v2) config -- same as kambi_shared.py
_PG_CONFIG = {
    "host": "localhost", "dbname": "picksdb", "user": "picksuser",
    "password": "password", "port": 5432,
}

# Core market_type/league pairs that should ALWAYS have rows in-season if the
# collector pipeline is healthy -- the exact class of gap the handicap-market
# bugs (Run Line/Moneyline/Point Spread/Puck Line/Asian Handicap) exposed.
# NHL/soccer entries are informational-only (off-season for large stretches of
# the year) -- flagged but not treated as a hard failure.
_HARD_MARKET_CHECKS = [
    ("Run Line", "MLB"),
    ("Moneyline", "MLB"),
]
_SOFT_MARKET_CHECKS = [
    ("Point Spread", "NFL"),
    ("Point Spread", "NCAAF"),
    ("Puck Line", "NHL"),
    ("Asian Handicap", None),  # any soccer league
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

_MLB_DB      = str(_DIR / "mlb_data.db")
_ALERTS_DB   = str(_DIR / "alerts.db")

_TODAY_UTC = datetime.now(timezone.utc).date().isoformat()

# ── Discord ───────────────────────────────────────────────────────────────────

def _discord_post(webhook_url: str, content: str) -> bool:
    if not webhook_url:
        return False
    try:
        data = json.dumps({"content": content}).encode("utf-8")
        req  = urllib.request.Request(
            webhook_url, data=data,
            headers={"Content-Type": "application/json", "User-Agent": "ObServatory/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as exc:
        log.error("Discord post failed: %s", exc)
        return False


# ── individual checks ─────────────────────────────────────────────────────────

def _last_line_date(log_path: Path) -> str | None:
    """Return the date in the most recent line of a log file, or None."""
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", line)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def _log_contains_today(log_path: Path, pattern: str | None = None) -> bool:
    """True if the log file contains today's date (optionally matching a pattern too)."""
    if not log_path.exists():
        return False
    today = _TODAY_UTC
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if today not in text:
            return False
        if pattern and pattern not in text:
            return False
        return True
    except Exception:
        return False


def _sync_odds_matched_today() -> bool:
    """True if sync_odds.log shows > 0 matched games for today's date."""
    log_path = _LOG_DIR / "sync_odds.log"
    if not log_path.exists():
        return False
    today = _TODAY_UTC
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        # Find blocks for today, look for '0 games matched' vs 'N games matched'
        blocks = text.split("[sync_odds]")
        for block in reversed(blocks):
            if today in block:
                m = re.search(r"(\d+) games matched", block)
                if m:
                    return int(m.group(1)) > 0
    except Exception:
        pass
    return False


def _mlb_db_has_today_games() -> bool:
    """True if mlb_data.db.games has at least one game for today."""
    try:
        con = sqlite3.connect(_MLB_DB)
        row = con.execute(
            "SELECT COUNT(*) FROM games WHERE game_date = ?", (_TODAY_UTC,)
        ).fetchone()
        con.close()
        return row[0] > 0
    except Exception:
        return False


def _mlb_db_last_game_date() -> str | None:
    try:
        con = sqlite3.connect(_MLB_DB)
        row = con.execute("SELECT MAX(game_date) FROM games").fetchone()
        con.close()
        return row[0]
    except Exception:
        return None


def _wnba_boxscores_fresh(max_stale_days: int = 8) -> bool:
    """True if wehoop_boxscores.log ran within max_stale_days (weekly cron)."""
    log_path = _LOG_DIR / "wehoop_boxscores.log"
    if not log_path.exists():
        return False
    try:
        # Look for the most recent date in the log
        text = log_path.read_text(encoding="utf-8", errors="replace")
        dates = re.findall(r"(\d{4}-\d{2}-\d{2})", text)
        if not dates:
            return False
        last = max(dates)
        delta = (datetime.fromisoformat(_TODAY_UTC) - datetime.fromisoformat(last)).days
        return delta <= max_stale_days
    except Exception:
        return False


# ── new checks: the exact class of gap that hid the 2026-06-29 MLB API break ──

def _stale_scheduled_games(max_hours: int = 6) -> list[tuple]:
    """Games whose scheduled start was more than max_hours ago but are still
    'Scheduled' -- the exact signature of the 3-week silent status-updater
    break (games started, played, finished, and mlb_data.db never noticed).
    Returns list of (game_id, game_date, game_time_utc, hours_stale)."""
    try:
        con = sqlite3.connect(_MLB_DB)
        rows = con.execute("""
            SELECT game_id, game_date, game_time_utc FROM games
            WHERE status = 'Scheduled' AND game_time_utc IS NOT NULL
        """).fetchall()
        con.close()
    except Exception:
        return []
    now = datetime.now(timezone.utc)
    stale = []
    for gid, gdate, gtime in rows:
        try:
            start = datetime.fromisoformat(gtime.replace("Z", "+00:00"))
        except Exception:
            continue
        hours = (now - start).total_seconds() / 3600
        if hours > max_hours:
            stale.append((gid, gdate, gtime, round(hours, 1)))
    return stale


def _market_type_zero_rows(market_type: str, league: str | None, days: int = 7) -> int | None:
    """Row count for a market_type (optionally scoped to a league) in the last
    N days from props_snapshots_v2. Returns None if the DB is unreachable."""
    try:
        import psycopg2
    except ImportError:
        return None
    try:
        conn = psycopg2.connect(**_PG_CONFIG, connect_timeout=10)
        cur = conn.cursor()
        if league:
            cur.execute("""
                SELECT count(*) FROM props_snapshots_v2 p
                JOIN games g ON p.event_id = g.event_id
                WHERE p.market_type = %s AND g.league = %s
                  AND p.snapshot_time > now() - interval %s
            """, (market_type, league, f"{days} days"))
        else:
            cur.execute("""
                SELECT count(*) FROM props_snapshots_v2
                WHERE market_type = %s AND snapshot_time > now() - interval %s
            """, (market_type, f"{days} days"))
        n = cur.fetchone()[0]
        cur.close()
        conn.close()
        return n
    except Exception:
        return None


def _collector_last_run_age_hours(log_name: str) -> float | None:
    """Hours since the log file's own most recent date-stamped line. None if
    the log doesn't exist or can't be parsed."""
    log_path = _LOG_DIR / log_name
    if not log_path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
    except Exception:
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def run_checks() -> list[str]:
    """Return a list of failure messages (empty = all healthy)."""
    failures: list[str] = []

    # --- MLB ---
    if not _mlb_db_has_today_games():
        last = _mlb_db_last_game_date() or "unknown"
        failures.append(
            f"⚠️ **MLB statcast** — no games in mlb_data.db for {_TODAY_UTC}. "
            f"Last game date: `{last}`. "
            f"Check: `/home/picks/logs/mlb_statcast.log`"
        )

    if not _sync_odds_matched_today():
        failures.append(
            f"⚠️ **MLB sync_odds** — matched 0 games today ({_TODAY_UTC}). "
            f"Market features will be NULL; predictions may fire garbage edges. "
            f"Check: `/home/picks/logs/sync_odds.log`"
        )

    if not _log_contains_today(_LOG_DIR / "predict_mlb.log"):
        failures.append(
            f"⚠️ **predict_mlb** — no entry for {_TODAY_UTC} in predict_mlb.log. "
            f"Check: `/home/picks/logs/predict_mlb.log`"
        )

    # --- Tennis ---
    tennis_log = _LOG_DIR / "collect_tennis.log"
    if not _log_contains_today(tennis_log):
        last = _last_line_date(tennis_log) or "unknown"
        failures.append(
            f"⚠️ **Tennis scraper** — no entry for {_TODAY_UTC}. "
            f"Last run: `{last}`. "
            f"Check: `/home/picks/logs/collect_tennis.log`"
        )

    if not _log_contains_today(_LOG_DIR / "predict_tennis.log"):
        failures.append(
            f"⚠️ **predict_tennis** — no entry for {_TODAY_UTC}. "
            f"Check: `/home/picks/logs/predict_tennis.log`"
        )

    # --- WNBA ---
    if not _wnba_boxscores_fresh():
        failures.append(
            f"⚠️ **WNBA box scores** — wehoop_boxscores last ran >8 days ago. "
            f"Predictions running on stale rolling stats. "
            f"Check: `/home/picks/logs/wehoop_boxscores.log`"
        )

    # --- Stale-Scheduled games: the exact signature of the 2026-06-29 break ---
    stale_games = _stale_scheduled_games(max_hours=6)
    if stale_games:
        sample = ", ".join(str(g[0]) for g in stale_games[:5])
        oldest = max(g[3] for g in stale_games)
        failures.append(
            f"🔴 **Stale Scheduled games** — {len(stale_games)} game(s) started >6h ago but "
            f"still show status='Scheduled' (oldest: {oldest}h). This is the exact signature "
            f"of the 2026-06-29 MLB Stats API break (gameType=R 406, undetected for 3 weeks). "
            f"Sample game_ids: {sample}. Check the boxscore pass in mlb_statcast.log."
        )

    # --- Whitelisted market_type collection gaps (the handicap-bug class) ---
    for market_type, league in _HARD_MARKET_CHECKS:
        n = _market_type_zero_rows(market_type, league)
        if n is None:
            failures.append(
                f"⚠️ **Market check unreachable** — could not query props_snapshots_v2 for "
                f"'{market_type}'{f' ({league})' if league else ''}. Postgres connection issue?"
            )
        elif n == 0:
            failures.append(
                f"🔴 **Silent collection gap** — market_type='{market_type}'"
                f"{f' league={league}' if league else ''} has ZERO rows in the last 7 days. "
                f"This is the same failure class as the Run Line/Moneyline/Point Spread "
                f"misclassification bugs -- a market silently stopped landing correctly."
            )

    soft_notes = []
    for market_type, league in _SOFT_MARKET_CHECKS:
        n = _market_type_zero_rows(market_type, league)
        if n == 0:
            soft_notes.append(f"{market_type}{f'/{league}' if league else ''}")
    if soft_notes:
        log.info("Soft market checks at zero (informational, likely off-season): %s",
                  ", ".join(soft_notes))

    # --- Key collector staleness (statcast/status updater, odds collectors) ---
    collector_max_age = {
        "mlb_statcast.log": 14,   # runs 2x/day (7am, noon UTC)
        "mlb_lineups.log": 26,    # runs 1x/day (7:30am UTC)
        "core.log": 2,            # runs hourly
    }
    for log_name, max_age in collector_max_age.items():
        age = _collector_last_run_age_hours(log_name)
        if age is None:
            failures.append(f"⚠️ **{log_name}** — log file missing or unreadable, can't verify last run.")
        elif age > max_age:
            failures.append(
                f"🔴 **{log_name}** — last touched {age:.1f}h ago (expected <{max_age}h). "
                f"Collector may be stalled or its cron may have stopped firing."
            )

    return failures


def main() -> None:
    log.info("Pipeline watchdog starting — %s", _TODAY_UTC)

    failures = run_checks()

    if not failures:
        log.info("All checks passed.")
        return

    log.warning("%d check(s) failed:", len(failures))
    for f in failures:
        log.warning("  %s", f)

    webhook = os.environ.get("DISCORD_WEBHOOK_RESULTS", "").strip()
    if not webhook:
        log.warning("DISCORD_WEBHOOK_RESULTS not set — alerts not posted to Discord.")
        return

    header = f"## 🔴 ObServatory Pipeline Alert — {_TODAY_UTC}\n"
    body   = "\n\n".join(failures)
    msg    = header + body

    if _discord_post(webhook, msg):
        log.info("Alert posted to Discord.")
    else:
        log.error("Failed to post alert to Discord.")


if __name__ == "__main__":
    main()
