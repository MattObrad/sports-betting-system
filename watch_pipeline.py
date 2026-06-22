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
