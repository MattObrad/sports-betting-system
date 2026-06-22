"""
grade_wnba.py — Auto-grade pending WNBA bets in alerts.db.

Pipeline:
    1. Query bet_alerts WHERE sport='WNBA' AND graded=0 AND alert_date < today
    2. Fetch actual player scores from ESPN public API
    3. Fetch closing odds from VPS Postgres props_snapshots (last snap before tipoff)
    4. Grade WIN/LOSS/PUSH, compute CLV, update the row

Run daily at 10am UTC (after WNBA games typically finish).
Cron: 0 10 * * * cd /home/picks && ALERTS_DB_PATH=/home/picks/alerts.db
      VPS_DB_HOST=localhost python3 grade_wnba.py >> /home/picks/logs/grade_wnba.log 2>&1

Usage:
    python grade_wnba.py
    python grade_wnba.py --date 2026-05-30   # re-grade a specific date
    python grade_wnba.py --dry-run           # show what would be graded, no writes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psycopg2

# ---------------------------------------------------------------------------
# Discord helper (inline — avoids import-path ambiguity with notify.py)
# ---------------------------------------------------------------------------
_COLOR_WIN  = 3066993   # green
_COLOR_LOSS = 15158332  # red
_COLOR_EVEN = 9807270   # grey


def _discord_post(webhook_url: str, payload: dict) -> bool:
    if not webhook_url:
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
            return resp.status in (200, 204)
    except Exception as exc:
        log.warning("Discord post failed: %s", exc)
        return False

_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_DIR / "logs" / "grade_wnba.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

ET = timedelta(hours=-4)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(_DIR / "config.json", encoding="utf-8") as _f:
    _CFG = json.load(_f)

_VPS = _CFG["vps_db"]

_ALERTS_DB = os.environ.get("ALERTS_DB_PATH") or str(_DIR / "alerts.db")

# ---------------------------------------------------------------------------
# ESPN helpers (mirrors backtest_wnba.py)
# ---------------------------------------------------------------------------
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={}"
ESPN_SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={}"


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=12) as r:
        return json.load(r)


def _norm(name: str) -> str:
    nfd = unicodedata.normalize("NFD", name.lower())
    return "".join(c for c in nfd if not unicodedata.combining(c))


def fetch_espn_scores(date_et: str) -> dict[str, float]:
    """Returns {normalized_player_name: points_scored}."""
    datekey = date_et.replace("-", "")
    board   = _fetch(ESPN_SCOREBOARD.format(datekey))
    results: dict[str, float] = {}
    for ev in board.get("events", []):
        status = ev.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("name", "")
        if "FINAL" not in status.upper():
            continue
        try:
            box = _fetch(ESPN_SUMMARY.format(ev["id"]))
        except Exception:
            continue
        for team_group in box.get("boxscore", {}).get("players", []):
            for stat_group in team_group.get("statistics", []):
                keys = stat_group.get("keys", [])
                try:
                    pts_idx = keys.index("points")
                except ValueError:
                    pts_idx = 1
                for athlete in stat_group.get("athletes", []):
                    player_name = athlete.get("athlete", {}).get("displayName", "")
                    stats = athlete.get("stats", [])
                    try:
                        results[_norm(player_name)] = float(stats[pts_idx])
                    except (IndexError, ValueError):
                        pass
    return results


# ---------------------------------------------------------------------------
# VPS Postgres: closing odds
# ---------------------------------------------------------------------------
_CLOSING_ODDS_SQL = """
SELECT DISTINCT ON (ps.player_name, ps.line)
    ps.over_odds AS closing_odds
FROM props_snapshots ps
JOIN games g ON g.event_id = ps.event_id
WHERE g.event_id  = %s
  AND ps.player_name = %s
  AND ps.line = %s
  AND ps.market_type = 'Player Points'
  AND ps.over_odds IS NOT NULL
  AND ps.snapshot_time < g.game_time
ORDER BY
    ps.player_name,
    ps.line,
    CASE WHEN SIGN(ps.over_odds) = SIGN(%s::integer) THEN 0 ELSE 1 END,
    ps.snapshot_time DESC,
    ps.over_odds DESC
"""

_GAME_ID_SQL = """
SELECT g.event_id, g.game_time
FROM games g
WHERE g.league = 'WNBA'
  AND DATE(g.game_time AT TIME ZONE 'America/New_York') = %s
"""


def fetch_closing_odds(pg_cur, event_id: str, player_name: str, line: float,
                       alert_odds: int) -> int | None:
    pg_cur.execute(_CLOSING_ODDS_SQL, (event_id, player_name, line, alert_odds))
    row = pg_cur.fetchone()
    return int(row[0]) if row else None


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

def decimal_from_american(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (-100 / odds + 1)


def implied_prob(odds: int) -> float:
    return 1.0 / decimal_from_american(odds)


def grade_bet(actual: float, line: float, direction: str) -> tuple[str, float | None]:
    """
    Returns (result, profit_units) for a flat $1 stake.
    direction='YES' means bet hits if actual >= line (milestone bet).
    """
    if direction == "YES":
        hit = actual >= line
        result = "WIN" if hit else "LOSS"
        return result, (decimal_from_american(-110) - 1.0) if hit else -1.0
    # OVER / UNDER for game totals
    if direction == "OVER":
        if actual > line:  return "WIN",  (decimal_from_american(-110) - 1.0)
        if actual < line:  return "LOSS", -1.0
        return "PUSH", 0.0
    if direction == "UNDER":
        if actual < line:  return "WIN",  (decimal_from_american(-110) - 1.0)
        if actual > line:  return "LOSS", -1.0
        return "PUSH", 0.0
    return "PENDING", None


def grade_with_actual_odds(
    actual: float, line: float, direction: str, open_odds: int
) -> tuple[str, float | None]:
    """Grade using the actual American odds stored in the alert (not -110 default)."""
    if direction == "YES":
        hit = actual >= line
        result = "WIN" if hit else "LOSS"
        return result, (decimal_from_american(open_odds) - 1.0) if hit else -1.0
    if direction == "OVER":
        if actual > line:  return "WIN",  (decimal_from_american(open_odds) - 1.0)
        if actual < line:  return "LOSS", -1.0
        return "PUSH", 0.0
    if direction == "UNDER":
        if actual < line:  return "WIN",  (decimal_from_american(open_odds) - 1.0)
        if actual > line:  return "LOSS", -1.0
        return "PUSH", 0.0
    return "PENDING", None


# ---------------------------------------------------------------------------
# Main grading loop
# ---------------------------------------------------------------------------

def grade_date(
    alerts_conn: sqlite3.Connection,
    pg_cur,
    date_et: str,
    dry_run: bool = False,
) -> dict:
    """Grade all ungraded WNBA bets for a given date. Returns summary counts."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Load ungraded bets for this date
    bets = alerts_conn.execute(
        "SELECT id, player_name, direction, line, odds, implied_prob, game_id "
        "FROM bet_alerts "
        "WHERE sport = 'WNBA' AND graded = 0 AND alert_date = ?",
        (date_et,),
    ).fetchall()

    if not bets:
        return {"date": date_et, "total": 0, "graded": 0, "pending": 0}

    log.info("  %s: %d ungraded bets to process", date_et, len(bets))

    # Fetch ESPN scores once for the whole date
    try:
        espn = fetch_espn_scores(date_et)
        log.info("  ESPN: %d player scores loaded for %s", len(espn), date_et)
    except Exception as exc:
        log.warning("  ESPN fetch failed for %s: %s — all bets remain PENDING", date_et, exc)
        return {"date": date_et, "total": len(bets), "graded": 0, "pending": len(bets)}

    graded = 0
    still_pending = 0

    for b in bets:
        bid          = b["id"]
        player_name  = b["player_name"]
        direction    = b["direction"]
        line         = float(b["line"])
        open_odds    = int(b["odds"])
        alert_impl   = float(b["implied_prob"]) if b["implied_prob"] else implied_prob(open_odds)
        game_id      = b["game_id"]

        # Actual score from ESPN
        actual = espn.get(_norm(player_name))
        if actual is None:
            log.debug("  No ESPN score for %s on %s", player_name, date_et)
            still_pending += 1
            continue

        # Closing odds from VPS — sign-matched to alert_odds cluster
        closing_odds = fetch_closing_odds(pg_cur, game_id, player_name, line, open_odds)
        closing_impl = implied_prob(closing_odds) if closing_odds is not None else None
        clv          = (closing_impl - alert_impl) if closing_impl is not None else None
        clv_beat     = (1 if clv > 0 else 0) if clv is not None else None

        # Grade
        result, profit = grade_with_actual_odds(actual, line, direction, open_odds)
        if result == "PENDING":
            still_pending += 1
            continue

        log.info(
            "  %s %s %.1f+  actual=%.0f  %s  profit=%s  CLV=%s",
            player_name, direction, line, actual, result,
            f"{profit:+.2f}" if profit is not None else "?",
            f"{clv*100:+.2f}pp" if clv is not None else "n/a",
        )

        if not dry_run:
            alerts_conn.execute(
                """
                UPDATE bet_alerts SET
                    closing_odds    = ?,
                    closing_implied = ?,
                    actual_result   = ?,
                    result          = ?,
                    profit_units    = ?,
                    clv             = ?,
                    clv_beat        = ?,
                    graded          = 1,
                    graded_at       = ?
                WHERE id = ?
                """,
                (closing_odds, closing_impl, actual, result, profit,
                 clv, clv_beat, now, bid),
            )
            graded += 1

    if not dry_run:
        alerts_conn.commit()

    return {
        "date":    date_et,
        "total":   len(bets),
        "graded":  graded,
        "pending": still_pending,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _post_wnba_results_embed(
    alerts_conn: sqlite3.Connection,
    webhook: str,
    dates: list,
) -> None:
    """Post per-bet results embed(s) for each graded date."""
    for date_et in dates:
        rows = alerts_conn.execute(
            """SELECT player_name, direction, line, odds, result, profit_units, clv
               FROM bet_alerts
               WHERE sport = 'WNBA' AND alert_date = ?
                 AND result IN ('WIN', 'LOSS', 'PUSH', 'PENDING')
               ORDER BY result != 'PENDING', player_name""",
            (date_et,),
        ).fetchall()

        if not rows:
            continue

        resolved = [r for r in rows if r["result"] != "PENDING"]
        pending  = [r for r in rows if r["result"] == "PENDING"]

        net   = sum((r["profit_units"] or 0.0) for r in resolved)
        color = _COLOR_WIN if net >= 0 else _COLOR_LOSS

        lines = []
        for r in resolved + pending:
            emoji   = {"WIN": "✅", "LOSS": "❌", "PUSH": "🔄"}.get(r["result"], "⏳")
            outcome = {"WIN": "WON", "LOSS": "LOST", "PUSH": "PUSH"}.get(r["result"], "PENDING")
            pname   = r["player_name"]
            dir_    = r["direction"]
            ln      = float(r["line"])

            bet_s   = f"{ln:.1f}+ pts" if dir_ == "YES" else f"{dir_} {ln:.1f}"
            odds_s  = f" @ {int(r['odds']):+d}" if r["odds"] is not None else ""

            if r["result"] == "PENDING":
                lines.append(f"⏳ {pname} — {bet_s} — PENDING")
                continue

            pnl   = f"{r['profit_units']:+.2f}u" if r["profit_units"] is not None else ""
            clv_s = (f"CLV: {float(r['clv'])*100:+.2f}pp"
                     if r["clv"] is not None else "")
            line1 = f"{emoji} {pname} — {bet_s} — {outcome} {pnl}{odds_s}"
            lines.append(line1 + (f"\n{clv_s}" if clv_s else ""))

        wins   = sum(1 for r in resolved if r["result"] == "WIN")
        losses = sum(1 for r in resolved if r["result"] == "LOSS")
        clvs   = [float(r["clv"]) * 100 for r in resolved if r["clv"] is not None]
        avg_clv_s = f"  |  Avg CLV: {sum(clvs)/len(clvs):+.1f}pp" if clvs else ""
        stat  = f"📊 {wins}-{losses} | {net:+.2f}u{avg_clv_s}  |  ObServatory WNBA Model"

        CHUNK = 4
        chunks = [lines[i:i+CHUNK] for i in range(0, max(len(lines), 1), CHUNK)]
        n = len(chunks)
        for idx, chunk in enumerate(chunks):
            suffix = f" ({idx+1}/{n})" if n > 1 else ""
            desc   = "\n\n".join(chunk) + f"\n\n{stat}"
            _discord_post(webhook, {"embeds": [{
                "title":       f"🏀 WNBA Results — {date_et}{suffix}",
                "color":       color,
                "description": desc,
                "footer":      {"text": "ObServatory WNBA Model"},
            }]})
            log.info("Discord WNBA results embed sent for %s%s.", date_et, suffix)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Auto-grade pending WNBA bets.")
    p.add_argument("--date",    default=None, metavar="YYYY-MM-DD",
                   help="Grade only this specific date (default: all ungraded < today)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be graded without writing anything")
    p.add_argument("--alerts-db", default=None)
    args = p.parse_args(argv)

    alerts_db = args.alerts_db or _ALERTS_DB
    today_et  = (datetime.now(timezone.utc) + ET).date().isoformat()

    log.info("=== grade_wnba starting (alerts.db: %s) ===", alerts_db)
    if args.dry_run:
        log.info("DRY RUN — no writes")

    # Connect to alerts.db
    if not Path(alerts_db).exists():
        log.error("alerts.db not found: %s  (run setup_alerts_db.py first)", alerts_db)
        return 1
    alerts_conn = sqlite3.connect(alerts_db)
    alerts_conn.row_factory = sqlite3.Row

    # Determine dates to grade
    if args.date:
        dates = [args.date]
    else:
        rows = alerts_conn.execute(
            "SELECT DISTINCT alert_date FROM bet_alerts "
            "WHERE sport = 'WNBA' AND graded = 0 AND alert_date < ? "
            "ORDER BY alert_date",
            (today_et,),
        ).fetchall()
        dates = [r["alert_date"] for r in rows]

    if not dates:
        log.info("No ungraded WNBA bets before %s.", today_et)
        alerts_conn.close()
        return 0

    log.info("Dates to grade: %s", dates)

    # Connect to VPS for closing odds
    try:
        pg = psycopg2.connect(
            host=os.environ.get("VPS_DB_HOST") or _VPS["host"],
            port=_VPS["port"],
            database=_VPS["database"],
            user=_VPS["user"],
            password=_VPS["password"],
        )
        pg_cur = pg.cursor()
        log.info("Connected to VPS Postgres.")
    except Exception as exc:
        log.warning("VPS Postgres unavailable (%s) — closing odds will be NULL.", exc)
        pg, pg_cur = None, None

    # Grade each date
    total_graded = 0
    for d in dates:
        summary = grade_date(alerts_conn, pg_cur, d, dry_run=args.dry_run)
        total_graded += summary["graded"]
        log.info(
            "  %s: %d/%d graded, %d still pending",
            d, summary["graded"], summary["total"], summary["pending"],
        )

    if pg:
        pg.close()

    log.info("=== Done. %d bets graded total. ===", total_graded)

    if total_graded > 0 and not args.dry_run:
        webhook = os.environ.get("DISCORD_WEBHOOK_RESULTS", "").strip()
        if webhook:
            # Dedup guard: prevent double Discord posts when morning + evening
            # crons overlap, or on manual re-runs of the same date.
            # Stored in alerts.db _collector_state; --date flag bypasses dedup
            # so a deliberate re-run (e.g. backfill) always re-sends.
            alerts_conn.execute(
                "CREATE TABLE IF NOT EXISTS _collector_state "
                "(key TEXT PRIMARY KEY, value TEXT)"
            )
            alerts_conn.commit()
            _manual = args.date is not None
            for d in dates:
                _dedup_key = f"results_discord_sent:WNBA:{d}"
                already = (not _manual) and bool(alerts_conn.execute(
                    "SELECT 1 FROM _collector_state WHERE key = ?",
                    (_dedup_key,),
                ).fetchone())
                if already:
                    log.info("Discord already sent for WNBA %s — skipping duplicate.", d)
                    continue
                _post_wnba_results_embed(alerts_conn, webhook, [d])
                alerts_conn.execute(
                    "INSERT OR REPLACE INTO _collector_state (key, value) VALUES (?, ?)",
                    (_dedup_key, "1"),
                )
                alerts_conn.commit()

    alerts_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
