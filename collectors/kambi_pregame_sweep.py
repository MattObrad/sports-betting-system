#!/usr/bin/env python3
"""
kambi_pregame_sweep.py — guaranteed closing-line collector for MLB.

Queries picksdb for any MLB game starting within the next WINDOW_MINUTES,
then calls process_props_for_event() for each so we always capture a true
T-0 to T-5min closing snapshot regardless of where the hourly cron lands.

Cron:
  */5 * * * * cd /home/picks && python3 collectors/kambi_pregame_sweep.py >> /home/picks/logs/pregame_sweep.log 2>&1

Fast path: if no games in window, prints one line and exits in <1s.
"""
import sys
import time
import argparse
from datetime import datetime, timezone

from kambi_shared import db_connect, ensure_tables, process_props_for_event, SLEEP_SECS

WINDOW_MINUTES = 10
MLB_LEAGUE     = "MLB"


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def _upcoming_mlb(conn, window_minutes):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT event_id, home_team, away_team, game_time
            FROM   games
            WHERE  league    = %s
              AND  game_time BETWEEN NOW()
                                 AND NOW() + (%s * INTERVAL '1 minute')
            ORDER  BY game_time
        """, (MLB_LEAGUE, window_minutes))
        return cur.fetchall()


def _next_mlb(conn):
    """Nearest future MLB event — used by --dry-run when window is empty."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT event_id, home_team, away_team, game_time
            FROM   games
            WHERE  league    = %s
              AND  game_time > NOW()
            ORDER  BY game_time
            LIMIT  1
        """, (MLB_LEAGUE,))
        return cur.fetchone()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-game prop sweep: collect closing MLB props within 10min of first pitch"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print which events would be swept without hitting the Kambi API",
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    conn = db_connect()
    try:
        ensure_tables(conn)
        events = _upcoming_mlb(conn, WINDOW_MINUTES)

        # ── Fast path ────────────────────────────────────────────────────────
        if not events:
            if args.dry_run:
                nxt = _next_mlb(conn)
                if nxt:
                    eid, home, away, gt = nxt
                    print(
                        f"[{ts}] dry-run: no events in {WINDOW_MINUTES}-min window.\n"
                        f"  Next MLB: {away} @ {home}  game_time={gt}  event_id={eid}"
                    )
                else:
                    print(f"[{ts}] dry-run: no events in window, no upcoming MLB in DB")
            else:
                print(f"[{ts}] pregame_sweep: no games in window, exiting")
            return

        # ── Events found ─────────────────────────────────────────────────────
        print(f"[{ts}] pregame_sweep: {len(events)} MLB event(s) in {WINDOW_MINUTES}-min window")
        for event_id, home, away, game_time in events:
            print(f"  event_id={event_id}  {away} @ {home}  starts={game_time}")

        if args.dry_run:
            print(
                f"[{ts}] dry-run: would call process_props_for_event() for "
                f"{len(events)} event(s) — skipping API"
            )
            return

        # ── Collect ──────────────────────────────────────────────────────────
        total_rows = 0
        with conn.cursor() as cur:
            for i, (event_id, home, away, game_time) in enumerate(events):
                if i > 0:
                    time.sleep(SLEEP_SECS)
                rows, _ = process_props_for_event(cur, str(event_id))
                total_rows += rows
                print(f"  {away} @ {home} ({event_id}): {rows} prop row(s)")
        conn.commit()
        print(
            f"[{ts}] pregame_sweep: done — "
            f"{total_rows} rows inserted across {len(events)} event(s)"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
