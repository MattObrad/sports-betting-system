#!/usr/bin/env python3
"""
sync_odds.py — Daily cron: pull today's MLB odds from Postgres into SQLite.

Suggested crontab entry on the VPS (runs at 09:30 and 14:30 ET daily):
    30 13 * * * /usr/bin/python3 /home/picks/sync_odds.py >> /home/picks/logs/sync_odds.log 2>&1
    30 18 * * * /usr/bin/python3 /home/picks/sync_odds.py >> /home/picks/logs/sync_odds.log 2>&1

Pulls MLB Total Runs (over/under) snapshots for today's games from the local
Postgres picksdb, pivots Over/Under rows into single rows, maps to SQLite
game_ids, and inserts new rows into mlb_data.db with INSERT OR IGNORE.

No CLI flags needed — edit the constants below if paths or credentials change.
"""

import os
import sys
import sqlite3
from datetime import date, timedelta

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed.  Run: pip install psycopg2-binary", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config — edit these if anything changes on the VPS
# ---------------------------------------------------------------------------

SQLITE_DB = "/home/picks/mlb_data.db"
PG_HOST   = "localhost"
PG_PORT   = 5432
PG_DB     = "picksdb"
PG_USER   = "picksuser"
# Password: env var takes precedence so the plaintext default is never required
# in production.  Set PICKSDB_PASSWORD in /etc/environment or the cron env.
PG_PASS   = os.environ.get("PICKSDB_PASSWORD", "password")

# ---------------------------------------------------------------------------
# Team name aliases
# ---------------------------------------------------------------------------

# Kambi VPS team names look like "TOR Blue Jays" or "Athletics" (no prefix).
# The 3-letter prefix matches our SQLite team_code in most cases; this dict
# handles the exceptions.
_PREFIX_ALIASES: dict[str, str] = {
    "WAS": "WSH",   # Washington Nationals
    "ARI": "AZ",    # Arizona Diamondbacks
    "CHW": "CWS",   # Chicago White Sox
    "SFG": "SF",    # San Francisco Giants (rare)
    "SDP": "SD",    # San Diego Padres (rare)
    "TBR": "TB",    # Tampa Bay Rays (rare)
    "KAN": "KC",    # Kansas City Royals (rare)
}

# Kambi → SQLite team_code mapping.
# Handles: full-name-only teams, 2-letter prefixes, and CHI ambiguity.
_FULLNAME_MAP: dict[str, str] = {
    # Full-name entries (no 3-letter prefix)
    "Athletics":       "OAK",
    # 2-letter prefix teams — NY / LA are ambiguous, need full name
    "NY Yankees":      "NYY",
    "NY Mets":         "NYM",
    "LA Dodgers":      "LAD",
    "LA Angels":       "LAA",
    # CHI is shared by two teams — disambiguate by full name
    "CHI Cubs":        "CHC",
    "CHI White Sox":   "CWS",
    # 2-letter prefix teams that map directly to the same 2-letter SQLite code
    "SF Giants":       "SF",
    "SD Padres":       "SD",
    "TB Rays":         "TB",
    "KC Royals":       "KC",
}


def _team_to_code(vps_name: str) -> str | None:
    """
    Parse a Kambi VPS team name into our SQLite team_code.
    'TOR Blue Jays' → 'TOR'   (prefix lookup)
    'Athletics'     → 'OAK'   (full-name override)
    Returns None if the name cannot be resolved.
    """
    if not vps_name:
        return None
    name = vps_name.strip()
    if name in _FULLNAME_MAP:
        return _FULLNAME_MAP[name]
    parts = name.split()
    prefix = parts[0] if parts else ""
    if len(prefix) == 3 and prefix.isupper():
        return _PREFIX_ALIASES.get(prefix, prefix)
    # 2-letter prefix: pass through (e.g. SF→SF, SD→SD); ambiguous 2-letter
    # cases like NY/LA are handled above by the full-name map.
    if len(prefix) == 2 and prefix.isupper():
        return prefix
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    today_str = date.today().isoformat()
    print(f"[sync_odds] {today_str} — starting", flush=True)

    # ── 1. Fetch today's rows from Postgres ───────────────────────────────────
    try:
        pg = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DB,
            user=PG_USER, password=PG_PASS,
            connect_timeout=15,
        )
        pg.autocommit = True
    except Exception as exc:
        print(f"ERROR: cannot connect to Postgres: {exc}", flush=True)
        sys.exit(1)

    try:
        with pg.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    o.event_id,
                    o.snapshot_time,
                    o.outcome,
                    o.line,
                    o.odds,
                    g.home_team,
                    g.away_team,
                    g.game_time
                FROM odds_snapshots o
                JOIN games g ON g.event_id = o.event_id
                WHERE g.league        = 'MLB'
                  AND o.market_type   = 'Total Runs'
                  AND g.game_time::date = CURRENT_DATE
                ORDER BY o.event_id, o.snapshot_time, o.outcome
            """)
            pg_rows = cur.fetchall()
    finally:
        pg.close()

    print(f"  Postgres: {len(pg_rows)} Total Runs rows for today", flush=True)
    if not pg_rows:
        print("  Nothing to sync.", flush=True)
        return

    # ── 2. Pivot Over/Under rows → one row per (event_id, snapshot_time) ──────
    pivoted: dict = {}
    for r in pg_rows:
        key = (r["event_id"], str(r["snapshot_time"]))
        if key not in pivoted:
            pivoted[key] = {
                "event_id":      r["event_id"],
                "snapshot_time": str(r["snapshot_time"]),
                "total_line":    float(r["line"]) if r["line"] is not None else None,
                "home_team":     r["home_team"],
                "away_team":     r["away_team"],
                "game_time":     r["game_time"],
                "over_juice":    None,
                "under_juice":   None,
            }
        outcome = (r["outcome"] or "").strip().lower()
        if outcome == "over":
            pivoted[key]["over_juice"] = r["odds"]
        elif outcome == "under":
            pivoted[key]["under_juice"] = r["odds"]

    # Drop any pivot pair that is missing one side
    pivoted_list = [v for v in pivoted.values()
                    if v["over_juice"] is not None and v["under_juice"] is not None]
    n_incomplete = len(pivoted) - len(pivoted_list)
    if n_incomplete:
        print(f"  Dropped {n_incomplete} snapshots missing Over or Under side", flush=True)

    # ── 3. Mark is_opening / is_closing per event ─────────────────────────────
    # Earliest snapshot for the event = is_opening; latest = is_closing.
    event_times: dict = {}
    for p in pivoted_list:
        event_times.setdefault(p["event_id"], []).append(p["snapshot_time"])
    first_snap = {eid: min(ts) for eid, ts in event_times.items()}
    last_snap  = {eid: max(ts) for eid, ts in event_times.items()}

    # ── 4. Build SQLite game lookup ───────────────────────────────────────────
    try:
        sq = sqlite3.connect(SQLITE_DB)
        sq.execute("PRAGMA journal_mode=WAL")
    except Exception as exc:
        print(f"ERROR: cannot open SQLite {SQLITE_DB}: {exc}", flush=True)
        sys.exit(1)

    # {(game_date, home_code, away_code): game_id}
    game_idx: dict = {}
    for gid, gdate, hcode, acode in sq.execute("""
        SELECT g.game_id, g.game_date, ht.team_code, at.team_code
        FROM   games g
        JOIN   teams ht ON ht.team_id = g.home_team_id
        JOIN   teams at ON at.team_id = g.away_team_id
    """):
        game_idx[(gdate, hcode, acode)] = gid

    # ── 5. Match pivoted rows to SQLite game_ids ──────────────────────────────
    insert_rows: list[tuple] = []
    n_matched   = 0
    n_unmatched = 0

    for p in pivoted_list:
        home_code = _team_to_code(p["home_team"])
        away_code = _team_to_code(p["away_team"])
        if not home_code or not away_code:
            n_unmatched += 1
            continue

        # Try UTC date then ±1 day — handles late-night ET/UTC edge cases
        game_id   = None
        game_time = p["game_time"]          # datetime returned by psycopg2
        for offset in (0, -1, 1):
            d = (game_time + timedelta(days=offset)).strftime("%Y-%m-%d")
            game_id = game_idx.get((d, home_code, away_code))
            if game_id:
                break

        if game_id is None:
            n_unmatched += 1
            continue

        n_matched += 1
        snap_ts  = p["snapshot_time"]
        is_open  = 1 if snap_ts == first_snap[p["event_id"]] else 0
        is_close = 1 if snap_ts == last_snap[p["event_id"]] else 0

        insert_rows.append((
            game_id,
            "kambi",
            snap_ts,
            p["total_line"],
            p["over_juice"],
            p["under_juice"],
            is_open,
            is_close,
        ))

    # ── 6. INSERT OR IGNORE into SQLite odds_snapshots ────────────────────────
    inserted = 0
    if insert_rows:
        sq.execute("BEGIN")
        try:
            for row in insert_rows:
                cur = sq.execute("""
                    INSERT OR IGNORE INTO odds_snapshots
                      (game_id, book, snapshot_time, total_line,
                       over_juice, under_juice, is_opening, is_closing)
                    VALUES (?,?,?,?,?,?,?,?)
                """, row)
                inserted += cur.rowcount
            sq.execute("COMMIT")
        except Exception:
            sq.execute("ROLLBACK")
            sq.close()
            raise

    # ── 6b. Normalise is_opening / is_closing per touched game ────────────────
    # is_opening is set per daily batch (earliest snapshot IN THAT BATCH). Across
    # successive syncs a game accumulates MULTIPLE is_opening rows with different
    # lines, which biases the downstream opening-line read. Re-derive flags across
    # ALL snapshots of each touched game so exactly one opening (globally earliest)
    # and one closing (globally latest) survive.
    touched = sorted({r[0] for r in insert_rows})
    if touched:
        sq.execute("BEGIN")
        try:
            ph = ",".join("?" * len(touched))
            sq.execute(
                f"UPDATE odds_snapshots SET is_opening=0, is_closing=0 "
                f"WHERE game_id IN ({ph})", touched
            )
            sq.execute(
                f"""UPDATE odds_snapshots SET is_opening=1
                    WHERE (game_id, snapshot_time) IN (
                        SELECT game_id, MIN(snapshot_time) FROM odds_snapshots
                        WHERE game_id IN ({ph}) GROUP BY game_id
                    )""", touched
            )
            sq.execute(
                f"""UPDATE odds_snapshots SET is_closing=1
                    WHERE (game_id, snapshot_time) IN (
                        SELECT game_id, MAX(snapshot_time) FROM odds_snapshots
                        WHERE game_id IN ({ph}) GROUP BY game_id
                    )""", touched
            )
            sq.execute("COMMIT")
        except Exception:
            sq.execute("ROLLBACK")
            sq.close()
            raise

    sq.close()

    # ── 7. Summary ────────────────────────────────────────────────────────────
    skipped = len(insert_rows) - inserted
    print(f"  {n_matched} games matched  |  {n_unmatched} unmatched", flush=True)
    print(f"  {inserted} rows inserted  |  {skipped} already existed (skipped)", flush=True)
    print(f"[sync_odds] done", flush=True)


if __name__ == "__main__":
    main()
