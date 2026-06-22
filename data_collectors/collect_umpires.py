"""
collect_umpires.py

Populates: umpires, umpire_game, games.home_plate_umpire_id, _collector_state

Three data flows:

  1. Game-level pass  (--backfill or daily mode)
     UmpScorecards /api/games?startDate=&endDate=  -- year-by-year, 2017-present
     Upserts into umpires (by name) and umpire_game (joined on game_pk).
     Fields stored per game:
       total_run_impact  -> zone_size_score
       overall_accuracy  -> overall_accuracy (new column)
       called_pitches / called_correct / called_wrong
       n_challenged / n_overturned
       home+away score   -> total_runs

  2. Career stats pass  (--umpscorecards or daily mode)
     UmpScorecards /api/umpires  -- all-time career aggregates, no date filter
     Saves to _collector_state['umpscorecards_career'] keyed by normalized name.
     engineer_features.py reads this cache for ump_career_* feature columns:
       total_run_impact_mean  -> ump_career_runs_per_game
       overall_accuracy_wmean -> ump_career_over_rate
       consistency_wmean      -> ump_zone_size_score
       n                      -> ump_games_sampled

  3. Assignment pass  (daily mode only -- for upcoming/today's games)
     MLB Stats API /schedule?hydrate=officials
     Writes games.home_plate_umpire_id for today + next 7 days so that
     predict_mlb.py can join career stats against today's games.

Usage:
  python data_collectors/collect_umpires.py                    # daily (all 3 passes)
  python data_collectors/collect_umpires.py --backfill         # game pass 2017-present
  python data_collectors/collect_umpires.py --backfill --start 2022-01-01
  python data_collectors/collect_umpires.py --umpscorecards    # career cache only
  python data_collectors/collect_umpires.py --show-cache       # inspect cache, then exit
"""

import argparse
import datetime
import json
import os
import signal
import sqlite3
import sys
import threading
import time

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH           = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mlb_data.db"))
MLB_API           = "https://statsapi.mlb.com/api/v1"
UMP_API           = "https://umpscorecards.com"
MODEL_FIRST_YEAR  = 2017   # our training data floor

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown = threading.Event()


def _handle_sigint(sig, frame):
    print("\n[interrupt] Ctrl-C -- finishing current batch and saving progress...")
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_sigint)

# ---------------------------------------------------------------------------
# HTTP sessions
# ---------------------------------------------------------------------------

_mlb_session = requests.Session()
_mlb_session.headers["User-Agent"] = "mlb-totals-model/1.0 (research)"

_ump_session = requests.Session()
_ump_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         f"{UMP_API}/umpires",
})


def mlb_get(path: str, params: dict = None, retries: int = 3) -> dict:
    """MLB Stats API GET with exponential-backoff retry."""
    url = f"{MLB_API}/{path}"
    for attempt in range(retries):
        try:
            r = _mlb_session.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 4 ** attempt
            print(f"\n  [retry {attempt+1}] {exc} -- waiting {wait}s", flush=True)
            time.sleep(wait)


def ump_get(path: str, params: dict = None, retries: int = 3, timeout: int = 30):
    """UmpScorecards API GET; returns parsed JSON (list or dict)."""
    url = f"{UMP_API}{path}"
    for attempt in range(retries):
        try:
            r = _ump_session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 4 ** attempt
            print(f"\n  [retry {attempt+1}] {exc} -- waiting {wait}s", flush=True)
            time.sleep(wait)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def ensure_state_table(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS _collector_state (
            key TEXT PRIMARY KEY, value TEXT
        )
    """)
    con.commit()


def get_state(con: sqlite3.Connection, key: str, default=None):
    row = con.execute("SELECT value FROM _collector_state WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_state(con: sqlite3.Connection, key: str, value):
    con.execute(
        """INSERT INTO _collector_state(key,value) VALUES(?,?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
        (key, json.dumps(value, default=str)),
    )
    con.commit()


def ensure_umpire_game_columns(con: sqlite3.Connection):
    """
    Add UmpScorecards-specific columns to umpire_game if they don't exist yet.
    Safe to call on every run (skips columns that are already present).
    """
    existing = {row[1] for row in con.execute("PRAGMA table_info(umpire_game)").fetchall()}
    new_cols = [
        ("called_pitches",   "INTEGER"),
        ("called_correct",   "INTEGER"),
        ("called_wrong",     "INTEGER"),
        ("overall_accuracy", "REAL"),
        ("n_challenged",     "INTEGER"),
        ("n_overturned",     "INTEGER"),
    ]
    added = []
    for col, ctype in new_cols:
        if col not in existing:
            con.execute(f"ALTER TABLE umpire_game ADD COLUMN {col} {ctype}")
            added.append(col)
    if added:
        con.commit()
        print(f"  Schema migrated: added {added} to umpire_game")

# ---------------------------------------------------------------------------
# Umpire helpers
# ---------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    return name.strip().lower() if name else ""


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def get_or_create_umpire(con: sqlite3.Connection, umpire_name: str) -> int:
    """
    Return umpire_id for umpire_name, inserting a new row if needed.
    Used for UmpScorecards data which provides names but not MLB person IDs.
    """
    row = con.execute(
        "SELECT umpire_id FROM umpires WHERE umpire_name=?", (umpire_name,)
    ).fetchone()
    if row:
        return row["umpire_id"]
    con.execute(
        "INSERT OR IGNORE INTO umpires(umpire_name, active) VALUES(?, 1)", (umpire_name,)
    )
    return con.execute(
        "SELECT umpire_id FROM umpires WHERE umpire_name=?", (umpire_name,)
    ).fetchone()["umpire_id"]


def update_game_umpire(con: sqlite3.Connection, game_id: int, umpire_id: int):
    """Backfill games.home_plate_umpire_id only if currently NULL."""
    con.execute(
        "UPDATE games SET home_plate_umpire_id=? WHERE game_id=? AND home_plate_umpire_id IS NULL",
        (umpire_id, game_id),
    )

# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------


def upsert_ump_game_row(con: sqlite3.Connection, row: dict):
    """
    Upsert one UmpScorecards /api/games row into umpire_game.
    row must have _umpire_id pre-resolved.
    """
    home  = row.get("home_score")
    away  = row.get("away_score")
    total = (home + away) if (home is not None and away is not None) else None

    con.execute("""
        INSERT INTO umpire_game(
            game_id, game_date, umpire_id, total_runs,
            zone_size_score, overall_accuracy,
            called_pitches, called_correct, called_wrong,
            n_challenged, n_overturned,
            source)
        VALUES(
            :game_id, :game_date, :umpire_id, :total_runs,
            :zone_size_score, :overall_accuracy,
            :called_pitches, :called_correct, :called_wrong,
            :n_challenged, :n_overturned,
            'umpscorecards')
        ON CONFLICT(game_id, umpire_id) DO UPDATE SET
            zone_size_score  = COALESCE(excluded.zone_size_score,  zone_size_score),
            overall_accuracy = COALESCE(excluded.overall_accuracy, overall_accuracy),
            called_pitches   = COALESCE(excluded.called_pitches,   called_pitches),
            called_correct   = COALESCE(excluded.called_correct,   called_correct),
            called_wrong     = COALESCE(excluded.called_wrong,     called_wrong),
            n_challenged     = COALESCE(excluded.n_challenged,     n_challenged),
            n_overturned     = COALESCE(excluded.n_overturned,     n_overturned),
            total_runs       = COALESCE(excluded.total_runs,       total_runs)
    """, {
        "game_id":          row["game_pk"],
        "game_date":        row.get("date", "")[:10],
        "umpire_id":        row["_umpire_id"],
        "total_runs":       total,
        "zone_size_score":  row.get("total_run_impact"),    # run impact of all calls
        "overall_accuracy": row.get("overall_accuracy"),    # % called pitches correct
        "called_pitches":   row.get("called_pitches"),
        "called_correct":   row.get("called_correct"),
        "called_wrong":     row.get("called_wrong"),
        "n_challenged":     row.get("n_challenged"),
        "n_overturned":     row.get("n_overturned"),
    })


def upsert_umpire_by_mlb_id(con: sqlite3.Connection, umpire_id: int, umpire_name: str):
    """Insert/update umpire using the integer MLB person ID (daily assignment pass)."""
    con.execute("""
        INSERT INTO umpires(umpire_id, umpire_name, active) VALUES(?,?,1)
        ON CONFLICT(umpire_id) DO UPDATE SET umpire_name=excluded.umpire_name
    """, (umpire_id, umpire_name))


def upsert_umpire_game_assignment(
    con: sqlite3.Connection, game_id: int, game_date: str, umpire_id: int
):
    """Minimal insert for the daily MLB API assignment pass."""
    con.execute("""
        INSERT OR IGNORE INTO umpire_game(game_id, game_date, umpire_id, source)
        VALUES(?, ?, ?, 'mlbapi')
    """, (game_id, game_date, umpire_id))

# ---------------------------------------------------------------------------
# UmpScorecards fetchers
# ---------------------------------------------------------------------------


def fetch_ump_games(start_date: str, end_date: str) -> list[dict]:
    """
    GET /api/games?startDate=&endDate= from UmpScorecards.
    Returns Regular-season rows that are not marked failed.
    NOTE: date filter is ignored by the API -- use startDate/endDate.
    """
    data = ump_get("/api/games", params={"startDate": start_date, "endDate": end_date})
    rows = data.get("rows", []) if isinstance(data, dict) else (data or [])
    return [r for r in rows if r.get("type") == "R" and not r.get("failed", False)]


def fetch_ump_career() -> list[dict]:
    """
    GET /api/umpires (all-time, no date filter) from UmpScorecards.
    Returns career aggregate rows for every umpire in the database.
    """
    data = ump_get("/api/umpires")
    return data.get("rows", []) if isinstance(data, dict) else (data or [])

# ---------------------------------------------------------------------------
# MLB Stats API helpers (daily assignment pass)
# ---------------------------------------------------------------------------


def fetch_schedule_with_officials(start_date: str, end_date: str) -> list[dict]:
    data = mlb_get("schedule", {
        "sportId":   1,
        "startDate": start_date,
        "endDate":   end_date,
        "gameType":  "R",
        "hydrate":   "officials,team",
    })
    return [g for block in data.get("dates", []) for g in block.get("games", [])]


def extract_hp_umpire(game: dict) -> tuple:
    for official in game.get("officials", []):
        if official.get("officialType", "").strip() == "Home Plate":
            person = official.get("official", {})
            return person.get("id"), person.get("fullName")
    return None, None

# ---------------------------------------------------------------------------
# Game-level pass  (UmpScorecards /api/games)
# ---------------------------------------------------------------------------


def run_games_pass(start_date: str, end_date: str):
    """
    Pull per-game scorecard data from UmpScorecards and upsert into umpire_game.
    Fetches year-by-year (~2,400 games/request) to stay inside API return limits.
    Skips games already written from UmpScorecards.
    Skips games not in the games table (no FK to anchor against).
    """
    con = get_db()
    ensure_state_table(con)
    ensure_umpire_game_columns(con)

    start = datetime.date.fromisoformat(start_date)
    end   = datetime.date.fromisoformat(end_date)

    # Year-by-year chunks (season-level granularity avoids API truncation)
    chunks: list[tuple[str, str]] = []
    for yr in range(start.year, end.year + 1):
        cs = max(start, datetime.date(yr,  1,  1))
        ce = min(end,   datetime.date(yr, 12, 31))
        if cs <= ce:
            chunks.append((cs.isoformat(), ce.isoformat()))

    print(f"\n[umpires-games] {start_date} -> {end_date}  ({len(chunks)} yearly chunk(s))")

    # Pre-build lookup sets to avoid repeated DB queries inside the inner loop
    done_ids: set[int] = set(
        r[0] for r in con.execute(
            "SELECT DISTINCT game_id FROM umpire_game WHERE source='umpscorecards'"
        ).fetchall()
    )
    known_game_ids: set[int] = set(
        r[0] for r in con.execute("SELECT game_id FROM games").fetchall()
    )

    print(f"  Already done (umpscorecards source): {len(done_ids)}")
    print(f"  Games in DB available for FK join  : {len(known_game_ids)}")

    # Per-name umpire_id cache avoids repeated SELECT for the same umpire
    ump_id_cache: dict[str, int] = {}

    total_written = 0
    total_no_fk   = 0
    total_errors  = 0

    for chunk_idx, (cs, ce) in enumerate(chunks, 1):
        if _shutdown.is_set():
            print(f"\n[umpires-games] Stopped after chunk {chunk_idx - 1}.")
            break

        print(f"  [{chunk_idx:>2}/{len(chunks)}]  {cs[:4]}  ", end="", flush=True)

        try:
            rows = fetch_ump_games(cs, ce)
        except Exception as exc:
            print(f"FETCH ERROR: {exc}")
            continue

        new_rows = [
            r for r in rows
            if r["game_pk"] not in done_ids
        ]
        no_fk = sum(1 for r in new_rows if r["game_pk"] not in known_game_ids)
        to_do = [r for r in new_rows if r["game_pk"] in known_game_ids]

        print(
            f"{len(rows)} from API | "
            f"{len(to_do)} to write | "
            f"{len(rows) - len(new_rows)} already done | "
            f"{no_fk} no FK",
            flush=True,
        )

        written = 0
        for row in to_do:
            ump_name = (row.get("umpire") or "").strip()
            if not ump_name:
                continue

            if ump_name not in ump_id_cache:
                ump_id_cache[ump_name] = get_or_create_umpire(con, ump_name)
            row["_umpire_id"] = ump_id_cache[ump_name]

            try:
                upsert_ump_game_row(con, row)
                update_game_umpire(con, row["game_pk"], row["_umpire_id"])
                done_ids.add(row["game_pk"])
                written += 1
            except Exception as exc:
                print(f"\n    [warn] game {row['game_pk']}: {exc}")
                total_errors += 1

        con.commit()
        total_written += written
        total_no_fk   += no_fk
        time.sleep(0.5)   # polite delay between yearly fetches

    print()
    print(f"[umpires-games] Pass complete.")
    print(f"  Written   : {total_written}")
    print(f"  No FK     : {total_no_fk} (UmpScorecards game not in our games table)")
    print(f"  Errors    : {total_errors}")
    print(f"  Umpires   : {len(ump_id_cache)} unique names resolved")
    con.close()

# ---------------------------------------------------------------------------
# Career stats pass  (UmpScorecards /api/umpires)
# ---------------------------------------------------------------------------


def run_umpscorecards_pass():
    """
    Fetch all-time career stats from UmpScorecards /api/umpires and cache in
    _collector_state['umpscorecards_career'] keyed by normalized umpire name.

    Field mapping to game_features columns:
      total_run_impact_mean  -> ump_career_runs_per_game
      overall_accuracy_wmean -> ump_career_over_rate    (percentage, 0-100)
      consistency_wmean      -> ump_zone_size_score
      n                      -> ump_games_sampled
    """
    con = get_db()
    ensure_state_table(con)

    print("\n[umpires-career] Fetching UmpScorecards /api/umpires (all-time)...")

    try:
        career_rows = fetch_ump_career()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print("  Career cache not updated -- using existing cache if present.")
        con.close()
        return

    if not career_rows:
        print("  Empty response -- career cache not updated.")
        con.close()
        return

    print(f"  {len(career_rows)} umpires returned from API")

    cache: dict = {}
    for row in career_rows:
        name = (row.get("umpire") or "").strip()
        if not name:
            continue
        cache[normalize_name(name)] = {
            "name":                    name,
            "ump_career_runs_per_game": row.get("total_run_impact_mean"),
            "ump_career_over_rate":     row.get("overall_accuracy_wmean"),
            "ump_zone_size_score":      row.get("consistency_wmean"),
            "ump_games_sampled":        row.get("n"),
            "weighted_score":           row.get("weighted_score"),
            "fetched_at":               now_iso(),
        }

    # Merge with existing cache: new data wins, don't erase umpires missing from
    # this fetch (retired umpires may disappear from the live API response).
    existing = get_state(con, "umpscorecards_career", {})
    existing.update(cache)
    set_state(con, "umpscorecards_career", existing)
    set_state(con, "umpscorecards_last_fetched", now_iso())

    # Coverage summary
    def filled(field):
        return sum(1 for v in cache.values() if v.get(field) is not None)

    print(f"  Cached {len(cache)} umpires in _collector_state:")
    print(f"    ump_career_runs_per_game : {filled('ump_career_runs_per_game')}/{len(cache)}")
    print(f"    ump_career_over_rate     : {filled('ump_career_over_rate')}/{len(cache)}")
    print(f"    ump_zone_size_score      : {filled('ump_zone_size_score')}/{len(cache)}")
    print(f"    ump_games_sampled        : {filled('ump_games_sampled')}/{len(cache)}")

    # Cross-reference: umpires in umpires table with no UmpScorecards entry
    db_umps = con.execute("SELECT umpire_name FROM umpires").fetchall()
    if db_umps:
        missing = [
            r["umpire_name"] for r in db_umps
            if normalize_name(r["umpire_name"]) not in cache
        ]
        if missing:
            print(f"\n  [note] {len(missing)} umpire(s) in umpires table not in UmpScorecards:")
            for name in missing[:8]:
                print(f"    {name}")
            if len(missing) > 8:
                print(f"    ... and {len(missing) - 8} more")

    con.close()
    print("[umpires-career] Pass complete.")

# ---------------------------------------------------------------------------
# Daily assignment pass  (MLB Stats API -- for today + upcoming games)
# ---------------------------------------------------------------------------


def run_assignment_pass(start_date: str, end_date: str):
    """
    Pull today's/upcoming umpire assignments from the MLB Stats API.
    UmpScorecards data lags a few hours after game completion, so the
    assignment pass ensures today's games have an umpire linked before
    predict_mlb.py runs at 10am ET.
    """
    con = get_db()
    ensure_state_table(con)
    ensure_umpire_game_columns(con)

    print(f"\n[umpires-assign] {start_date} -> {end_date} (MLB Stats API)...")

    done_mlb: set[int] = set(
        r[0] for r in con.execute(
            "SELECT DISTINCT game_id FROM umpire_game WHERE source='mlbapi'"
        ).fetchall()
    )

    try:
        games = fetch_schedule_with_officials(start_date, end_date)
    except Exception as exc:
        print(f"  ERROR fetching schedule: {exc}")
        con.close()
        return

    assigned = 0
    for g in games:
        game_pk = g["gamePk"]
        gdate   = g.get("gameDate", "")[:10]
        if game_pk in done_mlb:
            continue
        ump_id, ump_name = extract_hp_umpire(g)
        if not ump_id:
            continue
        upsert_umpire_by_mlb_id(con, ump_id, ump_name)
        upsert_umpire_game_assignment(con, game_pk, gdate, ump_id)
        update_game_umpire(con, game_pk, ump_id)
        assigned += 1

    con.commit()
    print(f"  Assigned: {assigned} new game(s)")
    con.close()
    print("[umpires-assign] Pass complete.")

# ---------------------------------------------------------------------------
# Cache display
# ---------------------------------------------------------------------------


def print_cache(n: int = 12):
    con = get_db()
    ensure_state_table(con)
    cache = get_state(con, "umpscorecards_career", {})
    last  = get_state(con, "umpscorecards_last_fetched", "never")

    print(f"\n  umpscorecards_career cache")
    print(f"  Last fetched : {last}")
    print(f"  Total entries: {len(cache)}")

    if not cache:
        print("  (empty)")
        con.close()
        return

    print()
    hdr = f"  {'Umpire':<26} {'G':>5} {'Runs/G':>7} {'OvrAcc%':>8} {'Zone':>7} {'Score':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for key, v in list(cache.items())[:n]:
        name   = v.get("name", key)[:25]
        games  = v.get("ump_games_sampled")
        rpg    = v.get("ump_career_runs_per_game")
        over   = v.get("ump_career_over_rate")
        zone   = v.get("ump_zone_size_score")
        score  = v.get("weighted_score")

        g_str    = f"{games:>5}"   if games  is not None else "  n/a"
        rpg_str  = f"{rpg:>7.3f}" if rpg    is not None else "    n/a"
        over_str = f"{over:>8.2f}" if over   is not None else "     n/a"
        zone_str = f"{zone:>7.2f}" if zone   is not None else "    n/a"
        score_str= f"{score:>6.1f}" if score  is not None else "   n/a"

        print(f"  {name:<26} {g_str} {rpg_str} {over_str} {zone_str} {score_str}")

    if len(cache) > n:
        print(f"  ... and {len(cache) - n} more")

    con.close()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Collect MLB umpire data from UmpScorecards API and MLB Stats API"
    )
    parser.add_argument("--backfill", action="store_true",
                        help="Game-level pass: pull UmpScorecards /api/games 2017-present")
    parser.add_argument("--start", metavar="YYYY-MM-DD",
                        help="Backfill start date (default: 2017-03-20)")
    parser.add_argument("--end", metavar="YYYY-MM-DD",
                        help="Backfill end date (default: yesterday)")
    parser.add_argument("--umpscorecards", action="store_true",
                        help="Career stats pass only: pull /api/umpires and update cache")
    parser.add_argument("--show-cache", action="store_true",
                        help="Print the cached career stats and exit")
    parser.add_argument("--date", metavar="YYYY-MM-DD",
                        help="Daily mode: override today's date")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        sys.exit(f"[error] Database not found at {DB_PATH}. Run setup_db.py first.")

    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    print("=" * 60)
    print("MLB Umpire Collector")
    print(f"  DB: {DB_PATH}")
    print("=" * 60)

    # --show-cache: inspect and exit
    if args.show_cache:
        print_cache(n=15)
        return

    # --umpscorecards: career cache only
    if args.umpscorecards:
        run_umpscorecards_pass()
        print("\nDone.")
        return

    # --backfill: historical game-level pass + career cache
    if args.backfill:
        start_date = args.start or f"{MODEL_FIRST_YEAR}-03-20"
        end_date   = args.end   or yesterday.isoformat()
        print(f"  Mode: backfill {start_date} -> {end_date}")
        run_games_pass(start_date, end_date)
        run_umpscorecards_pass()
        print("\nDone.")
        return

    # Daily mode: recent game scores + today's assignment + career cache refresh
    date_str   = args.date or today.isoformat()
    game_start = (datetime.date.fromisoformat(date_str) - datetime.timedelta(days=2)).isoformat()
    game_end   = yesterday.isoformat()
    assign_end = (datetime.date.fromisoformat(date_str) + datetime.timedelta(days=7)).isoformat()
    print(f"  Mode: daily ({date_str})")

    # Pick up the last 2 days of UmpScorecards data (same-day data may not be scored yet)
    if game_start <= game_end:
        run_games_pass(game_start, game_end)

    # Today + 7 days of assignments from MLB Stats API
    run_assignment_pass(date_str, assign_end)

    # Refresh career cache weekly (or always in daily mode)
    run_umpscorecards_pass()

    print("\nDone.")


if __name__ == "__main__":
    main()
