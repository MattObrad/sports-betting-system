"""
collect_lineups.py

Populates: lineups, players, games.home_starter_id / away_starter_id

Sources:
  - /schedule?hydrate=lineups,probablePitcher  (daily mode)
  - /game/{gamePk}/boxscore                    (backfill mode)
  - /people?personIds=...                      (player metadata, batched)

Lineup timing note:
  Official lineups drop ~30-90 min before first pitch. The VPS cron runs
  predictions at 10am UTC / 6am ET — well before most lineups are confirmed.
  Games with no confirmed lineup are flagged; predict_mlb.py uses roster
  fallback for those. Re-run this script closer to first pitch to pick up
  confirmed lineups before late games.

Usage:
  python data_collectors/collect_lineups.py                    # today (daily cron)
  python data_collectors/collect_lineups.py --date 2024-09-15  # specific date
  python data_collectors/collect_lineups.py --backfill         # 2023-01-01 to yesterday
  python data_collectors/collect_lineups.py --backfill --start 2022-01-01
  python data_collectors/collect_lineups.py --backfill --season 2023
"""

import argparse
import datetime
import os
import signal
import sqlite3
import sys
import threading
import time

import requests

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mlb_data.db"))
MLB_API = "https://statsapi.mlb.com/api/v1"

SKIP_STATUSES = {"Postponed", "Cancelled", "Suspended"}

# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown = threading.Event()


def _handle_sigint(sig, frame):
    print("\n[interrupt] Ctrl-C — finishing current item and saving progress...")
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_sigint)

# ── DB helpers ────────────────────────────────────────────────────────────────


def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def get_known_player_ids(con: sqlite3.Connection) -> set[int]:
    return {r[0] for r in con.execute("SELECT player_id FROM players").fetchall()}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers["User-Agent"] = "mlb-totals-model/1.0 (research)"


def http_get(url: str, params: dict = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 4 ** attempt
            print(f"\n  [retry {attempt+1}] {exc} — waiting {wait}s", flush=True)
            time.sleep(wait)


# ── ETA tracker ───────────────────────────────────────────────────────────────


class ETA:
    def __init__(self, total: int, label: str = "items"):
        self.total = total
        self.done = 0
        self.label = label
        self._start = time.time()

    def tick(self):
        self.done += 1

    def line(self) -> str:
        elapsed = time.time() - self._start
        pct = 100.0 * self.done / self.total if self.total else 0
        if self.done == 0:
            eta_str = "calculating..."
        else:
            secs_left = (self.total - self.done) * elapsed / self.done
            eta_str = str(datetime.timedelta(seconds=int(secs_left)))
        return f"  {self.done}/{self.total} {self.label} ({pct:.1f}%) — ETA: {eta_str}"


# ── Helpers ───────────────────────────────────────────────────────────────────


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _get_pos(obj: dict) -> str:
    """Extract position abbreviation from a player dict (handles multiple API shapes)."""
    for key in ("primaryPosition", "position"):
        pos = obj.get(key)
        if isinstance(pos, dict):
            return pos.get("abbreviation", "")
    return ""


# ── MLB Stats API — daily schedule ───────────────────────────────────────────


def fetch_daily_games(date_str: str) -> list[dict]:
    # gameType=R in the REQUEST now gets rejected by MLB's API with a 406
    # (isolated 2026-07-18: any gameType param, singular or plural, any casing,
    # any header combination, triggers it -- sportId+date alone still works).
    # Filter to regular season CLIENT-SIDE on the response's gameType field
    # instead, so the regular-season-only guarantee doesn't depend on a
    # request param MLB's API may reject again in the future.
    data = http_get(f"{MLB_API}/schedule", {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher,team",  # lineups hydration removed (MLB API now returns 406)
    })
    games = [g for block in data.get("dates", []) for g in block.get("games", [])]
    return [g for g in games if g.get("gameType") == "R"]


def parse_daily_game(g: dict, game_date: str) -> dict:
    """
    Parse one game dict from the schedule (lineups+probablePitcher hydration).

    Returns:
      game_id       : int
      lineup_rows   : list of dicts ready for lineups table
      starter_ids   : {'home': id, 'away': id}  (probable pitchers)
      new_player_ids: set of player IDs seen (for metadata fetch)
      lineup_confirmed: bool — True when official lineup was present
    """
    game_pk = g["gamePk"]
    teams = g.get("teams", {})
    lineups_data = g.get("lineups", {})

    lineup_rows: list[dict] = []
    starter_ids: dict[str, int] = {}
    new_pids: set[int] = set()
    lineup_confirmed = False

    for side in ("home", "away"):
        td = teams.get(side, {})
        team_id = td.get("team", {}).get("id")

        # Probable / confirmed starting pitcher
        prob = td.get("probablePitcher") or {}
        prob_id = prob.get("id")
        if prob_id:
            starter_ids[side] = prob_id
            new_pids.add(prob_id)

        # Official batting order (only present when lineup is submitted)
        players_key = "homePlayers" if side == "home" else "awayPlayers"
        players = lineups_data.get(players_key) or []
        if not players:
            continue  # not yet released — predict_mlb.py uses roster fallback

        lineup_confirmed = True

        for p in players:
            # Player ID: API returns either {id, ...} or {person: {id, ...}, ...}
            pid = p.get("id") or (p.get("person") or {}).get("id")
            if not pid:
                continue

            bo_raw = str(p.get("battingOrder") or "0")
            try:
                bo = int(bo_raw) // 100  # "300" → 3
            except ValueError:
                bo = 0
            if not 1 <= bo <= 9:
                continue

            new_pids.add(pid)
            lineup_rows.append({
                "game_id":      game_pk,
                "game_date":    game_date,
                "team_id":      team_id,
                "player_id":    pid,
                "batting_order": bo,
                "position":     _get_pos(p),
                "confirmed":    1,
                "updated_at":   now_iso(),
            })

    return {
        "game_id":          game_pk,
        "lineup_rows":      lineup_rows,
        "starter_ids":      starter_ids,
        "new_player_ids":   new_pids,
        "lineup_confirmed": lineup_confirmed,
    }


# ── MLB Stats API — boxscore (backfill) ──────────────────────────────────────


def fetch_boxscore_lineup(game_pk: int, game_date: str) -> dict:
    """
    Extract the starting batting order from a completed game's box score.
    Filters to battingOrder % 100 == 0 to exclude substitutes.
    Returns same shape as parse_daily_game.
    """
    bs = http_get(f"{MLB_API}/game/{game_pk}/boxscore")
    teams_data = bs.get("teams", {})

    lineup_rows: list[dict] = []
    starter_ids: dict[str, int] = {}
    new_pids: set[int] = set()

    for side in ("home", "away"):
        td = teams_data.get(side, {})
        team_id = (td.get("team") or {}).get("id")
        players = td.get("players", {})
        pitcher_order = td.get("pitchers", [])

        if pitcher_order:
            starter_ids[side] = pitcher_order[0]
            new_pids.add(pitcher_order[0])

        for pid_key, pdata in players.items():
            bo_raw = str(pdata.get("battingOrder") or "")
            if not bo_raw:
                continue
            try:
                bo_int = int(bo_raw)
            except ValueError:
                continue

            # Substitutes have non-round battingOrder (e.g. 301 = sub in slot 3)
            if bo_int % 100 != 0:
                continue
            bo = bo_int // 100
            if not 1 <= bo <= 9:
                continue

            pid = (pdata.get("person") or {}).get("id")
            if not pid:
                continue

            new_pids.add(pid)
            lineup_rows.append({
                "game_id":       game_pk,
                "game_date":     game_date,
                "team_id":       team_id,
                "player_id":     pid,
                "batting_order": bo,
                "position":      _get_pos(pdata),
                "confirmed":     1,
                "updated_at":    now_iso(),
            })

    return {
        "game_id":          game_pk,
        "lineup_rows":      lineup_rows,
        "starter_ids":      starter_ids,
        "new_player_ids":   new_pids,
        "lineup_confirmed": bool(lineup_rows),
    }


# ── MLB Stats API — player metadata ──────────────────────────────────────────


def fetch_players_batch(player_ids: list[int]) -> list[dict]:
    """Batch fetch player metadata from /people endpoint, 100 IDs per call."""
    all_people: list[dict] = []
    total_chunks = (len(player_ids) + 99) // 100
    for i, chunk in enumerate(chunks(player_ids, 100), 1):
        ids_str = ",".join(str(pid) for pid in chunk)
        print(f"\r  Fetching player metadata batch {i}/{total_chunks}...", end="", flush=True)
        try:
            data = http_get(f"{MLB_API}/people", {"personIds": ids_str})
            all_people.extend(data.get("people", []))
        except Exception as exc:
            print(f"\n  [warning] player batch {i} failed: {exc}", flush=True)
        time.sleep(0.3)
    print()
    return all_people


def parse_player_row(p: dict) -> dict:
    return {
        "player_id":  p.get("id"),
        "name_first": p.get("firstName", ""),
        "name_last":  p.get("lastName", ""),
        "throws":     (p.get("pitchHand") or {}).get("code", ""),
        "bats":       (p.get("batSide") or {}).get("code", ""),
        "position":   (p.get("primaryPosition") or {}).get("abbreviation", ""),
        "active":     1 if p.get("active", True) else 0,
    }


# ── DB writes ─────────────────────────────────────────────────────────────────


def upsert_lineup_rows(con: sqlite3.Connection, rows: list[dict]):
    for r in rows:
        con.execute("""
            INSERT INTO lineups(
                game_id, game_date, team_id, player_id,
                batting_order, position, confirmed, updated_at)
            VALUES(
                :game_id,:game_date,:team_id,:player_id,
                :batting_order,:position,:confirmed,:updated_at)
            ON CONFLICT(game_id, team_id, batting_order) DO UPDATE SET
                player_id    = excluded.player_id,
                position     = excluded.position,
                confirmed    = excluded.confirmed,
                updated_at   = excluded.updated_at
        """, r)


def upsert_players(con: sqlite3.Connection, rows: list[dict]):
    for r in rows:
        if not r.get("player_id"):
            continue
        con.execute("""
            INSERT INTO players(
                player_id, name_first, name_last, throws, bats, position, active)
            VALUES(
                :player_id,:name_first,:name_last,:throws,:bats,:position,:active)
            ON CONFLICT(player_id) DO UPDATE SET
                active   = excluded.active,
                position = CASE WHEN excluded.position != '' THEN excluded.position
                                ELSE position END
        """, r)


def update_game_starters(con: sqlite3.Connection, game_id: int, starter_ids: dict):
    """Write probable/confirmed starters to games table; never overwrites a set value."""
    if sid := starter_ids.get("home"):
        con.execute("""
            UPDATE games SET home_starter_id=?
            WHERE game_id=? AND home_starter_id IS NULL
        """, (sid, game_id))
    if sid := starter_ids.get("away"):
        con.execute("""
            UPDATE games SET away_starter_id=?
            WHERE game_id=? AND away_starter_id IS NULL
        """, (sid, game_id))


def _flush_players(con: sqlite3.Connection, new_pids: set[int], known_pids: set[int]):
    """Batch-fetch and upsert any player IDs not yet in the players table."""
    unseen = new_pids - known_pids
    if not unseen:
        return
    print(f"  Fetching metadata for {len(unseen)} new players...")
    people = fetch_players_batch(list(unseen))
    rows = [parse_player_row(p) for p in people]
    upsert_players(con, rows)
    con.commit()
    known_pids.update(unseen)
    print(f"  {len(rows)} player(s) upserted into players table.")


# ── Daily mode ────────────────────────────────────────────────────────────────


def run_daily(date_str: str):
    con = get_db()
    print(f"\n[lineups] Daily — {date_str}")

    games = fetch_daily_games(date_str)
    games = [
        g for g in games
        if g.get("status", {}).get("detailedState", "") not in SKIP_STATUSES
    ]

    if not games:
        print("  No games scheduled (or all postponed/cancelled).")
        con.close()
        return

    print(f"  {len(games)} game(s) on schedule")

    known_pids = get_known_player_ids(con)
    all_new_pids: set[int] = set()
    confirmed_count = 0

    for g in games:
        if _shutdown.is_set():
            break
        parsed = parse_daily_game(g, date_str)
        upsert_lineup_rows(con, parsed["lineup_rows"])
        update_game_starters(con, parsed["game_id"], parsed["starter_ids"])
        all_new_pids.update(parsed["new_player_ids"])
        if parsed["lineup_confirmed"]:
            confirmed_count += 1
        con.commit()

    missing = len(games) - confirmed_count
    print(f"  Lineups confirmed : {confirmed_count}/{len(games)}")
    print(f"  Probable pitchers written to games table")
    if missing:
        print(f"  [{missing} game(s) have no confirmed lineup — "
              f"predict_mlb.py will use roster fallback for those]")

    _flush_players(con, all_new_pids, known_pids)
    con.close()
    print("[lineups] Daily pass complete.")


# ── Backfill mode ─────────────────────────────────────────────────────────────


def run_backfill(start_date: str, end_date: str):
    con = get_db()
    print(f"\n[lineups] Backfill — {start_date} → {end_date}")

    # Only process final games not already in lineups table
    rows = con.execute("""
        SELECT g.game_id, g.game_date
        FROM games g
        WHERE g.game_date BETWEEN ? AND ?
          AND g.status = 'Final'
          AND g.game_id NOT IN (SELECT DISTINCT game_id FROM lineups)
        ORDER BY g.game_date
    """, (start_date, end_date)).fetchall()

    if not rows:
        print("  No games to backfill.")
        con.close()
        return

    print(f"  {len(rows)} games need batting orders")
    print("  (Ctrl-C saves progress and exits cleanly)")

    known_pids = get_known_player_ids(con)
    all_new_pids: set[int] = set()
    eta = ETA(len(rows), "games")
    errors: list[tuple] = []

    for row in rows:
        if _shutdown.is_set():
            print(f"\n[lineups] Stopped. {eta.done} games saved.")
            break

        game_pk   = row["game_id"]
        game_date = row["game_date"]

        try:
            parsed = fetch_boxscore_lineup(game_pk, game_date)
            upsert_lineup_rows(con, parsed["lineup_rows"])
            update_game_starters(con, game_pk, parsed["starter_ids"])
            all_new_pids.update(parsed["new_player_ids"])
            con.commit()
        except Exception as exc:
            con.rollback()
            errors.append((game_pk, str(exc)))

        eta.tick()
        print(f"\r{eta.line()}", end="", flush=True)
        time.sleep(0.3)

    print()

    if errors:
        print(f"  [warnings] {len(errors)} games failed:")
        for gid, msg in errors[:10]:
            print(f"    game {gid}: {msg}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    _flush_players(con, all_new_pids, known_pids)
    con.close()
    print("[lineups] Backfill complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Collect MLB starting lineups and probable pitchers into SQLite"
    )
    parser.add_argument("--date", type=str, metavar="YYYY-MM-DD",
                        help="Date to collect in daily mode (default: today)")
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill historical batting orders from box scores")
    parser.add_argument("--season", type=int, metavar="YYYY",
                        help="Backfill a full season (sets --start/--end automatically)")
    parser.add_argument("--start", type=str, metavar="YYYY-MM-DD",
                        help="Backfill start date (default: 2023-01-01)")
    parser.add_argument("--end", type=str, metavar="YYYY-MM-DD",
                        help="Backfill end date (default: yesterday)")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        sys.exit(f"[error] Database not found at {DB_PATH}. Run setup_db.py first.")

    today = datetime.date.today()

    print("=" * 60)
    print("MLB Lineup Collector")
    print(f"  DB: {DB_PATH}")

    if args.backfill or args.season:
        if args.season:
            start_date = f"{args.season}-03-20"
            end_date   = f"{args.season}-11-05"
        else:
            start_date = args.start or "2023-01-01"
            end_date   = args.end   or (today - datetime.timedelta(days=1)).isoformat()
        print(f"  Mode: backfill {start_date} → {end_date}")
        print("=" * 60)
        run_backfill(start_date, end_date)
    else:
        date_str = args.date or today.isoformat()
        print(f"  Mode: daily — {date_str}")
        print("=" * 60)
        run_daily(date_str)

    print("\nDone.")


if __name__ == "__main__":
    main()
