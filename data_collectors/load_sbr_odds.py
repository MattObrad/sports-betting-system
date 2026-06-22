"""
load_sbr_odds.py — Load SportsBookReview historical totals into odds_snapshots.

Sources:
  Pre-built dataset (2021-03-20 to 2025-08-16):
    https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/download/dataset/mlb_odds_dataset.json

  For 2019-2020: run the SBR scraper separately and pass --json FILE.

Usage:
  python data_collectors/load_sbr_odds.py                    # download + load full dataset
  python data_collectors/load_sbr_odds.py --json FILE        # load from local JSON file
  python data_collectors/load_sbr_odds.py --dry-run          # match stats only, no writes
  python data_collectors/load_sbr_odds.py --season 2023      # restrict to one season
  python data_collectors/load_sbr_odds.py --clear            # DELETE existing rows then reload
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, date as date_type, timedelta, timezone

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR    = os.path.dirname(__file__)
DB_PATH = os.path.normpath(os.path.join(_DIR, "..", "mlb_data.db"))
CACHE_DIR = os.path.normpath(os.path.join(_DIR, "..", "data", "odds_cache"))

DATASET_URL = (
    "https://github.com/ArnavSaraogi/mlb-odds-scraper"
    "/releases/download/dataset/mlb_odds_dataset.json"
)
DATASET_FILENAME = "mlb_odds_dataset.json"

# ── Book priority ──────────────────────────────────────────────────────────────
# Lower index = higher priority.  SBR sportsbook names are lowercase strings.
BOOK_PRIORITY = [
    "pinnacle",
    "draftkings",
    "fanduel",
    "bet365",
    "caesars",
    "betmgm",
    "pointsbet",
    "barstool",
    "unibet",
    "bovada",
]

# ── Team code alias map ────────────────────────────────────────────────────────
# Maps SBR shortName → our teams.team_code.
# Populated from known MLB abbreviation differences; extended after dry-runs.
SBR_TO_OUR: dict[str, str] = {
    # SBR uses 3-letter codes; our DB uses what the MLB Stats API returns
    "ARI": "AZ",    # Arizona Diamondbacks
    "CHW": "CWS",   # Chicago White Sox (SBR sometimes uses CHW)
    "CWS": "CWS",   # already matches
    "WAS": "WSH",   # Washington Nationals
    "WSH": "WSH",   # already matches
    "KAN": "KC",    # Kansas City Royals (rare long-form)
    "SFG": "SF",    # San Francisco Giants (older SBR style)
    "SDP": "SD",    # San Diego Padres
    "TBR": "TB",    # Tampa Bay Rays
    "ATH": "OAK",   # Athletics (if SBR uses ATH)
    "OAK": "OAK",   # already matches
    # Everything else should match 1:1 with our team_code
}


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_dataset(dest_path: str) -> str:
    """Stream-download the 80 MB JSON with a progress bar. Returns local path."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {DATASET_URL}")
    print(f"  -> {dest_path}")

    resp = requests.get(DATASET_URL, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk = 65_536

    with open(dest_path, "wb") as fh:
        for data in resp.iter_content(chunk_size=chunk):
            fh.write(data)
            downloaded += len(data)
            if total:
                pct = downloaded * 100 // total
                mb  = downloaded / 1_048_576
                print(f"\r  {pct:3d}%  {mb:6.1f} MB", end="", flush=True)
    print()
    print(f"  Downloaded {downloaded / 1_048_576:.1f} MB")
    return dest_path


# =============================================================================
# DB HELPERS
# =============================================================================

def build_game_index(con: sqlite3.Connection, season: int | None) -> dict:
    """
    Returns {(date_str, home_code, away_code): [game_id, ...]}

    List because doubleheaders can have two games on the same date for the
    same matchup.  We build team_code lookups via the teams table.
    """
    season_filter = f"AND g.season = {season}" if season else ""
    rows = con.execute(f"""
        SELECT g.game_id, g.game_date,
               ht.team_code AS home_code,
               at.team_code AS away_code
        FROM games g
        JOIN teams ht ON ht.team_id = g.home_team_id
        JOIN teams at ON at.team_id = g.away_team_id
        WHERE g.status IN ('Final', 'Completed Early', 'Postponed')
          {season_filter}
    """).fetchall()

    idx: dict = {}
    for game_id, game_date, home_code, away_code in rows:
        key = (game_date, home_code, away_code)
        idx.setdefault(key, []).append(game_id)
    return idx


def build_team_code_set(con: sqlite3.Connection) -> set:
    return {r[0] for r in con.execute("SELECT team_code FROM teams")}


# =============================================================================
# BOOK SELECTION
# =============================================================================

def pick_book(odds_list: list) -> dict | None:
    """
    From a list of {sportsbook, openingLine, currentLine} dicts,
    return the entry for the highest-priority book that has a non-null
    total on both opening and closing lines.
    Falls back to the first usable entry if no priority book found.
    """
    if not odds_list:
        return None

    usable = [
        b for b in odds_list
        if b
        and (b.get("openingLine") or {}).get("total") is not None
        and (b.get("currentLine") or {}).get("total") is not None
    ]
    if not usable:
        return None

    book_map = {b["sportsbook"].lower(): b for b in usable if b.get("sportsbook")}

    for priority_book in BOOK_PRIORITY:
        if priority_book in book_map:
            return book_map[priority_book]

    # No priority book — return the first usable entry
    return usable[0]


# =============================================================================
# MATCHING
# =============================================================================

def resolve_code(sbr_code: str, valid_codes: set) -> str | None:
    """Map an SBR shortName to our team_code, or None if unknown."""
    if sbr_code in valid_codes:
        return sbr_code
    mapped = SBR_TO_OUR.get(sbr_code)
    if mapped and mapped in valid_codes:
        return mapped
    return None


def match_game(
    date_str: str,
    sbr_home: str,
    sbr_away: str,
    valid_codes: set,
    game_index: dict,
) -> tuple[int | None, str]:
    """
    Returns (game_id, reason_string).
    reason: 'ok' | 'ok_offset' | 'no_alias' | 'not_in_db' | 'doubleheader_ambiguous'

    SBR uses the scheduled date; our DB uses the actual play date.
    Postponed games (very common in COVID era) are stored a day later in our DB.
    We try the exact date first, then +1 day, then -1 day.
    98% of date mismatches are +1 day (game postponed from scheduled to next day).
    """
    home = resolve_code(sbr_home, valid_codes)
    away = resolve_code(sbr_away, valid_codes)

    if home is None or away is None:
        missing = []
        if home is None: missing.append(f"home={sbr_home!r}")
        if away is None: missing.append(f"away={sbr_away!r}")
        return None, f"no_alias:{','.join(missing)}"

    d = date_type.fromisoformat(date_str)
    for offset, label in [(0, "ok"), (1, "ok_offset+1"), (-1, "ok_offset-1")]:
        key = ((d + timedelta(offset)).isoformat(), home, away)
        games = game_index.get(key, [])
        if games:
            if len(games) == 1:
                return games[0], label
            # Doubleheader — take game with lower id (game 1 of the day)
            return min(games), "doubleheader_ambiguous"

    return None, f"not_in_db:{date_str}:{home}v{away}"


# =============================================================================
# MAIN LOAD LOOP
# =============================================================================

def load_dataset(
    con: sqlite3.Connection,
    data: dict,
    dry_run: bool,
    season: int | None,
) -> dict:
    """
    Iterate every game in the JSON, match to our games table, write 2 rows.
    Returns stats dict.
    """
    valid_codes = build_team_code_set(con)
    game_index  = build_game_index(con, season)

    stats = {
        "dates_seen":      0,
        "games_seen":      0,
        "no_totals_odds":  0,
        "no_alias":        0,
        "not_in_db":       0,
        "dh_ambiguous":    0,
        "matched":         0,
        "matched_offset":  0,   # matched via ±1 day fallback
        "rows_written":    0,
        "season_skipped":  0,
        "unmatched_pairs": {},   # (sbr_home, sbr_away) -> count
    }

    insert_rows: list[tuple] = []

    for date_str in sorted(data.keys()):
        # Filter by season if requested
        try:
            yr = int(date_str[:4])
        except ValueError:
            continue
        if season and yr != season:
            stats["season_skipped"] += len(data[date_str])
            continue

        stats["dates_seen"] += 1
        games_on_date = data[date_str]

        for game in games_on_date:
            stats["games_seen"] += 1

            game_view = game.get("gameView", {})
            game_type = game_view.get("gameType", "R")
            if game_type != "R":
                # Skip spring training, all-star, playoffs (our model is reg-season)
                stats["season_skipped"] += 1
                continue

            sbr_home = (game_view.get("homeTeam") or {}).get("shortName", "")
            sbr_away = (game_view.get("awayTeam") or {}).get("shortName", "")
            start_dt  = game_view.get("startDate", f"{date_str}T18:00:00+00:00")

            # Get totals odds
            totals_list = (game.get("odds") or {}).get("totals", [])
            chosen = pick_book(totals_list)
            if chosen is None:
                stats["no_totals_odds"] += 1
                continue

            # Match to game_id
            game_id, reason = match_game(
                date_str, sbr_home, sbr_away, valid_codes, game_index
            )
            if game_id is None:
                if reason.startswith("no_alias"):
                    stats["no_alias"] += 1
                    pair = (sbr_away, sbr_home)
                    stats["unmatched_pairs"][pair] = stats["unmatched_pairs"].get(pair, 0) + 1
                else:
                    stats["not_in_db"] += 1
                continue

            if reason == "doubleheader_ambiguous":
                stats["dh_ambiguous"] += 1
            if "offset" in reason:
                stats["matched_offset"] += 1

            stats["matched"] += 1

            opening = chosen["openingLine"]
            closing  = chosen["currentLine"]
            book_name = chosen.get("sportsbook", "sbr").lower()

            # Opening row
            if opening.get("total") is not None:
                insert_rows.append((
                    game_id,
                    book_name,
                    f"{date_str}T00:00:00",       # proxy: start of game day
                    opening["total"],
                    opening.get("overOdds"),
                    opening.get("underOdds"),
                    1,   # is_opening
                    0,   # is_closing
                ))

            # Closing row
            if closing.get("total") is not None:
                insert_rows.append((
                    game_id,
                    book_name,
                    start_dt,
                    closing["total"],
                    closing.get("overOdds"),
                    closing.get("underOdds"),
                    0,   # is_opening
                    1,   # is_closing
                ))

    stats["rows_written"] = len(insert_rows)

    if not dry_run and insert_rows:
        con.execute("BEGIN")
        try:
            con.executemany("""
                INSERT OR IGNORE INTO odds_snapshots
                  (game_id, book, snapshot_time, total_line,
                   over_juice, under_juice, is_opening, is_closing)
                VALUES (?,?,?,?,?,?,?,?)
            """, insert_rows)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    return stats


# =============================================================================
# REPORTING
# =============================================================================

def print_report(stats: dict, dry_run: bool) -> None:
    mode = "[DRY RUN] " if dry_run else ""
    total = stats["games_seen"]
    print(f"\n{mode}=== SBR Odds Load Report ===")
    print(f"  Dates processed       : {stats['dates_seen']}")
    print(f"  Games seen (regular)  : {total}")
    print(f"  Skipped (non-regular) : {stats['season_skipped']}")
    print()
    print(f"  Matched to DB         : {stats['matched']:>6}  ({_pct(stats['matched'], total)})")
    print(f"    of which via +/-1d  : {stats['matched_offset']:>6}  (postponement date shift)")
    print(f"  No totals odds        : {stats['no_totals_odds']:>6}  ({_pct(stats['no_totals_odds'], total)})  (spring/cancelled)")
    print(f"  Unknown team code     : {stats['no_alias']:>6}  ({_pct(stats['no_alias'], total)})")
    print(f"  Not in games table    : {stats['not_in_db']:>6}  ({_pct(stats['not_in_db'], total)})")
    print(f"  Doubleheader (used first) : {stats['dh_ambiguous']:>4}")
    print()
    action = "Would write" if dry_run else "Rows written"
    print(f"  {action} to odds_snapshots : {stats['rows_written']}")

    if stats["unmatched_pairs"]:
        top = sorted(stats["unmatched_pairs"].items(), key=lambda x: -x[1])[:15]
        print(f"\n  Top unmatched team pairs (add to SBR_TO_OUR map):")
        for (away, home), n in top:
            print(f"    away={away!r:6s}  home={home!r:6s}  x{n}")


def _pct(n: int, total: int) -> str:
    return f"{n * 100 / total:.1f}%" if total else "n/a"


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Load SportsBookReview historical MLB totals into odds_snapshots"
    )
    ap.add_argument(
        "--json", type=str, default=None,
        help="Path to local JSON file (skip download)"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Parse and match but do not write to DB"
    )
    ap.add_argument(
        "--season", type=int, default=None,
        help="Load only this season (e.g. --season 2023)"
    )
    ap.add_argument(
        "--clear", action="store_true",
        help="DELETE all existing odds_snapshots rows before loading"
    )
    ap.add_argument("--db", type=str, default=DB_PATH)
    args = ap.parse_args()

    # ── Resolve JSON file ──────────────────────────────────────────────────────
    if args.json:
        json_path = args.json
        if not os.path.exists(json_path):
            print(f"ERROR: file not found: {json_path}")
            sys.exit(1)
        print(f"Using local file: {json_path}")
    else:
        os.makedirs(CACHE_DIR, exist_ok=True)
        json_path = os.path.join(CACHE_DIR, DATASET_FILENAME)
        if os.path.exists(json_path):
            size_mb = os.path.getsize(json_path) / 1_048_576
            print(f"Found cached file ({size_mb:.1f} MB): {json_path}")
        else:
            download_dataset(json_path)

    # ── Load JSON ──────────────────────────────────────────────────────────────
    print(f"Parsing JSON ...", end="", flush=True)
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f" {len(data)} dates loaded")

    # ── DB ─────────────────────────────────────────────────────────────────────
    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")

    if args.clear and not args.dry_run:
        n = con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
        con.execute("DELETE FROM odds_snapshots")
        con.commit()
        print(f"Cleared {n} existing odds_snapshots rows")

    existing = con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    if existing:
        print(f"Note: {existing} rows already in odds_snapshots (INSERT OR IGNORE will skip dupes)")

    # ── Run ────────────────────────────────────────────────────────────────────
    print(f"Matching games and {'simulating' if args.dry_run else 'writing'} rows ...")
    stats = load_dataset(con, data, dry_run=args.dry_run, season=args.season)

    print_report(stats, args.dry_run)

    con.close()


if __name__ == "__main__":
    main()
