"""
WNBA player box-score collector (sportsdataverse / "wehoop" data).

Pipeline (each phase its own transaction):
    Load   : sportsdataverse.wnba.load_wnba_player_boxscore(seasons=...)
    Filter : season_type IN (2,3), drop All-Star/Olympic pseudo-teams
    Phase A: upsert teams (backfill abbreviation + wehoop_team_id)
    Phase B: upsert games (backfill scores + wehoop_game_id + status='final')
    Phase C: upsert players (creates new player rows)
    Phase D: insert wnba_player_box_scores (INSERT OR IGNORE)

Idempotent — rerunning the same seasons is a no-op once data is loaded.

Filtered + unmatched rows are routed to CSVs in logs/ rather than dropped:
    logs/wehoop_filtered_allstar.csv  - rows excluded by franchise allowlist
    logs/wehoop_unmatched.csv         - games where team_id couldn't be resolved

Usage:
    python wehoop_boxscores.py
    python wehoop_boxscores.py --seasons 2024,2025,2026
"""

import argparse
import csv
import logging
import os
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Standalone path / DB setup  (replaces src.common.ssl_setup + src.common.database)
# ---------------------------------------------------------------------------

# Script's own directory — used for logs/ and CSVs regardless of cwd.
_DIR = Path(__file__).resolve().parent

# ssl_setup inline: point Python's SSL at certifi's CA bundle before any
# HTTPS call.  Fixes Windows cert failures; harmless no-op on Linux/VPS.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE",        certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE",   certifi.where())
except ImportError:
    pass  # certifi absent — let downstream raise its own cert error if needed

# DB path: SPORTS_DB_PATH env var (set by VPS cron) → sports.db beside this script.
DEFAULT_DB_PATH = Path(os.environ.get("SPORTS_DB_PATH") or str(_DIR / "sports.db"))


def connect(db_path: Path = None) -> sqlite3.Connection:
    """Open a sqlite3 connection with FK enforcement and Row factory."""
    p = Path(db_path) if db_path else DEFAULT_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

# ---------------------------------------------------------------------------

import polars as pl  # noqa: E402
import sportsdataverse.wnba as wnba  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPORT = "Basketball"
LEAGUE = "WNBA"

DEFAULT_SEASONS = list(range(2021, 2026))   # 2021..2025
KEEP_SEASON_TYPES = (2, 3)                   # regular + playoffs

# Explicit allowlist of real WNBA franchises (NOT a regex). This excludes
# All-Star pseudo-teams ("TEAM CLARK", "Team Stewart", "Team WNBA", ...) which
# sportsdataverse files under season_type=2 alongside regular-season games.
# 15 = 13 historical franchises (through 2025) + Portland Fire and Toronto
# Tempo, the 2026 expansion teams. UPDATE THIS SET WHEN THE WNBA EXPANDS,
# RELOCATES, OR REBRANDS A FRANCHISE.
WNBA_FRANCHISES = frozenset({
    "Atlanta Dream",
    "Chicago Sky",
    "Connecticut Sun",
    "Dallas Wings",
    "Golden State Valkyries",
    "Indiana Fever",
    "Las Vegas Aces",
    "Los Angeles Sparks",
    "Minnesota Lynx",
    "New York Liberty",
    "Phoenix Mercury",
    "Portland Fire",
    "Seattle Storm",
    "Toronto Tempo",
    "Washington Mystics",
})

LOG_FILE      = _DIR / "logs" / "wehoop_boxscores.log"
FILTERED_CSV  = _DIR / "logs" / "wehoop_filtered_allstar.csv"
UNMATCHED_CSV = _DIR / "logs" / "wehoop_unmatched.csv"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_name(s) -> str:
    """For matching teams/players across sources. Idempotent."""
    if s is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    no_accent = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = re.sub(r"[^\w\s]", " ", no_accent.lower())
    return re.sub(r"\s+", " ", lowered).strip()


def parse_plus_minus(v):
    """wehoop stores plus_minus as String ('-4', '+3', ''). Cast to int."""
    if v is None or v == "":
        return None
    try:
        return int(str(v).lstrip("+"))
    except (ValueError, TypeError):
        return None


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_int(v):
    """Tolerant int cast; returns None on missing/invalid (polars Boolean→int included)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Load + filter
# ---------------------------------------------------------------------------
def load_seasons(seasons: list[int]) -> pl.DataFrame:
    log.info("Loading sportsdataverse player boxscores for seasons %s", seasons)
    df = wnba.load_wnba_player_boxscore(seasons=list(seasons))
    log.info("Loaded %d rows × %d columns.", df.height, df.width)
    return df


def filter_season_types(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only allowed season_types. Log the drop count."""
    before = df.height
    df = df.filter(pl.col("season_type").is_in(list(KEEP_SEASON_TYPES)))
    dropped = before - df.height
    if dropped:
        log.info("season_type filter dropped %d rows (allowed=%s).", dropped, KEEP_SEASON_TYPES)
    return df


def filter_allstar(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Keep only rows where BOTH team_display_name and opponent_team_display_name
    are in the WNBA_FRANCHISES allowlist. Excluded rows are returned separately
    for routing to logs/wehoop_filtered_allstar.csv.
    """
    allow = list(WNBA_FRANCHISES)
    mask = (
        pl.col("team_display_name").is_in(allow)
        & pl.col("opponent_team_display_name").is_in(allow)
    )
    kept = df.filter(mask)
    dropped = df.filter(~mask)
    log.info("Franchise allowlist: kept %d rows, dropped %d (non-franchise).",
             kept.height, dropped.height)
    return kept, dropped


def write_dropped_to_csv(df: pl.DataFrame, path: Path, kind: str) -> None:
    """Append a polars DataFrame to a CSV for review (header on first write)."""
    if df.height == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    cols = df.columns
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(cols)
        for row in df.iter_rows():
            writer.writerow(["" if v is None else v for v in row])
    log.info("Wrote %d %s rows to %s", df.height, kind, path)


# ---------------------------------------------------------------------------
# Phase A — teams
# ---------------------------------------------------------------------------
def phase_a_teams(conn, df: pl.DataFrame) -> dict[int, int]:
    """
    Upsert teams. Returns wehoop_team_id -> internal_team_id map.

    Matching order:
        1) Exact canonical_name (Kambi rows already use 'Dallas Wings' style)
        2) Normalized canonical_name (case/punct/whitespace-insensitive)
        3) New row

    On exact or normalized match: backfill abbreviation & wehoop_team_id
    using COALESCE so existing non-null values are preserved.
    """
    now = now_utc()

    # Union of (own team, opponent team) so we always populate both perspectives.
    seen: dict[int, tuple[str, str | None]] = {}
    for prefix in ("", "opponent_"):
        sub = df.select([
            pl.col(f"{prefix}team_id").alias("id"),
            pl.col(f"{prefix}team_display_name").alias("name"),
            pl.col(f"{prefix}team_abbreviation").alias("abbr"),
        ]).unique()
        for r in sub.iter_rows(named=True):
            if r["id"] is None or not r["name"]:
                continue
            seen.setdefault(r["id"], (r["name"], r["abbr"]))

    log.info("Phase A: %d distinct teams in wehoop data.", len(seen))

    existing = conn.execute(
        "SELECT internal_team_id, canonical_name, abbreviation, wehoop_team_id "
        "FROM teams WHERE sport=? AND league=?",
        (SPORT, LEAGUE),
    ).fetchall()
    by_exact = {r["canonical_name"]: r for r in existing}
    by_norm = {normalize_name(r["canonical_name"]): r for r in existing}

    team_map: dict[int, int] = {}
    inserted = backfilled = 0

    cur = conn.cursor()
    try:
        for wehoop_id, (display_name, abbr) in sorted(seen.items()):
            norm = normalize_name(display_name)

            row = by_exact.get(display_name) or by_norm.get(norm)
            matched_kind = None
            if display_name in by_exact:
                matched_kind = "exact"
            elif norm in by_norm:
                matched_kind = "normalized"

            if row and matched_kind == "normalized":
                log.warning(
                    "Team normalized-but-not-exact match: wehoop=%r vs db=%r "
                    "(internal_team_id=%d). Merging to existing row; no duplicate created.",
                    display_name, row["canonical_name"], row["internal_team_id"],
                )

            if row:
                tid = row["internal_team_id"]
                team_map[wehoop_id] = tid
                if row["abbreviation"] is None or row["wehoop_team_id"] is None:
                    cur.execute(
                        "UPDATE teams SET "
                        "  abbreviation   = COALESCE(abbreviation,   ?), "
                        "  wehoop_team_id = COALESCE(wehoop_team_id, ?) "
                        "WHERE internal_team_id = ?",
                        (abbr, wehoop_id, tid),
                    )
                    backfilled += 1
            else:
                cur.execute(
                    "INSERT INTO teams "
                    "(sport, league, canonical_name, abbreviation, wehoop_team_id, first_seen_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (SPORT, LEAGUE, display_name, abbr, wehoop_id, now),
                )
                tid = cur.lastrowid
                team_map[wehoop_id] = tid
                new_row = {
                    "internal_team_id": tid,
                    "canonical_name": display_name,
                    "abbreviation": abbr,
                    "wehoop_team_id": wehoop_id,
                }
                by_exact[display_name] = new_row
                by_norm[norm] = new_row
                inserted += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    log.info("Phase A: %d teams inserted, %d existing teams backfilled.",
             inserted, backfilled)
    return team_map


# ---------------------------------------------------------------------------
# Phase B — games
# ---------------------------------------------------------------------------
def phase_b_games(conn, df: pl.DataFrame, team_map: dict[int, int]) -> tuple[dict[int, int], list[dict]]:
    """
    Upsert games. Returns (wehoop_game_id -> internal_game_id, unmatched_rows).

    Match on (sport, league, game_date, home_team_id, away_team_id).
    Existing match: UPDATE wehoop_game_id, scores, status='final'.
    New: INSERT with status='final'.
    Unresolved teams: routed to wehoop_unmatched.csv, NOT silently dropped.
    """
    now = now_utc()

    # One row per game; pick first row's perspective for home/away resolution.
    games = (
        df.group_by("game_id")
          .agg([
              pl.col("game_date").first(),
              pl.col("season").first(),
              pl.col("team_id").first(),
              pl.col("opponent_team_id").first(),
              pl.col("team_score").first(),
              pl.col("opponent_team_score").first(),
              pl.col("home_away").first(),
              pl.col("team_display_name").first(),
              pl.col("opponent_team_display_name").first(),
          ])
    )
    log.info("Phase B: %d distinct games in wehoop data.", games.height)

    existing = conn.execute(
        "SELECT internal_game_id, game_date, home_team_id, away_team_id "
        "FROM games WHERE sport=? AND league=?",
        (SPORT, LEAGUE),
    ).fetchall()
    by_exact = {
        (r["game_date"], r["home_team_id"], r["away_team_id"]): r["internal_game_id"]
        for r in existing
    }

    game_map: dict[int, int] = {}
    unmatched: list[dict] = []
    inserted = backfilled = 0

    cur = conn.cursor()
    try:
        for g in games.iter_rows(named=True):
            wehoop_game_id = g["game_id"]
            game_date = g["game_date"].isoformat() if g["game_date"] else None
            season = g["season"]
            row_team_iid = team_map.get(g["team_id"])
            opp_team_iid = team_map.get(g["opponent_team_id"])

            if row_team_iid is None or opp_team_iid is None:
                unmatched.append({
                    "wehoop_game_id": wehoop_game_id,
                    "game_date": game_date,
                    "season": season,
                    "team_display_name": g["team_display_name"],
                    "team_id": g["team_id"],
                    "team_resolved_to": row_team_iid,
                    "opponent_display_name": g["opponent_team_display_name"],
                    "opponent_id": g["opponent_team_id"],
                    "opponent_resolved_to": opp_team_iid,
                    "reason": "team_id not in team_map after Phase A",
                })
                continue

            if g["home_away"] == "home":
                home_id, away_id = row_team_iid, opp_team_iid
                home_score, away_score = g["team_score"], g["opponent_team_score"]
            else:
                home_id, away_id = opp_team_iid, row_team_iid
                home_score, away_score = g["opponent_team_score"], g["team_score"]

            key = (game_date, home_id, away_id)
            if key in by_exact:
                internal_game_id = by_exact[key]
                cur.execute(
                    "UPDATE games SET "
                    "  wehoop_game_id   = COALESCE(wehoop_game_id, ?), "
                    "  home_final_score = COALESCE(home_final_score, ?), "
                    "  away_final_score = COALESCE(away_final_score, ?), "
                    "  status           = 'final' "
                    "WHERE internal_game_id = ?",
                    (wehoop_game_id, home_score, away_score, internal_game_id),
                )
                backfilled += 1
            else:
                cur.execute(
                    "INSERT INTO games "
                    "(sport, league, season, game_date, home_team_id, away_team_id, "
                    " wehoop_game_id, home_final_score, away_final_score, status, first_seen_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'final', ?)",
                    (SPORT, LEAGUE, season, game_date, home_id, away_id,
                     wehoop_game_id, home_score, away_score, now),
                )
                internal_game_id = cur.lastrowid
                by_exact[key] = internal_game_id
                inserted += 1
            game_map[wehoop_game_id] = internal_game_id
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    log.info("Phase B: %d games inserted, %d existing games backfilled, %d unmatched.",
             inserted, backfilled, len(unmatched))
    return game_map, unmatched


# ---------------------------------------------------------------------------
# Phase C — players
# ---------------------------------------------------------------------------
def phase_c_players(conn, df: pl.DataFrame) -> dict[int, int]:
    """
    Upsert players. Returns wehoop_athlete_id -> internal_player_id.

    Match order:
        1) wehoop_athlete_id   (most reliable across sources)
        2) (sport, league, normalized_name)  (Kambi-created stub backfill)
        3) New row

    On normalized-name match: backfill wehoop_athlete_id; log only if the
    raw name differs from the existing full_name (genuine fuzzy match).
    """
    now = now_utc()

    players = df.select([
        pl.col("athlete_id"),
        pl.col("athlete_display_name"),
    ]).unique()
    log.info("Phase C: %d distinct players in wehoop data.", players.height)

    existing = conn.execute(
        "SELECT internal_player_id, full_name, normalized_name, wehoop_athlete_id "
        "FROM players WHERE sport=? AND league=?",
        (SPORT, LEAGUE),
    ).fetchall()
    by_wehoop = {
        r["wehoop_athlete_id"]: r for r in existing if r["wehoop_athlete_id"] is not None
    }
    by_norm = {r["normalized_name"]: r for r in existing}

    player_map: dict[int, int] = {}
    inserted = backfilled = 0

    cur = conn.cursor()
    try:
        for r in players.iter_rows(named=True):
            wehoop_id = r["athlete_id"]
            name = r["athlete_display_name"]
            if wehoop_id is None or not name:
                continue
            norm = normalize_name(name)

            if wehoop_id in by_wehoop:
                player_map[wehoop_id] = by_wehoop[wehoop_id]["internal_player_id"]
            elif norm in by_norm:
                existing_row = by_norm[norm]
                pid = existing_row["internal_player_id"]
                player_map[wehoop_id] = pid
                if existing_row["wehoop_athlete_id"] is None:
                    if existing_row["full_name"] != name:
                        log.warning(
                            "Player normalized-but-not-exact match: wehoop=%r vs db=%r "
                            "(internal_player_id=%d). Backfilling wehoop_athlete_id.",
                            name, existing_row["full_name"], pid,
                        )
                    cur.execute(
                        "UPDATE players SET wehoop_athlete_id = ? "
                        "WHERE internal_player_id = ? AND wehoop_athlete_id IS NULL",
                        (wehoop_id, pid),
                    )
                    by_wehoop[wehoop_id] = existing_row
                    backfilled += 1
            else:
                cur.execute(
                    "INSERT INTO players "
                    "(sport, league, full_name, normalized_name, wehoop_athlete_id, first_seen_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (SPORT, LEAGUE, name, norm, wehoop_id, now),
                )
                pid = cur.lastrowid
                player_map[wehoop_id] = pid
                new_row = {
                    "internal_player_id": pid,
                    "full_name": name,
                    "normalized_name": norm,
                    "wehoop_athlete_id": wehoop_id,
                }
                by_wehoop[wehoop_id] = new_row
                by_norm[norm] = new_row
                inserted += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    log.info("Phase C: %d players inserted, %d existing players backfilled.",
             inserted, backfilled)
    return player_map


# ---------------------------------------------------------------------------
# Phase D — box scores
# ---------------------------------------------------------------------------
BOX_INSERT_SQL = """
INSERT OR IGNORE INTO wnba_player_box_scores (
    sport, league, internal_game_id, internal_player_id, internal_team_id,
    game_date, started, minutes, points, rebounds,
    offensive_rebounds, defensive_rebounds, assists, steals, blocks,
    turnovers, fouls, fg_made, fg_attempted, three_made, three_attempted,
    ft_made, ft_attempted, plus_minus,
    source, source_box_score_id, collected_at, did_not_play, reason
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def phase_d_boxscores(conn, df: pl.DataFrame,
                      team_map: dict[int, int],
                      game_map: dict[int, int],
                      player_map: dict[int, int]) -> tuple[int, int, int]:
    """Insert box scores. Returns (inserted, ignored, skipped_unresolved)."""
    now = now_utc()
    inserted = ignored = skipped = 0

    cur = conn.cursor()
    try:
        for r in df.iter_rows(named=True):
            internal_game_id = game_map.get(r["game_id"])
            internal_player_id = player_map.get(r["athlete_id"])
            internal_team_id = team_map.get(r["team_id"])
            if not (internal_game_id and internal_player_id and internal_team_id):
                skipped += 1
                continue

            dnp = 1 if r.get("did_not_play") else 0
            reason = r.get("reason") if dnp else None  # only meaningful on DNPs

            game_date = r["game_date"].isoformat() if r["game_date"] else None

            cur.execute(BOX_INSERT_SQL, (
                SPORT, LEAGUE, internal_game_id, internal_player_id, internal_team_id,
                game_date,
                1 if r.get("starter") else 0,
                r.get("minutes"),
                r.get("points"),
                r.get("rebounds"),
                r.get("offensive_rebounds"),
                r.get("defensive_rebounds"),
                r.get("assists"),
                r.get("steals"),
                r.get("blocks"),
                r.get("turnovers"),
                r.get("fouls"),
                r.get("field_goals_made"),
                r.get("field_goals_attempted"),
                r.get("three_point_field_goals_made"),
                r.get("three_point_field_goals_attempted"),
                r.get("free_throws_made"),
                r.get("free_throws_attempted"),
                parse_plus_minus(r.get("plus_minus")),
                "sportsdataverse",
                f"{r['game_id']}:{r['athlete_id']}",
                now,
                dnp,
                reason,
            ))
            if cur.rowcount == 1:
                inserted += 1
            else:
                ignored += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    log.info("Phase D: %d box scores inserted, %d ignored (dedup), %d skipped (unresolved FK).",
             inserted, ignored, skipped)
    return inserted, ignored, skipped


# ---------------------------------------------------------------------------
# Unmatched-row writer
# ---------------------------------------------------------------------------
def write_unmatched(rows: list[dict]) -> None:
    if not rows:
        return
    UNMATCHED_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not UNMATCHED_CSV.exists() or UNMATCHED_CSV.stat().st_size == 0
    with UNMATCHED_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    log.warning("Wrote %d unmatched game rows to %s", len(rows), UNMATCHED_CSV)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(seasons: list[int]) -> int:
    log.info("=== wehoop_boxscores starting (seasons=%s) ===", seasons)
    log.info("DB: %s", DEFAULT_DB_PATH)

    df = load_seasons(seasons)
    df = filter_season_types(df)
    df, dropped = filter_allstar(df)
    if dropped.height:
        write_dropped_to_csv(dropped, FILTERED_CSV, "filtered (non-franchise)")

    if df.height == 0:
        log.warning("Nothing left after filters. Exiting.")
        return 0

    conn = connect()
    try:
        team_map = phase_a_teams(conn, df)
        game_map, unmatched = phase_b_games(conn, df, team_map)
        if unmatched:
            write_unmatched(unmatched)
        player_map = phase_c_players(conn, df)
        phase_d_boxscores(conn, df, team_map, game_map, player_map)
    finally:
        conn.close()

    log.info("=== Done ===")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEASONS),
        help=f"Comma-separated season list (default: {DEFAULT_SEASONS[0]}-{DEFAULT_SEASONS[-1]})",
    )
    args = parser.parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    sys.exit(run(seasons))
