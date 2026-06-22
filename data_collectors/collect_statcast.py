"""
collect_statcast.py

Populates: games, team_batting_game, pitcher_game_log, bullpen_game,
           reliever_appearance, _collector_state

Sources:
  - MLB Stats API (requests) — schedule + box scores
  - pybaseball statcast()   — hard_hit_pct, barrel_pct, avg_fastball_velo

Usage:
  python data_collectors/collect_statcast.py                        # yesterday + today
  python data_collectors/collect_statcast.py --start 2017-04-02    # backfill from date
  python data_collectors/collect_statcast.py --start 2024-09-01 --end 2024-10-01
  python data_collectors/collect_statcast.py --season 2023
  python data_collectors/collect_statcast.py --statcast-only --start 2024-04-01
  python data_collectors/collect_statcast.py --boxscore-only --start 2024-04-01
  python data_collectors/collect_statcast.py --workers 8
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mlb_data.db"))
MLB_API = "http://statsapi.mlb.com/api/v1"
SEASON_START = 2017
DEFAULT_WORKERS = 4

# FIP constants by season (source: FanGraphs guts page)
FIP_CONSTANTS = {
    2017: 3.158, 2018: 3.161, 2019: 3.214, 2020: 3.179,
    2021: 3.163, 2022: 3.127, 2023: 3.069, 2024: 3.093, 2025: 3.10,
}

# wOBA linear weights by season (source: FanGraphs guts page)
WOBA_WEIGHTS = {
    2017: {"bb": 0.693, "hbp": 0.723, "s": 0.877, "d": 1.232, "t": 1.552, "hr": 2.000, "scale": 0.321},
    2018: {"bb": 0.690, "hbp": 0.720, "s": 0.880, "d": 1.247, "t": 1.578, "hr": 2.031, "scale": 0.314},
    2019: {"bb": 0.690, "hbp": 0.720, "s": 0.884, "d": 1.261, "t": 1.601, "hr": 2.101, "scale": 0.334},
    2020: {"bb": 0.690, "hbp": 0.728, "s": 0.883, "d": 1.264, "t": 1.597, "hr": 2.037, "scale": 0.322},
    2021: {"bb": 0.688, "hbp": 0.718, "s": 0.882, "d": 1.254, "t": 1.581, "hr": 2.054, "scale": 0.330},
    2022: {"bb": 0.689, "hbp": 0.720, "s": 0.888, "d": 1.271, "t": 1.616, "hr": 2.101, "scale": 0.320},
    2023: {"bb": 0.690, "hbp": 0.722, "s": 0.883, "d": 1.244, "t": 1.569, "hr": 2.004, "scale": 0.323},
    2024: {"bb": 0.690, "hbp": 0.722, "s": 0.883, "d": 1.244, "t": 1.569, "hr": 2.004, "scale": 0.323},
    2025: {"bb": 0.690, "hbp": 0.722, "s": 0.883, "d": 1.244, "t": 1.569, "hr": 2.004, "scale": 0.323},
}

FASTBALL_TYPES = {"FF", "SI", "FC", "FT", "FA"}

# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown = threading.Event()


def _handle_sigint(sig, frame):
    print("\n[interrupt] Ctrl-C received — finishing current item and saving progress...")
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_sigint)

# ── DB helpers ────────────────────────────────────────────────────────────────


def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def ensure_state_table(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS _collector_state (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    con.commit()


def get_state(con: sqlite3.Connection, key: str, default=None):
    row = con.execute("SELECT value FROM _collector_state WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_state(con: sqlite3.Connection, key: str, value):
    con.execute(
        """INSERT INTO _collector_state(key, value) VALUES(?,?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
        (key, json.dumps(value)),
    )
    con.commit()


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
            wait = 4**attempt
            print(f"\n  [retry {attempt+1}] {exc} — waiting {wait}s", flush=True)
            time.sleep(wait)


# ── ETA tracker ───────────────────────────────────────────────────────────────


class ETA:
    def __init__(self, total: int, label: str = "items"):
        self.total = total
        self.done = 0
        self.label = label
        self._start = time.time()

    def tick(self, n: int = 1):
        self.done += n

    def line(self) -> str:
        elapsed = time.time() - self._start
        pct = 100.0 * self.done / self.total if self.total else 0
        if self.done == 0:
            eta_str = "calculating..."
        else:
            rate = self.done / elapsed
            secs_left = (self.total - self.done) / rate
            eta_str = str(datetime.timedelta(seconds=int(secs_left)))
        return f"  {self.done}/{self.total} {self.label} ({pct:.1f}%) — ETA: {eta_str}"


# ── Date helpers ──────────────────────────────────────────────────────────────


def date_range_weeks(start: datetime.date, end: datetime.date) -> list[tuple[str, str]]:
    chunks, cur = [], start
    while cur <= end:
        chunk_end = min(cur + datetime.timedelta(days=6), end)
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + datetime.timedelta(days=1)
    return chunks


def date_range_months(start: datetime.date, end: datetime.date) -> list[tuple[str, str]]:
    """
    Split [start, end] into calendar-month chunks.
    Keeps each fetch small enough that the MLB Stats API does not truncate results.
    A full MLB month is ~400-450 games; the API silently caps multi-year requests.
    """
    chunks, cur = [], start
    while cur <= end:
        if cur.month == 12:
            first_of_next = datetime.date(cur.year + 1, 1, 1)
        else:
            first_of_next = datetime.date(cur.year, cur.month + 1, 1)
        chunk_end = min(first_of_next - datetime.timedelta(days=1), end)
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = first_of_next
    return chunks


# ── Stat helpers ──────────────────────────────────────────────────────────────


def ip_to_float(ip_str) -> float:
    """Convert baseball IP notation to decimal ('6.2' → 6.667)."""
    try:
        parts = str(ip_str).split(".")
        full = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return round(full + thirds / 3.0, 4)
    except Exception:
        return 0.0


def calc_woba(bb, hbp, h, d, t, hr, ab, sf, season: int) -> float | None:
    w = WOBA_WEIGHTS.get(season, WOBA_WEIGHTS[2024])
    singles = max(h - d - t - hr, 0)
    denom = ab + bb + hbp + sf
    if denom == 0:
        return None
    num = w["bb"] * bb + w["hbp"] * hbp + w["s"] * singles + w["d"] * d + w["t"] * t + w["hr"] * hr
    return round(num / denom, 3)


def calc_fip(hr, bb, hbp, k, ip, season: int) -> float | None:
    if ip == 0:
        return None
    c = FIP_CONSTANTS.get(season, 3.10)
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + c, 2)


# ── MLB Stats API ─────────────────────────────────────────────────────────────


def fetch_schedule(start_date: str, end_date: str) -> list[dict]:
    data = http_get(f"{MLB_API}/schedule", {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "gameType": "R",
        "hydrate": "team,venue,linescore",
    })
    return [g for block in data.get("dates", []) for g in block.get("games", [])]


def parse_schedule_row(g: dict) -> dict:
    gdate = g.get("gameDate", "")[:10]
    teams = g.get("teams", {})
    home, away = teams.get("home", {}), teams.get("away", {})
    h_score = home.get("score")
    a_score = away.get("score")
    total = (h_score + a_score) if (h_score is not None and a_score is not None) else None
    dh_flag = g.get("doubleHeader", "N")
    game_num = g.get("gameNumber", 1)
    dh_val = 0 if dh_flag == "N" else game_num
    dn = g.get("dayNight", "")
    return {
        "game_id":            g["gamePk"],
        "game_date":          gdate,
        "game_time_utc":      g.get("gameDate"),
        "day_night":          "D" if dn.lower().startswith("d") else "N",
        "home_team_id":       home.get("team", {}).get("id"),
        "away_team_id":       away.get("team", {}).get("id"),
        "venue_id":           g.get("venue", {}).get("id"),
        "series_game_number": g.get("seriesGameNumber", 1),
        "doubleheader":       dh_val,
        "status":             g.get("status", {}).get("detailedState", ""),
        "home_score":         h_score,
        "away_score":         a_score,
        "total_runs":         total,
        "season":             int(gdate[:4]),
    }


def fetch_boxscore(game_pk: int) -> dict:
    return http_get(f"{MLB_API}/game/{game_pk}/boxscore")


def parse_boxscore(game_pk: int, game_date: str, bs: dict, season: int) -> dict:
    """
    Returns dict of lists ready for DB inserts:
      batting_rows, pitcher_rows, bullpen_rows, reliever_rows, starter_ids
    """
    result = {
        "batting_rows": [], "pitcher_rows": [],
        "bullpen_rows": [], "reliever_rows": [],
        "starter_ids": {},
    }
    teams_data = bs.get("teams", {})
    home_tid = teams_data.get("home", {}).get("team", {}).get("id")
    away_tid = teams_data.get("away", {}).get("team", {}).get("id")
    fip_c = FIP_CONSTANTS.get(season, 3.10)

    for side in ("home", "away"):
        td = teams_data.get(side, {})
        team_id = td.get("team", {}).get("id")
        opp_id = away_tid if side == "home" else home_tid
        is_home = 1 if side == "home" else 0
        players = td.get("players", {})
        pitcher_order = td.get("pitchers", [])  # ordered by appearance; [0] = starter
        bat = td.get("teamStats", {}).get("batting", {})

        # ── Batting row ───────────────────────────────────────────────────────
        ab   = bat.get("atBats", 0) or 0
        h    = bat.get("hits", 0) or 0
        d    = bat.get("doubles", 0) or 0
        t    = bat.get("triples", 0) or 0
        hr   = bat.get("homeRuns", 0) or 0
        bb   = bat.get("baseOnBalls", 0) or 0
        hbp  = bat.get("hitByPitch", 0) or 0
        sf   = bat.get("sacFlies", 0) or 0
        so   = bat.get("strikeOuts", 0) or 0
        pa   = bat.get("plateAppearances", 0) or 0
        runs = bat.get("runs", 0) or 0

        # Use API-provided OBP/SLG strings when available; they're authoritative
        def _parse_pct(val):
            try:
                return float(val) if val not in (None, "", ".---") else None
            except (ValueError, TypeError):
                return None

        obp = _parse_pct(bat.get("obp"))
        slg = _parse_pct(bat.get("slg"))
        woba = calc_woba(bb, hbp, h, d, t, hr, ab, sf, season)
        k_pct  = round(so / pa, 3) if pa > 0 else None
        bb_pct = round(bb / pa, 3) if pa > 0 else None

        result["batting_rows"].append({
            "game_id": game_pk, "game_date": game_date,
            "team_id": team_id, "is_home": is_home, "opponent_id": opp_id,
            "runs_scored": runs, "hits": h, "doubles": d, "triples": t,
            "home_runs": hr, "walks": bb, "strikeouts": so,
            "at_bats": ab, "plate_appearances": pa,
            "obp": obp, "slg": slg, "woba": woba,
            "k_pct": k_pct, "bb_pct": bb_pct,
            # hard_hit_pct, barrel_pct, wrc_plus_season → Statcast / weekly FG pull
        })

        # ── Pitching rows ─────────────────────────────────────────────────────
        starter_id = pitcher_order[0] if pitcher_order else None
        result["starter_ids"][side] = starter_id

        bullpen = {"ip": 0.0, "n": 0, "runs": 0, "er": 0, "k": 0, "bb": 0, "hbp": 0, "hr": 0}

        for i, pid in enumerate(pitcher_order):
            pdata = players.get(f"ID{pid}", {})
            ps = pdata.get("stats", {}).get("pitching", {})
            if not ps:
                continue

            ip  = ip_to_float(ps.get("inningsPitched", "0.0"))
            er  = ps.get("earnedRuns", 0) or 0
            r   = ps.get("runs", 0) or 0
            k   = ps.get("strikeOuts", 0) or 0
            bb_p = ps.get("baseOnBalls", 0) or 0
            hbp_p = ps.get("hitBatsmen", 0) or 0
            h_p  = ps.get("hits", 0) or 0
            hr_p = ps.get("homeRuns", 0) or 0
            pitches = ps.get("pitchesThrown") or ps.get("numberOfPitches")

            era_game = round(er / ip * 9, 2) if ip > 0 else None
            fip_game = calc_fip(hr_p, bb_p, hbp_p, k, ip, season)

            if i == 0:
                # Starter
                result["pitcher_rows"].append({
                    "game_id": game_pk, "game_date": game_date,
                    "pitcher_id": pid, "team_id": team_id, "opponent_id": opp_id,
                    "is_home": is_home,
                    "innings_pitched": ip, "runs_allowed": r, "earned_runs": er,
                    "strikeouts": k, "walks": bb_p, "hits_allowed": h_p,
                    "hr_allowed": hr_p, "pitches_thrown": pitches,
                    "era_game": era_game, "fip_game": fip_game,
                    # avg_fastball_velo, days_rest → patched separately
                })
            else:
                # Reliever
                bullpen["ip"]   += ip
                bullpen["n"]    += 1
                bullpen["runs"] += r
                bullpen["er"]   += er
                bullpen["k"]    += k
                bullpen["bb"]   += bb_p
                bullpen["hbp"]  += hbp_p
                bullpen["hr"]   += hr_p

                result["reliever_rows"].append({
                    "game_id": game_pk, "game_date": game_date,
                    "pitcher_id": pid, "team_id": team_id,
                    "innings_pitched": ip, "pitches_thrown": pitches,
                })

        bip = bullpen["ip"]
        era_rel = round(bullpen["er"] / bip * 9, 2) if bip > 0 else None
        fip_rel = calc_fip(bullpen["hr"], bullpen["bb"], bullpen["hbp"], bullpen["k"], bip, season)

        result["bullpen_rows"].append({
            "game_id": game_pk, "game_date": game_date, "team_id": team_id,
            "relief_ip": round(bip, 2),
            "relief_pitchers_used": bullpen["n"],
            "relief_runs": bullpen["runs"], "relief_er": bullpen["er"],
            "relief_k": bullpen["k"], "relief_bb": bullpen["bb"],
            "era_relief": era_rel, "fip_relief": fip_rel,
        })

    return result


# ── Statcast ──────────────────────────────────────────────────────────────────


def fetch_statcast_week(start_dt: str, end_dt: str):
    """Worker function: pull one week of Statcast. Returns (start, end, df|Exception)."""
    try:
        import pybaseball  # import inside worker for thread safety
        pybaseball.cache.enable()
        df = pybaseball.statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
        return start_dt, end_dt, df
    except Exception as exc:
        return start_dt, end_dt, exc


def aggregate_statcast(df) -> tuple[dict, dict]:
    """
    Returns:
      team_metrics  : {(game_pk, is_home): {hard_hit_pct, barrel_pct}}
      pitcher_metrics: {(game_pk, pitcher_id): {avg_fastball_velo}}
    """
    if df is None or df.empty:
        return {}, {}

    team_m, pitcher_m = {}, {}

    # Batted ball metrics (launch_speed not null)
    batted = df[df["launch_speed"].notna()].copy()
    if not batted.empty:
        # Top of inning = away team batting; Bot = home team batting
        batted["is_home"] = (batted["inning_topbot"] == "Bot").astype(int)
        batted["hard_hit"] = (batted["launch_speed"] >= 95).astype(float)
        # pybaseball ≥2.2 removed the 'barrel' column; use launch_speed_angle==6 instead
        if "barrel" in batted.columns:
            batted["barrel_v"] = batted["barrel"].fillna(0).clip(0, 1)
        elif "launch_speed_angle" in batted.columns:
            batted["barrel_v"] = (batted["launch_speed_angle"] == 6).astype(float)
        else:
            batted["barrel_v"] = 0.0

        grp = batted.groupby(["game_pk", "is_home"]).agg(
            hard_hit_pct=("hard_hit", "mean"),
            barrel_pct=("barrel_v", "mean"),
        ).reset_index()

        for _, row in grp.iterrows():
            key = (int(row["game_pk"]), int(row["is_home"]))
            team_m[key] = {
                "hard_hit_pct": round(float(row["hard_hit_pct"]), 4),
                "barrel_pct":   round(float(row["barrel_pct"]), 4),
            }

    # Fastball velocity per pitcher per game
    fb = df[df["pitch_type"].isin(FASTBALL_TYPES) & df["release_speed"].notna()]
    if not fb.empty:
        grp2 = fb.groupby(["game_pk", "pitcher"])["release_speed"].mean().reset_index()
        for _, row in grp2.iterrows():
            key = (int(row["game_pk"]), int(row["pitcher"]))
            pitcher_m[key] = {"avg_fastball_velo": round(float(row["release_speed"]), 1)}

    return team_m, pitcher_m


# ── DB writes ─────────────────────────────────────────────────────────────────


def extract_teams_venues(games: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Pull unique team and venue rows from a raw schedule response.
    Returns (team_rows, venue_rows) with enough fields to satisfy FK constraints.
    The schedule is already hydrated with team+venue, so no extra API calls needed.
    """
    seen_teams: dict = {}
    seen_venues: dict = {}

    for g in games:
        for side in ("home", "away"):
            t = g.get("teams", {}).get(side, {}).get("team", {})
            tid = t.get("id")
            if tid and tid not in seen_teams:
                seen_teams[tid] = {
                    "team_id":   tid,
                    # abbreviation ("NYY") preferred; fall back to teamCode or str(id)
                    "team_code": t.get("abbreviation") or t.get("teamCode") or str(tid),
                    "team_name": t.get("name") or str(tid),
                    "city":      t.get("locationName"),
                    "league":    (t.get("league") or {}).get("name"),
                    "division":  (t.get("division") or {}).get("name"),
                    "venue_id":  (t.get("venue") or {}).get("id"),
                }

        v = g.get("venue", {})
        vid = v.get("id")
        if vid and vid not in seen_venues:
            seen_venues[vid] = {
                "venue_id":   vid,
                "venue_name": v.get("name") or str(vid),
            }

    return list(seen_teams.values()), list(seen_venues.values())


def seed_teams(con: sqlite3.Connection, rows: list[dict]):
    """
    Insert team rows extracted from the schedule. INSERT OR IGNORE preserves any
    richer data (city, lat/long, etc.) already seeded by a dedicated seed script.
    """
    for r in rows:
        con.execute("""
            INSERT OR IGNORE INTO teams(team_id, team_code, team_name, city, league, division)
            VALUES(:team_id, :team_code, :team_name, :city, :league, :division)
        """, r)
    con.commit()


def seed_venues(con: sqlite3.Connection, rows: list[dict]):
    """
    Insert venue rows extracted from the schedule. INSERT OR IGNORE preserves any
    richer data (lat/long, altitude, roof_type, cf_direction_deg) already seeded.
    """
    for r in rows:
        con.execute("""
            INSERT OR IGNORE INTO venues(venue_id, venue_name)
            VALUES(:venue_id, :venue_name)
        """, r)
    con.commit()


def upsert_game(con: sqlite3.Connection, row: dict):
    con.execute("""
        INSERT INTO games(
            game_id, game_date, game_time_utc, day_night,
            home_team_id, away_team_id, venue_id, series_game_number,
            doubleheader, status, home_score, away_score, total_runs, season)
        VALUES(
            :game_id,:game_date,:game_time_utc,:day_night,
            :home_team_id,:away_team_id,:venue_id,:series_game_number,
            :doubleheader,:status,:home_score,:away_score,:total_runs,:season)
        ON CONFLICT(game_id) DO UPDATE SET
            status=excluded.status,
            home_score=excluded.home_score,
            away_score=excluded.away_score,
            total_runs=excluded.total_runs
    """, row)


def upsert_batting(con: sqlite3.Connection, rows: list[dict]):
    for r in rows:
        con.execute("""
            INSERT INTO team_batting_game(
                game_id, game_date, team_id, is_home, opponent_id,
                runs_scored, hits, doubles, triples, home_runs,
                walks, strikeouts, at_bats, plate_appearances,
                obp, slg, woba, k_pct, bb_pct)
            VALUES(
                :game_id,:game_date,:team_id,:is_home,:opponent_id,
                :runs_scored,:hits,:doubles,:triples,:home_runs,
                :walks,:strikeouts,:at_bats,:plate_appearances,
                :obp,:slg,:woba,:k_pct,:bb_pct)
            ON CONFLICT(game_id, team_id) DO UPDATE SET
                runs_scored=excluded.runs_scored,
                obp=excluded.obp, slg=excluded.slg, woba=excluded.woba
        """, r)


def upsert_pitcher(con: sqlite3.Connection, rows: list[dict]):
    for r in rows:
        con.execute("""
            INSERT INTO pitcher_game_log(
                game_id, game_date, pitcher_id, team_id, opponent_id, is_home,
                innings_pitched, runs_allowed, earned_runs, strikeouts, walks,
                hits_allowed, hr_allowed, pitches_thrown, era_game, fip_game)
            VALUES(
                :game_id,:game_date,:pitcher_id,:team_id,:opponent_id,:is_home,
                :innings_pitched,:runs_allowed,:earned_runs,:strikeouts,:walks,
                :hits_allowed,:hr_allowed,:pitches_thrown,:era_game,:fip_game)
            ON CONFLICT(game_id, pitcher_id) DO NOTHING
        """, r)


def upsert_bullpen(con: sqlite3.Connection, rows: list[dict]):
    for r in rows:
        con.execute("""
            INSERT INTO bullpen_game(
                game_id, game_date, team_id, relief_ip, relief_pitchers_used,
                relief_runs, relief_er, relief_k, relief_bb, era_relief, fip_relief)
            VALUES(
                :game_id,:game_date,:team_id,:relief_ip,:relief_pitchers_used,
                :relief_runs,:relief_er,:relief_k,:relief_bb,:era_relief,:fip_relief)
            ON CONFLICT(game_id, team_id) DO NOTHING
        """, r)


def upsert_relievers(con: sqlite3.Connection, rows: list[dict]):
    for r in rows:
        con.execute("""
            INSERT OR IGNORE INTO reliever_appearance(
                game_id, game_date, pitcher_id, team_id, innings_pitched, pitches_thrown)
            VALUES(:game_id,:game_date,:pitcher_id,:team_id,:innings_pitched,:pitches_thrown)
        """, r)


def set_starter_ids(con: sqlite3.Connection, game_pk: int, starter_ids: dict):
    con.execute("""
        UPDATE games SET home_starter_id=?, away_starter_id=? WHERE game_id=?
    """, (starter_ids.get("home"), starter_ids.get("away"), game_pk))


def backfill_days_rest(con: sqlite3.Connection, game_date: str):
    """Compute days_rest for starters whose rest is still NULL on this date."""
    rows = con.execute("""
        SELECT pitcher_id, game_id FROM pitcher_game_log
        WHERE game_date=? AND days_rest IS NULL
    """, (game_date,)).fetchall()

    d1 = datetime.date.fromisoformat(game_date)
    for row in rows:
        prev = con.execute("""
            SELECT game_date FROM pitcher_game_log
            WHERE pitcher_id=? AND game_date < ?
            ORDER BY game_date DESC LIMIT 1
        """, (row["pitcher_id"], game_date)).fetchone()

        if prev:
            d0 = datetime.date.fromisoformat(prev["game_date"])
            rest = (d1 - d0).days - 1
            con.execute("""
                UPDATE pitcher_game_log SET days_rest=? WHERE game_id=? AND pitcher_id=?
            """, (rest, row["game_id"], row["pitcher_id"]))


def patch_statcast_batting(con: sqlite3.Connection, team_m: dict):
    for (game_pk, is_home), vals in team_m.items():
        con.execute("""
            UPDATE team_batting_game
            SET hard_hit_pct=?, barrel_pct=?
            WHERE game_id=? AND is_home=? AND hard_hit_pct IS NULL
        """, (vals["hard_hit_pct"], vals["barrel_pct"], game_pk, is_home))


def patch_statcast_velo(con: sqlite3.Connection, pitcher_m: dict):
    for (game_pk, pitcher_id), vals in pitcher_m.items():
        con.execute("""
            UPDATE pitcher_game_log
            SET avg_fastball_velo=?
            WHERE game_id=? AND pitcher_id=? AND avg_fastball_velo IS NULL
        """, (vals["avg_fastball_velo"], game_pk, pitcher_id))


# ── Box score pass ────────────────────────────────────────────────────────────


def run_boxscore_pass(start_date: str, end_date: str):
    """
    Fetch box scores for every Final game in [start_date, end_date].

    The MLB Stats API schedule endpoint silently truncates multi-year requests
    at roughly 2,500 results.  We chunk by calendar month (≈400-450 games each)
    to stay well inside that limit -- the same pattern the Statcast pass uses
    for weekly chunks.
    """
    con = get_db()
    ensure_state_table(con)

    start  = datetime.date.fromisoformat(start_date)
    end    = datetime.date.fromisoformat(end_date)
    chunks = date_range_months(start, end)

    print(f"\n[boxscore] {start_date} -> {end_date}  ({len(chunks)} monthly chunks)")

    # Build the done set once -- game_ids already fully written to batting table.
    # We extend it as we go so month-boundary duplicates are not re-fetched.
    done: set[int] = set(
        r[0] for r in con.execute(
            "SELECT DISTINCT game_id FROM team_batting_game"
        ).fetchall()
    )
    print(f"  {len(done)} game(s) already in DB -- will skip")

    total_written = 0
    total_errors: list[tuple[int, str]] = []

    for chunk_idx, (cs, ce) in enumerate(chunks, 1):
        if _shutdown.is_set():
            print(f"\n[boxscore] Stopped at user request after chunk {chunk_idx-1}.")
            break

        chunk_games = fetch_schedule(cs, ce)

        # Seed teams/venues from each chunk so FK constraints are always satisfied.
        t_rows, v_rows = extract_teams_venues(chunk_games)
        seed_venues(con, v_rows)
        seed_teams(con, t_rows)

        to_do = [
            g for g in chunk_games
            if g.get("status", {}).get("abstractGameState") == "Final"
            and g["gamePk"] not in done
        ]

        # Progress prefix: "[ 23/107]  2018-11"
        prefix = f"[{chunk_idx:>3}/{len(chunks)}]  {cs[:7]}"

        if not to_do:
            print(f"  {prefix}  {len(chunk_games)} scheduled | 0 new -- skip", flush=True)
            continue

        print(
            f"  {prefix}  {len(chunk_games)} scheduled | {len(to_do)} new Final games",
            flush=True,
        )

        eta    = ETA(len(to_do), "games")
        errors: list[tuple[int, str]] = []

        for g in to_do:
            if _shutdown.is_set():
                break

            game_pk   = g["gamePk"]
            game_date = g.get("gameDate", "")[:10]
            season    = int(game_date[:4])

            try:
                upsert_game(con, parse_schedule_row(g))
                bs     = fetch_boxscore(game_pk)
                parsed = parse_boxscore(game_pk, game_date, bs, season)

                upsert_batting(con, parsed["batting_rows"])
                upsert_pitcher(con, parsed["pitcher_rows"])
                upsert_bullpen(con, parsed["bullpen_rows"])
                upsert_relievers(con, parsed["reliever_rows"])
                set_starter_ids(con, game_pk, parsed["starter_ids"])
                backfill_days_rest(con, game_date)
                con.commit()
                done.add(game_pk)

            except Exception as exc:
                con.rollback()
                errors.append((game_pk, str(exc)))

            eta.tick()
            print(f"\r    {eta.line()}", end="", flush=True)
            time.sleep(0.3)

        print()  # newline after inline progress
        total_written += eta.done
        total_errors.extend(errors)

        if errors:
            print(f"    [warn] {len(errors)} game(s) failed in this chunk:")
            for gid, msg in errors[:5]:
                print(f"      game {gid}: {msg}")
            if len(errors) > 5:
                print(f"      ... and {len(errors)-5} more")

    print()
    if total_errors:
        print(f"[boxscore] {len(total_errors)} total error(s) across all chunks.")
    print(f"[boxscore] Pass complete -- {total_written} game(s) written.")
    con.close()


# ── Schedule seed pass ───────────────────────────────────────────────────────


def run_schedule_seed_pass(date_str: str):
    """
    Seed the games table with all games scheduled for date_str regardless of
    status (Preview, Pre-Game, Scheduled, etc.).

    This is called in daily mode so predict_mlb.py can find today's games
    before they are played.  Box scores and stats are NOT fetched — only the
    games row (home/away teams, venue, game_time_utc, status).

    Uses INSERT OR IGNORE so it never overwrites a row that was already written
    by run_boxscore_pass with a Final status and completed stats.
    """
    con = get_db()
    ensure_state_table(con)

    print(f"\n[schedule-seed] Seeding scheduled games for {date_str}")

    games = fetch_schedule(date_str, date_str)
    if not games:
        print(f"  No games returned by MLB API for {date_str}")
        con.close()
        return

    t_rows, v_rows = extract_teams_venues(games)
    seed_venues(con, v_rows)
    seed_teams(con, t_rows)

    inserted = 0
    skipped  = 0
    for g in games:
        row = parse_schedule_row(g)
        cur = con.execute("""
            INSERT OR IGNORE INTO games(
                game_id, game_date, game_time_utc, day_night,
                home_team_id, away_team_id, venue_id, series_game_number,
                doubleheader, status, home_score, away_score, total_runs, season)
            VALUES(
                :game_id,:game_date,:game_time_utc,:day_night,
                :home_team_id,:away_team_id,:venue_id,:series_game_number,
                :doubleheader,:status,:home_score,:away_score,:total_runs,:season)
        """, row)
        if cur.rowcount:
            inserted += 1
        else:
            skipped += 1

    con.commit()
    print(
        f"  {len(games)} games fetched | "
        f"{inserted} new rows inserted | "
        f"{skipped} already existed (skipped)"
    )
    con.close()


# ── Statcast pass ─────────────────────────────────────────────────────────────


def run_statcast_pass(start_date: str, end_date: str, workers: int):
    con = get_db()
    ensure_state_table(con)

    start = datetime.date.fromisoformat(start_date)
    end   = datetime.date.fromisoformat(end_date)
    all_weeks = date_range_weeks(start, end)

    patched: set = set(get_state(con, "statcast_patched_weeks", []))
    to_fetch = [(s, e) for s, e in all_weeks if s not in patched]

    print(f"\n[statcast] {len(all_weeks)} weeks in range | {len(patched)} already done "
          f"| {len(to_fetch)} to fetch | workers={workers}")

    if not to_fetch:
        print("  Nothing to do.")
        con.close()
        return

    eta     = ETA(len(to_fetch), "weeks")
    db_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_statcast_week, s, e): (s, e) for s, e in to_fetch}

        for future in as_completed(futures):
            if _shutdown.is_set():
                print("\n[statcast] Shutdown — cancelling remaining futures.")
                pool.shutdown(wait=False, cancel_futures=True)
                break

            s_dt, e_dt, result = future.result()

            if isinstance(result, Exception):
                print(f"\n  [error] statcast {s_dt}->{e_dt}: {result}", flush=True)
            else:
                try:
                    team_m, pitcher_m = aggregate_statcast(result)
                    with db_lock:
                        patch_statcast_batting(con, team_m)
                        patch_statcast_velo(con, pitcher_m)
                        patched.add(s_dt)
                        set_state(con, "statcast_patched_weeks", list(patched))
                        con.commit()
                except Exception as exc:
                    print(f"\n  [error] aggregating {s_dt}->{e_dt}: {exc}", flush=True)

            eta.tick()
            print(f"\r{eta.line()}", end="", flush=True)

    print()
    con.close()
    print("[statcast] Pass complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Collect MLB game data (schedule, box scores, Statcast) into SQLite"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--season", type=int, metavar="YYYY",
                      help="Collect a full season (e.g. 2023)")
    mode.add_argument("--start", type=str, metavar="YYYY-MM-DD",
                      help="Start date (default: yesterday)")

    parser.add_argument("--end", type=str, metavar="YYYY-MM-DD",
                        help="End date (default: today)")
    parser.add_argument("--statcast-only", action="store_true",
                        help="Skip box score pass; only patch Statcast metrics")
    parser.add_argument("--boxscore-only", action="store_true",
                        help="Skip Statcast pass")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel Statcast workers (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    today = datetime.date.today()
    is_daily_mode = (args.start is None and args.season is None)

    if args.season:
        start_date = f"{args.season}-03-20"
        end_date   = f"{args.season}-11-05"
    else:
        start_date = args.start or (today - datetime.timedelta(days=1)).isoformat()
        end_date   = args.end   or today.isoformat()

    # Hard floor: no data before 2017
    if start_date < f"{SEASON_START}-01-01":
        start_date = f"{SEASON_START}-03-20"

    print("=" * 60)
    print("MLB Statcast Collector")
    print(f"  Range   : {start_date} -> {end_date}")
    print(f"  DB      : {DB_PATH}")
    print(f"  Workers : {args.workers}")
    mode_str = ("statcast only" if args.statcast_only
                else "boxscore only" if args.boxscore_only
                else "full (boxscore + statcast)")
    print(f"  Mode    : {mode_str}")
    if is_daily_mode:
        print(f"  Daily   : yes  (will also seed today's scheduled games)")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        sys.exit(f"[error] Database not found at {DB_PATH}. Run setup_db.py first.")

    if not args.statcast_only:
        run_boxscore_pass(start_date, end_date)

    if not args.boxscore_only:
        run_statcast_pass(start_date, end_date, args.workers)

    # In daily mode, also seed today's scheduled games (status=Preview/Pre-Game)
    # so predict_mlb.py can find them in the games table before they're played.
    # This is separate from the boxscore pass, which only processes Final games.
    if is_daily_mode and not args.statcast_only:
        run_schedule_seed_pass(today.isoformat())

    print("\nDone.")


if __name__ == "__main__":
    main()
