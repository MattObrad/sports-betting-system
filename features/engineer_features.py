"""
engineer_features.py — Compute and cache game_features from raw collected data.

Usage:
    python features/engineer_features.py              # incremental (missing games only)
    python features/engineer_features.py --full       # recompute all games
    python features/engineer_features.py --season 2023
    python features/engineer_features.py --date 2024-05-28
    python features/engineer_features.py --version v1.1
"""

import os
import sys
import json
import math
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import sqlite3

try:
    import pybaseball
    pybaseball.cache.enable()
    HAS_PYBASEBALL = True
except ImportError:
    HAS_PYBASEBALL = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(__file__)
DB_PATH = os.path.normpath(os.path.join(_DIR, "..", "mlb_data.db"))

FEATURE_VERSION = "v1.0"

# Staleness guard: a starter's days_rest above this is impossible mid-season and
# means the rolling L3/L5 window crossed the off-season (incomplete backfill).
# Such rolling-form features are nulled in compute_pitcher_features().
MAX_DAYS_REST = 60

FIP_CONSTANTS = {
    2017: 3.19, 2018: 3.16, 2019: 3.36, 2020: 3.34,
    2021: 3.26, 2022: 2.96, 2023: 3.24, 2024: 3.17, 2025: 3.20,
}

FG_TEAM_MAP = {
    "Angels": "LAA",      "Astros": "HOU",      "Athletics": "OAK",
    "Blue Jays": "TOR",   "Braves": "ATL",       "Brewers": "MIL",
    "Cardinals": "STL",   "Cubs": "CHC",         "Diamondbacks": "ARI",
    "Dodgers": "LAD",     "Giants": "SF",        "Guardians": "CLE",
    "Indians": "CLE",     "Mariners": "SEA",     "Marlins": "MIA",
    "Mets": "NYM",        "Nationals": "WSH",    "Orioles": "BAL",
    "Padres": "SD",       "Phillies": "PHI",     "Pirates": "PIT",
    "Rangers": "TEX",     "Rays": "TB",          "Red Sox": "BOS",
    "Reds": "CIN",        "Rockies": "COL",      "Royals": "KC",
    "Tigers": "DET",      "Twins": "MIN",        "White Sox": "CWS",
    "Yankees": "NYY",
}

# =============================================================================
# UTILITIES
# =============================================================================

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles."""
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def juice_to_prob(juice: int) -> float:
    """American odds → implied probability."""
    if juice > 0:
        return 100.0 / (juice + 100)
    return abs(juice) / (abs(juice) + 100.0)


def _safe_era(er_sum: pd.Series, ip_sum: pd.Series) -> np.ndarray:
    return np.where(ip_sum > 0, er_sum / ip_sum * 9.0, np.nan)


def _safe_rate(num: pd.Series, denom: pd.Series, mult: float = 9.0) -> np.ndarray:
    return np.where(denom > 0, num / denom * mult, np.nan)


# =============================================================================
# BREF + SAVANT LOADERS  (FanGraphs permanently 403-blocked as of 2026)
# =============================================================================

def load_bref_pitcher_stats(seasons: list) -> pd.DataFrame:
    """
    Season-level pitcher stats from Baseball Reference.
    FIP and BB9 are computed from raw counts (HR, BB, SO, IP) — BBRef doesn't
    publish FIP directly but provides all the inputs.
    Returns: [name, mlb_id, fip_bref, so9_bref, bb9_bref, era_bref, ip_bref, season]
    Used as the xFIP proxy for pre-2015 seasons where Savant xERA isn't available.
    """
    if not HAS_PYBASEBALL:
        log.warning("pybaseball not installed — BBRef pitcher stats will be NULL")
        return pd.DataFrame()
    frames = []
    for s in seasons:
        try:
            df = pybaseball.pitching_stats_bref(s)
            df = df[df["IP"].notna() & (df["IP"] >= 5)].copy()

            # Compute FIP from raw counts using the project's FIP constants
            c = FIP_CONSTANTS.get(s, 3.20)
            if all(col in df.columns for col in ["HR", "BB", "SO"]):
                ip = df["IP"].replace(0, np.nan)
                df["fip_bref"] = (13 * df["HR"] + 3 * df["BB"] - 2 * df["SO"]) / ip + c

            # BB/9 from raw BB + IP
            if "BB" in df.columns:
                ip = df["IP"].replace(0, np.nan)
                df["bb9_bref"] = df["BB"] / ip * 9.0

            df = df.rename(columns={
                "Name":  "name",
                "ERA":   "era_bref",
                "SO9":   "so9_bref",
                "IP":    "ip_bref",
                "mlbID": "mlb_id",
            })
            keep = [c for c in ["name", "mlb_id", "fip_bref", "so9_bref",
                                 "bb9_bref", "era_bref", "ip_bref"] if c in df.columns]
            df = df[keep].copy()
            df["season"] = s
            frames.append(df)
            log.info("BBRef pitcher stats %d: %d rows (FIP computed from raw counts)", s, len(df))
        except Exception as exc:
            log.warning("BBRef pitcher stats %d failed: %s", s, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_savant_pitcher_xstats(seasons: list) -> pd.DataFrame:
    """
    Season-level xERA from Baseball Savant (statcast_pitcher_expected_stats).
    xERA is a better predictor of future ERA than xFIP or SIERA.
    Available from 2015 onwards (start of the Statcast era).
    player_id is the MLBAM player ID — same as pitcher_id in pitcher_game_log,
    so this can be joined directly without name matching.
    Returns: [name, player_id, xera, est_woba, season]
    """
    if not HAS_PYBASEBALL:
        return pd.DataFrame()
    frames = []
    for s in seasons:
        if s < 2015:
            log.info("Savant xStats not available pre-2015 — skipping %d (BBRef FIP used instead)", s)
            continue
        try:
            df = pybaseball.statcast_pitcher_expected_stats(s, minPA=10)
            # Name is stored as a single 'last_name, first_name' column — parse to 'First Last'
            nc = "last_name, first_name"
            if nc in df.columns:
                df["name"] = df[nc].str.split(", ").apply(
                    lambda x: f"{x[1].strip()} {x[0].strip()}" if len(x) == 2 else x[0].strip()
                )
            keep = [c for c in ["name", "player_id", "xera", "est_woba"] if c in df.columns]
            df = df[keep].copy()
            df["season"] = s
            frames.append(df)
            log.info("Savant xStats %d: %d pitchers (xERA)", s, len(df))
        except Exception as exc:
            log.warning("Savant pitcher xStats %d failed: %s", s, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def combine_pitcher_xstats(bref: pd.DataFrame, savant: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BBRef and Savant stats into the schema that compute_pitcher_features expects:
        [name, player_id (MLBAM), season, xfip, siera]

    Priority: Savant xERA is used where available (2015+).
    BBRef computed FIP is the fallback (pre-2015 or when Savant is missing a pitcher).
    siera is left NULL — no equivalent replacement without FanGraphs.
    """
    rows = []

    # Index Savant by (player_id, season) for fast lookup
    sav_by_id: dict = {}
    sav_by_name: dict = {}
    if not savant.empty and "player_id" in savant.columns:
        for _, r in savant.iterrows():
            sav_by_id[(int(r["player_id"]), int(r["season"]))] = r
            if "name" in savant.columns and pd.notna(r.get("name")):
                sav_by_name[(str(r["name"]).lower().strip(), int(r["season"]))] = r

    # Index BBRef by (name, season)
    bref_by_name: dict = {}
    if not bref.empty and "name" in bref.columns:
        for _, r in bref.iterrows():
            if pd.notna(r.get("name")):
                bref_by_name[(str(r["name"]).lower().strip(), int(r["season"]))] = r

    # Build unified set of (pitcher, season) keys from both sources
    keys: set = set()
    if not savant.empty:
        for _, r in savant.iterrows():
            keys.add((r.get("name", ""), r.get("player_id"), int(r["season"])))
    if not bref.empty:
        for _, r in bref.iterrows():
            keys.add((r.get("name", ""), None, int(r["season"])))

    for name, player_id, season in keys:
        # Prefer Savant xERA as xfip substitute
        xfip_val = np.nan
        sav_row = None
        if player_id is not None:
            sav_row = sav_by_id.get((int(player_id), season))
        if sav_row is None and pd.notna(name) and name:
            sav_row = sav_by_name.get((str(name).lower().strip(), season))
        if sav_row is not None:
            xfip_val = sav_row.get("xera", np.nan)

        # Fallback: BBRef computed FIP
        if np.isnan(xfip_val) and pd.notna(name) and name:
            bref_row = bref_by_name.get((str(name).lower().strip(), season))
            if bref_row is not None:
                xfip_val = bref_row.get("fip_bref", np.nan)

        rows.append({
            "name":      name,
            "player_id": int(player_id) if player_id is not None else np.nan,
            "season":    season,
            "xfip":      xfip_val,
            "siera":     np.nan,   # no equivalent replacement
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["name", "player_id", "season", "xfip", "siera"]
    )


def load_fg_team_wrc(seasons: list) -> pd.DataFrame:
    """
    wRC+ source removed (FanGraphs 403-blocked).
    wrc_plus feature will be NULL in game_features until a BBRef replacement is wired up.
    Function retained as a stub so call sites don't need to change.
    """
    log.info("wRC+ source (FanGraphs) unavailable — wrc_plus will be NULL.")
    return pd.DataFrame()


# =============================================================================
# DB DATA LOADERS
# =============================================================================

def load_db_data(con: sqlite3.Connection, where_clause: str = "") -> dict:
    """Load all raw tables into DataFrames."""
    game_filter = f"WHERE {where_clause}" if where_clause else ""

    games = pd.read_sql(f"""
        SELECT game_id, game_date, season, home_team_id, away_team_id,
               venue_id, home_starter_id, away_starter_id,
               home_plate_umpire_id, series_game_number, day_night,
               doubleheader, status
        FROM games
        {game_filter}
        ORDER BY game_date
    """, con, parse_dates=["game_date"])

    bat = pd.read_sql("""
        SELECT b.game_id, b.game_date, b.team_id, b.is_home, b.opponent_id,
               b.opposing_starter_throws, b.runs_scored,
               b.obp, b.slg, b.woba, b.hard_hit_pct, b.barrel_pct,
               b.k_pct, b.bb_pct,
               g.season
        FROM team_batting_game b
        JOIN games g USING (game_id)
        ORDER BY b.team_id, b.game_date
    """, con, parse_dates=["game_date"])

    pit = pd.read_sql("""
        SELECT p.game_id, p.game_date, p.pitcher_id, p.team_id, p.opponent_id,
               p.is_home, p.innings_pitched, p.earned_runs,
               p.strikeouts, p.walks, p.hr_allowed,
               p.avg_fastball_velo, p.days_rest,
               g.season
        FROM pitcher_game_log p
        JOIN games g USING (game_id)
        ORDER BY p.pitcher_id, p.game_date
    """, con, parse_dates=["game_date"])

    bul = pd.read_sql("""
        SELECT b.game_id, b.game_date, b.team_id, b.era_relief, b.fip_relief,
               g.season
        FROM bullpen_game b
        JOIN games g USING (game_id)
        ORDER BY b.team_id, b.game_date
    """, con, parse_dates=["game_date"])

    ump_game = pd.read_sql("""
        SELECT game_id, umpire_id, zone_size_score, over_under_result
        FROM umpire_game
        ORDER BY game_date
    """, con)

    umpires = pd.read_sql("SELECT umpire_id, umpire_name FROM umpires", con)

    weather = pd.read_sql("""
        SELECT game_id, temperature_f, wind_speed_mph, wind_to_cf,
               precipitation_prob, humidity_pct, roof_closed
        FROM weather
    """, con)

    venues = pd.read_sql("""
        SELECT venue_id, latitude, longitude, altitude_ft, roof_type
        FROM venues
    """, con)

    teams = pd.read_sql("SELECT team_id, team_code FROM teams", con)

    pf = pd.read_sql("SELECT venue_id, season, pf_runs FROM park_factors", con)

    players = pd.read_sql(
        "SELECT player_id, name_first, name_last, throws FROM players", con
    )

    odds = pd.read_sql("""
        SELECT game_id, snapshot_time, total_line, over_juice, under_juice,
               is_opening, is_closing
        FROM odds_snapshots
        ORDER BY game_id, snapshot_time
    """, con)
    # Normalise snapshot_time to UTC-aware datetime so sort_values() is
    # chronologically correct across both SBR ("2021-04-01T00:00:00") and
    # VPS Kambi ("2026-05-29 02:02:34.385170+00:00") timestamp formats.
    if not odds.empty:
        odds["snapshot_time"] = pd.to_datetime(
            odds["snapshot_time"], format="mixed", utc=True
        )

    travel = pd.read_sql("""
        SELECT team_id, game_date, travel_dist_miles, days_since_last_game
        FROM team_travel
        ORDER BY team_id, game_date
    """, con, parse_dates=["game_date"])

    return dict(
        games=games, bat=bat, pit=pit, bul=bul,
        ump_game=ump_game, umpires=umpires, weather=weather,
        venues=venues, teams=teams, pf=pf, players=players,
        odds=odds, travel=travel,
    )


def load_ump_scorecards_cache(con: sqlite3.Connection) -> dict:
    """Load UmpScorecards career cache from _collector_state."""
    try:
        row = con.execute(
            "SELECT value FROM _collector_state WHERE key = 'umpscorecards_career'"
        ).fetchone()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return {}


# =============================================================================
# FEATURE BLOCK: TEAM OFFENSE
# =============================================================================

def compute_team_offense(bat_df: pd.DataFrame, teams_df: pd.DataFrame,
                         fg_wrc: pd.DataFrame) -> pd.DataFrame:
    """
    Returns (game_id, team_id, is_home) + rolling batting features.
    shift(1) on every window ensures no lookahead.
    """
    df = bat_df.sort_values(["team_id", "game_date"]).copy()
    g = df.groupby("team_id", group_keys=False)

    # All-games rolling means
    for n, sfx in [(5, "5"), (10, "10"), (30, "30")]:
        df[f"rpg_roll{sfx}"] = g["runs_scored"].transform(
            lambda x, n=n: x.rolling(n, min_periods=1).mean().shift(1)
        )
    df["rpg_szn"] = df.groupby(["team_id", "season"], group_keys=False)["runs_scored"].transform(
        lambda x: x.expanding(1).mean().shift(1)
    )

    for col, out_sfx in [
        ("obp",          "obp"),
        ("slg",          "slg"),
        ("woba",         "woba"),
        ("k_pct",        "k_pct"),
        ("bb_pct",       "bb_pct"),
        ("hard_hit_pct", "hard_hit"),
        ("barrel_pct",   "barrel"),
    ]:
        for n, sfx in [(5, "5"), (30, "30")]:
            df[f"{out_sfx}_roll{sfx}"] = g[col].transform(
                lambda x, n=n: x.rolling(n, min_periods=1).mean().shift(1)
            )

    # vs RHP / vs LHP splits (rolling 30 within each hand subset)
    for hand, col_out in [("R", "rpg_vs_rhp_30"), ("L", "rpg_vs_lhp_30")]:
        sub = df[df["opposing_starter_throws"] == hand].copy()
        sub[col_out] = sub.groupby("team_id", group_keys=False)["runs_scored"].transform(
            lambda x: x.rolling(30, min_periods=1).mean().shift(1)
        )
        df = df.merge(sub[["game_id", "team_id", col_out]], on=["game_id", "team_id"], how="left")

    # Home / road location splits (rolling 30 within location subset)
    for loc_is_home, col_out in [(1, "rpg_home_30"), (0, "rpg_away_30")]:
        sub = df[df["is_home"] == loc_is_home].copy()
        sub[col_out] = sub.groupby("team_id", group_keys=False)["runs_scored"].transform(
            lambda x: x.rolling(30, min_periods=1).mean().shift(1)
        )
        df = df.merge(sub[["game_id", "team_id", col_out]], on=["game_id", "team_id"], how="left")

    # FanGraphs wRC+ — use season Y-1 FG data for season Y games
    if fg_wrc is not None and len(fg_wrc):
        df = df.merge(teams_df[["team_id", "team_code"]], on="team_id", how="left")
        fg_shifted = fg_wrc.copy()
        fg_shifted["season"] = fg_shifted["season"] + 1  # Y-1 data → labelled as season Y
        df = df.merge(
            fg_shifted[["team_code", "season", "wrc_plus"]],
            on=["team_code", "season"], how="left",
        )
    else:
        if "team_code" not in df.columns:
            df["team_code"] = np.nan
        df["wrc_plus"] = np.nan

    keep = [
        "game_id", "team_id", "is_home",
        "rpg_roll5", "rpg_roll10", "rpg_roll30", "rpg_szn",
        "obp_roll5", "obp_roll30", "slg_roll5", "slg_roll30",
        "woba_roll5", "woba_roll30", "wrc_plus",
        "k_pct_roll30", "bb_pct_roll30", "hard_hit_roll30", "barrel_roll30",
        "rpg_vs_rhp_30", "rpg_vs_lhp_30", "rpg_home_30", "rpg_away_30",
    ]
    return df[[c for c in keep if c in df.columns]]


# =============================================================================
# FEATURE BLOCK: TEAM DEFENSE
# =============================================================================

def compute_team_defense(bat_df: pd.DataFrame, bul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns (game_id, team_id) + rolling runs-allowed and bullpen ERA/FIP.
    Runs allowed for team X = opponent's runs_scored in same game.
    """
    opp = bat_df[["game_id", "team_id", "runs_scored"]].rename(
        columns={"team_id": "opponent_id", "runs_scored": "runs_allowed"}
    )
    df = bat_df[["game_id", "game_date", "team_id", "opponent_id"]].copy()
    df = df.merge(opp, on=["game_id", "opponent_id"], how="left")
    df = df.sort_values(["team_id", "game_date"])

    g = df.groupby("team_id", group_keys=False)
    for n, sfx in [(5, "5"), (10, "10"), (30, "30")]:
        df[f"ra_roll{sfx}"] = g["runs_allowed"].transform(
            lambda x, n=n: x.rolling(n, min_periods=1).mean().shift(1)
        )

    # Bullpen rolling ERA / FIP (last 10 starts)
    bul = bul_df.sort_values(["team_id", "game_date"]).copy()
    bg = bul.groupby("team_id", group_keys=False)
    bul["bul_era_roll10"] = bg["era_relief"].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1)
    )
    bul["bul_fip_roll10"] = bg["fip_relief"].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1)
    )

    df = df.merge(
        bul[["game_id", "team_id", "bul_era_roll10", "bul_fip_roll10"]],
        on=["game_id", "team_id"], how="left",
    )
    return df[["game_id", "team_id", "ra_roll5", "ra_roll10", "ra_roll30",
               "bul_era_roll10", "bul_fip_roll10"]]


# =============================================================================
# FEATURE BLOCK: BULLPEN LOAD
# =============================================================================

def compute_bullpen_load(con: sqlite3.Connection, status_clause: str) -> pd.DataFrame:
    """
    IP thrown by bullpen in the 3 calendar days prior to each game, per team.

    Drives off the games table (home + away) rather than bullpen_game so that
    UNPLAYED games (Pre-Game/Scheduled), which have no bullpen_game row yet,
    still receive a point-in-time bullpen-load value.  The reliever_appearance
    join is strictly before game_date (r.game_date < gt.game_date), so no
    same-day leakage.
    """
    rows = con.execute(f"""
        SELECT gt.game_id, gt.team_id,
               COALESCE(SUM(r.innings_pitched), 0.0) AS bullpen_ip_3days
        FROM (
            SELECT game_id, home_team_id AS team_id, game_date FROM games WHERE {status_clause}
            UNION ALL
            SELECT game_id, away_team_id AS team_id, game_date FROM games WHERE {status_clause}
        ) gt
        LEFT JOIN reliever_appearance r
            ON  r.team_id  = gt.team_id
            AND r.game_date >= date(gt.game_date, '-3 days')
            AND r.game_date <  gt.game_date
        GROUP BY gt.game_id, gt.team_id
    """).fetchall()
    return pd.DataFrame(rows, columns=["game_id", "team_id", "bullpen_ip_3days"])


# =============================================================================
# PRE-GAME PLACEHOLDERS  (fix for all-NULL live features)
# =============================================================================

def build_pregame_placeholders(
    unplayed_games: pd.DataFrame,
    players_df: pd.DataFrame,
    pit_hist: pd.DataFrame,
) -> tuple:
    """
    Build synthetic box-score rows for UNPLAYED games (Pre-Game/Preview/Scheduled).

    Why this exists
    ---------------
    Every rolling feature in this module is computed by attaching a
    `.rolling(n).shift(1)` value to a team's / pitcher's own box-score row.
    An unplayed game has no box-score row, so the joins in assemble_features()
    return NULL for every rolling feature — the model would score today's games
    on an empty vector.

    The fix: append placeholder rows for the upcoming games to the bat/pit/bul
    frames BEFORE the rolling blocks run.  All stat columns are NaN, but the
    placeholder carries the keys the rolling/split logic needs (team_id, is_home,
    opponent_id, opposing_starter_throws, pitcher_id, season, game_date).

    Because every window uses `.shift(1)`, a placeholder's own NaN stats are
    never read for its own feature value — only the prior N COMPLETED games are
    used.  This guarantees the live feature vector is computed by the *identical*
    code path as the training features (no train/serve skew).

    Returns (bat_ph, pit_ph, bul_ph) DataFrames (possibly empty).
    """
    throws = {}
    if players_df is not None and len(players_df):
        throws = dict(zip(players_df["player_id"], players_df["throws"]))

    # Most recent completed appearance per pitcher → days_rest for the placeholder.
    last_app = {}
    if pit_hist is not None and len(pit_hist):
        last_app = pit_hist.groupby("pitcher_id")["game_date"].max().to_dict()

    bat_rows, pit_rows, bul_rows = [], [], []
    for g in unplayed_games.itertuples(index=False):
        gid, gdate, season = g.game_id, g.game_date, g.season
        home, away = g.home_team_id, g.away_team_id
        hsp, asp = g.home_starter_id, g.away_starter_id

        # Offense + bullpen: one row per team.  opposing_starter_throws is the
        # OTHER team's probable starter's handedness (drives the vs-LHP/RHP split).
        for team, opp, is_home, opp_sp in [
            (home, away, 1, asp),
            (away, home, 0, hsp),
        ]:
            bat_rows.append({
                "game_id": gid, "game_date": gdate, "team_id": team,
                "is_home": is_home, "opponent_id": opp,
                "opposing_starter_throws": throws.get(opp_sp),
                "runs_scored": np.nan, "obp": np.nan, "slg": np.nan,
                "woba": np.nan, "hard_hit_pct": np.nan, "barrel_pct": np.nan,
                "k_pct": np.nan, "bb_pct": np.nan, "season": season,
            })
            bul_rows.append({
                "game_id": gid, "game_date": gdate, "team_id": team,
                "era_relief": np.nan, "fip_relief": np.nan, "season": season,
            })

        # Pitcher: one row per probable starter (skip TBD/unknown starters).
        for pid, team, opp, is_home in [
            (hsp, home, away, 1),
            (asp, away, home, 0),
        ]:
            if pid is None or (isinstance(pid, float) and math.isnan(pid)):
                continue
            dr = np.nan
            la = last_app.get(pid)
            if la is not None and pd.notna(la):
                dr = (pd.Timestamp(gdate) - pd.Timestamp(la)).days
            pit_rows.append({
                "game_id": gid, "game_date": gdate, "pitcher_id": pid,
                "team_id": team, "opponent_id": opp, "is_home": is_home,
                "innings_pitched": np.nan, "earned_runs": np.nan,
                "strikeouts": np.nan, "walks": np.nan, "hr_allowed": np.nan,
                "avg_fastball_velo": np.nan, "days_rest": dr, "season": season,
            })

    return (
        pd.DataFrame(bat_rows),
        pd.DataFrame(pit_rows),
        pd.DataFrame(bul_rows),
    )


# =============================================================================
# FEATURE BLOCK: PITCHER
# =============================================================================

def compute_pitcher_features(
    pit_df: pd.DataFrame,
    fg_pit: pd.DataFrame,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns (pitcher_id, game_id) + rolling ERA/FIP/K9/BB9/HR9/velo + career vs opp.
    xFIP and SIERA come from FanGraphs prior season (Y-1).
    career_vs_opp_pa = 0 (not NULL) when pitcher has never faced this opponent.
    """
    df = pit_df.sort_values(["pitcher_id", "game_date"]).copy()
    g = df.groupby("pitcher_id", group_keys=False)

    # Rolling ERA: sum(ER)/sum(IP)*9 avoids Jensen's inequality from averaging rates
    for n, sfx in [(3, "l3"), (5, "l5")]:
        er_s = g["earned_runs"].transform(lambda x, n=n: x.rolling(n, min_periods=1).sum().shift(1))
        ip_s = g["innings_pitched"].transform(lambda x, n=n: x.rolling(n, min_periods=1).sum().shift(1))
        df[f"era_{sfx}"] = _safe_era(er_s, ip_s)

    # Season ERA (expanding within season)
    sg = df.groupby(["pitcher_id", "season"], group_keys=False)
    er_szn = sg["earned_runs"].transform(lambda x: x.expanding(1).sum().shift(1))
    ip_szn = sg["innings_pitched"].transform(lambda x: x.expanding(1).sum().shift(1))
    df["era_szn"] = _safe_era(er_szn, ip_szn)

    # Rolling FIP last 5 (HR, BB, K, IP sums)
    for col in ["hr_allowed", "walks", "strikeouts", "innings_pitched"]:
        df[f"_s5_{col}"] = g[col].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    df["_fip_c"] = df["season"].map(FIP_CONSTANTS).fillna(3.20)
    ip5 = df["_s5_innings_pitched"]
    df["fip_l5"] = np.where(
        ip5 > 0,
        (13 * df["_s5_hr_allowed"] + 3 * df["_s5_walks"] - 2 * df["_s5_strikeouts"]) / ip5 + df["_fip_c"],
        np.nan,
    )
    df["k9_l5"]  = _safe_rate(df["_s5_strikeouts"], ip5, 9.0)
    df["bb9_l5"] = _safe_rate(df["_s5_walks"],      ip5, 9.0)
    df["hr9_l5"] = _safe_rate(df["_s5_hr_allowed"], ip5, 9.0)

    df["avg_velo_l5"] = g["avg_fastball_velo"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )

    # Home/road season ERA
    for loc_val, era_col in [(1, "era_home_szn"), (0, "era_away_szn")]:
        sub = df[df["is_home"] == loc_val].copy()
        sg2 = sub.groupby(["pitcher_id", "season"], group_keys=False)
        er_l = sg2["earned_runs"].transform(lambda x: x.expanding(1).sum().shift(1))
        ip_l = sg2["innings_pitched"].transform(lambda x: x.expanding(1).sum().shift(1))
        sub[era_col] = _safe_era(er_l, ip_l)
        df = df.merge(sub[["game_id", "pitcher_id", era_col]], on=["game_id", "pitcher_id"], how="left")

    # Career ERA vs each opponent (cumulative ER/IP across all seasons, shift(1))
    df_vs = df.sort_values(["pitcher_id", "opponent_id", "game_date"]).copy()
    vg = df_vs.groupby(["pitcher_id", "opponent_id"], group_keys=False)
    vs_er = vg["earned_runs"].transform(lambda x: x.expanding(1).sum().shift(1))
    vs_ip = vg["innings_pitched"].transform(lambda x: x.expanding(1).sum().shift(1))
    df_vs["career_vs_opp_era"] = _safe_era(vs_er, vs_ip)
    # career_vs_opp_pa: approximate batters faced (IP * 4); explicitly 0 for no prior matchup
    df_vs["career_vs_opp_pa"] = np.where(vs_ip > 0, (vs_ip * 4).round(), 0.0)

    df = df.merge(
        df_vs[["game_id", "pitcher_id", "career_vs_opp_era", "career_vs_opp_pa"]],
        on=["game_id", "pitcher_id"], how="left",
    )
    df["career_vs_opp_pa"] = df["career_vs_opp_pa"].fillna(0).astype(int)

    # Always merge throws from players table (independent of FanGraphs availability)
    if players_df is not None and len(players_df):
        df = df.merge(
            players_df[["player_id", "throws"]].rename(columns={"player_id": "pitcher_id"}),
            on="pitcher_id", how="left",
        )
    else:
        if "throws" not in df.columns:
            df["throws"] = np.nan

    # External pitcher xStats: Savant xERA (preferred) + BBRef FIP (fallback)
    # fg_pit parameter now holds the output of combine_pitcher_xstats()
    if fg_pit is not None and len(fg_pit):
        ext = fg_pit.copy()
        ext["season"] = ext["season"] + 1  # Y-1 stats → applied to season Y games

        # Pass 1: join by MLBAM player_id (most reliable — no name ambiguity)
        if "player_id" in ext.columns:
            ext_id = ext.dropna(subset=["player_id"]).copy()
            ext_id["pitcher_id_join"] = ext_id["player_id"].astype(int)
            df = df.merge(
                ext_id[["pitcher_id_join", "season", "xfip", "siera"]],
                left_on=["pitcher_id", "season"],
                right_on=["pitcher_id_join", "season"],
                how="left",
            ).drop(columns=["pitcher_id_join"], errors="ignore")

        # Pass 2: name-match fallback for rows still missing xfip after ID join.
        # Build a (pitcher_id, season) → xfip lookup via player name, then fillna.
        if "xfip" not in df.columns:
            df["xfip"] = np.nan
        if "siera" not in df.columns:
            df["siera"] = np.nan

        if df["xfip"].isna().any() and players_df is not None and len(players_df):
            pl = players_df.copy()
            pl["name_full"] = (
                pl["name_first"].fillna("") + " " + pl["name_last"].fillna("")
            ).str.strip()
            ext_name = ext.dropna(subset=["name"]).copy()
            name_lookup = (
                pl[["player_id", "name_full"]]
                .rename(columns={"player_id": "pitcher_id"})
                .merge(
                    ext_name[["name", "season", "xfip", "siera"]].rename(
                        columns={"name": "name_full"}
                    ),
                    on="name_full", how="inner",
                )
                .dropna(subset=["xfip"])
                .drop_duplicates(subset=["pitcher_id", "season"], keep="first")
            )
            if not name_lookup.empty:
                df = df.merge(
                    name_lookup[["pitcher_id", "season", "xfip", "siera"]],
                    on=["pitcher_id", "season"], how="left", suffixes=("", "_nm"),
                )
                df["xfip"]  = df["xfip"].fillna(df.get("xfip_nm",  np.nan))
                df["siera"] = df["siera"].fillna(df.get("siera_nm", np.nan))
                df = df.drop(columns=["xfip_nm", "siera_nm"], errors="ignore")
    else:
        df["xfip"] = np.nan
        df["siera"] = np.nan

    # Drop temp columns
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # ── Staleness guard ──────────────────────────────────────────────────────
    # days_rest > MAX_DAYS_REST is impossible mid-season; it means the rolling
    # L3/L5 window reached back across the off-season because the pitcher has no
    # recent collected start (e.g. an incomplete 2026 backfill). Those rolling
    # features are then year-old data masquerading as current form — the exact
    # bug behind game 822727 (Meyer days_rest=366 → era_l5=5.96 from May 2025).
    # Null the rolling-form features so the model treats them as MISSING rather
    # than trusting stale numbers. xfip/siera (prior-season FanGraphs, keyed by
    # season) and career_vs_opp (cumulative, shift(1)-correct) are NOT stale and
    # are preserved. days_rest itself is kept as an informative flag.
    STALE_ROLLING_COLS = [
        "era_l3", "era_l5", "fip_l5",
        "k9_l5", "bb9_l5", "hr9_l5", "avg_velo_l5",
    ]
    stale = df["days_rest"] > MAX_DAYS_REST
    n_stale = int(stale.sum())
    if n_stale:
        log.warning(
            "Staleness guard: %d pitcher-game row(s) have days_rest > %d -- "
            "nulling rolling-form features (era_l3/l5, fip_l5, k9/bb9/hr9_l5, velo).",
            n_stale, MAX_DAYS_REST,
        )
        for col in STALE_ROLLING_COLS:
            if col in df.columns:
                df.loc[stale, col] = np.nan

    keep = [
        "game_id", "pitcher_id",
        "era_l3", "era_l5", "era_szn", "fip_l5", "xfip", "siera",
        "k9_l5", "bb9_l5", "hr9_l5", "avg_velo_l5", "days_rest",
        "career_vs_opp_era", "career_vs_opp_pa",
        "era_home_szn", "era_away_szn", "throws",
    ]
    return df[[c for c in keep if c in df.columns]]


# =============================================================================
# FEATURE BLOCK: UMPIRE
# =============================================================================

def compute_umpire_features(
    ump_game_df: pd.DataFrame,
    umpires_df: pd.DataFrame,
    ump_cache: dict,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns (game_id) + umpire tendency features.
    ump_games_sampled = 0 (not NULL) for umpires with no UmpScorecards data.
    """
    game_umps = games_df[["game_id", "home_plate_umpire_id"]].merge(
        umpires_df[["umpire_id", "umpire_name"]],
        left_on="home_plate_umpire_id", right_on="umpire_id", how="left",
    )

    rows = []
    for _, row in game_umps.iterrows():
        name = row["umpire_name"]
        entry = ump_cache.get(name.lower(), {}) if pd.notna(name) else {}
        rows.append({
            "game_id":                  int(row["game_id"]),
            "ump_career_over_rate":     entry.get("ump_career_over_rate"),
            "ump_career_runs_per_game": entry.get("ump_career_runs_per_game"),
            "ump_zone_size_score":      entry.get("ump_zone_size_score"),
            "ump_games_sampled":        int(entry.get("ump_games_sampled", 0)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# FEATURE BLOCK: WEATHER + PARK
# =============================================================================

def compute_weather_park(
    wx_df: pd.DataFrame,
    venues_df: pd.DataFrame,
    pf_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Returns (game_id) + weather and park features. Park factors use Y-1 season."""
    df = games_df[["game_id", "venue_id", "season"]].copy()
    df = df.merge(wx_df, on="game_id", how="left")
    df = df.merge(venues_df[["venue_id", "altitude_ft", "roof_type"]], on="venue_id", how="left")

    # Park factors: use prior season to avoid leakage
    pf_prev = pf_df.copy()
    pf_prev["season"] = pf_prev["season"] + 1
    df = df.merge(pf_prev[["venue_id", "season", "pf_runs"]], on=["venue_id", "season"], how="left")

    df["is_dome"] = (df["roof_type"] == "dome").astype(int)

    return df[["game_id", "temperature_f", "wind_speed_mph", "wind_to_cf",
               "precipitation_prob", "humidity_pct", "roof_closed",
               "pf_runs", "altitude_ft", "is_dome"]]


# =============================================================================
# FEATURE BLOCK: MARKET
# =============================================================================

def compute_market_features(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns (game_id) + market features derived from opening and closing lines.

    Existing features (unchanged):
      opening_total     — line when first posted
      current_total     — line at game time (closing)
      line_movement     — current_total - opening_total (positive = line moved up)
      over_juice        — closing over juice (American odds integer)
      under_juice       — closing under juice
      implied_over_prob — closing over probability including vig
      hours_since_open  — time between opening and closing snapshot

    New features:
      implied_total     — closing line adjusted for juice asymmetry.
                          Strips vig from over/under juice to get no-vig probabilities,
                          then shifts the posted line proportional to how far the
                          no-vig over probability deviates from 50%.
                          When over is more expensive (-120 vs -100), the market
                          consensus run total is slightly higher than the posted line.
                          Formula: current_total + (no_vig_over_prob - 0.5)
                          At -110/-110: adjustment = 0 (symmetric).
                          At -120/-100: no_vig_over ≈ 0.545, adjustment ≈ +0.045.

      juice_imbalance   — over_juice - under_juice (closing, American odds integers).
                          0 = symmetric (-110/-110).
                          Negative = over is more expensive (sharp money on under,
                          or public pounding the over and book raising its price).
                          Positive = under is more expensive.
                          Example: -120 over, -100 under → imbalance = -20.
    """
    if odds_df.empty:
        return pd.DataFrame(columns=[
            "game_id", "opening_total", "current_total", "line_movement",
            "over_juice", "under_juice", "implied_over_prob", "hours_since_open",
            "implied_total", "juice_imbalance",
        ])

    # Opening row: first snapshot marked is_opening=1
    opening = (
        odds_df[odds_df["is_opening"] == 1]
        .sort_values("snapshot_time")
        .groupby("game_id")
        .first()
        .reset_index()
        .rename(columns={"total_line": "opening_total", "snapshot_time": "open_time"})
    )

    # Closing row: last snapshot by time (is_closing=1 or final snapshot)
    closing = (
        odds_df[odds_df["is_closing"] == 1]
        .sort_values("snapshot_time")
        .groupby("game_id")
        .last()
        .reset_index()
        .rename(columns={"total_line": "current_total"})
    )
    # Fallback: if no is_closing rows, use the last overall snapshot
    if closing.empty:
        closing = (
            odds_df.sort_values("snapshot_time")
            .groupby("game_id").last().reset_index()
            .rename(columns={"total_line": "current_total"})
        )

    df = closing[["game_id", "current_total", "over_juice", "under_juice", "snapshot_time"]].merge(
        opening[["game_id", "opening_total", "open_time"]], on="game_id", how="left"
    )

    # ── Existing derived features ────────────────────────────────────────────
    df["line_movement"] = df["current_total"] - df["opening_total"]

    df["implied_over_prob"] = df["over_juice"].apply(
        lambda j: juice_to_prob(int(j)) if pd.notna(j) else np.nan
    )

    # Strip timezone info before subtracting.
    # format='mixed' handles both SBR ("2021-04-01T00:00:00") and VPS Kambi
    # ("2026-05-29 02:02:34.385170+00:00") timestamp formats in the same column.
    t_close = pd.to_datetime(df["snapshot_time"], format="mixed", utc=True).dt.tz_localize(None)
    t_open  = pd.to_datetime(df["open_time"],     format="mixed", utc=True).dt.tz_localize(None)
    df["hours_since_open"] = (t_close - t_open).dt.total_seconds() / 3600.0

    # ── New: implied_total ───────────────────────────────────────────────────
    # Strip vig from closing juice to get no-vig over probability.
    # Then adjust posted line proportionally: each percentage point above 50%
    # corresponds to the market slightly favouring the over beyond the posted number.
    def _implied_total(row) -> float | None:
        oj = row["over_juice"]
        uj = row["under_juice"]
        ct = row["current_total"]
        if pd.isna(oj) or pd.isna(uj) or pd.isna(ct):
            return np.nan
        p_over  = juice_to_prob(int(oj))
        p_under = juice_to_prob(int(uj))
        total_p = p_over + p_under
        if total_p <= 0:
            return float(ct)
        no_vig_over = p_over / total_p          # no-vig over probability
        # Shift: each +1% above 50% → +0.01 run adjustment on the implied total.
        # This is a subtle signal; the main value is directional.
        return float(ct) + (no_vig_over - 0.5)

    df["implied_total"] = df.apply(_implied_total, axis=1)

    # ── New: juice_imbalance ─────────────────────────────────────────────────
    # Negative = over more expensive (public/sharp on over).
    # Positive = under more expensive.
    df["juice_imbalance"] = np.where(
        df["over_juice"].notna() & df["under_juice"].notna(),
        df["over_juice"].astype(float) - df["under_juice"].astype(float),
        np.nan,
    )

    return df[["game_id", "opening_total", "current_total", "line_movement",
               "over_juice", "under_juice", "implied_over_prob", "hours_since_open",
               "implied_total", "juice_imbalance"]]


# =============================================================================
# FEATURE BLOCK: SCHEDULE / TRAVEL
# =============================================================================

def compute_schedule_features(
    games_df: pd.DataFrame,
    travel_df: pd.DataFrame,
) -> pd.DataFrame:
    """Returns (game_id) + days rest, travel distance, series info, day/night."""
    df = games_df[["game_id", "game_date", "home_team_id", "away_team_id",
                   "series_game_number", "day_night"]].copy()

    home_t = travel_df.rename(columns={
        "team_id": "home_team_id",
        "travel_dist_miles": "home_travel_dist_miles",
        "days_since_last_game": "home_days_rest",
    })
    away_t = travel_df.rename(columns={
        "team_id": "away_team_id",
        "travel_dist_miles": "away_travel_dist_miles",
        "days_since_last_game": "away_days_rest",
    })

    df = df.merge(
        home_t[["home_team_id", "game_date", "home_travel_dist_miles", "home_days_rest"]],
        on=["home_team_id", "game_date"], how="left",
    )
    df = df.merge(
        away_t[["away_team_id", "game_date", "away_travel_dist_miles", "away_days_rest"]],
        on=["away_team_id", "game_date"], how="left",
    )
    df["is_day_game"] = (df["day_night"] == "D").astype(int)

    return df[["game_id", "home_days_rest", "away_days_rest",
               "home_travel_dist_miles", "away_travel_dist_miles",
               "series_game_number", "is_day_game"]]


# =============================================================================
# ASSEMBLY
# =============================================================================

# Column rename maps for offense block
_OFF_RENAME_HOME = {
    "rpg_roll5":       "home_runs_per_game_5",
    "rpg_roll10":      "home_runs_per_game_10",
    "rpg_roll30":      "home_runs_per_game_30",
    "rpg_szn":         "home_runs_per_game_szn",
    "obp_roll5":       "home_obp_5",
    "obp_roll30":      "home_obp_30",
    "slg_roll5":       "home_slg_5",
    "slg_roll30":      "home_slg_30",
    "woba_roll5":      "home_woba_5",
    "woba_roll30":     "home_woba_30",
    "wrc_plus":        "home_wrc_plus_szn",
    "k_pct_roll30":    "home_k_pct_30",
    "bb_pct_roll30":   "home_bb_pct_30",
    "hard_hit_roll30": "home_hard_hit_30",
    "barrel_roll30":   "home_barrel_30",
    "rpg_vs_rhp_30":   "home_runs_vs_rhp_30",
    "rpg_vs_lhp_30":   "home_runs_vs_lhp_30",
    "rpg_home_30":     "home_home_runs_per_game_30",
}
_OFF_RENAME_AWAY = {
    "rpg_roll5":       "away_runs_per_game_5",
    "rpg_roll10":      "away_runs_per_game_10",
    "rpg_roll30":      "away_runs_per_game_30",
    "rpg_szn":         "away_runs_per_game_szn",
    "obp_roll5":       "away_obp_5",
    "obp_roll30":      "away_obp_30",
    "slg_roll5":       "away_slg_5",
    "slg_roll30":      "away_slg_30",
    "woba_roll5":      "away_woba_5",
    "woba_roll30":     "away_woba_30",
    "wrc_plus":        "away_wrc_plus_szn",
    "k_pct_roll30":    "away_k_pct_30",
    "bb_pct_roll30":   "away_bb_pct_30",
    "hard_hit_roll30": "away_hard_hit_30",
    "barrel_roll30":   "away_barrel_30",
    "rpg_vs_rhp_30":   "away_runs_vs_rhp_30",
    "rpg_vs_lhp_30":   "away_runs_vs_lhp_30",
    "rpg_away_30":     "away_away_runs_per_game_30",
}
_DEF_RENAME_HOME = {
    "ra_roll5":        "home_ra_per_game_5",
    "ra_roll10":       "home_ra_per_game_10",
    "ra_roll30":       "home_ra_per_game_30",
    "bul_era_roll10":  "home_bullpen_era_10",
    "bul_fip_roll10":  "home_bullpen_fip_10",
}
_DEF_RENAME_AWAY = {
    "ra_roll5":        "away_ra_per_game_5",
    "ra_roll10":       "away_ra_per_game_10",
    "ra_roll30":       "away_ra_per_game_30",
    "bul_era_roll10":  "away_bullpen_era_10",
    "bul_fip_roll10":  "away_bullpen_fip_10",
}
_PIT_RENAME = {
    "era_l3":            "_sp_era_l3",
    "era_l5":            "_sp_era_l5",
    "era_szn":           "_sp_era_szn",
    "fip_l5":            "_sp_fip_l5",
    "xfip":              "_sp_xfip_l5",
    "siera":             "_sp_siera_szn",
    "k9_l5":             "_sp_k9_l5",
    "bb9_l5":            "_sp_bb9_l5",
    "hr9_l5":            "_sp_hr9_l5",
    "avg_velo_l5":       "_sp_avg_velo_l5",
    "days_rest":         "_sp_days_rest",
    "career_vs_opp_era": "_sp_career_vs_opp_era",
    "career_vs_opp_pa":  "_sp_career_vs_opp_pa",
    "throws":            "_sp_throws",
}


def _join_side(games: pd.DataFrame, team_col: str,
               df: pd.DataFrame, rename: dict, bp_load: pd.DataFrame,
               bp_col: str) -> pd.DataFrame:
    """Join a team-level block to games via team_col, returning wide game-level df."""
    # Offense: filter by is_home flag (already in off_df via is_home column)
    # Defense/bullpen: join by (game_id, team_id)
    merged = games[["game_id", team_col]].merge(
        df, left_on=["game_id", team_col], right_on=["game_id", "team_id"], how="left"
    ).drop(columns=["team_id", team_col], errors="ignore")
    merged = merged.rename(columns=rename)

    bp = games[["game_id", team_col]].merge(
        bp_load, left_on=["game_id", team_col], right_on=["game_id", "team_id"], how="left"
    ).drop(columns=["team_id", team_col], errors="ignore")
    bp = bp.rename(columns={"bullpen_ip_3days": bp_col})

    return merged.merge(bp, on="game_id", how="left")


def _join_pitcher_side(games: pd.DataFrame, starter_col: str,
                       pit_df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Join pitcher features for home or away starter."""
    gp = games[["game_id", starter_col]].merge(
        pit_df, left_on=["game_id", starter_col], right_on=["game_id", "pitcher_id"], how="left"
    ).drop(columns=["pitcher_id", starter_col], errors="ignore")

    # Apply generic rename then prepend side prefix
    gp = gp.rename(columns=_PIT_RENAME)
    rename2 = {c: f"{side}{c}" for c in gp.columns if c.startswith("_sp_")}
    gp = gp.rename(columns=rename2)

    # Home/road split ERA: home_sp_home_era_szn vs away_sp_away_era_szn
    if side == "home" and "era_home_szn" in pit_df.columns:
        src = games[["game_id", starter_col]].merge(
            pit_df[["game_id", "pitcher_id", "era_home_szn"]],
            left_on=["game_id", starter_col], right_on=["game_id", "pitcher_id"], how="left"
        )
        gp = gp.merge(src[["game_id", "era_home_szn"]].rename(
            columns={"era_home_szn": "home_sp_home_era_szn"}
        ), on="game_id", how="left")
    elif side == "away" and "era_away_szn" in pit_df.columns:
        src = games[["game_id", starter_col]].merge(
            pit_df[["game_id", "pitcher_id", "era_away_szn"]],
            left_on=["game_id", starter_col], right_on=["game_id", "pitcher_id"], how="left"
        )
        gp = gp.merge(src[["game_id", "era_away_szn"]].rename(
            columns={"era_away_szn": "away_sp_away_era_szn"}
        ), on="game_id", how="left")

    return gp


def assemble_features(
    games_df: pd.DataFrame,
    off_df: pd.DataFrame,
    def_df: pd.DataFrame,
    bp_load_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    ump_feat_df: pd.DataFrame,
    wx_park_df: pd.DataFrame,
    market_df: pd.DataFrame,
    sched_df: pd.DataFrame,
    version: str,
) -> pd.DataFrame:
    """Joins all feature blocks on game_id to produce one row per game."""
    base = games_df[["game_id"]].copy()

    # ── Offense: filter by is_home, then join to games via team_id ─────────
    home_off = games_df[["game_id", "home_team_id"]].merge(
        off_df[off_df["is_home"] == 1],
        left_on=["game_id", "home_team_id"], right_on=["game_id", "team_id"], how="left",
    ).drop(columns=["team_id", "home_team_id", "is_home"], errors="ignore"
    ).rename(columns=_OFF_RENAME_HOME)

    away_off = games_df[["game_id", "away_team_id"]].merge(
        off_df[off_df["is_home"] == 0],
        left_on=["game_id", "away_team_id"], right_on=["game_id", "team_id"], how="left",
    ).drop(columns=["team_id", "away_team_id", "is_home"], errors="ignore"
    ).rename(columns=_OFF_RENAME_AWAY)

    # ── Defense + bullpen ──────────────────────────────────────────────────
    home_def = _join_side(games_df, "home_team_id", def_df, _DEF_RENAME_HOME,
                          bp_load_df, "home_bullpen_ip_3days")
    away_def = _join_side(games_df, "away_team_id", def_df, _DEF_RENAME_AWAY,
                          bp_load_df, "away_bullpen_ip_3days")

    # ── Pitchers ───────────────────────────────────────────────────────────
    home_pit = _join_pitcher_side(games_df, "home_starter_id", pit_df, "home")
    away_pit = _join_pitcher_side(games_df, "away_starter_id", pit_df, "away")

    result = (
        base
        .merge(home_off,     on="game_id", how="left")
        .merge(away_off,     on="game_id", how="left")
        .merge(home_def,     on="game_id", how="left")
        .merge(away_def,     on="game_id", how="left")
        .merge(home_pit,     on="game_id", how="left")
        .merge(away_pit,     on="game_id", how="left")
        .merge(ump_feat_df,  on="game_id", how="left")
        .merge(wx_park_df,   on="game_id", how="left")
        .merge(market_df,    on="game_id", how="left")
        .merge(sched_df,     on="game_id", how="left")
    )

    result["feature_version"] = version
    result["computed_at"] = datetime.utcnow().isoformat()

    # Sparse feature guarantee: 0 not NULL for known-entity no-history cases
    if "ump_games_sampled" in result.columns:
        result["ump_games_sampled"] = result["ump_games_sampled"].fillna(0).astype(int)

    # Rename park factor column to match schema
    result = result.rename(columns={"pf_runs": "park_factor_runs"})

    return result


# =============================================================================
# WRITE
# =============================================================================

def write_features(con: sqlite3.Connection, feat_df: pd.DataFrame, version: str) -> int:
    """Upsert feature rows to game_features. Returns count written."""
    if feat_df.empty:
        return 0

    schema_cols = [
        row[1] for row in con.execute("PRAGMA table_info(game_features)").fetchall()
    ]
    write_cols = [c for c in schema_cols if c in feat_df.columns]
    if "game_id" not in write_cols or "feature_version" not in write_cols:
        raise RuntimeError("feat_df is missing game_id or feature_version column")

    df_out = feat_df[write_cols].copy()

    con.execute("BEGIN")
    try:
        col_names = ", ".join(write_cols)
        placeholders = ", ".join("?" * len(write_cols))
        sql = f"INSERT OR REPLACE INTO game_features ({col_names}) VALUES ({placeholders})"
        rows = [
            tuple(None if (isinstance(v, float) and math.isnan(v)) else v
                  for v in row)
            for row in df_out.itertuples(index=False)
        ]
        con.executemany(sql, rows)
        con.execute("COMMIT")
        log.info("Wrote %d rows to game_features (version=%s)", len(rows), version)
        return len(rows)
    except Exception:
        con.execute("ROLLBACK")
        raise


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute game_features from raw MLB data")
    ap.add_argument("--full",    action="store_true", help="Recompute all games")
    ap.add_argument("--season",  type=int,   help="Process only this season")
    ap.add_argument("--date",    type=str,   help="Process only this date (YYYY-MM-DD)")
    ap.add_argument("--version", type=str,   default=FEATURE_VERSION)
    ap.add_argument("--db",      type=str,   default=DB_PATH)
    args = ap.parse_args()

    if not HAS_PYBASEBALL:
        log.warning("pybaseball not installed — install with: pip install pybaseball")

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")

    version = args.version

    # ── Statuses to include ───────────────────────────────────────────────
    # Final / Completed Early: historical games with box score stats.
    # Pre-Game / Preview / Scheduled: today's and future games — features are
    # computed from historical rolling windows; actual_total is NULL (unplayed).
    # These rows are needed by predict_mlb.py for same-day predictions.
    INCLUDE_STATUSES = (
        "'Final'", "'Completed Early'",
        "'Pre-Game'", "'Preview'", "'Scheduled'",
    )
    status_clause = f"status IN ({', '.join(INCLUDE_STATUSES)})"

    # ── Determine which games to skip (incremental mode) ──────────────────
    # Correct incremental logic: compute the SET DIFFERENCE between candidate
    # game_ids (all processable games in the games table) and already-computed
    # game_ids (rows in game_features). Process only the difference.
    if not args.full:
        candidate_ids = {
            row[0] for row in con.execute(
                f"SELECT game_id FROM games WHERE {status_clause}"
            )
        }
        already_ids = {
            row[0] for row in con.execute(
                "SELECT game_id FROM game_features WHERE feature_version = ?", (version,)
            )
        }
        new_ids = candidate_ids - already_ids
        log.info(
            "Incremental mode: %d candidates | %d already computed | %d new to process",
            len(candidate_ids), len(already_ids), len(new_ids),
        )
    else:
        new_ids     = None   # None = process everything in the WHERE clause
        already_ids = set()

    # ── Build WHERE clause for game loading ───────────────────────────────
    conditions = [status_clause]
    if args.date:
        conditions.append(f"game_date = '{args.date}'")
    elif args.season:
        conditions.append(f"season = {args.season}")
    where = " AND ".join(conditions)

    # ── Load raw data ──────────────────────────────────────────────────────
    log.info("Loading raw data from %s ...", args.db)
    data = load_db_data(con, where_clause=where)
    games = data["games"]

    # Filter to only new game_ids (incremental) or apply --full logic
    if not args.full and new_ids is not None:
        games = games[games["game_id"].isin(new_ids)].copy()

    if games.empty:
        log.info("No new games to process. Done.")
        con.close()
        return

    log.info("Processing %d games ...", len(games))

    # ── Pre-game placeholders ───────────────────────────────────────────────
    # Unplayed games (Pre-Game/Preview/Scheduled) have no box-score rows, so the
    # rolling-feature joins would return NULL for every offense/defense/pitcher
    # feature.  Inject synthetic placeholder rows (NaN stats, real keys) into the
    # bat/pit/bul frames BEFORE the rolling blocks run, so the SAME rolling code
    # produces correct point-in-time pre-game features.  shift(1) guarantees a
    # placeholder's own NaN stats are never read for its own value.
    unplayed = games[~games["status"].isin(["Final", "Completed Early"])].copy()
    if not unplayed.empty:
        bat_ph, pit_ph, bul_ph = build_pregame_placeholders(
            unplayed, data["players"], data["pit"]
        )
        if not bat_ph.empty:
            data["bat"] = pd.concat([data["bat"], bat_ph], ignore_index=True)
        if not pit_ph.empty:
            data["pit"] = pd.concat([data["pit"], pit_ph], ignore_index=True)
        if not bul_ph.empty:
            data["bul"] = pd.concat([data["bul"], bul_ph], ignore_index=True)
        log.info(
            "Injected pre-game placeholders for %d unplayed game(s): "
            "+%d bat, +%d pit, +%d bul rows",
            len(unplayed), len(bat_ph), len(pit_ph), len(bul_ph),
        )

    # ── External pitcher stats (BBRef + Savant — replaces FanGraphs) ─────────
    bat_seasons = sorted(data["bat"]["season"].dropna().astype(int).unique().tolist())
    if bat_seasons:
        # Pull one extra season before the earliest so Y-1 data is available for season min
        ext_seasons = list(range(min(bat_seasons) - 1, max(bat_seasons) + 1))
    else:
        ext_seasons = []

    log.info("Loading BBRef pitcher stats for %d seasons ...", len(ext_seasons))
    bref_pit = load_bref_pitcher_stats(ext_seasons)

    log.info("Loading Savant pitcher xStats for %d seasons (2015+ only) ...", len(ext_seasons))
    savant_xstats = load_savant_pitcher_xstats(ext_seasons)

    fg_pit = combine_pitcher_xstats(bref_pit, savant_xstats)
    log.info(
        "Combined external pitcher stats: %d rows (%d with non-NULL xfip)",
        len(fg_pit),
        int(fg_pit["xfip"].notna().sum()) if not fg_pit.empty else 0,
    )

    log.info("wRC+ (FanGraphs removed) — wrc_plus will be NULL")
    fg_wrc = load_fg_team_wrc(ext_seasons)

    # ── UmpScorecards cache ────────────────────────────────────────────────
    ump_cache = load_ump_scorecards_cache(con)
    log.info("UmpScorecards cache: %d umpires", len(ump_cache))

    # ── Feature blocks ─────────────────────────────────────────────────────
    log.info("Computing offense features ...")
    off_df = compute_team_offense(data["bat"], data["teams"], fg_wrc)

    log.info("Computing defense features ...")
    def_df = compute_team_defense(data["bat"], data["bul"])

    log.info("Computing bullpen load (3-day IP) ...")
    bp_load = compute_bullpen_load(con, status_clause)

    log.info("Computing pitcher features ...")
    pit_df = compute_pitcher_features(data["pit"], fg_pit, data["players"])

    log.info("Computing umpire features ...")
    ump_feat = compute_umpire_features(
        data["ump_game"], data["umpires"], ump_cache, games
    )

    log.info("Computing weather and park features ...")
    wx_park = compute_weather_park(data["weather"], data["venues"], data["pf"], games)

    log.info("Computing market features ...")
    market = compute_market_features(data["odds"])

    log.info("Computing schedule features ...")
    sched = compute_schedule_features(games, data["travel"])

    # ── Assemble and write ─────────────────────────────────────────────────
    log.info("Assembling feature matrix ...")
    feat_df = assemble_features(
        games, off_df, def_df, bp_load, pit_df,
        ump_feat, wx_park, market, sched, version,
    )

    n = write_features(con, feat_df, version)
    log.info("Done. %d game_features rows written.", n)
    con.close()


if __name__ == "__main__":
    main()
