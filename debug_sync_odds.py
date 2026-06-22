"""
debug_sync_odds.py — diagnose why sync_odds fails to match games.

Shows actual Postgres team names for recent dates and tests whether
_team_to_code() can resolve them. Also shows the SQLite game_idx
sample to find format mismatches.
"""
import os, sys, sqlite3
from datetime import date, timedelta

try:
    import psycopg2, psycopg2.extras
except ImportError:
    sys.exit("psycopg2 not installed")

PG_HOST = "localhost"
PG_PORT = 5432
PG_DB   = "picksdb"
PG_USER = "picksuser"
PG_PASS = os.environ.get("PICKSDB_PASSWORD", "password")
SQLITE_DB = "/home/picks/mlb_data.db"

_PREFIX_ALIASES = {
    "WAS": "WSH", "ARI": "AZ", "CHW": "CWS",
    "SFG": "SF", "SDP": "SD", "TBR": "TB", "KAN": "KC",
}
_FULLNAME_MAP = {"Athletics": "OAK"}

def _team_to_code(vps_name):
    if not vps_name:
        return None
    name = vps_name.strip()
    if name in _FULLNAME_MAP:
        return _FULLNAME_MAP[name]
    parts = name.split()
    prefix = parts[0] if parts else ""
    if len(prefix) == 3 and prefix.isupper():
        return _PREFIX_ALIASES.get(prefix, prefix)
    return None

pg = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                       user=PG_USER, password=PG_PASS, connect_timeout=15)
pg.autocommit = True

# ── Show distinct team names in Postgres games table (last 7 days) ──────────
print("=== POSTGRES: distinct home/away team names last 7 days ===")
with pg.cursor() as cur:
    cur.execute("""
        SELECT DISTINCT home_team, away_team,
               game_time::date as game_date,
               league
        FROM games
        WHERE league = 'MLB'
          AND game_time::date >= CURRENT_DATE - 7
        ORDER BY game_date, home_team
        LIMIT 60
    """)
    rows = cur.fetchall()

bad_teams = set()
print(f"{'Date':12} {'Home':20} {'HomeCode':10} {'Away':20} {'AwayCode':10}")
print("-" * 75)
for home, away, gdate, league in rows:
    hc = _team_to_code(home)
    ac = _team_to_code(away)
    if hc is None:
        bad_teams.add(home)
    if ac is None:
        bad_teams.add(away)
    print(f"{str(gdate):12} {str(home):20} {str(hc):10} {str(away):20} {str(ac):10}")

if bad_teams:
    print(f"\n!! UNPARSEABLE TEAM NAMES: {sorted(bad_teams)}")
else:
    print("\nAll team names parsed OK")

# ── Show distinct Postgres game_time values for broken dates ─────────────────
print("\n=== POSTGRES: game_time values for Jun 19-22 ===")
with pg.cursor() as cur:
    cur.execute("""
        SELECT game_time, game_time::date as gdate, home_team, away_team
        FROM games
        WHERE league = 'MLB'
          AND game_time::date BETWEEN '2026-06-19' AND '2026-06-22'
        ORDER BY game_time
        LIMIT 30
    """)
    rows = cur.fetchall()
for gt, gd, ht, at in rows:
    hc = _team_to_code(ht)
    ac = _team_to_code(at)
    print(f"  game_time={gt!r}  date={gd}  {ht}({hc}) vs {at}({ac})")

# ── Show SQLite game_idx for same dates ──────────────────────────────────────
print("\n=== SQLITE: games table for Jun 19-22 ===")
sq = sqlite3.connect(SQLITE_DB)
sq_rows = sq.execute("""
    SELECT g.game_date, g.game_time_utc, ht.team_code, at.team_code, g.game_id
    FROM games g
    JOIN teams ht ON ht.team_id = g.home_team_id
    JOIN teams at ON at.team_id = g.away_team_id
    WHERE g.game_date BETWEEN '2026-06-19' AND '2026-06-22'
    ORDER BY g.game_date, g.game_time_utc
    LIMIT 30
""").fetchall()
for gdate, gtime, hc, ac, gid in sq_rows:
    print(f"  {gdate}  {gtime}  {hc}@{ac}  game_id={gid}")

# ── Cross-check: for each Postgres game on Jun 19, try to find in SQLite ────
print("\n=== CROSS-CHECK: Postgres Jun 19 games vs SQLite lookup ===")
sq_idx = {}
for gdate, _, hc, ac, gid in sq_rows:
    sq_idx[(gdate, hc, ac)] = gid

with pg.cursor() as cur:
    cur.execute("""
        SELECT DISTINCT home_team, away_team, game_time
        FROM games
        WHERE league = 'MLB'
          AND game_time::date = '2026-06-19'
    """)
    jun19 = cur.fetchall()

for ht, at, gt in jun19:
    hc = _team_to_code(ht)
    ac = _team_to_code(at)
    found = None
    if hc and ac:
        for offset in (0, -1, 1):
            d = (gt + timedelta(days=offset)).strftime("%Y-%m-%d")
            found = sq_idx.get((d, hc, ac))
            if found:
                break
    match_str = f"game_id={found}" if found else "NO MATCH"
    print(f"  {ht}({hc}) vs {at}({ac})  game_time={gt!r}  → {match_str}")

pg.close()
sq.close()
