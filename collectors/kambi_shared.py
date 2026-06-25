"""
kambi_shared.py — shared constants, helpers, and collection logic.
Import run_collection() from here; define SPORT_ENDPOINTS in each collector.
"""
import re
import time
import requests
import psycopg2

BASE_URL = "https://eu-offering-api.kambicdn.com/offering/v2018/potawuswirl"

DB_CONFIG = {
    "host":     "localhost",
    "dbname":   "picksdb",
    "user":     "picksuser",
    "password": "password",
    "port":     5432,
}

PARAMS = {
    "lang":      "en_US",
    "market":    "US",
    "client_id": 2,
    "channel_id": 1,
    "ncid":      1,
    "useCombined": "true",
}

SLEEP_SECS = 3

# Whitelist of listView criterion labels → normalized market_type.
# Anything not in this map is skipped (alternates, pitcher lines, etc.)
PRIMARY_MARKET_MAP = {
    # Basketball / Football (shared labels)
    "Point Spread":                         "Point Spread",
    "Moneyline":                            "Moneyline",
    "Total Points - Including Overtime":    "Total Points",
    "Total Points":                         "Total Points",
    # Baseball
    "Run Line":                             "Run Line",
    "Total Runs":                           "Total Runs",
    "Total Runs - Including Extra Innings": "Total Runs",
    # Hockey
    "Puck Line":                            "Puck Line",
    "Total Goals":                          "Total Goals",
    "Total Goals - Including Overtime":     "Total Goals",
    # Soccer
    "Match":                                "Match Result",
    "Asian Handicap":                       "Asian Handicap",
    # Total Goals shared with hockey above
}

# Maps patterns in criterion labels to clean market_type names for props.
# More-specific patterns must come before broader ones (e.g. PRA before Points).
PROP_PATTERNS = [
    # Basketball
    (re.compile(r"Points, Rebounds & Assists", re.I), "Player PRA"),
    (re.compile(r"double-double",              re.I), "Player Double-Double"),
    (re.compile(r"triple-double",              re.I), "Player Triple-Double"),
    (re.compile(r"Points Scored|Points By The Player", re.I), "Player Points"),
    (re.compile(r"Rebounds By The Player",     re.I), "Player Rebounds"),
    (re.compile(r"Assists By The Player",      re.I), "Player Assists"),
    (re.compile(r"Three-Point Field Goals",    re.I), "Player Threes"),
    # Baseball — player props (Alt lines before main: "2+ Strikeouts thrown" also
    # matches the main pattern, so Alt must win first)
    (re.compile(r"\d\+\s*Strikeouts thrown",           re.I), "Player Strikeouts Alt"),
    (re.compile(r"Strikeouts thrown by the Player",     re.I), "Player Strikeouts"),
    (re.compile(r"Home Run",                           re.I), "Player Home Runs"),
    (re.compile(r"Total Bases",                        re.I), "Player Total Bases"),
    (re.compile(r"\bRBI",                              re.I), "Player RBIs"),
    (re.compile(r"Earned Run",                         re.I), "Player Earned Runs"),
    (re.compile(r"\d\+\s*Hits by the Player",          re.I), "Player Hits Alt"),
    (re.compile(r"Total Hits by the Player",           re.I), "Player Hits"),
    (re.compile(r"Stolen Bases? by the Player",        re.I), "Player Stolen Bases"),
    (re.compile(r"Runs Scored by the Player",          re.I), "Player Runs"),
    (re.compile(r"Doubles? by the Player",             re.I), "Player Doubles"),
    (re.compile(r"Outs Recorded by the Player",        re.I), "Player Outs"),
    # Baseball — inning/team markets (First 5 and First 3 before Inning 1)
    (re.compile(r"Total Runs - First 5",               re.I), "Team Runs First 5"),
    (re.compile(r"Total Runs - First 3",               re.I), "Team Runs First 3"),
    (re.compile(r"Total Runs - Inning 1",              re.I), "Team Runs Inning 1"),
    (re.compile(r"to Score a Run - Inning 1",          re.I), "Team Score Inning 1"),
    # American Football
    (re.compile(r"Passing Yards",              re.I), "Player Passing Yards"),
    (re.compile(r"Rushing Yards",              re.I), "Player Rushing Yards"),
    (re.compile(r"Receiving Yards",            re.I), "Player Receiving Yards"),
    (re.compile(r"Touchdown",                  re.I), "Player Touchdowns"),
    (re.compile(r"Reception",                  re.I), "Player Receptions"),
    (re.compile(r"Interception",               re.I), "Player Interceptions"),
    (re.compile(r"Passing Attempts",           re.I), "Player Pass Attempts"),
    (re.compile(r"Completions",                re.I), "Player Completions"),
    # Hockey
    (re.compile(r"Goal Scorer",                re.I), "Player Goal Scorer"),
    (re.compile(r"Shots? on (Goal|Net)",       re.I), "Player Shots"),
    (re.compile(r"Hockey Assists?",            re.I), "Player Assists"),
]

DDL = """
CREATE TABLE IF NOT EXISTS games (
    event_id  VARCHAR(50) PRIMARY KEY,
    sport     VARCHAR(50),
    league    VARCHAR(10),
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    game_time TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id            SERIAL PRIMARY KEY,
    event_id      VARCHAR(50) REFERENCES games(event_id),
    snapshot_time TIMESTAMPTZ DEFAULT NOW(),
    market_type   VARCHAR(50),
    outcome       VARCHAR(50),
    line          NUMERIC(6,2),
    odds          INTEGER
);

CREATE TABLE IF NOT EXISTS props_snapshots (
    id            SERIAL PRIMARY KEY,
    event_id      VARCHAR(50) REFERENCES games(event_id),
    snapshot_time TIMESTAMPTZ DEFAULT NOW(),
    player_name   VARCHAR(100),
    market_type   VARCHAR(50),
    line          NUMERIC(6,2),
    over_odds     INTEGER,
    under_odds    INTEGER
);

CREATE TABLE IF NOT EXISTS props_snapshots_v2 (
    id            SERIAL PRIMARY KEY,
    event_id      VARCHAR(50) REFERENCES games(event_id),
    snapshot_time TIMESTAMPTZ DEFAULT NOW(),
    player_name   VARCHAR(100),
    market_type   VARCHAR(50),
    line          NUMERIC(6,2),
    over_odds     INTEGER,
    under_odds    INTEGER,
    side          VARCHAR(10),
    UNIQUE (event_id, player_name, market_type, line, snapshot_time, side)
);
"""


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def db_connect():
    return psycopg2.connect(**DB_CONFIG)


def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def upsert_game(cur, event_id, sport, league, home_team, away_team, game_time):
    cur.execute("""
        INSERT INTO games (event_id, sport, league, home_team, away_team, game_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id) DO UPDATE
            SET sport     = EXCLUDED.sport,
                league    = EXCLUDED.league,
                home_team = EXCLUDED.home_team,
                away_team = EXCLUDED.away_team,
                game_time = EXCLUDED.game_time
    """, (event_id, sport, league, home_team, away_team, game_time))


def insert_snapshot(cur, event_id, market_type, outcome_label, line, odds):
    cur.execute("""
        INSERT INTO odds_snapshots (event_id, market_type, outcome, line, odds)
        VALUES (%s, %s, %s, %s, %s)
    """, (event_id, market_type, outcome_label, line, odds))


def insert_prop_snapshot(cur, event_id, player_name, market_type, line, over_odds, side="", under_odds=None):
    cur.execute("""
        INSERT INTO props_snapshots_v2 (event_id, player_name, market_type, line, over_odds, under_odds, side)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id, player_name, market_type, line, snapshot_time, side) DO NOTHING
    """, (event_id, player_name, market_type, line, over_odds, under_odds, side))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_teams(event_name):
    """Split 'Away Team @ Home Team' into (home, away)."""
    if " @ " in event_name:
        away, home = event_name.split(" @ ", 1)
        return home.strip(), away.strip()
    return event_name, ""


def classify_prop_market(criterion_label):
    """Map a criterion label to a clean market_type string."""
    for pattern, market_type in PROP_PATTERNS:
        if pattern.search(criterion_label):
            return market_type
    return "Player Prop"


def format_odds(odds_american, odds_decimal):
    if odds_american:
        val = int(odds_american)
        american = f"+{val}" if val > 0 else str(val)
        if odds_decimal:
            dec = odds_decimal / 1000
            return f"{american} ({dec:.3f})"
        return american
    if odds_decimal:
        dec = odds_decimal / 1000
        if dec >= 2.0:
            american = f"+{int(round((dec - 1) * 100))}"
        else:
            american = str(int(round(-100 / (dec - 1))))
        return f"{american} ({dec:.3f})"
    return "N/A"


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------

def fetch_json(url, params=None):
    resp = requests.get(url, params=params or PARAMS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_event_markets(event_id):
    url = f"{BASE_URL}/betoffer/event/{event_id}.json"
    return fetch_json(url)


# ---------------------------------------------------------------------------
# DB write: listView event → games + odds_snapshots
# ---------------------------------------------------------------------------

def process_list_view_event(cur, event_wrapper, sport, league):
    ev        = event_wrapper.get("event", {})
    event_id  = str(ev.get("id", ""))
    name      = ev.get("name", "")
    game_time = ev.get("start")  # ISO 8601 — psycopg2 parses it directly

    home_team, away_team = parse_teams(name)
    upsert_game(cur, event_id, sport, league, home_team, away_team, game_time)

    odds_rows = 0
    for offer in event_wrapper.get("betOffers", []):
        raw_label   = offer.get("criterion", {}).get("label", "")
        market_type = PRIMARY_MARKET_MAP.get(raw_label)
        if market_type is None:
            continue

        for outcome in offer.get("outcomes", []):
            odds_am  = outcome.get("oddsAmerican")
            raw_line = outcome.get("line")
            label    = outcome.get("label", "")

            if odds_am is None:
                continue

            odds = int(odds_am)
            line = (raw_line / 1000) if raw_line else None

            insert_snapshot(cur, event_id, market_type, label, line, odds)
            odds_rows += 1

    return odds_rows


# ---------------------------------------------------------------------------
# DB write: betoffer response → props_snapshots
# ---------------------------------------------------------------------------

def process_props_for_event(cur, event_id):
    """Fetch full markets for event_id and write player props.

    Returns (rows_written, seen_offer_types) where seen_offer_types is a set
    of every betOfferType.name encountered (used for end-of-run discovery logging).
    Rows are only written when an outcome has a participant (player name).
    """
    try:
        data = fetch_event_markets(event_id)
    except requests.HTTPError as e:
        print(f"    Error fetching props for event {event_id}: {e}")
        return 0, set()

    rows             = 0
    seen_offer_types = set()

    for offer in data.get("betOffers", []):
        offer_type      = offer.get("betOfferType", {}).get("name", "Unknown")
        criterion_label = offer.get("criterion", {}).get("label", "")
        market_type     = classify_prop_market(criterion_label)

        seen_offer_types.add(offer_type)

        for outcome in offer.get("outcomes", []):
            player_name = outcome.get("participant")
            if not player_name:
                continue

            odds_am  = outcome.get("oddsAmerican")
            raw_line = outcome.get("line")
            side     = outcome.get("label", "")

            if odds_am is None:
                continue

            over_odds = int(odds_am)
            line      = (raw_line / 1000) if raw_line else None

            insert_prop_snapshot(cur, event_id, player_name, market_type, line, over_odds, side)
            rows += 1

    return rows, seen_offer_types


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_separator(char="-", width=72):
    print(char * width)


def print_game_lines(event):
    ev       = event.get("event", {})
    name     = ev.get("name", "Unknown")
    start    = ev.get("start", "")
    event_id = ev.get("id")
    print(f"\n  EVENT: {name}  |  Start: {start}  |  ID: {event_id}")

    bet_offers = event.get("betOffers", [])
    if not bet_offers:
        print("    (no game lines in listView)")
        return

    for offer in bet_offers:
        criterion    = offer.get("criterion", {})
        market_label = criterion.get("label", offer.get("betOfferType", {}).get("name", "Unknown Market"))
        suspended    = offer.get("suspended")
        status_tag   = " [SUSPENDED]" if suspended else ""
        print(f"\n    Market: {market_label}{status_tag}")

        for outcome in offer.get("outcomes", []):
            label       = outcome.get("label", "")
            participant = outcome.get("participant", "")
            status      = outcome.get("status", "")
            odds_am     = outcome.get("oddsAmerican")
            odds_dec    = outcome.get("odds")
            line        = outcome.get("line")

            display_name = participant if participant else label
            line_str     = f"  line: {line / 1000:.1f}" if line is not None else ""
            odds_str     = format_odds(odds_am, odds_dec)
            status_str   = "" if status == "OPEN" else f" [{status}]"

            print(f"      {display_name:<32} {odds_str}{line_str}{status_str}")


def print_full_markets(event_id, event_name):
    print(f"\n  --- Full markets for: {event_name} ---")
    try:
        data = fetch_event_markets(event_id)
    except requests.HTTPError as e:
        print(f"    Error fetching markets: {e}")
        return

    bet_offers = data.get("betOffers", [])
    if not bet_offers:
        print("    (no markets found)")
        return

    for offer in bet_offers:
        criterion    = offer.get("criterion", {})
        market_label = criterion.get("label", offer.get("betOfferType", {}).get("name", "Unknown"))
        suspended    = offer.get("suspended")
        status_tag   = " [SUSPENDED]" if suspended else ""
        print(f"\n    [{market_label}]{status_tag}")

        for outcome in offer.get("outcomes", []):
            label       = outcome.get("label", "")
            participant = outcome.get("participant", "")
            status      = outcome.get("status", "")
            odds_am     = outcome.get("oddsAmerican")
            odds_dec    = outcome.get("odds")
            line        = outcome.get("line")

            display_name = participant if participant else label
            line_str     = f"  line: {line / 1000:.1f}" if line is not None else ""
            odds_str     = format_odds(odds_am, odds_dec)
            status_str   = "" if status == "OPEN" else f" [{status}]"

            print(f"      {display_name:<32} {odds_str}{line_str}{status_str}")


# ---------------------------------------------------------------------------
# Per-sport collection
# ---------------------------------------------------------------------------

def collect_sport(conn, sport_name, endpoint, sport_key):
    url = f"{BASE_URL}/listView/{endpoint}.json"
    try:
        data = fetch_json(url)
    except requests.RequestException as e:
        print(f"  No events found for {sport_name} ({e})")
        return None

    events = data.get("events", [])
    if not events:
        print(f"  No events found for {sport_name}")
        return None

    print_separator("=")
    print(f"  {sport_name} ODDS")
    print_separator("=")
    print(f"  Found {len(events)} event(s)\n")

    # ---- Phase 1: game lines ------------------------------------------------
    total_odds_rows = 0
    event_infos     = []  # (event_id, event_name) for props phase

    with conn.cursor() as cur:
        for i, event_wrapper in enumerate(events):
            print_separator()
            print_game_lines(event_wrapper)

            odds_rows = process_list_view_event(cur, event_wrapper, sport_key, sport_name)
            total_odds_rows += odds_rows

            event_id   = event_wrapper.get("event", {}).get("id")
            event_name = event_wrapper.get("event", {}).get("name", str(event_id))

            if event_id:
                event_infos.append((str(event_id), event_name))
                if i > 0:
                    time.sleep(SLEEP_SECS)
                print_full_markets(event_id, event_name)

    conn.commit()
    print(f"\n  DB: upserted {len(events)} game(s), inserted {total_odds_rows} odds snapshot(s) for {sport_name}")

    # ---- Phase 2: player props ----------------------------------------------
    print(f"\n  Collecting props for {len(event_infos)} {sport_name} event(s)...")
    total_prop_rows = 0
    all_offer_types = set()

    with conn.cursor() as cur:
        for event_id, event_name in event_infos:
            time.sleep(SLEEP_SECS)
            prop_rows, seen_types = process_props_for_event(cur, event_id)
            total_prop_rows += prop_rows
            all_offer_types |= seen_types
            print(f"    {event_name}: {prop_rows} prop row(s)")

    conn.commit()
    print(f"\n  DB: inserted {total_prop_rows} props snapshot(s) for {sport_name}")
    print()

    return {
        "games":       len(events),
        "odds":        total_odds_rows,
        "props":       total_prop_rows,
        "offer_types": all_offer_types,
    }


# ---------------------------------------------------------------------------
# Entry point — called by each collector with its own endpoint list
# ---------------------------------------------------------------------------

def run_collection(sport_endpoints):
    conn = db_connect()
    try:
        ensure_tables(conn)

        totals        = {"games": 0, "odds": 0, "props": 0}
        no_events     = []
        sport_results = {}   # sport_name -> offer_types set

        for sport_name, endpoint, sport_key in sport_endpoints:
            result = collect_sport(conn, sport_name, endpoint, sport_key)
            if result is None:
                no_events.append(sport_name)
            else:
                totals["games"] += result["games"]
                totals["odds"]  += result["odds"]
                totals["props"] += result["props"]
                sport_results[sport_name] = result["offer_types"]

        # ---- Run summary ----------------------------------------------------
        print_separator("=")
        print("  RUN SUMMARY")
        print_separator("=")
        print(f"  Total games upserted:           {totals['games']}")
        print(f"  Total odds snapshots inserted:  {totals['odds']}")
        print(f"  Total props snapshots inserted: {totals['props']}")

        if no_events:
            print(f"\n  Sports with no events: {', '.join(no_events)}")
        else:
            print("\n  Sports with no events: none")

        if sport_results:
            print("\n  betOfferType.name values seen per sport:")
            for sport_name, offer_types in sport_results.items():
                types_str = ", ".join(sorted(offer_types)) if offer_types else "(none)"
                print(f"    {sport_name:<12} {types_str}")
        print_separator("=")

    finally:
        conn.close()
