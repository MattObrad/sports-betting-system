"""
collect_weather.py

Populates: weather, venues (latitude, longitude, cf_direction_deg, roof_type)

Sources:
  - Open-Meteo API (free, no key): archive for history, forecast for upcoming
  - MLB Stats API /venues/{id}: coordinates seeded on first run

Batching strategy:
  Games are grouped by venue; one API call covers 90 days at a time.
  Full 2017-present backfill ≈ 240 Open-Meteo calls instead of 19k+.

Historical precipitation_prob:
  Open-Meteo archive has no forecast probability. We use a binary proxy:
  precip_prob = 1.0 if actual precip > 0.1 mm, else 0.0.

wind_to_cf:
  Component of wind along home-plate → CF axis (mph).
  Positive = blowing out toward wall (higher totals).
  Negative = blowing in from CF (lower totals).
  Requires venues.cf_direction_deg — hardcoded below, four parks flagged.

Usage:
  python data_collectors/collect_weather.py                     # today + 7 days
  python data_collectors/collect_weather.py --date 2024-09-15   # specific date
  python data_collectors/collect_weather.py --backfill          # 2017-present
  python data_collectors/collect_weather.py --backfill --start 2023-01-01
  python data_collectors/collect_weather.py --season 2022
"""

import argparse
import datetime
import math
import os
import signal
import sqlite3
import sys
import threading
import time
from collections import defaultdict

import requests

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH      = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mlb_data.db"))
MLB_API      = "https://statsapi.mlb.com/api/v1"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CHUNK_DAYS   = 90   # days per Open-Meteo API call for historical data

# ── CF Directions ─────────────────────────────────────────────────────────────
# Compass bearing FROM home plate TO center field.
# Positive wind_to_cf = wind blowing out toward CF wall  → more runs
# Negative wind_to_cf = wind blowing in from CF          → fewer runs
#
# ⚠ FOUR PARKS NEED VERIFICATION — flagged below.
# To correct: change the integer next to the park name and re-run.
# Lookup venue name: SELECT DISTINCT venue_name FROM venues ORDER BY venue_name;

CF_DIRECTIONS: dict[str, int | None] = {
    # ── American League East ─────────────────────────────────────────────────
    "Yankee Stadium":                     5,
    "Fenway Park":                       35,   # ⚠ VERIFY: unusual NE-SW park footprint;
                                               #   CF runs roughly NNE from home plate.
                                               #   Risk: sign flip vs expected ~215° inverse.
    "Oriole Park at Camden Yards":       345,
    "Camden Yards":                      345,
    "Tropicana Field":                  None,  # fixed dome — weather irrelevant
    "Rogers Centre":                      45,  # retractable roof

    # ── American League Central ──────────────────────────────────────────────
    "Guaranteed Rate Field":             315,
    "Progressive Field":                  50,
    "Comerica Park":                      45,
    "Kauffman Stadium":                   55,
    "Target Field":                      330,

    # ── American League West ─────────────────────────────────────────────────
    "Globe Life Field":                    5,  # retractable (opened 2020)
    "Globe Life Park in Arlington":        5,  # pre-2020 name — same bearing
    "Minute Maid Park":                   40,  # retractable
    "Angel Stadium":                     305,
    "Angel Stadium of Anaheim":          305,
    "T-Mobile Park":                      10,  # retractable (renamed 2019)
    "Safeco Field":                       10,  # pre-2019 name
    "Oakland Coliseum":                  310,
    "Oakland-Alameda County Coliseum":   310,
    "RingCentral Coliseum":             310,
    "Sutter Health Park":                315,  # Sacramento — A's 2025+

    # ── National League East ─────────────────────────────────────────────────
    "Citizens Bank Park":                  5,
    "Citi Field":                         15,
    "Nationals Park":                     10,
    "Truist Park":                       315,
    "SunTrust Park":                     315,  # pre-2020 name
    "loanDepot park":                    230,  # ⚠ VERIFY: retractable; unusual SW orientation
                                               #   facing Biscayne Bay — may need adjustment.
    "Marlins Park":                      230,  # pre-2021 name

    # ── National League Central ──────────────────────────────────────────────
    "Wrigley Field":                      80,  # ⚠ VERIFY: CF is roughly E; the famous
                                               #   in/out Lake Michigan wind — getting this
                                               #   wrong flips the sign of wind_to_cf entirely.
    "Great American Ball Park":          245,  # ⚠ VERIFY: very unusual orientation; home plate
                                               #   faces NE so CF is roughly SW (~245°).
                                               #   Opposite of most parks — confirm before use.
    "American Family Field":              50,  # retractable (renamed 2021)
    "Miller Park":                        50,  # pre-2021 name
    "Busch Stadium":                     355,
    "PNC Park":                           30,

    # ── National League West ─────────────────────────────────────────────────
    "Dodger Stadium":                    310,
    "Oracle Park":                       340,  # ⚠ VERIFY: bay-facing, McCovey Cove behind CF;
                                               #   wind patterns are complex and shift seasonally.
    "AT&T Park":                         340,  # pre-2019 name
    "Chase Field":                        60,  # retractable
    "Coors Field":                        50,
    "Petco Park":                        320,
}

# Weather is never relevant — skip fetch, write roof_closed=1
FIXED_DOMES: set[str] = {"Tropicana Field"}

# Retractable roofs — fetch weather but default roof_closed=0 (assume open)
RETRACTABLE_ROOFS: set[str] = {
    "Rogers Centre", "Globe Life Field", "Minute Maid Park",
    "T-Mobile Park", "Safeco Field", "American Family Field",
    "Miller Park", "Chase Field", "loanDepot park", "Marlins Park",
}

# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown = threading.Event()


def _handle_sigint(sig, frame):
    print("\n[interrupt] Ctrl-C — finishing current API call and saving progress...")
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_sigint)

# ── DB helpers ────────────────────────────────────────────────────────────────


def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


# ── HTTP helpers ──────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers["User-Agent"] = "mlb-totals-model/1.0 (research)"


def http_get(url: str, params: dict = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=30)
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
            return f"  0/{self.total} {self.label} (0.0%) — ETA: calculating..."
        secs_left = (self.total - self.done) * elapsed / self.done
        eta_str = str(datetime.timedelta(seconds=int(secs_left)))
        return f"  {self.done}/{self.total} {self.label} ({pct:.1f}%) — ETA: {eta_str}"


# ── Date helpers ──────────────────────────────────────────────────────────────


def date_chunks(start: datetime.date, end: datetime.date, days: int = CHUNK_DAYS):
    chunks, cur = [], start
    while cur <= end:
        chunk_end = min(cur + datetime.timedelta(days=days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + datetime.timedelta(days=1)
    return chunks


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


# ── Weather math ──────────────────────────────────────────────────────────────


def calc_wind_to_cf(wind_from_deg, wind_speed_mph, cf_dir_deg) -> float | None:
    """
    Wind component (mph) along the home-plate → CF axis.
    Positive = blowing out (increases run scoring).
    Negative = blowing in (decreases run scoring).

    wind_from_deg: meteorological convention — direction wind is coming FROM.
    cf_dir_deg:    compass bearing from home plate to center field.
    """
    if any(v is None for v in (wind_from_deg, wind_speed_mph, cf_dir_deg)):
        return None
    wind_going_deg = (wind_from_deg + 180) % 360
    angle_diff = math.radians(wind_going_deg - cf_dir_deg)
    return round(wind_speed_mph * math.cos(angle_diff), 2)


def wmo_to_conditions(code) -> str:
    """Map WMO weather interpretation code to a human-readable string."""
    if code is None:
        return "Unknown"
    c = int(code)
    if c == 0:        return "Clear"
    if c <= 2:        return "Partly Cloudy"
    if c == 3:        return "Overcast"
    if c <= 49:       return "Fog"
    if c <= 67:       return "Rain"
    if c <= 77:       return "Snow"
    if c <= 82:       return "Showers"
    if c <= 99:       return "Thunderstorm"
    return "Unknown"


def safe_idx(lst: list, i: int):
    """Return lst[i] or None if out of range."""
    return lst[i] if lst and i < len(lst) else None


def game_to_utc_hour(g) -> str | None:
    """
    Convert game start time to Open-Meteo hourly key 'YYYY-MM-DDTHH:00'.
    Falls back to day/night heuristic when game_time_utc is missing.
    """
    gtu = g["game_time_utc"] if isinstance(g, dict) else g[2]
    if gtu:
        try:
            dt = datetime.datetime.fromisoformat(str(gtu).replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%dT%H:00")
        except (ValueError, AttributeError):
            pass
    # Fallback heuristics (UTC)
    date_str = g["game_date"] if isinstance(g, dict) else g[1]
    dn = (g["day_night"] if isinstance(g, dict) else g[3]) or "N"
    hour = 17 if str(dn).upper().startswith("D") else 23
    return f"{date_str}T{hour:02d}:00"


# ── Open-Meteo API ────────────────────────────────────────────────────────────


def fetch_open_meteo(lat: float, lon: float, start_date: str, end_date: str,
                     is_forecast: bool) -> dict[str, dict]:
    """
    Fetch hourly weather from Open-Meteo for a lat/lon + date range.
    Returns a dict: {'YYYY-MM-DDTHH:00' → {field: value, ...}}
    """
    base = {
        "latitude":        lat,
        "longitude":       lon,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone":        "GMT",
    }

    if is_forecast:
        today = datetime.date.today()
        end   = datetime.date.fromisoformat(end_date)
        days  = max(1, min((end - today).days + 2, 16))
        params = {
            **base,
            "forecast_days": days,
            "hourly": ",".join([
                "temperature_2m", "precipitation_probability",
                "wind_speed_10m", "wind_direction_10m",
                "relative_humidity_2m", "weathercode", "precipitation",
            ]),
        }
        url = FORECAST_URL
    else:
        params = {
            **base,
            "start_date": start_date,
            "end_date":   end_date,
            "hourly": ",".join([
                "temperature_2m", "wind_speed_10m", "wind_direction_10m",
                "relative_humidity_2m", "weathercode", "precipitation",
            ]),
        }
        url = ARCHIVE_URL

    data = http_get(url, params)
    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])

    lookup = {}
    for i, t in enumerate(times):
        lookup[t] = {
            "temperature_f":       safe_idx(hourly.get("temperature_2m", []), i),
            "wind_speed_mph":      safe_idx(hourly.get("wind_speed_10m", []), i),
            "wind_dir_deg":        safe_idx(hourly.get("wind_direction_10m", []), i),
            "humidity_pct":        safe_idx(hourly.get("relative_humidity_2m", []), i),
            "weathercode":         safe_idx(hourly.get("weathercode", []), i),
            "precipitation_mm":    safe_idx(hourly.get("precipitation", []), i),
            "precip_prob_raw":     safe_idx(hourly.get("precipitation_probability", []), i),
        }
    return lookup


# ── Row builder ───────────────────────────────────────────────────────────────


def build_weather_row(game_id: int, hw: dict, cf_dir, is_forecast: bool,
                      is_retractable: bool) -> dict:
    temp       = hw.get("temperature_f")
    wind_speed = hw.get("wind_speed_mph")
    wind_dir   = hw.get("wind_dir_deg")
    humidity   = hw.get("humidity_pct")
    wcode      = hw.get("weathercode")
    precip_mm  = hw.get("precipitation_mm")
    prob_raw   = hw.get("precip_prob_raw")

    # Precipitation probability
    if is_forecast and prob_raw is not None:
        precip_prob = round(prob_raw / 100.0, 2)
    elif precip_mm is not None:
        precip_prob = 1.0 if precip_mm > 0.1 else 0.0  # binary proxy for historical
    else:
        precip_prob = None

    return {
        "game_id":           game_id,
        "temperature_f":     round(temp, 1)       if temp       is not None else None,
        "wind_speed_mph":    round(wind_speed, 1)  if wind_speed  is not None else None,
        "wind_dir_deg":      round(wind_dir, 0)    if wind_dir    is not None else None,
        "wind_to_cf":        calc_wind_to_cf(wind_dir, wind_speed, cf_dir),
        "precipitation_prob": precip_prob,
        "humidity_pct":      round(humidity, 0)    if humidity    is not None else None,
        "conditions":        wmo_to_conditions(wcode),
        "roof_closed":       0,  # fixed domes handled separately; retractable defaults open
        "fetched_at":        now_iso(),
    }


# ── DB writes ─────────────────────────────────────────────────────────────────


def upsert_weather(con: sqlite3.Connection, row: dict):
    con.execute("""
        INSERT INTO weather(
            game_id, temperature_f, wind_speed_mph, wind_dir_deg, wind_to_cf,
            precipitation_prob, humidity_pct, conditions, roof_closed, fetched_at)
        VALUES(
            :game_id,:temperature_f,:wind_speed_mph,:wind_dir_deg,:wind_to_cf,
            :precipitation_prob,:humidity_pct,:conditions,:roof_closed,:fetched_at)
        ON CONFLICT(game_id) DO UPDATE SET
            temperature_f      = excluded.temperature_f,
            wind_speed_mph     = excluded.wind_speed_mph,
            wind_dir_deg       = excluded.wind_dir_deg,
            wind_to_cf         = excluded.wind_to_cf,
            precipitation_prob = excluded.precipitation_prob,
            humidity_pct       = excluded.humidity_pct,
            conditions         = excluded.conditions,
            roof_closed        = excluded.roof_closed,
            fetched_at         = excluded.fetched_at
    """, row)


# ── Venue seeding ─────────────────────────────────────────────────────────────


def seed_venues(con: sqlite3.Connection, venue_ids: list[int]):
    """
    For each venue_id in games: ensure venues row exists with lat/lon.
    Fetches from MLB Stats API on first encounter.
    Also seeds cf_direction_deg and roof_type from our hardcoded dicts.
    """
    existing = {r[0] for r in con.execute("SELECT venue_id FROM venues").fetchall()}
    need_coord = []

    for vid in venue_ids:
        if vid is None:
            continue
        row = con.execute(
            "SELECT latitude, longitude FROM venues WHERE venue_id=?", (vid,)
        ).fetchone()
        if row is None or row["latitude"] is None:
            need_coord.append(vid)

    if not need_coord:
        return

    print(f"\n  Seeding coordinates for {len(need_coord)} venue(s)...")
    for vid in need_coord:
        try:
            data = http_get(f"{MLB_API}/venues/{vid}", {"hydrate": "location"})
            venues = data.get("venues", [])
            if not venues:
                continue
            v = venues[0]
            name = v.get("name", "")
            loc  = (v.get("location") or {})
            coords = loc.get("defaultCoordinates") or {}
            lat  = coords.get("latitude")
            lon  = coords.get("longitude")
            if not lat or not lon:
                print(f"    [warning] venue {vid} ({name}): no coordinates in API")
                continue

            cf_dir    = CF_DIRECTIONS.get(name)  # None if unknown
            roof_type = ("dome"        if name in FIXED_DOMES else
                         "retractable" if name in RETRACTABLE_ROOFS else
                         "open")

            con.execute("""
                INSERT INTO venues(venue_id, venue_name, latitude, longitude,
                                   cf_direction_deg, roof_type)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(venue_id) DO UPDATE SET
                    venue_name      = excluded.venue_name,
                    latitude        = excluded.latitude,
                    longitude       = excluded.longitude,
                    cf_direction_deg = CASE WHEN excluded.cf_direction_deg IS NOT NULL
                                           THEN excluded.cf_direction_deg
                                           ELSE cf_direction_deg END,
                    roof_type       = CASE WHEN excluded.roof_type IS NOT NULL
                                           THEN excluded.roof_type
                                           ELSE roof_type END
            """, (vid, name, lat, lon, cf_dir, roof_type))
            con.commit()
            cf_note = f"CF={cf_dir}°" if cf_dir is not None else "CF=unknown"
            print(f"    {name} ({vid}): {lat:.4f}, {lon:.4f}  {cf_note}  roof={roof_type}")
        except Exception as exc:
            print(f"    [error] venue {vid}: {exc}")
        time.sleep(0.3)


# ── Main fetch orchestration ──────────────────────────────────────────────────


def run_fetch(start_date: str, end_date: str, refresh_upcoming: bool):
    con = get_db()
    today = datetime.date.today()

    # For daily mode: delete stale forecast rows so we re-fetch fresh predictions
    if refresh_upcoming:
        con.execute("""
            DELETE FROM weather WHERE game_id IN (
                SELECT game_id FROM games WHERE game_date BETWEEN ? AND ?
            )
        """, (today.isoformat(), end_date))
        con.commit()

    # Games needing weather: final games (history) or any non-cancelled upcoming games
    games = con.execute("""
        SELECT g.game_id, g.game_date, g.game_time_utc, g.day_night,
               g.venue_id, g.status,
               v.latitude, v.longitude, v.venue_name,
               v.cf_direction_deg, v.roof_type, v.altitude_ft
        FROM games g
        LEFT JOIN venues v ON g.venue_id = v.venue_id
        WHERE g.game_date BETWEEN ? AND ?
          AND g.status NOT IN ('Postponed', 'Cancelled', 'Suspended')
          AND (g.status = 'Final' OR g.game_date >= ?)
          AND g.game_id NOT IN (SELECT game_id FROM weather)
        ORDER BY g.venue_id, g.game_date
    """, (start_date, end_date, today.isoformat())).fetchall()

    if not games:
        print("  No games need weather data.")
        con.close()
        return

    print(f"  {len(games)} games need weather data")

    # Seed venue coordinates for any venue we haven't seen yet
    venue_ids = list({g["venue_id"] for g in games if g["venue_id"]})
    seed_venues(con, venue_ids)

    # Reload venue info after seeding
    venue_info: dict[int, dict] = {}
    for vid in venue_ids:
        row = con.execute("""
            SELECT latitude, longitude, venue_name, cf_direction_deg, roof_type
            FROM venues WHERE venue_id=?
        """, (vid,)).fetchone()
        if row:
            venue_info[vid] = dict(row)

    # Group games by venue
    by_venue: dict[int, list] = defaultdict(list)
    for g in games:
        by_venue[g["venue_id"]].append(g)

    # Pre-compute all (venue, chunk) work items so we have a total for ETA
    # Fixed domes are handled inline without an API call.
    work_items: list[tuple] = []
    dome_games: list = []

    for vid, vgames in by_venue.items():
        vi   = venue_info.get(vid, {})
        name = vi.get("venue_name", "")

        if name in FIXED_DOMES:
            dome_games.extend(vgames)
            continue

        if not vi.get("latitude") or not vi.get("longitude"):
            print(f"  [skip] venue {vid} ({name}): no coordinates — add lat/lon to venues table")
            continue

        dates      = sorted(datetime.date.fromisoformat(g["game_date"]) for g in vgames)
        min_date   = dates[0]
        max_date   = dates[-1]

        for chunk_s, chunk_e in date_chunks(min_date, max_date):
            chunk_games = [
                g for g in vgames
                if chunk_s <= datetime.date.fromisoformat(g["game_date"]) <= chunk_e
            ]
            if chunk_games:
                work_items.append((vi, vid, chunk_s, chunk_e, chunk_games))

    # Write fixed-dome rows immediately (no API needed)
    for g in dome_games:
        upsert_weather(con, {"game_id": g["game_id"], "roof_closed": 1, "fetched_at": now_iso(),
                              "temperature_f": None, "wind_speed_mph": None, "wind_dir_deg": None,
                              "wind_to_cf": None, "precipitation_prob": None,
                              "humidity_pct": None, "conditions": None})
    if dome_games:
        con.commit()
        print(f"  {len(dome_games)} fixed-dome game(s) written (roof_closed=1, weather NULL)")

    print(f"  {len(work_items)} API call(s) needed (batched by venue × 90-day chunk)")

    if not work_items:
        con.close()
        return

    eta    = ETA(len(work_items), "chunks")
    errors = []

    for vi, vid, chunk_s, chunk_e, chunk_games in work_items:
        if _shutdown.is_set():
            print(f"\n[weather] Stopped. {eta.done}/{eta.total} chunks done, progress saved.")
            break

        name         = vi.get("venue_name", f"venue_{vid}")
        lat          = vi["latitude"]
        lon          = vi["longitude"]
        cf_dir       = vi.get("cf_direction_deg")
        is_retract   = name in RETRACTABLE_ROOFS
        is_forecast  = chunk_s >= today

        try:
            lookup = fetch_open_meteo(lat, lon, chunk_s.isoformat(), chunk_e.isoformat(),
                                      is_forecast)

            written = 0
            for g in chunk_games:
                hour_key = game_to_utc_hour(g)
                hw = lookup.get(hour_key)
                if hw is None:
                    # Hour not in response — shouldn't happen with correct chunking
                    errors.append((g["game_id"], f"hour {hour_key} missing from response"))
                    continue
                row = build_weather_row(g["game_id"], hw, cf_dir, is_forecast, is_retract)
                upsert_weather(con, row)
                written += 1

            con.commit()

        except Exception as exc:
            errors.append((f"{name} {chunk_s}→{chunk_e}", str(exc)))
            con.rollback()

        eta.tick()
        print(f"\r{eta.line()}", end="", flush=True)
        time.sleep(1.0)  # respectful delay; Open-Meteo is free, don't hammer it

    print()

    if errors:
        print(f"  [warnings] {len(errors)} errors:")
        for ident, msg in errors[:10]:
            print(f"    {ident}: {msg}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")

    con.close()
    print("[weather] Fetch complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Collect game-time weather from Open-Meteo into SQLite"
    )
    parser.add_argument("--date", type=str, metavar="YYYY-MM-DD",
                        help="Single date (daily mode override)")
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch historical weather 2017-present")
    parser.add_argument("--season", type=int, metavar="YYYY",
                        help="Fetch one full season")
    parser.add_argument("--start", type=str, metavar="YYYY-MM-DD",
                        help="Backfill start date (default: 2017-03-20)")
    parser.add_argument("--end", type=str, metavar="YYYY-MM-DD",
                        help="Backfill end date (default: yesterday)")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        sys.exit(f"[error] Database not found at {DB_PATH}. Run setup_db.py first.")

    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    print("=" * 60)
    print("MLB Weather Collector")
    print(f"  DB: {DB_PATH}")

    if args.season:
        start_date       = f"{args.season}-03-20"
        end_date         = f"{args.season}-11-05"
        refresh_upcoming = False
        label = f"season {args.season}"
    elif args.backfill:
        start_date       = args.start or "2017-03-20"
        end_date         = args.end   or yesterday.isoformat()
        refresh_upcoming = False
        label = f"backfill {start_date} → {end_date}"
    elif args.date:
        start_date       = args.date
        end_date         = args.date
        refresh_upcoming = True
        label = f"single date {args.date}"
    else:
        # Daily default: today + 7 days
        start_date       = today.isoformat()
        end_date         = (today + datetime.timedelta(days=7)).isoformat()
        refresh_upcoming = True
        label = f"today + 7 days ({start_date} → {end_date})"

    print(f"  Mode: {label}")
    print(f"  Refresh upcoming: {refresh_upcoming}")
    print("=" * 60)

    run_fetch(start_date, end_date, refresh_upcoming)
    print("\nDone.")


if __name__ == "__main__":
    main()
