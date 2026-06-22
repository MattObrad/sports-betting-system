"""
scrape_sbr_odds.py — Scrape SportsBookReview totals for historical seasons
and load directly into odds_snapshots via the load_sbr_odds logic.

Based on: https://github.com/ArnavSaraogi/mlb-odds-scraper (MIT)

Usage:
  python data_collectors/scrape_sbr_odds.py --start 2019-05-03 --end 2020-11-01
  python data_collectors/scrape_sbr_odds.py --season 2019
  python data_collectors/scrape_sbr_odds.py --season 2020
  python data_collectors/scrape_sbr_odds.py --dry-run --start 2019-05-03 --end 2019-06-01

Notes:
  - SBR data is reliably available from 2019-05-03 onward.
  - Use --concurrent 3 (default) to stay under SBR's rate limit.
  - The pre-built dataset (load_sbr_odds.py) already covers 2021-2025.
  - This script is intended for 2019-2020 only.
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import sqlite3
from datetime import datetime, timedelta, date as date_type

import aiohttp
import requests

# ── Project imports ────────────────────────────────────────────────────────────
_DIR = os.path.dirname(__file__)
sys.path.insert(0, _DIR)
from load_sbr_odds import (
    build_game_index, build_team_code_set, pick_book,
    match_game, load_dataset, print_report,
    DB_PATH, CACHE_DIR,
)

# ── Constants ──────────────────────────────────────────────────────────────────
SBR_BASE   = "https://www.sportsbookreview.com/betting-odds/mlb-baseball"
MLB_API    = "https://statsapi.mlb.com/api/v1"
NEXT_DATA  = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5; rv:128.0) Gecko/20100101 Firefox/128.0",
]


# =============================================================================
# DATE HELPERS
# =============================================================================

def date_range(start: str, end: str) -> list[str]:
    """All dates (YYYY-MM-DD) from start to end inclusive."""
    d   = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end,   "%Y-%m-%d").date()
    out = []
    while d <= end_d:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def season_bounds(year: int) -> tuple[str, str]:
    """Approximate regular season bounds. SBR only reliable from 2019-05-03."""
    starts = {
        2019: "2019-05-03",  # SBR reliable start
        2020: "2020-07-23",  # COVID shortened season
    }
    ends = {
        2019: "2019-09-29",
        2020: "2020-09-27",
    }
    start = starts.get(year, f"{year}-04-01")
    end   = ends.get(year,   f"{year}-09-30")
    return start, end


# =============================================================================
# MLB SCHEDULE (game type lookup)
# =============================================================================

def get_mlb_schedule(start: str, end: str) -> dict:
    """
    Returns {date_str: {(norm_away, norm_home): game_type}}.
    Used to filter out spring training, playoffs etc.
    """
    def norm(name: str) -> str:
        return (name.lower()
                .replace(".", "").replace("'", "")
                .replace("-", " ").replace("&", "and").strip())

    schedule = {}
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    cur = s
    while cur <= e:
        nxt = min(cur.replace(year=cur.year + 1) - timedelta(days=1), e)
        url = (f"{MLB_API}/schedule?sportId=1"
               f"&startDate={cur.strftime('%Y-%m-%d')}"
               f"&endDate={nxt.strftime('%Y-%m-%d')}")
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            for di in resp.json().get("dates", []):
                date_s = di["date"]
                schedule.setdefault(date_s, {})
                for g in di.get("games", []):
                    away = norm(g["teams"]["away"]["team"]["name"])
                    home = norm(g["teams"]["home"]["team"]["name"])
                    schedule[date_s][(away, home)] = g.get("gameType", "R")
        except Exception as exc:
            print(f"  [schedule] {cur.date()} - {nxt.date()}: {exc}")
        cur = nxt + timedelta(days=1)
    return schedule


# =============================================================================
# ASYNC SCRAPER
# =============================================================================

async def fetch_html(session: aiohttp.ClientSession, url: str,
                     semaphore: asyncio.Semaphore,
                     retries: int = 3) -> str | None:
    for attempt in range(retries):
        async with semaphore:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept-Language": "en-US,en;q=0.9",
            }
            try:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    print(f"  HTTP {resp.status}: {url}")
            except Exception as exc:
                print(f"  fetch error ({attempt+1}/{retries}): {exc}")
        if attempt < retries - 1:
            await asyncio.sleep(2 + random.uniform(0, 2))
    return None


def parse_totals(html: str, date_str: str, schedule_day: dict) -> list[dict]:
    """
    Parse __NEXT_DATA__ JSON from SBR totals page.
    Returns list of game dicts matching the load_sbr_odds JSON schema:
      {gameView: {startDate, awayTeam: {shortName}, homeTeam: {shortName}, gameType},
       odds: {totals: [{sportsbook, openingLine, currentLine}]}}
    """
    m = NEXT_DATA.search(html)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
        game_rows = (data.get("props", {})
                        .get("pageProps", {})
                        .get("oddsTables", [{}])[0]
                        .get("oddsTableModel", {})
                        .get("gameRows", []))
    except (json.JSONDecodeError, KeyError, IndexError):
        return []

    games = []
    for gr in game_rows:
        try:
            gv  = gr.get("gameView", {})
            away_name = (gv.get("awayTeam") or {}).get("fullName", "")
            home_name = (gv.get("homeTeam") or {}).get("fullName", "")
            away_short = (gv.get("awayTeam") or {}).get("shortName", "")
            home_short = (gv.get("homeTeam") or {}).get("shortName", "")

            # Determine game type
            def _norm(n):
                return (n.lower().replace(".", "").replace("'", "")
                         .replace("-", " ").replace("&", "and").strip())
            gtype = schedule_day.get((_norm(away_name), _norm(home_name)), "R")

            odds_views = []
            for ov in (gr.get("oddsViews") or []):
                if ov is None:
                    continue
                opening = ov.get("openingLine") or {}
                closing  = ov.get("currentLine") or {}
                odds_views.append({
                    "sportsbook":  ov.get("sportsbook", "unknown"),
                    "openingLine": {
                        "total":     opening.get("total"),
                        "overOdds":  opening.get("overOdds"),
                        "underOdds": opening.get("underOdds"),
                    },
                    "currentLine": {
                        "total":     closing.get("total"),
                        "overOdds":  closing.get("overOdds"),
                        "underOdds": closing.get("underOdds"),
                    },
                })

            games.append({
                "gameView": {
                    "startDate": gv.get("startDate", f"{date_str}T18:00:00+00:00"),
                    "awayTeam":  {"shortName": away_short},
                    "homeTeam":  {"shortName": home_short},
                    "gameType":  gtype,
                },
                "odds": {"totals": odds_views},
            })
        except Exception:
            continue
    return games


async def scrape_dates(dates: list[str], schedule: dict,
                       max_concurrent: int, fast: bool) -> dict:
    """Scrape all dates, return {date_str: [game, ...]} matching JSON schema."""
    semaphore = asyncio.Semaphore(max_concurrent)
    base_delay = 0.5 if fast else 1.5
    results: dict = {}

    async with aiohttp.ClientSession() as session:
        total = len(dates)
        for i, date_str in enumerate(dates):
            url = f"{SBR_BASE}/totals/full-game/?date={date_str}"
            html = await fetch_html(session, url, semaphore)
            if html:
                day_schedule = schedule.get(date_str, {})
                games = parse_totals(html, date_str, day_schedule)
                if games:
                    results[date_str] = games
            pct = (i + 1) * 100 // total
            print(f"\r  [{i+1:4d}/{total}] {date_str}  {pct:3d}%"
                  f"  dates_with_games={len(results)}", end="", flush=True)
            if i < total - 1:
                await asyncio.sleep(base_delay + random.uniform(0, base_delay * 0.5))

    print()
    return results


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scrape SBR MLB totals for 2019-2020 and load into odds_snapshots"
    )
    ap.add_argument("--start",      type=str, help="Start date YYYY-MM-DD")
    ap.add_argument("--end",        type=str, help="End date YYYY-MM-DD")
    ap.add_argument("--season",     type=int, help="Scrape a full season (2019 or 2020)")
    ap.add_argument("--concurrent", type=int, default=3,
                    help="Max concurrent requests (default 3; keep low to avoid blocks)")
    ap.add_argument("--fast",       action="store_true", help="Shorter sleep between requests")
    ap.add_argument("--dry-run",    action="store_true", help="Match but don't write to DB")
    ap.add_argument("--save-json",  type=str, default=None,
                    help="Also save scraped data to this JSON file")
    ap.add_argument("--db",         type=str, default=DB_PATH)
    args = ap.parse_args()

    # Resolve date range
    if args.season:
        start, end = season_bounds(args.season)
    elif args.start and args.end:
        start, end = args.start, args.end
    elif args.start:
        start = args.start
        end   = date_type.today().isoformat()
    else:
        ap.print_help()
        sys.exit(1)

    print(f"Scraping SBR totals: {start} -> {end}")
    print(f"Concurrent: {args.concurrent}  Fast: {args.fast}")

    dates = date_range(start, end)
    print(f"Date range: {len(dates)} days")

    # Fetch MLB schedule for game type classification
    print("Fetching MLB schedule for game type lookup...")
    schedule = get_mlb_schedule(start, end)
    print(f"  Schedule loaded: {len(schedule)} dates")

    # Scrape SBR
    print("Scraping SBR...")
    data = asyncio.run(scrape_dates(dates, schedule, args.concurrent, args.fast))
    print(f"Scraped {sum(len(v) for v in data.values())} games from {len(data)} dates")

    if not data:
        print("No data scraped — check if SBR is blocking or date range has no games")
        sys.exit(1)

    # Optionally save JSON
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"Saved scraped data -> {args.save_json}")

    # Load into DB using shared loader logic
    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")

    existing = con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    print(f"  Existing odds_snapshots rows: {existing}")

    stats = load_dataset(con, data, dry_run=args.dry_run, season=None)
    print_report(stats, args.dry_run)
    con.close()


if __name__ == "__main__":
    main()
