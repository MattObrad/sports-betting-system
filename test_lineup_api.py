"""Test MLB Stats API hydration options to find the 406 source."""
import requests, sys

session = requests.Session()
session.headers["User-Agent"] = "mlb-totals-model/1.0 (research)"

tests = [
    ("WITHOUT lineups", "probablePitcher,team"),
    ("WITH    lineups", "lineups,probablePitcher,team"),
    ("NO     hydrate",  None),
]

for label, hydrate in tests:
    params = {"sportId": 1, "date": "2026-06-22", "gameType": "R"}
    if hydrate:
        params["hydrate"] = hydrate
    try:
        r = session.get("https://statsapi.mlb.com/api/v1/schedule",
                        params=params, timeout=10)
        extra = ""
        if r.status_code == 200:
            data = r.json()
            games = [g for block in data.get("dates", [])
                       for g in block.get("games", [])]
            extra = f" — {len(games)} games"
            if games:
                teams = games[0].get("teams", {})
                starter_key = list((teams.get("home", {}).get("probablePitcher") or {}).keys())[:3]
                extra += f", pitcher keys={starter_key}"
        print(f"{label}: HTTP {r.status_code}{extra}")
    except Exception as e:
        print(f"{label}: ERROR — {e}")
