"""
kambi_collector_core.py — major leagues and marquee events.

Cron suggestion: run every few hours for live line monitoring.
"""
from kambi_shared import run_collection

# (display_name, listView path, sport_key)
# display_name stored in games.league — must be <= 10 chars.
SPORT_ENDPOINTS = [
    # American Football
    ("NFL",       "american_football/nfl",                   "american_football"),
    ("NCAAF",     "american_football/ncaaf",                 "american_football"),
    # Baseball
    ("MLB",       "baseball/mlb",                            "baseball"),
    # Basketball
    ("NBA",       "basketball/nba",                          "basketball"),
    ("NCAAB",     "basketball/ncaab",                        "basketball"),
    ("WNBA",      "basketball/wnba",                         "basketball"),
    # Hockey
    ("NHL",       "ice_hockey/nhl",                          "ice_hockey"),
    # Soccer
    ("UCL",       "football/champions_league",               "soccer"),
    ("EPL",       "football/england/premier_league",         "soccer"),
    ("MLS",       "football/usa/mls",                        "soccer"),
    ("WC 2026",   "football/world_cup_2026",                 "soccer"),
    # Tennis — Grand Slam main draws (men + women)
    ("Fr. Open",  "tennis/grand_slam/french_open",           "tennis"),
    ("FO Women",  "tennis/grand_slam/french_open_women",     "tennis"),
    ("Wimbledon", "tennis/grand_slam/wimbledon",             "tennis"),
    ("Wimb. W",   "tennis/grand_slam/wimbledon_women",       "tennis"),
    ("US Open",   "tennis/grand_slam/us_open",               "tennis"),
    ("US Open W", "tennis/grand_slam/us_open_women",         "tennis"),
    # Golf
    ("PGA Tour",  "golf/pga_tour",                           "golf"),
    # MMA
    ("UFC",       "ufc_mma/ufc",                             "mma"),
    # Boxing
    ("Boxing",    "boxing/upcoming_fights",                  "boxing"),
]

if __name__ == "__main__":
    run_collection(SPORT_ENDPOINTS)
