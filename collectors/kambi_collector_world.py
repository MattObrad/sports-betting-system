"""
kambi_collector_world.py — international, minor, and secondary leagues.

Cron suggestion: run once or twice daily (lower priority than core).
"""
from kambi_shared import run_collection

# (display_name, listView path, sport_key)
# display_name stored in games.league — must be <= 10 chars.
SPORT_ENDPOINTS = [
    # American Football
    ("CFL",       "american_football/cfl",                        "american_football"),
    ("UFL",       "american_football/ufl",                        "american_football"),
    # Baseball — international + college
    ("KBO",       "baseball/south_korea/kbo_league",              "baseball"),
    ("NPB",       "baseball/japan/npb",                           "baseball"),
    ("NCAA-B",    "baseball/ncaa",                                "baseball"),
    # Basketball — women's college + international
    ("NCAAW",     "basketball/ncaaw",                             "basketball"),
    ("EuroLeag",  "basketball/euroleague",                        "basketball"),
    # Hockey — minor + international
    ("AHL",       "ice_hockey/ahl",                               "ice_hockey"),
    ("IIHF-WC",   "ice_hockey/world_championship",                "ice_hockey"),
    # Soccer — secondary leagues
    ("La Liga",   "football/spain/la_liga",                       "soccer"),
    ("Ligue 1",   "football/france/ligue_1",                      "soccer"),
    ("Copa Lib",  "football/copa_libertadores",                   "soccer"),
    ("NWSL",      "football/usa/nwsl__w_",                        "soccer"),
    ("Bras-A",    "football/brazil/brasileirao_serie_a",          "soccer"),
    # Tennis — non-slam
    ("Challengr", "tennis/challenger",                            "tennis"),
    ("ITF Men",   "tennis/itf_men",                               "tennis"),
    ("ITF Women", "tennis/itf_women",                             "tennis"),
    # Golf — non-PGA
    ("LIV Golf",  "golf/liv_golf",                                "golf"),
    ("LPGA Tour", "golf/lpga_tour",                               "golf"),
    ("DP World",  "golf/dp_world_tour",                           "golf"),
    ("Korn Ferr", "golf/korn_ferry_tour",                         "golf"),
    # Motorsports
    ("NASCAR",    "motorsports/nascar/cup_series",                "motorsports"),
    ("Formula 1", "formula_1/race",                               "motorsports"),
    # Other
    ("IPL",       "cricket/ipl",                                  "cricket"),
    ("AFL",       "australian_rules/afl",                         "australian_rules"),
    ("NRL",       "rugby_league/nrl",                             "rugby_league"),
    ("Darts-PL",  "darts/premier_league_darts",                   "darts"),
    ("PLL",       "lacrosse/premier_lacrosse_league",             "lacrosse"),
]

if __name__ == "__main__":
    run_collection(SPORT_ENDPOINTS)
