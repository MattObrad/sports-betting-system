"""
fix_sync_odds.py — patch _team_to_code() in sync_odds.py with the full alias map.

Bugs found by debug_sync_odds.py:
  1. 2-letter Kambi prefixes (NY, LA, SF, SD, TB, KC) all return None because
     the code requires len(prefix)==3.
  2. CHI is ambiguous: Cubs→CHC, White Sox→CWS in SQLite; current code
     passes CHI through unchanged, matching neither.

Fix: expand _FULLNAME_MAP with all ambiguous/2-letter team names.
"""
import re

SYNC_ODDS_PATH = '/home/picks/sync_odds.py'
src = open(SYNC_ODDS_PATH).read()

OLD_MAPS = '''# Teams that appear with no 3-letter prefix at all
_FULLNAME_MAP: dict[str, str] = {
    "Athletics": "OAK",
}'''

NEW_MAPS = '''# Kambi → SQLite team_code mapping.
# Handles: full-name-only teams, 2-letter prefixes, and CHI ambiguity.
_FULLNAME_MAP: dict[str, str] = {
    # Full-name entries (no 3-letter prefix)
    "Athletics":       "OAK",
    # 2-letter prefix teams — NY / LA are ambiguous, need full name
    "NY Yankees":      "NYY",
    "NY Mets":         "NYM",
    "LA Dodgers":      "LAD",
    "LA Angels":       "LAA",
    # CHI is shared by two teams — disambiguate by full name
    "CHI Cubs":        "CHC",
    "CHI White Sox":   "CWS",
    # 2-letter prefix teams that map directly to the same 2-letter SQLite code
    "SF Giants":       "SF",
    "SD Padres":       "SD",
    "TB Rays":         "TB",
    "KC Royals":       "KC",
}'''

if OLD_MAPS not in src:
    print("ERROR: could not find old _FULLNAME_MAP block — check sync_odds.py manually")
    raise SystemExit(1)

new_src = src.replace(OLD_MAPS, NEW_MAPS)

# Also update _team_to_code to handle 2-letter pass-through for any
# remaining 2-letter codes not in FULLNAME_MAP
OLD_FUNC = '''    if len(prefix) == 3 and prefix.isupper():
        return _PREFIX_ALIASES.get(prefix, prefix)
    return None'''

NEW_FUNC = '''    if len(prefix) == 3 and prefix.isupper():
        return _PREFIX_ALIASES.get(prefix, prefix)
    # 2-letter prefix: pass through (e.g. SF→SF, SD→SD); ambiguous 2-letter
    # cases like NY/LA are handled above by the full-name map.
    if len(prefix) == 2 and prefix.isupper():
        return prefix
    return None'''

if OLD_FUNC not in new_src:
    print("WARNING: could not find old _team_to_code tail — skipping 2-letter fallback patch")
else:
    new_src = new_src.replace(OLD_FUNC, NEW_FUNC)

with open(SYNC_ODDS_PATH, 'w') as f:
    f.write(new_src)
print("sync_odds.py patched successfully")

# Verify by running the function on all known problem names
exec(compile(new_src, SYNC_ODDS_PATH, 'exec'))
test_names = [
    ("NY Yankees",   "NYY"),
    ("NY Mets",      "NYM"),
    ("LA Dodgers",   "LAD"),
    ("LA Angels",    "LAA"),
    ("CHI Cubs",     "CHC"),
    ("CHI White Sox","CWS"),
    ("SF Giants",    "SF"),
    ("SD Padres",    "SD"),
    ("TB Rays",      "TB"),
    ("KC Royals",    "KC"),
    ("Athletics",    "OAK"),
    ("ATL Braves",   "ATL"),
    ("MIL Brewers",  "MIL"),
    ("TOR Blue Jays","TOR"),
    ("WAS Nationals","WSH"),
    ("ARI Diamondbacks","AZ"),
]
print("\n=== VERIFICATION ===")
all_ok = True
for name, expected in test_names:
    got = _team_to_code(name)
    ok = "OK" if got == expected else f"FAIL (got {got!r})"
    print(f"  {name:20s} → {got:5s}  {ok}")
    if got != expected:
        all_ok = False
print("\n" + ("All mappings correct." if all_ok else "SOME MAPPINGS WRONG — fix before deploying!"))
