"""
replay_tennis_fresh.py -- Fresh tennis replay from raw Elo + Kambi props.

PART A: Historical Elo accuracy (all matches 2020+, no odds needed)
PART B: Elo edge vs Kambi moneylines (May 27+, graded against actual results)

Goes directly to Postgres for odds — bypasses predictions table.
This catches ALL Kambi matches, not just ones where predict_tennis.py ran.
"""
import sqlite3, json, unicodedata, bisect
from collections import defaultdict
from difflib import SequenceMatcher
import psycopg2

tennis = sqlite3.connect('/home/picks/tennis.db')
cfg    = json.load(open('/home/picks/tennis_config.json'))
pg     = psycopg2.connect(host='localhost', dbname='picksdb',
                          user='picksuser', password='password')
cur    = pg.cursor()

betting  = cfg.get('betting', cfg)
min_prob = betting.get('min_model_prob', 0.80)
min_edge = betting.get('min_edge', 0.08)
max_odds = betting.get('max_odds_american', 300)
extreme  = betting.get('extreme_edge', 0.35)
min_career = betting.get('min_career_matches', 10)

print('=== TENNIS FRESH REPLAY ===')
print(f'Config: min_prob={min_prob}, min_edge={min_edge}, max_odds=±{max_odds}, extreme={extreme}')
print()

# ── Name normalisation / matching ─────────────────────────────────────────
def norm(n):
    if not n: return ''
    nfd = unicodedata.normalize('NFD', n.lower())
    return ''.join(c for c in nfd if not unicodedata.combining(c)).strip()

def names_match(a, b, thresh=0.82):
    na, nb = norm(a), norm(b)
    if na == nb: return True
    return SequenceMatcher(None, na, nb).ratio() >= thresh

# ── Load all player Elo into memory (avoids per-match DB queries) ─────────
print('Loading player_elo into memory...', end='', flush=True)
elo_schema = [r[1] for r in tennis.execute('PRAGMA table_info(player_elo)').fetchall()]
print(f' columns: {elo_schema}')

# Build {player_name: [(match_date, overall_elo, matches_played), ...]} sorted by date
elo_history = defaultdict(list)
for row in tennis.execute('SELECT player_name, match_date, overall_elo, matches_played FROM player_elo ORDER BY player_name, match_date').fetchall():
    player, date, elo, mp = row
    elo_history[player].append((date, elo, mp))

total_players = len(elo_history)
total_elo_rows = sum(len(v) for v in elo_history.values())
print(f'Elo history: {total_players} players, {total_elo_rows:,} rows')

def get_elo_before(player, before_date):
    """Get latest Elo for player strictly before before_date. Returns (elo, matches_played) or None."""
    hist = elo_history.get(player)
    if not hist:
        return None
    dates = [h[0] for h in hist]
    idx = bisect.bisect_left(dates, before_date) - 1
    if idx < 0:
        return None
    return hist[idx][1], hist[idx][2]  # (elo, matches_played)

def get_elo_fuzzy(player, before_date):
    """Exact match first, then fuzzy if not found."""
    result = get_elo_before(player, before_date)
    if result:
        return result
    # Try fuzzy match against known players
    best_ratio = 0.0
    best_name  = None
    np_norm = norm(player)
    for known in elo_history.keys():
        ratio = SequenceMatcher(None, np_norm, norm(known)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name  = known
    if best_ratio >= 0.82:
        return get_elo_before(best_name, before_date)
    return None

# ════════════════════════════════════════════════════════════════════════════
# PART A: Historical Elo accuracy (2020+, no Kambi odds needed)
# ════════════════════════════════════════════════════════════════════════════
print()
print('=' * 60)
print('PART A: ELO ACCURACY ON HISTORICAL MATCHES (2020+)')
print('=' * 60)

matches = tennis.execute('''
    SELECT tourney_date, winner_name, loser_name, surface
    FROM matches
    WHERE tourney_date >= '2020-01-01'
    AND score IS NOT NULL
    AND score NOT LIKE '%W/O%'
    AND score NOT LIKE '%RET%'
    ORDER BY tourney_date
''').fetchall()
print(f'Historical matches 2020+: {len(matches):,}')

# Calibration buckets: (0.50-0.60, 0.60-0.70, ..., 0.90+)
brier_bins = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.01)]
bin_counts = defaultdict(lambda: {'n':0,'correct':0,'brier_sum':0.0})

correct = total_elo = 0
brier_elo = brier_base = 0.0

for tourney_date, winner, loser, surface in matches:
    we = get_elo_before(winner, tourney_date)
    le = get_elo_before(loser,  tourney_date)
    if not we or not le: continue
    w_elo, w_mp = we
    l_elo, l_mp = le
    if w_mp < 10 or l_mp < 10: continue

    p_win = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
    total_elo += 1
    brier_elo  += (p_win - 1) ** 2
    brier_base += (0.5 - 1)   ** 2
    if p_win > 0.5: correct += 1

    for lo, hi in brier_bins:
        if lo <= p_win < hi:
            bin_counts[(lo,hi)]['n'] += 1
            bin_counts[(lo,hi)]['correct'] += 1
            bin_counts[(lo,hi)]['brier_sum'] += (p_win - 1) ** 2
            break

if total_elo:
    print(f'\nMatches with valid Elo (both ≥10 matches): {total_elo:,}')
    print(f'Elo accuracy (picks higher-prob side):  {correct/total_elo:.1%}')
    print(f'Elo Brier score:      {brier_elo/total_elo:.4f}')
    print(f'Baseline Brier (50/50): {brier_base/total_elo:.4f}')
    print(f'Brier improvement:    {(brier_base-brier_elo)/total_elo:.4f}')
    print()
    print(f'{"Prob bucket":12s}  {"n":>8}  {"accuracy":>9}  {"Brier":>8}')
    print('-' * 45)
    for lo, hi in brier_bins:
        b = bin_counts[(lo,hi)]
        if b['n'] == 0: continue
        acc  = b['correct'] / b['n']
        bscr = b['brier_sum'] / b['n']
        print(f'{lo:.0%}-{hi:.0%}      {b["n"]:>8,}  {acc:>8.1%}  {bscr:>8.4f}')

# ════════════════════════════════════════════════════════════════════════════
# PART B: Kambi odds + Elo edge (May 27+)
# ════════════════════════════════════════════════════════════════════════════
print()
print('=' * 60)
print('PART B: KAMBI ODDS + ELO EDGE (May 27+ validation window)')
print('=' * 60)

# Get all ITF Women moneyline snapshots from Postgres
cur.execute("""
    SELECT DISTINCT ON (g.event_id, p.player_name)
        g.event_id,
        g.game_time::date AS match_date,
        p.player_name,
        p.over_odds
    FROM props_snapshots p
    JOIN games g ON p.event_id = g.event_id
    WHERE g.league = 'ITF Women'
    AND   p.line IS NULL
    AND   p.over_odds IS NOT NULL
    AND   p.snapshot_time >= '2026-05-27'
    ORDER BY g.event_id, p.player_name, p.snapshot_time ASC
""")
snapshots = cur.fetchall()
print(f'ITF Women moneyline snapshots (opening, distinct by event+player): {len(snapshots)}')

# Group by event_id
from itertools import groupby
from operator import itemgetter

events = defaultdict(list)
for event_id, match_date, player, odds in snapshots:
    events[event_id].append({'date': str(match_date), 'player': player, 'odds': int(odds)})

print(f'Unique match events: {len(events)}')

# ── Process each match event ─────────────────────────────────────────────
qualifying_bets = []
skipped = defaultdict(int)

for event_id, players in events.items():
    if len(players) != 2:
        skipped['not_2_players'] += 1
        continue

    match_date = players[0]['date']
    p1 = players[0]
    p2 = players[1]

    # Elo lookup
    e1 = get_elo_fuzzy(p1['player'], match_date)
    e2 = get_elo_fuzzy(p2['player'], match_date)

    if not e1 or not e2:
        skipped['no_elo'] += 1
        continue
    if e1[1] < min_career or e2[1] < min_career:
        skipped['insufficient_career'] += 1
        continue

    elo1, elo2 = e1[0], e2[0]

    # Win probabilities (Elo formula)
    p1_model = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    p2_model = 1 - p1_model

    # De-vig Kambi odds
    def imp(odds):
        return 100 / (100 + odds) if odds > 0 else abs(odds) / (abs(odds) + 100)

    imp1, imp2 = imp(p1['odds']), imp(p2['odds'])
    overround = imp1 + imp2
    if overround < 1.0:
        skipped['sub_100_overround'] += 1
        continue

    p1_fair = imp1 / overround
    p2_fair = imp2 / overround

    p1_edge = p1_model - p1_fair
    p2_edge = p2_model - p2_fair

    # Pick the side with positive edge
    if p1_edge >= p2_edge:
        alert_prob, alert_edge, alert_odds = p1_model, p1_edge, p1['odds']
        alert_player, opp_player = p1['player'], p2['player']
    else:
        alert_prob, alert_edge, alert_odds = p2_model, p2_edge, p2['odds']
        alert_player, opp_player = p2['player'], p1['player']

    # Apply filters
    if alert_prob < min_prob:
        skipped['below_min_prob'] += 1
        continue
    if alert_edge < min_edge:
        skipped['below_min_edge'] += 1
        continue
    if abs(alert_edge) > extreme:
        skipped['extreme_edge_blocked'] += 1
        continue

    # EV check
    dec = 1 + alert_odds/100 if alert_odds > 0 else 1 + 100/abs(alert_odds)
    ev = alert_prob * dec - 1
    if ev <= 0:
        skipped['negative_ev'] += 1
        continue

    odds_capped = (abs(alert_odds) > max_odds)
    if odds_capped:
        skipped['odds_capped_but_included'] += 1
        # Include in results but flag (so we can show WITH vs WITHOUT cap)

    # Find actual result in tennis.db
    result_val = None
    for w_name, l_name in tennis.execute('''
        SELECT winner_name, loser_name FROM matches
        WHERE tourney_date BETWEEN date(?, '-3 days') AND date(?, '+3 days')
        AND score IS NOT NULL AND score NOT LIKE '%W/O%'
    ''', (match_date, match_date)).fetchall():
        win_match  = names_match(alert_player, w_name) and names_match(opp_player, l_name)
        loss_match = names_match(alert_player, l_name) and names_match(opp_player, w_name)
        if win_match:
            result_val = 'WIN'; break
        if loss_match:
            result_val = 'LOSS'; break

    if result_val is None:
        skipped['no_result'] += 1

    qualifying_bets.append({
        'date':         match_date,
        'player':       alert_player,
        'opponent':     opp_player,
        'prob':         alert_prob,
        'edge':         alert_edge,
        'odds':         alert_odds,
        'ev':           ev,
        'odds_capped':  odds_capped,
        'result':       result_val,
    })

print(f'\nSkip breakdown: {dict(skipped)}')
print(f'Qualifying bets (including ungraded): {len(qualifying_bets)}')

# ── Results ────────────────────────────────────────────────────────────────
graded  = [b for b in qualifying_bets if b['result'] is not None]
pending = [b for b in qualifying_bets if b['result'] is None]
wins    = sum(1 for b in graded if b['result'] == 'WIN')
losses  = sum(1 for b in graded if b['result'] == 'LOSS')
staked  = wins + losses

def calc_profit(bets):
    p = 0.0
    for b in bets:
        if b['result'] == 'WIN':
            p += b['odds']/100 if b['odds'] > 0 else 100/abs(b['odds'])
        elif b['result'] == 'LOSS':
            p -= 1.0
    return p

total_profit = calc_profit(graded)
roi = total_profit / staked * 100 if staked else 0

print(f'\n=== TENNIS FRESH REPLAY RESULTS ===')
print(f'Graded:  {wins}W-{losses}L')
print(f'Pending: {len(pending)} (no result found yet)')
print(f'Profit:  {total_profit:+.2f}u')
print(f'ROI:     {roi:+.1f}%')
print(f'$20 flat: wagered ${staked*20} | net ${total_profit*20:+.2f}')

# ── WITH vs WITHOUT odds cap ───────────────────────────────────────────────
print(f'\n=== WITH vs WITHOUT ODDS CAP (±{max_odds}) ===')
capped_graded   = [b for b in graded if not b['odds_capped']]    # only those within cap
all_graded      = graded
for label, bets in [('With cap (≤±300)', capped_graded), ('Without cap (all)', all_graded)]:
    w = sum(1 for b in bets if b['result']=='WIN')
    l = sum(1 for b in bets if b['result']=='LOSS')
    p = calc_profit(bets)
    r = p/(w+l)*100 if (w+l) else 0
    print(f'  {label}: {w}W-{l}L | {p:+.2f}u | {r:+.1f}% ROI | n={len(bets)}')

# ── By confidence bucket ───────────────────────────────────────────────────
print(f'\n=== BY CONFIDENCE BUCKET ===')
for lo, hi in [(0.80,0.85),(0.85,0.90),(0.90,0.95),(0.95,1.01)]:
    bucket = [b for b in graded if lo <= b['prob'] < hi]
    if not bucket: continue
    w  = sum(1 for b in bucket if b['result']=='WIN')
    l  = sum(1 for b in bucket if b['result']=='LOSS')
    p  = calc_profit(bucket)
    r  = p/(w+l)*100 if (w+l) else 0
    ap = sum(b['prob'] for b in bucket)/len(bucket)
    print(f'  {lo:.0%}-{hi:.0%}: n={len(bucket)} | avg_prob={ap:.1%} | {w}W-{l}L | {p:+.2f}u | {r:+.1f}%')

# ── Monthly ────────────────────────────────────────────────────────────────
print(f'\n=== MONTHLY ===')
monthly = defaultdict(lambda: {'W':0,'L':0,'pending':0,'profit':0.0})
for b in qualifying_bets:
    m = b['date'][:7]
    if b['result'] == 'WIN':   monthly[m]['W'] += 1; monthly[m]['profit'] += calc_profit([b])
    elif b['result'] == 'LOSS': monthly[m]['L'] += 1; monthly[m]['profit'] += calc_profit([b])
    else:                       monthly[m]['pending'] += 1
for month in sorted(monthly):
    m = monthly[month]
    staked_m = m['W'] + m['L']
    roi_m = m['profit']/staked_m*100 if staked_m else 0
    print(f'  {month}: {m["W"]}W-{m["L"]}L-{m["pending"]}P | {m["profit"]:+.2f}u | {roi_m:+.1f}%')

# ── Compare to predictions table (sanity check) ───────────────────────────
print(f'\n=== COMPARISON vs PREDICTIONS TABLE ===')
stored = tennis.execute('''
    SELECT COUNT(*) FROM predictions WHERE prediction_date >= '2026-05-27'
''').fetchone()[0]
print(f'  Stored predictions:         {stored}')
print(f'  Fresh Postgres events:      {len(events)}')
print(f'  Fresh qualifying bets:      {len(qualifying_bets)} (no odds cap: all above filters)')
missed = len(events) - stored
print(f'  Events not in predictions:  {missed} (matches Kambi had but predict_tennis.py missed)')
