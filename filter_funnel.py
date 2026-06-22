"""
filter_funnel.py -- Tennis filter funnel diagnostic.

Shows exactly how many predictions are blocked at each filter stage,
with special focus on the odds cap impact hypothesis.
Reads from tennis.db predictions table only — no Postgres needed.
"""
import sqlite3, json
from collections import Counter, defaultdict

tennis = sqlite3.connect('/home/picks/tennis.db')
cfg    = json.load(open('/home/picks/tennis_config.json'))

betting    = cfg.get('betting', cfg)
min_prob   = betting.get('min_model_prob',    0.80)
min_edge   = betting.get('min_edge',          0.08)
max_odds   = betting.get('max_odds_american',  300)
extreme    = betting.get('extreme_edge',       0.35)
min_career = betting.get('min_career_matches',  10)

print('=== TENNIS FILTER FUNNEL DIAGNOSTIC ===')
print(f'  min_model_prob:      {min_prob}')
print(f'  min_edge:            {min_edge}')
print(f'  max_odds_american:   ±{max_odds}')
print(f'  extreme_edge:        {extreme}')
print(f'  min_career_matches:  {min_career}')
print()

rows = tennis.execute('''
    SELECT player1_model_prob, player2_model_prob,
           player1_edge, player2_edge,
           player1_kambi_odds, player2_kambi_odds,
           prediction_date
    FROM predictions
    WHERE prediction_date >= '2026-05-27'
    AND player1_model_prob IS NOT NULL
''').fetchall()

print(f'Total predictions since May 27: {len(rows)}')

# ── Per-side funnel (evaluate both players independently) ─────────────────
# This shows the true "how many bet-candidates exist" picture
total_sides   = 0
fail_prob_n   = fail_edge_n = fail_odds_n = fail_extreme_n = fail_ev_n = 0
qualify_n     = 0

for r in rows:
    mp1, mp2, e1, e2, o1, o2, date = r
    for mp, edge, odds in [(mp1, e1, o1), (mp2, e2, o2)]:
        if mp is None or edge is None or odds is None:
            continue
        total_sides += 1
        if mp < min_prob:          fail_prob_n += 1;    continue
        if edge < min_edge:        fail_edge_n += 1;    continue
        if abs(odds) > max_odds:   fail_odds_n += 1;    continue
        if abs(edge) > extreme:    fail_extreme_n += 1; continue
        dec = 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
        if mp * dec - 1 <= 0:      fail_ev_n += 1;      continue
        qualify_n += 1

print(f'\n=== PER-SIDE FILTER FUNNEL ({total_sides} total bet-sides evaluated) ===')
def pct(n): return f'{n/total_sides*100:.0f}%'
print(f'  1. Fail prob < {min_prob}:          {fail_prob_n:4d}  ({pct(fail_prob_n)})')
print(f'  2. Fail edge < {min_edge}:          {fail_edge_n:4d}  ({pct(fail_edge_n)})')
print(f'  3. Fail odds > ±{max_odds}:         {fail_odds_n:4d}  ({pct(fail_odds_n)})  ← KEY QUESTION')
print(f'  4. Fail extreme > {extreme}:        {fail_extreme_n:4d}  ({pct(fail_extreme_n)})')
print(f'  5. Fail EV <= 0:                {fail_ev_n:4d}  ({pct(fail_ev_n)})')
print(f'  QUALIFY:                        {qualify_n:4d}  ({qualify_n/total_sides*100:.1f}%)')

# ── High-confidence analysis ──────────────────────────────────────────────
high_conf = sum(1 for r in rows if max(r[0] or 0, r[1] or 0) >= min_prob)
print(f'\n=== HIGH-CONFIDENCE PREDICTIONS (max_prob >= {min_prob}) ===')
print(f'  Predictions with at least one side >= {min_prob}: {high_conf}/{len(rows)} ({high_conf/len(rows)*100:.0f}%)')

# Odds distribution for HIGH-PROB sides (regardless of other filters)
print(f'\n=== ODDS DISTRIBUTION FOR HIGH-PROB SIDES ===')
print(f'  (every side with prob >= {min_prob}, no other filter)')
odds_buckets = Counter()
for r in rows:
    mp1, mp2, e1, e2, o1, o2, date = r
    for mp, odds in [(mp1, o1), (mp2, o2)]:
        if mp is None or mp < min_prob or odds is None:
            continue
        # Bin by 100-unit intervals
        if odds >= 0:
            lo = (odds // 100) * 100
            key = f'+{lo:3d} to +{lo+100:3d}'
        else:
            lo = (abs(odds) // 100) * 100
            key = f'-{lo+100:3d} to -{lo:3d}  '
        odds_buckets[key] += 1

total_high = sum(odds_buckets.values())
print(f'  {"Odds Range":22s}  {"Count":>5}  {"% of high-prob":>14}  {"Blocked?":>10}')
print(f'  {"-"*60}')
for bucket in sorted(odds_buckets.keys()):
    count = odds_buckets[bucket]
    # Approximate middle of bucket to check cap
    try:
        parts = bucket.strip().replace('+','').split(' to ')
        mid = (int(parts[0]) + int(parts[1])) // 2
        blocked = '← BLOCKED' if abs(mid) > max_odds else ''
    except Exception:
        blocked = ''
    bar = '█' * min(count, 40)
    print(f'  {bucket:22s}  {count:>5}  {count/total_high*100:>13.0f}%  {blocked:>10}  {bar}')

# ── Odds cap impact: best-side selection (mirrors replay_tennis.py logic) ──
print(f'\n=== ODDS CAP IMPACT (best-edge-side selection per match) ===')

def best_side(r):
    mp1, mp2, e1, e2, o1, o2, date = r
    if (e1 or -99) >= (e2 or -99):
        return mp1, e1, o1
    return mp2, e2, o2

qualify_no_cap = 0
qualify_with_cap = 0
for r in rows:
    mp, edge, odds = best_side(r)
    if mp is None or edge is None or odds is None: continue
    if mp < min_prob: continue
    if edge < min_edge: continue
    if abs(edge) > extreme: continue
    dec = 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
    if mp * dec - 1 <= 0: continue
    qualify_no_cap += 1
    if abs(odds) <= max_odds:
        qualify_with_cap += 1

blocked_by_cap = qualify_no_cap - qualify_with_cap
print(f'  Qualifying WITH  odds cap (±{max_odds}):    {qualify_with_cap:4d}')
print(f'  Qualifying WITHOUT odds cap:              {qualify_no_cap:4d}')
print(f'  Blocked by odds cap:                      {blocked_by_cap:4d} ({blocked_by_cap/max(qualify_no_cap,1)*100:.0f}% of potential bets)')

print(f'\n=== BET COUNT AT DIFFERENT ODDS CAPS ===')
for cap in [150, 200, 250, 300, 350, 400, 500, 750, 999]:
    count = 0
    for r in rows:
        mp, edge, odds = best_side(r)
        if mp is None or edge is None or odds is None: continue
        if mp < min_prob: continue
        if edge < min_edge: continue
        if abs(odds) > cap: continue
        if abs(edge) > extreme: continue
        dec = 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
        if mp * dec - 1 <= 0: continue
        count += 1
    marker = ' ← CURRENT' if cap == max_odds else ''
    print(f'  max_odds=±{cap:3d}:  {count:4d} qualifying bets{marker}')

# ── What do the qualifying bets actually look like? ───────────────────────
print(f'\n=== QUALIFYING BET PROFILE (no odds cap, extreme_edge <= {extreme}) ===')
details = []
for r in rows:
    mp, edge, odds = best_side(r)
    mp1, mp2, e1, e2, o1, o2, date = r
    if mp is None or edge is None or odds is None: continue
    if mp < min_prob: continue
    if edge < min_edge: continue
    if abs(edge) > extreme: continue
    dec = 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
    if mp * dec - 1 <= 0: continue
    details.append({'prob': mp, 'edge': edge, 'odds': odds, 'date': date,
                    'capped': abs(odds) > max_odds})

if details:
    avg_prob = sum(d['prob'] for d in details) / len(details)
    avg_edge = sum(d['edge'] for d in details) / len(details)
    avg_odds = sum(d['odds'] for d in details) / len(details)
    n_capped = sum(1 for d in details if d['capped'])
    print(f'  Total bets (no cap):  {len(details)}')
    print(f'  Avg model prob:       {avg_prob:.1%}')
    print(f'  Avg edge:             {avg_edge:.1%}')
    print(f'  Avg Kambi odds:       {avg_odds:+.0f}')
    print(f'  Blocked by ±{max_odds} cap:   {n_capped}')
    print()
    print(f'  Monthly breakdown (no cap / pass cap):')
    by_month = defaultdict(lambda: {'total':0, 'passed':0})
    for d in details:
        m = d['date'][:7]
        by_month[m]['total'] += 1
        if not d['capped']:
            by_month[m]['passed'] += 1
    for month in sorted(by_month):
        m = by_month[month]
        print(f'    {month}:  {m["passed"]:3d} passed cap / {m["total"]:3d} total')

print()
print('HYPOTHESIS CHECK:')
print(f'  If odds cap is the main bottleneck, blocked_by_cap should be large.')
print(f'  If prob/edge is the main bottleneck, fail_prob/fail_edge dominate.')
print(f'  Adjust max_odds_american in tennis_config.json to change the cap.')
