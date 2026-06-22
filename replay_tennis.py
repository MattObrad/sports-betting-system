"""
replay_tennis.py -- Clean historical tennis replay at multiple extreme_edge thresholds.

Tests extreme_threshold in [0.20, 0.25, 0.35] to find the right filtering level.
All other filters (min_prob=0.80, min_edge=0.08, max_odds=300, min_career=10) held constant.
No alerts.db.
"""
import sqlite3, json, unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

cfg    = json.load(open('/home/picks/tennis_config.json'))
tennis = sqlite3.connect('/home/picks/tennis.db')

MIN_PROB           = cfg['betting'].get('min_model_prob',    0.80)
MIN_EDGE           = cfg['betting'].get('min_edge',          0.08)
MAX_ODDS           = cfg['betting'].get('max_odds_american',  300)
MIN_CAREER_MATCHES = cfg['betting'].get('min_career_matches',  10)
FUZZY_THRESHOLD    = 0.82
MATCH_WINDOW_DAYS  = 10
EXTREME_THRESHOLDS = [0.20, 0.25, 0.35]

print('=== TENNIS CLEAN REPLAY — EXTREME EDGE THRESHOLD SWEEP ===')
print(f'Fixed: min_prob={MIN_PROB}, min_edge={MIN_EDGE}, max_odds=+{MAX_ODDS}, min_career={MIN_CAREER_MATCHES}')
print()

# ── helpers ───────────────────────────────────────────────────────────────────
def _norm(name: str) -> str:
    nfd = unicodedata.normalize('NFD', name.lower())
    return ''.join(c for c in nfd if not unicodedata.combining(c)).strip()

def _names_match(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    sa = na.replace('-',' ').replace('.','').split()
    sb = nb.replace('-',' ').replace('.','').split()
    if len(sa) >= 2 and len(sa[-1]) == 1 and len(sb) >= 2:
        if ' '.join(sa[:-1]) == ' '.join(sb[1:]) and sa[-1] == sb[0][0]:
            return True
    if len(sb) >= 2 and len(sb[-1]) == 1 and len(sa) >= 2:
        if ' '.join(sb[:-1]) == ' '.join(sa[1:]) and sb[-1] == sa[0][0]:
            return True
    return SequenceMatcher(None, na, nb).ratio() >= FUZZY_THRESHOLD

_career_cache: dict = {}
def career_matches(player: str, before_date: str) -> int:
    key = (player, before_date)
    if key not in _career_cache:
        _career_cache[key] = tennis.execute(
            "SELECT COUNT(*) FROM matches WHERE (winner_name=? OR loser_name=?) AND tourney_date < ?",
            (player, player, before_date)
        ).fetchone()[0]
    return _career_cache[key]

def find_result(player: str, opponent: str, pred_date: str):
    from datetime import datetime, timedelta
    lo = (datetime.fromisoformat(pred_date) - timedelta(days=MATCH_WINDOW_DAYS)).date().isoformat()
    hi = (datetime.fromisoformat(pred_date) + timedelta(days=MATCH_WINDOW_DAYS)).date().isoformat()
    rows = tennis.execute(
        "SELECT winner_name, loser_name FROM matches WHERE tourney_date BETWEEN ? AND ? AND score IS NOT NULL AND score NOT LIKE '%W/O%'",
        (lo, hi)
    ).fetchall()
    for winner, loser in rows:
        if _names_match(winner, player):
            if opponent and not _names_match(loser, opponent): continue
            return 'WIN', winner, loser
        if _names_match(loser, player):
            if opponent and not _names_match(winner, opponent): continue
            return 'LOSS', winner, loser
    return None, None, None

def profit_for(result: str, odds: int) -> float:
    if result == 'WIN':  return (odds / 100) if odds > 0 else (100 / abs(odds))
    if result == 'LOSS': return -1.0
    return 0.0

# ── load all predictions once ─────────────────────────────────────────────────
preds = tennis.execute('''
    SELECT id, prediction_date, event_id,
           player1_name, player2_name,
           player1_model_prob, player2_model_prob,
           player1_kambi_odds, player2_kambi_odds,
           player1_fair_prob,  player2_fair_prob,
           player1_edge,       player2_edge,
           surface, tourney_level
    FROM predictions
    WHERE player1_model_prob IS NOT NULL
    ORDER BY prediction_date
''').fetchall()

print(f'Total stored predictions: {len(preds)}')
high_conf = sum(1 for p in preds if max(p[5] or 0, p[6] or 0) >= MIN_PROB)
print(f'With prob >= {MIN_PROB}: {high_conf}')
print()

def run_replay(extreme_thresh: float):
    qualifying = []
    skipped = defaultdict(int)

    for row in preds:
        (pid, pred_date, event_id,
         p1_name, p2_name,
         p1_prob, p2_prob,
         p1_odds, p2_odds,
         p1_fair, p2_fair,
         p1_edge, p2_edge,
         surface, level) = row

        # Extreme flag at given threshold
        if abs(p1_edge or 0) > extreme_thresh or abs(p2_edge or 0) > extreme_thresh:
            skipped['extreme_flag'] += 1
            continue

        bet_player = bet_opp = bet_prob = bet_edge = bet_odds = None
        for (player, opp, prob, edge, odds) in [
            (p1_name, p2_name, p1_prob, p1_edge, p1_odds),
            (p2_name, p1_name, p2_prob, p2_edge, p2_odds),
        ]:
            if prob is None or edge is None or odds is None: continue
            if prob < MIN_PROB: continue
            if edge < MIN_EDGE: continue
            if odds > MAX_ODDS: continue
            if bet_prob is None or prob > bet_prob:
                bet_player, bet_opp = player, opp
                bet_prob, bet_edge, bet_odds = prob, edge, odds

        if bet_player is None:
            skipped['below_threshold'] += 1
            continue

        if career_matches(bet_player, pred_date) < MIN_CAREER_MATCHES:
            skipped['career_matches'] += 1
            continue

        result, w_name, l_name = find_result(bet_player, bet_opp, pred_date)
        if result is None:
            skipped['ungraded'] += 1
            qualifying.append({
                'pred_date': pred_date, 'player': bet_player, 'opp': bet_opp,
                'prob': bet_prob, 'edge': bet_edge, 'odds': bet_odds,
                'surface': surface, 'result': 'PENDING', 'profit': None,
            })
            continue

        qualifying.append({
            'pred_date': pred_date, 'player': bet_player, 'opp': bet_opp,
            'prob': bet_prob, 'edge': bet_edge, 'odds': bet_odds,
            'surface': surface, 'result': result,
            'profit': profit_for(result, bet_odds),
        })

    return qualifying, skipped

# ── threshold sweep summary ───────────────────────────────────────────────────
print('=' * 72)
print(f'{"Extreme":>8} | {"Total":>6} | {"Graded":>7} | {"Record":>10} | {"ROI":>8} | {"Pending":>8}')
print('-' * 72)

full_results = {}
for thresh in EXTREME_THRESHOLDS:
    bets, skipped = run_replay(thresh)
    graded  = [b for b in bets if b['result'] != 'PENDING']
    pending = [b for b in bets if b['result'] == 'PENDING']
    wins    = sum(1 for b in graded if b['result'] == 'WIN')
    losses  = sum(1 for b in graded if b['result'] == 'LOSS')
    staked  = wins + losses
    profit  = sum(b['profit'] for b in graded)
    roi     = profit / staked * 100 if staked else 0
    record  = f'{wins}W-{losses}L'
    print(f'{thresh:>8.2f} | {len(bets):>6} | {len(graded):>7} | {record:>10} | {roi:>+7.1f}% | {len(pending):>8}')
    full_results[thresh] = (bets, graded, pending, wins, losses, profit, skipped)

print('=' * 72)
print()

# ── detailed breakdown per threshold ─────────────────────────────────────────
for thresh in EXTREME_THRESHOLDS:
    bets, graded, pending, wins, losses, profit, skipped = full_results[thresh]
    staked = wins + losses
    roi    = profit / staked * 100 if staked else 0
    blocked_by_extreme = skipped['extreme_flag']

    print(f'\n{"="*60}')
    print(f'EXTREME THRESHOLD = {thresh}')
    print(f'{"="*60}')
    print(f'Qualifying: {len(bets)} ({staked} graded, {len(pending)} pending)')
    print(f'Record: {wins}W-{losses}L | {profit:+.2f}u | {roi:+.1f}% ROI')
    print(f'Blocked by extreme flag: {blocked_by_extreme} (of {high_conf} high-confidence)')
    print(f'Skip breakdown: {dict(skipped)}')

    if not graded:
        print('  No graded bets.')
        continue

    # Confidence breakdown
    print('\nBy confidence:')
    for lo, hi in [(0.80,0.85),(0.85,0.90),(0.90,0.95),(0.95,1.01)]:
        bucket = [b for b in graded if lo <= b['prob'] < hi]
        if not bucket: continue
        bw = sum(1 for b in bucket if b['result'] == 'WIN')
        bp = sum(b['profit'] for b in bucket)
        print(f'  {lo:.0%}-{hi:.0%}: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u, {bp/len(bucket)*100:+.1f}%')

    # Edge breakdown
    print('\nBy edge size:')
    for lo, hi in [(0.08,0.12),(0.12,0.16),(0.16,0.20),(0.20,0.25),(0.25,0.35)]:
        bucket = [b for b in graded if lo <= b['edge'] < hi]
        if not bucket: continue
        bw = sum(1 for b in bucket if b['result'] == 'WIN')
        bp = sum(b['profit'] for b in bucket)
        print(f'  {lo:.0%}-{hi:.0%}: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u, {bp/len(bucket)*100:+.1f}%')

    # Surface breakdown
    surfs = defaultdict(lambda: {'W':0,'L':0,'profit':0.0})
    for b in graded:
        s = b['surface'] or 'Unknown'
        surfs[s]['W' if b['result']=='WIN' else 'L'] += 1
        surfs[s]['profit'] += b['profit']
    if surfs:
        print('\nBy surface:')
        for surf, m in sorted(surfs.items()):
            t = m['W'] + m['L']
            if t: print(f'  {surf}: {m["W"]}W-{m["L"]}L, {m["profit"]:+.2f}u')

    # Monthly
    monthly = defaultdict(lambda: {'W':0,'L':0,'profit':0.0,'pending':0})
    for b in bets:
        m = b['pred_date'][:7]
        if b['result'] == 'PENDING': monthly[m]['pending'] += 1
        elif b['result'] == 'WIN':   monthly[m]['W'] += 1; monthly[m]['profit'] += b['profit']
        else:                         monthly[m]['L'] += 1; monthly[m]['profit'] += b['profit']
    print('\nMonthly:')
    for month in sorted(monthly):
        m = monthly[month]
        t = m['W'] + m['L']
        roi_m = m['profit']/t*100 if t else 0
        print(f'  {month}: {m["W"]}W-{m["L"]}L-{m["pending"]}P | {m["profit"]:+.2f}u | {roi_m:+.1f}%')

    # All pending
    if pending:
        print(f'\nPending ({len(pending)}):')
        for b in pending[:15]:
            print(f'  {b["pred_date"]} {b["player"]} vs {b["opp"]} (prob={b["prob"]:.3f}, edge={b["edge"]:.3f})')

print()
print('VERDICT GUIDE:')
print('  - Positive ROI with n >= 20 graded: potentially real (but still small sample)')
print('  - If higher extreme_thresh → better ROI: the blocked high-edge bets are good bets')
print('  - If higher extreme_thresh → worse ROI: the 20% filter IS correctly blocking noise')
print('  - n=6 at 0.20 means we need 3-4 more weeks of live data before conclusions')
