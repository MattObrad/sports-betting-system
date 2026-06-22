"""
replay_mlb.py -- Clean historical MLB replay at multiple edge thresholds.

Tests min_edge_runs in [0.8, 1.0, 1.2, 1.5] to find the break-even point.
All other filters (min_confidence, extreme, NULL hard-block) held constant.
No alerts.db.
"""
import sqlite3, json
from collections import defaultdict

con = sqlite3.connect('/home/picks/mlb_data.db')
cfg = json.load(open('/home/picks/config.json'))

min_prob   = cfg['betting'].get('min_confidence', 0.55)
THRESHOLDS = [0.8, 1.0, 1.2, 1.5]

print('=== MLB CLEAN REPLAY — THRESHOLD SWEEP ===')
print(f'Fixed filters: min_confidence={min_prob}, extreme_block=>3.0, NULL market hard-block')
print()

# Load all gradeable MC results once
rows = con.execute('''
    SELECT
        mcr.run_date,
        mcr.game_id,
        mcr.market_line,
        mcr.predicted_total,
        mcr.ensemble_over_prob,
        mcr.juice,
        g.game_date,
        g.home_score + g.away_score as actual_total,
        gf.line_movement,
        gf.current_total,
        gf.over_juice,
        gf.under_juice,
        COALESCE(g.doubleheader, 0) as doubleheader
    FROM monte_carlo_results mcr
    JOIN games g ON mcr.game_id = g.game_id
    JOIN game_features gf ON mcr.game_id = gf.game_id
    WHERE g.home_score IS NOT NULL
    AND g.away_score IS NOT NULL
    AND mcr.predicted_total IS NOT NULL
    AND mcr.market_line IS NOT NULL
    AND mcr.market_line <= 13.0
    AND COALESCE(g.doubleheader, 0) = 0
    ORDER BY mcr.run_date
''').fetchall()

print(f'Base dataset: {len(rows)} games with MC predictions + final scores')

# Overall direction summary (no filters)
all_dir = con.execute('''
    SELECT
        SUM(CASE WHEN mcr.predicted_total > mcr.market_line THEN 1 ELSE 0 END) as over_pred,
        SUM(CASE WHEN mcr.predicted_total < mcr.market_line THEN 1 ELSE 0 END) as under_pred,
        ROUND(AVG(mcr.predicted_total - mcr.market_line), 3) as avg_edge,
        SUM(CASE WHEN mcr.predicted_total > mcr.market_line AND g.home_score+g.away_score > mcr.market_line THEN 1 ELSE 0 END) as over_wins,
        SUM(CASE WHEN mcr.predicted_total < mcr.market_line AND g.home_score+g.away_score < mcr.market_line THEN 1 ELSE 0 END) as under_wins,
        COUNT(*) as total
    FROM monte_carlo_results mcr
    JOIN games g ON mcr.game_id = g.game_id
    WHERE g.home_score IS NOT NULL AND mcr.market_line IS NOT NULL AND mcr.predicted_total IS NOT NULL
''').fetchone()
n_all = all_dir[5]
print(f'All MC predictions (no filter):')
print(f'  OVER: {all_dir[0]} ({all_dir[0]/n_all:.0%}) | win rate {all_dir[3]/all_dir[0]:.1%}')
print(f'  UNDER: {all_dir[1]} ({all_dir[1]/n_all:.0%}) | win rate {all_dir[4]/max(all_dir[1],1):.1%}')
print(f'  Avg signed edge: {all_dir[2]:+.3f} runs')

actual_over = con.execute('''
    SELECT
        SUM(CASE WHEN g.home_score+g.away_score > mcr.market_line THEN 1 ELSE 0 END),
        SUM(CASE WHEN g.home_score+g.away_score < mcr.market_line THEN 1 ELSE 0 END),
        COUNT(*)
    FROM monte_carlo_results mcr
    JOIN games g ON mcr.game_id = g.game_id
    WHERE g.home_score IS NOT NULL AND mcr.market_line IS NOT NULL
''').fetchone()
print(f'Actual market OVER rate: {actual_over[0]/actual_over[2]:.1%} | UNDER: {actual_over[1]/actual_over[2]:.1%}')
print()

def run_replay(min_edge: float):
    qualifying = []
    skipped_null = skipped_extreme = skipped_edge = skipped_conf = 0

    for row in rows:
        (run_date, game_id, market_line, predicted, over_prob, juice,
         game_date, actual, line_movement, current_total,
         over_juice, under_juice, doubleheader) = row

        run_edge  = predicted - market_line
        abs_edge  = abs(run_edge)
        direction = 'OVER' if run_edge > 0 else 'UNDER'
        confidence = over_prob if direction == 'OVER' else 1.0 - over_prob

        if line_movement is None and current_total is None:
            skipped_null += 1; continue
        if abs_edge > 3.0:
            skipped_extreme += 1; continue
        if abs_edge < min_edge:
            skipped_edge += 1; continue
        if confidence < min_prob:
            skipped_conf += 1; continue

        if direction == 'OVER':
            result = 'WIN' if actual > market_line else ('LOSS' if actual < market_line else 'PUSH')
            odds = int(over_juice) if over_juice else (int(juice) if juice else -110)
        else:
            result = 'WIN' if actual < market_line else ('LOSS' if actual > market_line else 'PUSH')
            odds = int(under_juice) if under_juice else -110

        if result == 'WIN':
            profit = (odds / 100) if odds > 0 else (100 / abs(odds))
        elif result == 'LOSS':
            profit = -1.0
        else:
            profit = 0.0

        qualifying.append({
            'date': game_date, 'direction': direction,
            'line': market_line, 'predicted': predicted, 'edge': run_edge,
            'confidence': confidence, 'actual': actual,
            'result': result, 'profit': profit, 'odds': odds,
        })

    return qualifying, skipped_null, skipped_extreme, skipped_edge, skipped_conf

# ── threshold sweep ───────────────────────────────────────────────────────────
print('=' * 70)
print(f'{"Threshold":>10} | {"n":>5} | {"Record":>12} | {"ROI":>8} | {"OVER%":>6} | {"OVER W%":>8} | {"UNDER W%":>9}')
print('-' * 70)

full_results = {}
for thresh in THRESHOLDS:
    bets, s_null, s_ext, s_edge, s_conf = run_replay(thresh)
    wins   = sum(1 for b in bets if b['result'] == 'WIN')
    losses = sum(1 for b in bets if b['result'] == 'LOSS')
    pushes = sum(1 for b in bets if b['result'] == 'PUSH')
    staked = wins + losses
    profit = sum(b['profit'] for b in bets)
    roi    = profit / staked * 100 if staked else 0
    overs  = [b for b in bets if b['direction'] == 'OVER']
    unders = [b for b in bets if b['direction'] == 'UNDER']
    ow = sum(1 for b in overs if b['result'] == 'WIN')
    uw = sum(1 for b in unders if b['result'] == 'WIN')
    over_pct = len(overs) / max(len(bets), 1) * 100
    owr = ow / max(len(overs), 1) * 100
    uwr = uw / max(len(unders), 1) * 100
    record = f'{wins}W-{losses}L-{pushes}P'
    print(f'{thresh:>10.1f} | {len(bets):>5} | {record:>12} | {roi:>+7.1f}% | {over_pct:>5.0f}% | {owr:>7.1f}% | {uwr:>8.1f}%')
    full_results[thresh] = (bets, wins, losses, profit, overs, unders)

print('=' * 70)
print()

# ── detailed breakdown for each threshold ─────────────────────────────────────
for thresh in THRESHOLDS:
    bets, wins, losses, profit, overs, unders = full_results[thresh]
    staked = wins + losses
    if not bets:
        print(f'\n--- min_edge={thresh} --- NO QUALIFYING BETS ---')
        continue
    roi = profit / staked * 100 if staked else 0
    print(f'\n--- min_edge={thresh} --- {len(bets)} bets | {wins}W-{losses}L | {roi:+.1f}% ROI ---')

    # Monthly
    monthly = defaultdict(lambda: {'W':0,'L':0,'profit':0.0})
    for b in bets:
        m = b['date'][:7]
        monthly[m]['W' if b['result']=='WIN' else 'L'] += 1
        monthly[m]['profit'] += b['profit']
    print('  Monthly:')
    for month in sorted(monthly):
        m = monthly[month]
        t = m['W'] + m['L']
        print(f'    {month}: {m["W"]}W-{m["L"]}L | {m["profit"]:+.2f}u | {m["profit"]/t*100:+.1f}%')

    # Edge buckets
    print('  By edge size:')
    for lo, hi in [(0.8,1.0),(1.0,1.2),(1.2,1.5),(1.5,2.0),(2.0,3.0)]:
        bucket = [b for b in bets if lo <= abs(b['edge']) < hi]
        if not bucket: continue
        bw = sum(1 for b in bucket if b['result'] == 'WIN')
        bp = sum(b['profit'] for b in bucket)
        print(f'    edge {lo:.1f}-{hi:.1f}: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u, {bp/len(bucket)*100:+.1f}%')

    # Confidence calibration
    print('  Confidence calibration:')
    for lo, hi in [(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,1.0)]:
        bucket = [b for b in bets if lo <= b['confidence'] < hi]
        if not bucket: continue
        act_wr = sum(1 for b in bucket if b['result'] == 'WIN') / len(bucket)
        print(f'    {lo:.0%}-{hi:.0%}: n={len(bucket)}, act_wr={act_wr:.1%}')

print()
print('Break-even win rate at -110 odds: 52.4%')
print('Break-even win rate at -115 odds: 53.5%')
print()
print('VERDICT GUIDE:')
print('  - Any threshold with ROI > 0 and n >= 20: potential signal, needs live CLV validation')
print('  - OVER% >> 50% with OVER win rate < 52%: model bias is unbeatable at any threshold')
print('  - UNDER win rate > 55% with n >= 10: consider flipping direction')
