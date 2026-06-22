import sqlite3, os

con = sqlite3.connect('/home/picks/mlb_data.db')

print('=== MLB MARKET DATA QUALITY (last 30 days) ===')
result = con.execute('''
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN mcr.market_line IS NOT NULL THEN 1 ELSE 0 END) as has_market_line,
        SUM(CASE WHEN mcr.market_line IS NULL THEN 1 ELSE 0 END) as null_market_line,
        ROUND(AVG(mcr.predicted_total), 3) as avg_predicted_residual,
        SUM(CASE WHEN mcr.predicted_total > 0 THEN 1 ELSE 0 END) as over_bias_count,
        ROUND(AVG(mcr.sim_over_prob), 3) as avg_sim_over_prob,
        ROUND(AVG(mcr.ensemble_over_prob), 3) as avg_ens_over_prob,
        SUM(CASE WHEN g.total_runs IS NOT NULL THEN 1 ELSE 0 END) as graded
    FROM monte_carlo_results mcr
    JOIN games g ON mcr.game_id = g.game_id
    WHERE g.game_date >= date('now', '-30 days')
    AND mcr.predicted_total IS NOT NULL
''').fetchone()

if result and result[0]:
    total = result[0]
    print(f'Total MC predictions: {total}')
    print(f'Has market_line:       {result[1]} ({result[1]/total*100:.0f}%)')
    print(f'NULL market_line:      {result[2]} ({result[2]/total*100:.0f}%)')
    print(f'Avg predicted residual:{result[3]:+.3f} (>0 = OVER bias)')
    print(f'Positive residual (OVER lean): {result[4]}/{total} ({result[4]/total*100:.0f}%)')
    print(f'Avg sim_over_prob:     {result[5]:.3f}')
    print(f'Avg ensemble_over_prob:{result[6]:.3f}')
    print(f'Graded (result known): {result[7]}')
else:
    print('No rows found in last 30 days')

print()
print('=== GAME_FEATURES LINE MOVEMENT (from sync_odds, last 30 days) ===')
lm_result = con.execute('''
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN gf.line_movement IS NOT NULL
            AND gf.line_movement != 0 THEN 1 ELSE 0 END) as real_line_movement,
        SUM(CASE WHEN gf.opening_total IS NOT NULL THEN 1 ELSE 0 END) as has_opening,
        SUM(CASE WHEN gf.current_total IS NOT NULL THEN 1 ELSE 0 END) as has_current,
        ROUND(AVG(gf.line_movement), 3) as avg_lm
    FROM game_features gf
    JOIN games g ON gf.game_id = g.game_id
    WHERE g.game_date >= date('now', '-30 days')
''').fetchone()
if lm_result and lm_result[0]:
    t = lm_result[0]
    print(f'Total game_features rows: {t}')
    print(f'Non-zero line_movement:   {lm_result[1]} ({lm_result[1]/t*100:.0f}%)')
    print(f'Has opening_total:        {lm_result[2]} ({lm_result[2]/t*100:.0f}%)')
    print(f'Has current_total:        {lm_result[3]} ({lm_result[3]/t*100:.0f}%)')
    print(f'Avg line_movement:        {lm_result[4] or 0:+.3f}')
else:
    print('No game_features rows in last 30 days')

print()
print('=== RECENT MC RESULTS (last 10 graded) ===')
recent = con.execute('''
    SELECT g.game_date,
           ht.team_abbrev, at.team_abbrev,
           mcr.predicted_total, mcr.market_line,
           mcr.sim_over_prob, mcr.ensemble_over_prob,
           g.total_runs,
           CASE WHEN mcr.predicted_total > 0 AND g.total_runs > mcr.market_line THEN 'WIN'
                WHEN mcr.predicted_total > 0 AND g.total_runs < mcr.market_line THEN 'LOSS'
                WHEN mcr.predicted_total > 0 AND g.total_runs = mcr.market_line THEN 'PUSH'
                ELSE 'UNDER-LEAN' END as simple_result
    FROM monte_carlo_results mcr
    JOIN games g ON mcr.game_id = g.game_id
    LEFT JOIN teams ht ON g.home_team_id = ht.team_id
    LEFT JOIN teams at ON g.away_team_id = at.team_id
    WHERE g.total_runs IS NOT NULL
    ORDER BY g.game_date DESC
    LIMIT 10
''').fetchall()
if recent:
    print(f'{"Date":12} {"Match":14} {"Pred+":>6} {"Mkt":>5} {"SimP":>5} {"EnsP":>5} {"Act":>4} {"Result":>10}')
    print('-'*70)
    for r in recent:
        date, home, away, pred, mkt, sim, ens, act, res = r
        match = f'{away or "?"[:5]}@{home or "?"[:5]}'
        print(f'{date:12} {match:14} {pred or 0:+6.2f} {mkt or 0:5.1f} {sim or 0:5.3f} {ens or 0:5.3f} {act or 0:4.1f} {res:>10}')
else:
    print('No graded rows found')

print()
print('=== SYNC_ODDS LOG (last 30 lines) ===')
log = '/home/picks/logs/sync_odds.log'
if os.path.exists(log):
    lines = open(log).readlines()
    for l in lines[-30:]:
        print(l.rstrip())
else:
    print('sync_odds.log not found at', log)
