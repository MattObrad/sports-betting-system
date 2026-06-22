import sqlite3

print('############ MLB ############')
con = sqlite3.connect('/home/picks/mlb_data.db')

# columns present in game_features
cols = [r[1] for r in con.execute("PRAGMA table_info(game_features)").fetchall()]
print('game_features has home_sp_days_rest:', 'home_sp_days_rest' in cols)
print('game_features has ump_career_over_rate:', 'ump_career_over_rate' in cols)

print()
print('=== RECENT PREDICTIONS (last 14 days) ===')
try:
    rows = con.execute('''
    SELECT mcr.run_date, COUNT(*) as games,
      SUM(CASE WHEN mcr.ensemble_over_prob > 0.55 THEN 1 ELSE 0 END) as over_55,
      SUM(CASE WHEN mcr.predicted_total > mcr.market_line THEN 1 ELSE 0 END) as pred_over_line
    FROM monte_carlo_results mcr
    WHERE mcr.run_date >= date('now','-21 days')
    GROUP BY mcr.run_date ORDER BY mcr.run_date DESC
    ''').fetchall()
    for r in rows: print(r)
except Exception as e:
    print('ERR', e)

print()
print('=== OVER/UNDER BIAS: all MC results last 30 days ===')
try:
    r = con.execute('''
    SELECT COUNT(*) n,
      SUM(CASE WHEN predicted_total > market_line THEN 1 ELSE 0 END) pred_over,
      SUM(CASE WHEN predicted_total < market_line THEN 1 ELSE 0 END) pred_under,
      ROUND(AVG(predicted_total - market_line),3) avg_signed_edge,
      ROUND(AVG(sigma_effective),3) avg_sigma
    FROM monte_carlo_results
    WHERE run_date >= date('now','-30 days') AND market_line IS NOT NULL
    ''').fetchone()
    print(r)
except Exception as e:
    print('ERR', e)

print()
print('=== EDGE BET HISTORY (direction bias) ===')
try:
    cols = [c[1] for c in con.execute("PRAGMA table_info(edge_bets)").fetchall()]
    print('edge_bets cols:', cols)
    rows = con.execute('''
    SELECT run_date, game_id, bet_direction, market_line, predicted_total,
      ROUND(predicted_total - market_line,2) run_edge
    FROM edge_bets ORDER BY run_date DESC LIMIT 25
    ''').fetchall()
    for r in rows: print(r)
    print('Direction counts:', con.execute(
      "SELECT bet_direction, COUNT(*) FROM edge_bets GROUP BY bet_direction").fetchall())
except Exception as e:
    print('ERR', e)

print()
print('=== PITCHER STALENESS (last 30 days) ===')
try:
    r = con.execute('''
    SELECT COUNT(*) total,
      SUM(CASE WHEN home_sp_days_rest > 60 THEN 1 ELSE 0 END) stale_home,
      SUM(CASE WHEN away_sp_days_rest > 60 THEN 1 ELSE 0 END) stale_away,
      ROUND(AVG(home_sp_era_l5),2) avg_home_era,
      ROUND(AVG(away_sp_era_l5),2) avg_away_era
    FROM game_features gf JOIN games g ON gf.game_id=g.game_id
    WHERE g.game_date >= date('now','-30 days')
    ''').fetchone()
    print(r)
except Exception as e:
    print('ERR', e)

print()
print('############ WNBA (sports.db) ############')
con2 = sqlite3.connect('/home/picks/sports.db')
cols = [r[1] for r in con2.execute("PRAGMA table_info(wnba_player_box_scores)").fetchall()]
print('box score cols:', cols)
print('Max game_date:', con2.execute('SELECT MAX(game_date) FROM wnba_player_box_scores').fetchone())
print('2026 rows:', con2.execute("SELECT COUNT(*) FROM wnba_player_box_scores WHERE game_date LIKE '2026%'").fetchone())
print()
print('=== MOST RECENT 12 GAME DATES (2026) ===')
for r in con2.execute('''
  SELECT game_date, COUNT(*) rows, COUNT(DISTINCT internal_player_id) players
  FROM wnba_player_box_scores WHERE game_date LIKE '2026%'
  GROUP BY game_date ORDER BY game_date DESC LIMIT 12''').fetchall():
    print(r)
print()
print('id_aliases kambi count:', con2.execute(
  "SELECT COUNT(*) FROM id_aliases WHERE source='kambi'").fetchone())
