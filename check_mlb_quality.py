import sqlite3
con = sqlite3.connect('/home/picks/mlb_data.db')

print('=== NULL RATES IN FEATURES (last 30 days) ===')
result = con.execute('''
SELECT
  COUNT(*) as total_games,
  ROUND(100.0*SUM(CASE WHEN line_movement IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_line_movement,
  ROUND(100.0*SUM(CASE WHEN current_total IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_current_total,
  ROUND(100.0*SUM(CASE WHEN home_sp_era_l5 IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_sp_era,
  ROUND(100.0*SUM(CASE WHEN ump_career_over_rate IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_ump,
  ROUND(100.0*SUM(CASE WHEN wind_to_cf IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_weather
FROM game_features gf
JOIN games g ON gf.game_id = g.game_id
WHERE g.game_date >= date('now', '-30 days')
''').fetchone()
print(result)

print()
print('=== RECENT PREDICTIONS (last 14 days) ===')
rows = con.execute('''
SELECT run_date, COUNT(*) as games_processed,
  SUM(CASE WHEN over_prob > 0.55 THEN 1 ELSE 0 END) as potential_edges,
  COUNT(DISTINCT CASE WHEN e.game_id IS NOT NULL THEN mcr.game_id END) as actual_alerts
FROM monte_carlo_results mcr
LEFT JOIN edge_bets e ON mcr.game_id = e.game_id
  AND mcr.run_date = e.run_date
WHERE mcr.run_date >= date('now', '-14 days')
GROUP BY mcr.run_date ORDER BY mcr.run_date DESC
''').fetchall()
for r in rows: print(r)

print()
print('=== PITCHER STALENESS (last 30 days) ===')
result = con.execute('''
SELECT
  COUNT(*) as total_games,
  SUM(CASE WHEN home_sp_days_rest > 60 THEN 1 ELSE 0 END) as stale_home_pitcher,
  SUM(CASE WHEN away_sp_days_rest > 60 THEN 1 ELSE 0 END) as stale_away_pitcher,
  ROUND(AVG(home_sp_era_l5),2) as avg_home_era,
  ROUND(AVG(away_sp_era_l5),2) as avg_away_era
FROM game_features gf
JOIN games g ON gf.game_id = g.game_id
WHERE g.game_date >= date('now', '-30 days')
''').fetchone()
print(result)

print()
print('=== EDGE BET HISTORY ===')
rows = con.execute('''
SELECT run_date, game_id, direction,
  market_line, predicted_total, over_prob,
  ROUND(predicted_total - market_line, 2) as run_edge
FROM edge_bets
ORDER BY run_date DESC
LIMIT 20
''').fetchall()
for r in rows: print(r)
