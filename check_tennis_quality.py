import sqlite3
con = sqlite3.connect('/home/picks/tennis.db')

print('=== TENNIS DATA FRESHNESS ===')
print('Max match date:', con.execute(
  'SELECT MAX(tourney_date) FROM matches'
).fetchone())
print('Total matches:', con.execute(
  'SELECT COUNT(*) FROM matches'
).fetchone())
print('June 2026 matches:', con.execute(
  "SELECT COUNT(*) FROM matches WHERE tourney_date LIKE '2026-06%'"
).fetchone())
print('Unresolved names:', con.execute(
  "SELECT COUNT(*) FROM matches WHERE notes='unresolved_name'"
).fetchone())

print()
print('=== ELO POOL HEALTH ===')
rows = con.execute('''
SELECT
  strftime('%Y', match_date) as year,
  COUNT(DISTINCT player_name) as unique_players,
  ROUND(AVG(overall_elo),1) as avg_elo,
  ROUND(MIN(overall_elo),1) as min_elo,
  ROUND(MAX(overall_elo),1) as max_elo
FROM player_elo
GROUP BY year
ORDER BY year DESC
LIMIT 5
''').fetchall()
for r in rows: print(r)

print()
print('=== RECENT PREDICTIONS ===')
rows = con.execute('''
SELECT prediction_date, COUNT(*) as total,
  SUM(CASE WHEN player1_model_prob >= 0.80
    OR player2_model_prob >= 0.80 THEN 1 ELSE 0 END) as above_80,
  SUM(CASE WHEN player1_edge >= 0.08
    OR player2_edge >= 0.08 THEN 1 ELSE 0 END) as edge_ok
FROM predictions
WHERE prediction_date >= date('now','-14 days')
GROUP BY prediction_date
ORDER BY prediction_date DESC
''').fetchall()
for r in rows: print(r)
