import sqlite3
con = sqlite3.connect('/home/picks/sports.db')

print('=== WNBA DATA FRESHNESS ===')
print('Max game date:', con.execute(
  'SELECT MAX(game_date) FROM wnba_player_box_scores'
).fetchone())
print('2026 rows:', con.execute(
  'SELECT COUNT(*) FROM wnba_player_box_scores WHERE season=2026'
).fetchone())
print('Teams in 2026:', con.execute(
  'SELECT COUNT(DISTINCT team_id) FROM wnba_player_box_scores WHERE season=2026'
).fetchone())
print('Players in 2026:', con.execute(
  'SELECT COUNT(DISTINCT athlete_id) FROM wnba_player_box_scores WHERE season=2026'
).fetchone())

print()
print('=== MOST RECENT GAME DATES ===')
rows = con.execute('''
SELECT game_date, COUNT(*) as player_rows
FROM wnba_player_box_scores
WHERE season=2026
GROUP BY game_date
ORDER BY game_date DESC
LIMIT 10
''').fetchall()
for r in rows: print(r)
