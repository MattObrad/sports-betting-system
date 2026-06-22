import psycopg2
con = psycopg2.connect(
    host='localhost', dbname='picksdb',
    user='picksuser', password='password'
)
cur = con.cursor()

print('=== ALL MARKET TYPES BY LEAGUE (last 14 days) ===')
cur.execute('''
SELECT g.league, p.market_type,
  COUNT(DISTINCT p.player_name) as unique_players,
  COUNT(DISTINCT p.line) as unique_lines,
  COUNT(*) as total_rows,
  MAX(p.snapshot_time) as most_recent
FROM props_snapshots p
JOIN games g ON p.event_id = g.event_id
WHERE p.snapshot_time > NOW() - INTERVAL '14 days'
GROUP BY g.league, p.market_type
ORDER BY g.league, total_rows DESC
''')
for r in cur.fetchall(): print(r)

print()
print('=== WNBA PROP TYPES DETAIL ===')
cur.execute('''
SELECT p.player_name, p.market_type, p.line,
  p.over_odds, p.snapshot_time
FROM props_snapshots p
JOIN games g ON p.event_id = g.event_id
WHERE g.league = 'WNBA'
AND p.snapshot_time > NOW() - INTERVAL '3 days'
ORDER BY p.snapshot_time DESC
LIMIT 50
''')
for r in cur.fetchall(): print(r)

print()
print('=== MLB PROP TYPES DETAIL ===')
cur.execute('''
SELECT p.player_name, p.market_type, p.line,
  p.over_odds, p.snapshot_time
FROM props_snapshots p
JOIN games g ON p.event_id = g.event_id
WHERE g.league = 'MLB'
AND p.market_type != 'Game'
AND p.snapshot_time > NOW() - INTERVAL '3 days'
ORDER BY p.market_type, p.snapshot_time DESC
LIMIT 50
''')
for r in cur.fetchall(): print(r)

print()
print('=== TENNIS PROP TYPES DETAIL ===')
cur.execute('''
SELECT p.player_name, p.market_type, p.line,
  p.over_odds, p.snapshot_time
FROM props_snapshots p
JOIN games g ON p.event_id = g.event_id
WHERE g.league IN ('ITF Women','ITF Men','Challengr')
AND p.snapshot_time > NOW() - INTERVAL '3 days'
ORDER BY p.market_type, p.snapshot_time DESC
LIMIT 50
''')
for r in cur.fetchall(): print(r)
