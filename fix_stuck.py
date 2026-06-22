import sqlite3
from datetime import datetime, timezone

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
a = sqlite3.connect('/home/picks/alerts.db')

print('=== BEFORE ===')
for r in a.execute("SELECT id,sport,player_name,result,profit_units,notified FROM bet_alerts WHERE id IN (108,109,110) OR (sport='TENNIS' AND result='PENDING')"):
    print(r)

# 1) Remove the 3 injected WNBA rows (notified=0, never placed; created by audit re-run of --date 2026-06-20)
a.execute("DELETE FROM bet_alerts WHERE id IN (108,109,110) AND notified=0 AND notified_at IS NULL")
print('Deleted injected WNBA rows:', a.total_changes)

# 2) Grade the stuck Miroshnichenko tennis bet: confirmed LOSS
#    tennis.db matches: 2026-06-12 Alexandra Vagramov beat Veronica Miroshnichenko 6-1 6-4
#    (grader missed it: Kambi 'Veronika' vs Sackmann 'Veronica' — k != c, not an accent)
cur = a.execute("""UPDATE bet_alerts SET result='LOSS', actual_result=0.0,
    profit_units=-1.0, graded=1, graded_at=?
    WHERE sport='TENNIS' AND result='PENDING' AND player_name LIKE '%iroshnich%'""", (now,))
print('Graded Miroshnichenko LOSS, rows:', cur.rowcount)
a.commit()

print()
print('=== AFTER: alerts summary ===')
for r in a.execute("SELECT sport,result,COUNT(*),ROUND(SUM(profit_units),2) FROM bet_alerts GROUP BY sport,result ORDER BY sport,result"):
    print(r)
print()
print('=== remaining PENDING ===')
for r in a.execute("SELECT sport,alert_date,player_name FROM bet_alerts WHERE result='PENDING'"):
    print(r)
a.close()

# 3) update tennis.db internal alerts table for consistency
t = sqlite3.connect('/home/picks/tennis.db')
t.execute("UPDATE alerts SET result='loss' WHERE player_name LIKE '%iroshnich%' AND result='pending'")
t.commit()
print('\ntennis.db alerts row updated.')

# 4) Investigate Kelsey Plum 06-17
print('\n=== Kelsey Plum investigation (sports.db box scores) ===')
s = sqlite3.connect('/home/picks/sports.db')
for r in s.execute("""SELECT bs.game_date, bs.points, bs.minutes, bs.did_not_play
   FROM wnba_player_box_scores bs JOIN id_aliases ia ON bs.internal_player_id=ia.internal_id
   WHERE ia.source='kambi' AND ia.source_id LIKE '%Plum%'
   ORDER BY bs.game_date DESC LIMIT 6"""):
    print(r)
print("Plum alias rows:", s.execute("SELECT source_id,internal_id FROM id_aliases WHERE source='kambi' AND source_id LIKE '%Plum%'").fetchall())
s.close()
