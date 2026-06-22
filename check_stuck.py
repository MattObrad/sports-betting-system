import sqlite3

print('======== ALERTS.DB current state ========')
con = sqlite3.connect('/home/picks/alerts.db')
print('--- summary by sport/result ---')
for r in con.execute('''SELECT sport,result,COUNT(*),ROUND(SUM(profit_units),2)
  FROM bet_alerts GROUP BY sport,result ORDER BY sport,result'''):
    print(r)

print()
print('--- WNBA 2026-06-20 rows (did my run inject these?) ---')
for r in con.execute('''SELECT id,player_name,line,odds,result,notified,graded,
  ROUND(profit_units,2),notified_at FROM bet_alerts
  WHERE sport='WNBA' AND alert_date='2026-06-20' ORDER BY id'''):
    print(r)

print()
print('--- still PENDING ---')
for r in con.execute('''SELECT sport,alert_date,player_name,result,notified,graded
  FROM bet_alerts WHERE result='PENDING' ORDER BY alert_date'''):
    print(r)

print()
print('======== TENNIS.DB: hunt for stuck Miroshnichenko 06-12 ========')
t = sqlite3.connect('/home/picks/tennis.db')
print('--- matches with Miroshnichenko name (any) around June ---')
for r in t.execute('''SELECT tourney_date,winner_name,loser_name,score,notes
  FROM matches WHERE (winner_name LIKE '%iroshnich%' OR loser_name LIKE '%iroshnich%')
  AND tourney_date LIKE '2026-06%' ORDER BY tourney_date'''):
    print(r)
print('--- Vagramov (opponent) around June ---')
for r in t.execute('''SELECT tourney_date,winner_name,loser_name,score
  FROM matches WHERE (winner_name LIKE '%agramov%' OR loser_name LIKE '%agramov%')
  AND tourney_date LIKE '2026-06%' ORDER BY tourney_date'''):
    print(r)

print()
print('--- the alert row in tennis.db alerts table ---')
for r in t.execute('''SELECT id,player_name,opponent_name,result,created_at
  FROM alerts WHERE player_name LIKE '%iroshnich%' '''):
    print(r)
