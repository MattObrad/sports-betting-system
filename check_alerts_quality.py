import sqlite3
con = sqlite3.connect('/home/picks/alerts.db')

print('=== FULL ALERTS SUMMARY ===')
rows = con.execute('''
SELECT sport, result, COUNT(*) as n,
  ROUND(SUM(profit_units),2) as units,
  ROUND(AVG(clv),4) as avg_clv,
  MIN(alert_date) as first,
  MAX(alert_date) as last
FROM bet_alerts
GROUP BY sport, result
ORDER BY sport, result
''').fetchall()
for r in rows: print(r)

print()
print('=== NULL RATES IN ALERTS ===')
rows = con.execute('''
SELECT sport,
  COUNT(*) as total,
  SUM(CASE WHEN odds IS NULL THEN 1 ELSE 0 END) as null_odds,
  SUM(CASE WHEN model_prob IS NULL THEN 1 ELSE 0 END) as null_prob,
  SUM(CASE WHEN predicted_value IS NULL THEN 1 ELSE 0 END) as null_pred,
  SUM(CASE WHEN clv IS NULL AND graded=1 THEN 1 ELSE 0 END) as null_clv_graded,
  SUM(CASE WHEN actual_result IS NULL AND graded=1 THEN 1 ELSE 0 END) as null_actual_graded
FROM bet_alerts GROUP BY sport
''').fetchall()
for r in rows: print(r)

print()
print('=== STUCK BETS (>3 days ungraded) ===')
rows = con.execute('''
SELECT sport, alert_date, player_name, result
FROM bet_alerts
WHERE graded=0 AND result='PENDING'
AND alert_date < date('now', '-3 days')
ORDER BY alert_date
''').fetchall()
for r in rows: print(r)
