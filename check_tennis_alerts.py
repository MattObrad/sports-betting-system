import sqlite3
con = sqlite3.connect('/home/picks/alerts.db')
rows = con.execute('''
SELECT alert_date, player_name, odds, result,
profit_units, clv, graded
FROM bet_alerts WHERE sport='TENNIS'
ORDER BY alert_date
''').fetchall()
wins = sum(1 for r in rows if r[3]=='WIN')
losses = sum(1 for r in rows if r[3]=='LOSS')
pending = sum(1 for r in rows if r[3]=='PENDING')
profit = sum(r[4] for r in rows if r[4])
print(f'TENNIS: {wins}W-{losses}L-{pending}P | {profit:+.2f}u')
print()
for r in rows: print(r)
