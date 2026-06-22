import sqlite3, os

con = sqlite3.connect('/home/picks/mlb_data.db')

# Check teams table schema
teams_cols = [r[1] for r in con.execute('PRAGMA table_info(teams)').fetchall()]
print(f'Teams columns: {teams_cols}')
abbrev_col = next((c for c in teams_cols if 'abbrev' in c.lower() or 'code' in c.lower() or 'name' in c.lower()), teams_cols[1] if len(teams_cols)>1 else 'team_id')
print(f'Using column: {abbrev_col}')

print()
print('=== RECENT MC RESULTS (last 10 graded) ===')
recent = con.execute(f'''
    SELECT g.game_date,
           ht.{abbrev_col}, at.{abbrev_col},
           mcr.predicted_total, mcr.market_line,
           mcr.sim_over_prob, mcr.ensemble_over_prob,
           g.total_runs
    FROM monte_carlo_results mcr
    JOIN games g ON mcr.game_id = g.game_id
    LEFT JOIN teams ht ON g.home_team_id = ht.team_id
    LEFT JOIN teams at ON g.away_team_id = at.team_id
    WHERE g.total_runs IS NOT NULL
    ORDER BY g.game_date DESC
    LIMIT 10
''').fetchall()
if recent:
    print(f'{"Date":12} {"Match":16} {"Pred":>6} {"Mkt":>5} {"SimP":>5} {"EnsP":>5} {"Act":>4} {"Result":>8}')
    print('-'*65)
    for r in recent:
        date, home, away, pred, mkt, sim_p, ens_p, act = r
        match = f'{str(away or "?")[:6]}@{str(home or "?")[:6]}'
        # OVER bet result: predicted > market → bet OVER; win if actual > market
        bet = 'OVER' if (pred or 0) > (mkt or 99) else 'UNDER'
        if mkt and act:
            res = 'WIN' if (act > mkt and bet == 'OVER') or (act < mkt and bet == 'UNDER') else 'LOSS' if act != mkt else 'PUSH'
        else:
            res = '?'
        print(f'{date:12} {match:16} {pred or 0:+6.2f} {mkt or 0:5.1f} {sim_p or 0:5.3f} {ens_p or 0:5.3f} {act or 0:4.1f} {res:>8}')
else:
    print('No graded rows')

print()
print('=== ALERT RECORD (from alerts.db) ===')
try:
    alert_con = sqlite3.connect('/home/picks/alerts.db')
    alerts = alert_con.execute('''
        SELECT result, COUNT(*) as n,
               SUM(CASE WHEN result='WIN' THEN 1.0 ELSE 0 END) as wins,
               SUM(CASE WHEN result='LOSS' THEN 1.0 ELSE 0 END) as losses
        FROM alerts
        WHERE sport='MLB'
        AND alert_date >= date('now', '-30 days')
        AND result IN ('WIN','LOSS','PUSH')
        GROUP BY result
    ''').fetchall()
    total_a = alert_con.execute(
        "SELECT COUNT(*), SUM(CASE WHEN result IS NULL OR result NOT IN ('WIN','LOSS','PUSH') THEN 1 ELSE 0 END) FROM alerts WHERE sport='MLB' AND alert_date >= date('now','-30 days')"
    ).fetchone()
    print(f'MLB alerts last 30 days: {total_a[0]} total, {total_a[1]} pending')
    for r in alerts:
        print(f'  {r[0]}: {r[1]}')
except Exception as e:
    print(f'alerts.db query failed: {e}')

print()
print('=== SYNC_ODDS LOG (last 30 lines) ===')
log = '/home/picks/logs/sync_odds.log'
if os.path.exists(log):
    lines = open(log).readlines()
    for l in lines[-30:]:
        print(l.rstrip())
else:
    print('No sync_odds.log at', log)
