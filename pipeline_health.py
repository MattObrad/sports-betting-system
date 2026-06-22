import sqlite3, psycopg2, json, os
from datetime import datetime, timedelta, timezone

pg = psycopg2.connect(host='localhost', dbname='picksdb',
    user='picksuser', password='password')
cur = pg.cursor()
mlb = sqlite3.connect('/home/picks/mlb_data.db')
sports = sqlite3.connect('/home/picks/sports.db')
tennis = sqlite3.connect('/home/picks/tennis.db')
alerts = sqlite3.connect('/home/picks/alerts.db')
now = datetime.now(timezone.utc)

print('=== PIPELINE HEALTH CHECK ===')
print(f'Run time: {now}')
print()

# 1. KAMBI COLLECTOR — is it still collecting?
print('--- KAMBI COLLECTION ---')
cur.execute('''
    SELECT league, MAX(snapshot_time) as last_snapshot,
    COUNT(*) as snapshots_last_hour
    FROM props_snapshots p
    JOIN games g ON p.event_id = g.event_id
    WHERE p.snapshot_time > NOW() - INTERVAL '2 hours'
    GROUP BY league ORDER BY last_snapshot DESC
''')
rows = cur.fetchall()
if rows:
    print(f'Active leagues in last 2 hours: {len(rows)}')
    for r in rows[:10]: print(f'  {r[0]}: last={r[1]}, count={r[2]}')
else:
    print('WARNING: No snapshots in last 2 hours — collector may be down')

# 2. MLB PIPELINE — all stages
print()
print('--- MLB PIPELINE ---')

# Games table freshness
last_game = mlb.execute(
    'SELECT MAX(game_date) FROM games'
).fetchone()[0]
print(f'Games table max date: {last_game}')
if last_game < (now - timedelta(days=2)).strftime('%Y-%m-%d'):
    print('WARNING: Games table stale — collect_statcast.py may be failing')

# Features freshness
last_features = mlb.execute('''
    SELECT MAX(g.game_date)
    FROM game_features gf JOIN games g ON gf.game_id=g.game_id
''').fetchone()[0]
print(f'Features max date: {last_features}')

# NULL rates in recent features
null_check = mlb.execute('''
    SELECT
        COUNT(*) as total,
        ROUND(100.0*SUM(CASE WHEN line_movement IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_line_movement,
        ROUND(100.0*SUM(CASE WHEN current_total IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_current,
        ROUND(100.0*SUM(CASE WHEN ump_career_over_rate IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_ump,
        ROUND(100.0*SUM(CASE WHEN home_sp_era_l5 IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_sp_era,
        ROUND(100.0*SUM(CASE WHEN wind_to_cf IS NULL THEN 1 ELSE 0 END)/COUNT(*),1) as pct_null_weather
    FROM game_features gf
    JOIN games g ON gf.game_id=g.game_id
    WHERE g.game_date >= date('now','-14 days')
''').fetchone()
print(f'Features NULL rates (last 14 days):')
print(f'  Total games: {null_check[0]}')
print(f'  line_movement NULL: {null_check[1]}%  (should be <40% ideally)')
print(f'  current_total NULL: {null_check[2]}%')
print(f'  ump NULL: {null_check[3]}%  (known issue — 97% NULL)')
print(f'  sp_era NULL: {null_check[4]}%  (should be <25%)')
print(f'  weather NULL: {null_check[5]}%  (should be <35%)')

# sync_odds health — check recent log
sync_log = '/home/picks/logs/sync_odds.log'
if os.path.exists(sync_log):
    lines = open(sync_log).readlines()[-20:]
    matched = [l for l in lines if 'matched' in l]
    if matched:
        print(f'sync_odds last result: {matched[-1].strip()}')
    else:
        print('WARNING: Cannot find match count in sync_odds.log')
else:
    print('WARNING: sync_odds.log not found')

# Predictions freshness
last_pred = mlb.execute(
    'SELECT MAX(run_date) FROM monte_carlo_results'
).fetchone()[0]
print(f'Last MLB prediction run: {last_pred}')
if last_pred < (now - timedelta(days=2)).strftime('%Y-%m-%d'):
    print('WARNING: Predictions stale — predict_mlb.py may be failing')

# OVER bias check
over_check = mlb.execute('''
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN predicted_total > market_line THEN 1 ELSE 0 END) as over_predictions,
        ROUND(AVG(predicted_total - market_line),3) as avg_signed_edge
    FROM monte_carlo_results
    WHERE run_date >= date('now','-14 days')
    AND predicted_total IS NOT NULL
    AND market_line IS NOT NULL
''').fetchone()
if over_check[0] > 0:
    over_pct = over_check[1]/over_check[0]*100
    print(f'OVER bias check (last 14 days): {over_pct:.0f}% OVER, avg edge {over_check[2]:+.3f}')
    if over_pct > 80:
        print('WARNING: Severe OVER bias detected — model still tilted')

print()
print('--- WNBA PIPELINE ---')

# Box scores freshness
last_bs = sports.execute(
    'SELECT MAX(game_date) FROM wnba_player_box_scores'
).fetchone()[0]
print(f'Box scores max date: {last_bs}')
lag = (now.date() - datetime.strptime(last_bs, '%Y-%m-%d').date()).days
print(f'Box score lag: {lag} days')
if lag > 7:
    print('WARNING: Box scores >7 days stale — wehoop refresh may have failed')

# 2026 player coverage — season col may not exist, use game_date
tables = [r[0] for r in sports.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
bs_cols = [r[1] for r in sports.execute("PRAGMA table_info(wnba_player_box_scores)").fetchall()]
if 'season' in bs_cols:
    coverage = sports.execute('''
        SELECT COUNT(DISTINCT athlete_id) as players,
        COUNT(DISTINCT game_date) as game_days
        FROM wnba_player_box_scores WHERE season=2026
    ''').fetchone()
else:
    coverage = sports.execute('''
        SELECT COUNT(DISTINCT athlete_id) as players,
        COUNT(DISTINCT game_date) as game_days
        FROM wnba_player_box_scores WHERE game_date LIKE '2026%'
    ''').fetchone()
print(f'2026 coverage: {coverage[0]} players, {coverage[1]} game days')

# Kambi WNBA props freshness
cur.execute('''
    SELECT MAX(p.snapshot_time),
    COUNT(DISTINCT p.player_name) as players_today
    FROM props_snapshots p
    JOIN games g ON p.event_id=g.event_id
    WHERE g.league='WNBA'
    AND p.snapshot_time > NOW() - INTERVAL '24 hours'
''')
wnba_props = cur.fetchone()
print(f'WNBA props last 24h: last={wnba_props[0]}, unique players={wnba_props[1]}')

# Alias coverage
cur.execute('''
    SELECT COUNT(DISTINCT p.player_name) as kambi_players
    FROM props_snapshots p
    JOIN games g ON p.event_id=g.event_id
    WHERE g.league='WNBA'
    AND p.snapshot_time > NOW() - INTERVAL '7 days'
''')
kambi_players = cur.fetchone()[0]
print(f'Kambi WNBA players (last 7 days): {kambi_players}')

alias_count = sports.execute(
    "SELECT COUNT(*) FROM id_aliases WHERE source='kambi'"
).fetchone()[0] if 'id_aliases' in tables else 0
print(f'Player aliases in sports.db: {alias_count}')
coverage_pct = (alias_count/kambi_players*100) if kambi_players else 0
print(f'Alias coverage: {coverage_pct:.0f}% ({alias_count}/{kambi_players})')
if coverage_pct < 80:
    print('WARNING: Low alias coverage — many WNBA props being skipped')

print()
print('--- TENNIS PIPELINE ---')

# Sackmann freshness
last_match = tennis.execute(
    'SELECT MAX(tourney_date) FROM matches'
).fetchone()[0]
print(f'Matches max date: {last_match}')
lag_t = (now.date() - datetime.strptime(last_match, '%Y-%m-%d').date()).days
print(f'Match data lag: {lag_t} days (should be <=2 with scraper)')
if lag_t > 3:
    print('WARNING: Tennis data >3 days stale — scraper may be failing')

# Elo health
elo_check = tennis.execute('''
    SELECT COUNT(DISTINCT player_name),
    ROUND(AVG(overall_elo),1) as avg_elo
    FROM player_elo
    WHERE match_date = (SELECT MAX(match_date) FROM player_elo)
''').fetchone()
print(f'Active Elo pool: {elo_check[0]} players, avg={elo_check[1]}')
if abs(elo_check[1] - 1500) > 100:
    print(f'WARNING: Elo pool mean {elo_check[1]} far from 1500 — possible inflation')

# Unresolved names
unresolved = tennis.execute(
    "SELECT COUNT(*) FROM matches WHERE notes='unresolved_name'"
).fetchone()[0]
print(f'Unresolved player names: {unresolved}')

# Prediction freshness
last_tennis_pred = tennis.execute(
    'SELECT MAX(prediction_date) FROM predictions'
).fetchone()[0]
print(f'Last tennis prediction: {last_tennis_pred}')
if last_tennis_pred and last_tennis_pred < (now - timedelta(days=2)).strftime('%Y-%m-%d'):
    print('WARNING: Tennis predictions stale')

# Kambi tennis props freshness
cur.execute('''
    SELECT MAX(p.snapshot_time),
    COUNT(DISTINCT g.event_id) as matches_today
    FROM props_snapshots p
    JOIN games g ON p.event_id=g.event_id
    WHERE g.league='ITF Women'
    AND p.snapshot_time > NOW() - INTERVAL '24 hours'
''')
tennis_props = cur.fetchone()
print(f'ITF Women props last 24h: last={tennis_props[0]}, matches={tennis_props[1]}')

print()
print('--- ALERTS DB HEALTH ---')
stuck = alerts.execute('''
    SELECT sport, COUNT(*) as stuck
    FROM bet_alerts
    WHERE graded=0 AND result='PENDING'
    AND alert_date < date('now','-3 days')
    GROUP BY sport
''').fetchall()
if stuck:
    print('WARNING: Stuck bets found:')
    for s in stuck: print(f'  {s[0]}: {s[1]} bets stuck >3 days')
else:
    print('No stuck bets found')

recent = alerts.execute('''
    SELECT sport, MAX(alert_date) as last_alert,
    COUNT(*) as total,
    SUM(CASE WHEN graded=1 THEN 1 ELSE 0 END) as graded
    FROM bet_alerts GROUP BY sport
''').fetchall()
print('Alert summary:')
for r in recent: print(f'  {r[0]}: last={r[1]}, total={r[2]}, graded={r[3]}')

print()
print('--- LOG FILE HEALTH ---')
logs = [
    'predict_mlb.log',
    'predict_tennis.log',
    'sync_odds.log',
    'scrape_itf.log',
    'scrape_itf_afternoon.log',
    'grade_tennis_results.log',
    'grade_wnba.log',
    'wehoop_refresh.log',
    'watchdog.log',
    'mlb_statcast.log',
]
for log in logs:
    path = f'/home/picks/logs/{log}'
    if os.path.exists(path):
        mtime = datetime.fromtimestamp(
            os.path.getmtime(path), tz=timezone.utc)
        age_hours = (now - mtime).total_seconds() / 3600
        status = 'OK' if age_hours < 26 else f'STALE ({age_hours:.0f}h old)'
        size = os.path.getsize(path)
        print(f'  {log}: {status}, {size/1024:.1f}KB')
    else:
        print(f'  {log}: MISSING')

print()
print('=== HEALTH CHECK COMPLETE ===')
print('Fix any WARNINGs above before running backtests.')
print('A clean backtest on a broken pipeline is meaningless.')
