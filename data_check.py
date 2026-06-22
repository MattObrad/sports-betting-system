import sqlite3, psycopg2

mlb = sqlite3.connect('/home/picks/mlb_data.db')
sports = sqlite3.connect('/home/picks/sports.db')
tennis = sqlite3.connect('/home/picks/tennis.db')
pg = psycopg2.connect(host='localhost', dbname='picksdb',
    user='picksuser', password='password')
cur = pg.cursor()

print('=== MLB ===')
print('Graded games (have final score):')
print(mlb.execute('''
    SELECT MIN(game_date), MAX(game_date), COUNT(*)
    FROM games WHERE home_score IS NOT NULL
    AND away_score IS NOT NULL
''').fetchone())

print('Games with features:')
print(mlb.execute('''
    SELECT MIN(g.game_date), MAX(g.game_date), COUNT(*)
    FROM game_features gf
    JOIN games g ON gf.game_id=g.game_id
    WHERE g.home_score IS NOT NULL
''').fetchone())

print('MC results with predictions AND final scores:')
print(mlb.execute('''
    SELECT MIN(mcr.run_date), MAX(mcr.run_date), COUNT(*)
    FROM monte_carlo_results mcr
    JOIN games g ON mcr.game_id = g.game_id
    WHERE mcr.predicted_total IS NOT NULL
    AND mcr.market_line IS NOT NULL
    AND g.home_score IS NOT NULL
    AND g.away_score IS NOT NULL
''').fetchone())

print('MC results: NULL market breakdown')
print(mlb.execute('''
    SELECT
        SUM(CASE WHEN gf.line_movement IS NULL AND gf.current_total IS NULL THEN 1 ELSE 0 END) as null_both,
        SUM(CASE WHEN gf.line_movement IS NOT NULL OR gf.current_total IS NOT NULL THEN 1 ELSE 0 END) as has_market,
        COUNT(*) as total
    FROM monte_carlo_results mcr
    JOIN games g ON mcr.game_id = g.game_id
    JOIN game_features gf ON mcr.game_id = gf.game_id
    WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    AND mcr.predicted_total IS NOT NULL AND mcr.market_line IS NOT NULL
''').fetchone())

print('Direction split (OVER/UNDER) across ALL gradeable MC results:')
print(mlb.execute('''
    SELECT
        SUM(CASE WHEN mcr.predicted_total > mcr.market_line THEN 1 ELSE 0 END) as over_pred,
        SUM(CASE WHEN mcr.predicted_total < mcr.market_line THEN 1 ELSE 0 END) as under_pred,
        ROUND(AVG(mcr.predicted_total - mcr.market_line),3) as avg_edge
    FROM monte_carlo_results mcr
    JOIN games g ON mcr.game_id = g.game_id
    WHERE g.home_score IS NOT NULL AND mcr.market_line IS NOT NULL
''').fetchone())

print()
print('=== WNBA ===')
print('Box scores:')
print(sports.execute('''
    SELECT MIN(game_date), MAX(game_date), COUNT(*)
    FROM wnba_player_box_scores
''').fetchone())

print('2026 season (game_date filter):')
print(sports.execute('''
    SELECT MIN(game_date), MAX(game_date),
    COUNT(DISTINCT internal_player_id), COUNT(*)
    FROM wnba_player_box_scores WHERE game_date LIKE '2026%'
''').fetchone())

print('Games in 2026 with >=5 players (actual game days):')
print(sports.execute('''
    SELECT COUNT(DISTINCT game_date)
    FROM wnba_player_box_scores
    WHERE game_date LIKE '2026%' AND did_not_play = 0
''').fetchone())

print()
print('=== WNBA KAMBI PROPS (Postgres) ===')
cur.execute('''
    SELECT MIN(p.snapshot_time), MAX(p.snapshot_time),
    COUNT(DISTINCT g.event_id) as events,
    COUNT(DISTINCT p.player_name) as players,
    COUNT(*) as total_snapshots
    FROM props_snapshots p
    JOIN games g ON p.event_id=g.event_id
    WHERE g.league='WNBA'
    AND p.bet_offer_type IN ('Player Points','Player Rebounds','Player Assists')
''')
print(cur.fetchone())

cur.execute('''
    SELECT g.game_time::date as game_date,
    COUNT(DISTINCT p.player_name) as players,
    COUNT(*) as snapshots
    FROM props_snapshots p
    JOIN games g ON p.event_id=g.event_id
    WHERE g.league='WNBA'
    AND p.bet_offer_type='Player Points'
    GROUP BY g.game_time::date
    ORDER BY g.game_time::date DESC LIMIT 20
''')
print('WNBA Player Points by game date (last 20):')
for r in cur.fetchall(): print(f'  {r[0]}: {r[1]} players, {r[2]} snapshots')

print()
print('=== TENNIS ===')
print('Matches with results (non-walkover):')
print(tennis.execute('''
    SELECT MIN(tourney_date), MAX(tourney_date), COUNT(*)
    FROM matches WHERE score IS NOT NULL
    AND score NOT LIKE '%W/O%' AND score NOT LIKE '%RET%'
''').fetchone())

print('Predictions stored (all time):')
print(tennis.execute('''
    SELECT MIN(prediction_date), MAX(prediction_date), COUNT(*)
    FROM predictions
    WHERE player1_model_prob IS NOT NULL
''').fetchone())

print('Predictions with qualifying threshold (>=0.80):')
print(tennis.execute('''
    SELECT MIN(prediction_date), MAX(prediction_date), COUNT(*)
    FROM predictions
    WHERE player1_model_prob >= 0.80 OR player2_model_prob >= 0.80
''').fetchone())

print('Predictions: how many can be matched to results?')
# Check overlap between prediction dates and match dates
print(tennis.execute('''
    SELECT COUNT(DISTINCT p.prediction_date) as pred_days,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN p.player1_model_prob >= 0.80 OR p.player2_model_prob >= 0.80 THEN 1 ELSE 0 END) as above_80
    FROM predictions p
''').fetchone())

print('Elo coverage:')
print(tennis.execute('''
    SELECT COUNT(DISTINCT player_name),
    MIN(match_date), MAX(match_date)
    FROM player_elo
''').fetchone())

print('Alerts in tennis.db (own grading table):')
tennis_tables = [r[0] for r in tennis.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('Tables:', tennis_tables)
if 'alerts' in tennis_tables:
    print(tennis.execute('''
        SELECT MIN(created_at), MAX(created_at), COUNT(*),
        SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) as losses,
        SUM(CASE WHEN result='pending' THEN 1 ELSE 0 END) as pending
        FROM alerts
    ''').fetchone())
