import sqlite3
con = sqlite3.connect('/home/picks/mlb_data.db')
print('=== monte_carlo_results columns ===')
cols = [r[1] for r in con.execute('PRAGMA table_info(monte_carlo_results)').fetchall()]
print(cols)
print()
print('=== games columns ===')
gcols = [r[1] for r in con.execute('PRAGMA table_info(games)').fetchall()]
print(gcols)
print()
print('=== sample MC row ===')
row = con.execute('SELECT * FROM monte_carlo_results LIMIT 1').fetchone()
if row:
    for k,v in zip(cols, row):
        print(f'  {k}: {v}')
