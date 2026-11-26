import sqlite3

DB_FILE = 'predictor.db'

with sqlite3.connect(DB_FILE) as con:
    rows = con.execute('''
        SELECT season, COUNT(*) AS matches,
               SUM(CASE WHEN home_goals IS NOT NULL AND away_goals IS NOT NULL THEN 1 ELSE 0 END) AS finished
        FROM matches
        GROUP BY season
        ORDER BY season
    ''').fetchall()


print('Premier League seasons stored in predictor.db')
print('-' * 50)
print(f'{"Season":<12}{"Matches":>10}{"Finished":>10}')
print('-' * 50)
for season, matches, finished in rows:
    print(f'{season:<12}{matches:>10}{finished:>10}')
print('-' * 50)
print('Total seasons:', len(rows))
print('Total matches:', sum(r[1] for r in rows))

# Highlight expected historical seasons that may be missing or incomplete.
expected = [f'{y}/{str(y+1)[-2:]}' for y in range(2016, 2027)]
existing = {r[0] for r in rows}
print('\nMissing season labels (if any):')
missing = [s for s in expected if s not in existing]
print(missing if missing else 'None detected by these labels')
