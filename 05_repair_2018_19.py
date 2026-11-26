"""Repair missing Premier League 2018/19 matches in predictor.db.
Downloads only 2018/19 (football-data.co.uk E0.csv), validates the CSV,
creates a SQLite backup, and inserts missing matches without duplicates.
Run from the predictor project folder: py 05_repair_2018_19.py
"""
import csv
import io
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime



DB_FILE = 'predictor.db'
SEASON = '2018/19'
CSV_URL = 'https://www.football-data.co.uk/mmz4281/1819/E0.csv'
EXPECTED_MATCHES = 380



def download_csv():
    try:
        import requests
    except ImportError:
        print('Missing requests. Install it first: py -m pip install requests')
        sys.exit(1)

    headers = {'User-Agent': 'Mozilla/5.0 (compatible; PredictorDataRepair/1.0)'}
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(CSV_URL, headers=headers, timeout=30)
            response.raise_for_status()
            # football-data CSV is usually UTF-8/ASCII, but tolerate BOM.
            text = response.content.decode('utf-8-sig', errors='replace')
            if 'HomeTeam' not in text or 'FTHG' not in text:
                raise ValueError('Downloaded content does not look like the expected football CSV.')
            return text
        except Exception as exc:
            last_error = exc
            print(f'Download attempt {attempt + 1}/3 failed: {exc}')
            if attempt < 2:
                time.sleep(2)
    print(f'Could not download 2018/19 data: {last_error}')
    sys.exit(1)


def parse_csv(text):
    reader = csv.DictReader(io.StringIO(text))
    required = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'}
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        print('CSV is missing required columns:', sorted(required - set(reader.fieldnames or [])))
        sys.exit(1)

    parsed = []
    for row in reader:
        if not row.get('HomeTeam') or not row.get('AwayTeam'):
            continue
        try:
            hg, ag = int(row['FTHG']), int(row['FTAG'])
            result = row['FTR'].strip().upper()
            if result not in {'H', 'D', 'A'}:
                continue
            date_raw = row['Date'].strip()
            date_value = None
            for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d'):
                try:
                    date_value = datetime.strptime(date_raw, fmt).strftime('%Y-%m-%d')
                    break
                except ValueError:
                    pass
            if date_value is None:
                continue
            parsed.append((date_value, SEASON, row['HomeTeam'].strip(), row['AwayTeam'].strip(), hg, ag, result))
        except (ValueError, KeyError, TypeError):
            continue

    # CSV should contain one row per fixture; guard against duplicates within source.
    unique = {}
    for match in parsed:
        key = (match[0], match[2], match[3])
        unique[key] = match
    matches = list(unique.values())
    matches.sort(key=lambda x: (x[0], x[2], x[3]))
    if len(matches) != EXPECTED_MATCHES:
        print(f'Validation failed: expected {EXPECTED_MATCHES} unique fixtures, found {len(matches)}.')
        print('No database changes were made. Check the downloaded CSV before retrying.')
        sys.exit(1)
    return matches


def main():
    if not os.path.isfile(DB_FILE):
        print(f'Database not found: {DB_FILE}. Run this script from your predictor project folder.')
        sys.exit(1)


    # Back up the database before any insert attempt.
    backup = DB_FILE + '.backup_before_2018_19_repair'
    if not os.path.exists(backup):
        shutil.copy2(DB_FILE, backup)
        print(f'Backup created: {backup}')
    else:
        print(f'Backup already exists; keeping it: {backup}')


    matches = parse_csv(download_csv())
    print(f'Validated {len(matches)} fixtures for {SEASON}.')


    con = sqlite3.connect(DB_FILE)
    try:
        con.execute('BEGIN')
        season_count = con.execute('SELECT COUNT(*) FROM matches WHERE season = ?', (SEASON,)).fetchone()[0]
        if season_count >= EXPECTED_MATCHES:
            print(f'Database already has {season_count} matches for {SEASON}. Nothing to insert.')
            con.rollback()
            return


        existing = set(con.execute(
            'SELECT date, home_team, away_team FROM matches WHERE season = ?', (SEASON,)
        ).fetchall())
        missing = [m for m in matches if (m[0], m[2], m[3]) not in existing]

        inserted = 0
        for match in missing:
            con.execute('''
                INSERT INTO matches (date, season, home_team, away_team, home_goals, away_goals, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', match)
            inserted += 1

        final_count = con.execute('SELECT COUNT(*) FROM matches WHERE season = ?', (SEASON,)).fetchone()[0]
        if final_count != EXPECTED_MATCHES:
            raise RuntimeError(f'After insert, season count is {final_count}, not {EXPECTED_MATCHES}. Rolling back.')
        con.commit()
        print(f'Inserted {inserted} missing matches. Database now has {final_count} matches for {SEASON}.')
        print('Repair completed successfully. Run py 04_check_seasons.py to verify all seasons.')
    except Exception as exc:
        con.rollback()
        print('Repair failed; database transaction was rolled back.')
        print('Error:', exc)
        print('Your backup is:', backup)
        sys.exit(1)
    finally:
        con.close()


if __name__ == '__main__':
    main()
