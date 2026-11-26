from pathlib import Path
import re

path = Path(__file__).resolve().parent / 'app.py'
source = path.read_text(encoding='utf-8')
original = source

# Add Bundesliga model path and league metadata, preserving existing mappings.
if 'BUNDESLIGA_MODEL_PATH' not in source:
    source = source.replace(
        'LALIGA_MODEL_PATH = BASE_DIR.parent / "models" / "laliga_model.joblib"',
        'LALIGA_MODEL_PATH = BASE_DIR.parent / "models" / "laliga_model.joblib"\nBUNDESLIGA_MODEL_PATH = BASE_DIR / "models" / "bundesliga_model.joblib'
    )
if '"BL": {"name": "Bundesliga"' not in source:
    source = source.replace(
        'LEAGUES = {"PL": {"name": "Premier League", "country": "England"}, "PD": {"name": "LaLiga", "country": "Spain"}}',
        'LEAGUES = {"PL": {"name": "Premier League", "country": "England"}, "PD": {"name": "LaLiga", "country": "Spain"}, "BL": {"name": "Bundesliga", "country": "Germany"}}'
    )

# Replace get_bundle only.
new_get_bundle = '''def get_bundle(league="PL"):
    code = league.upper()
    paths = {"PL": EPL_MODEL_PATH, "PD": LALIGA_MODEL_PATH, "BL": BUNDESLIGA_MODEL_PATH}
    path = paths.get(code)
    if path is None:
        raise HTTPException(status_code=400, detail="Unsupported league. Use PL, PD or BL.")
    if code not in _bundles:
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"{code} model file not found: {path}")
        try:
            _bundles[code] = joblib.load(path)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not load {code} model: {exc}")
    return _bundles[code]
'''
source, n = re.subn(r'def get_bundle\(league="PL"\):\n.*?\n\n\ndef _fetch_fixtures', new_get_bundle + '\n\n\ndef _fetch_fixtures', source, flags=re.S)
if n != 1:
    raise RuntimeError('Could not safely patch get_bundle; no app.py was changed.')

# Add Bundesliga model-info metadata branch.
old = '''    if league.upper() == "PD":
        info = {"model_name": "LaLiga Random Forest", "league": "LaLiga", "matches_used": bundle.get("matches_used"), "train_matches": bundle.get("train_matches"), "test_matches": bundle.get("test_matches"), "trained_at": bundle.get("trained_at"), "features": bundle.get("features", [])}
    return info'''
new = '''    if league.upper() == "PD":
        info = {"model_name": "LaLiga Random Forest", "league": "LaLiga", "matches_used": bundle.get("matches_used"), "train_matches": bundle.get("train_matches"), "test_matches": bundle.get("test_matches"), "trained_at": bundle.get("trained_at"), "features": bundle.get("features", [])}
    elif league.upper() == "BL":
        info = bundle.get("info", {}) or {"model_name": "Bundesliga Random Forest", "league": "Bundesliga", "matches_used": bundle.get("matches_used"), "train_matches": bundle.get("train_matches"), "test_matches": bundle.get("test_matches"), "trained_at": bundle.get("trained_at"), "features": bundle.get("features", [])}
    return info'''
if old not in source:
    raise RuntimeError('Could not safely patch model-info; no app.py was changed.')
source = source.replace(old, new, 1)

# Add Bundesliga team list branch.
old = '''    if code == "PL":
        names = sorted(bundle["state"].played.keys())
    else:
        from core import laliga_teams
        try: names = laliga_teams()
        except Exception as exc: raise HTTPException(status_code=503, detail=f"Could not load LaLiga team history: {exc}")'''
new = '''    if code == "PL":
        names = sorted(bundle["state"].played.keys())
    elif code == "PD":
        from core import laliga_teams
        try: names = laliga_teams()
        except Exception as exc: raise HTTPException(status_code=503, detail=f"Could not load LaLiga team history: {exc}")
    else:
        from core import bundesliga_teams
        try: names = bundesliga_teams()
        except Exception as exc: raise HTTPException(status_code=503, detail=f"Could not load Bundesliga team history: {exc}")'''
if old not in source:
    raise RuntimeError('Could not safely patch teams endpoint; no app.py was changed.')
source = source.replace(old, new, 1)

# Add Bundesliga prediction branch while retaining existing PL and PD behavior.
old = '''        else:
            from core import predict_laliga_match
            probabilities = predict_laliga_match(bundle, home, away, request.match_date.isoformat())'''
new = '''        elif code == "PD":
            from core import predict_laliga_match
            probabilities = predict_laliga_match(bundle, home, away, request.match_date.isoformat())
        else:
            from core import bundesliga_teams, predict_bundesliga_match
            known_names = {name.casefold(): name for name in bundesliga_teams()}
            if home.casefold() not in known_names:
                raise HTTPException(status_code=404, detail=f"Unknown home team: {home}")
            if away.casefold() not in known_names:
                raise HTTPException(status_code=404, detail=f"Unknown away team: {away}")
            home, away = known_names[home.casefold()], known_names[away.casefold()]
            probabilities = predict_bundesliga_match(bundle, home, away, request.match_date.isoformat())'''
if old not in source:
    raise RuntimeError('Could not safely patch predict endpoint; no app.py was changed.')
source = source.replace(old, new, 1)

# Fixture team-name source and prediction routing.
old = '''    known_names = bundle["state"].played.keys() if code == "PL" else __import__("core").laliga_teams()'''
new = '''    if code == "PL":
        known_names = bundle["state"].played.keys()
    elif code == "PD":
        known_names = __import__("core").laliga_teams()
    else:
        known_names = __import__("core").bundesliga_teams()'''
if old not in source:
    raise RuntimeError('Could not safely patch fixture team source; no app.py was changed.')
source = source.replace(old, new, 1)
old = '''                else:
                    from core import predict_laliga_match
                    result = predict_laliga_match(bundle, h_model, a_model, match_date.isoformat())'''
new = '''                elif code == "PD":
                    from core import predict_laliga_match
                    result = predict_laliga_match(bundle, h_model, a_model, match_date.isoformat())
                else:
                    from core import predict_bundesliga_match
                    result = predict_bundesliga_match(bundle, h_model, a_model, match_date.isoformat())'''
if old not in source:
    raise RuntimeError('Could not safely patch fixture prediction branch; no app.py was changed.')
source = source.replace(old, new, 1)
source = source.replace('Available leagues: PL, PD.', 'Available leagues: PL, PD, BL.')

if source == original:
    raise RuntimeError('No changes made; app.py may already include Bundesliga or differ from expected version.')

# Write backup first, then updated file.
backup = path.with_name('app.py.backup_before_bundesliga')
if not backup.exists():
    backup.write_text(original, encoding='utf-8')
path.write_text(source, encoding='utf-8')
print('Bundesliga support patch applied to app.py.')
print(f'Backup created/kept at: {backup}')
print('Next: run `py -m py_compile app.py`, then restart Uvicorn.')
