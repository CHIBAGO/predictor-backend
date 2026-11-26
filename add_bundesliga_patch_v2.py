from pathlib import Path
import shutil

app_path = Path("app.py")
if not app_path.exists():
    raise SystemExit("app.py not found. Run this script from the predictor project folder.")

source = app_path.read_text(encoding="utf-8")
backup = Path("app.py.backup_before_bundesliga_v2")
if not backup.exists():
    shutil.copy2(app_path, backup)

def replace_once(old, new, label):
    global source
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"Stopped safely: expected exactly one match for {label}, found {count}. app.py was not changed.")
    source = source.replace(old, new, 1)

replace_once(
    'LALIGA_MODEL_PATH = BASE_DIR.parent / "models" / "laliga_model.joblib"\n',
    'LALIGA_MODEL_PATH = BASE_DIR.parent / "models" / "laliga_model.joblib"\nBUNDESLIGA_MODEL_PATH = BASE_DIR / "models" / "bundesliga_model.joblib"\n',
    "model path"
)
replace_once(
    'LEAGUES = {"PL": {"name": "Premier League", "country": "England"}, "PD": {"name": "LaLiga", "country": "Spain"}}\n',
    'LEAGUES = {"PL": {"name": "Premier League", "country": "England"}, "PD": {"name": "LaLiga", "country": "Spain"}, "BL": {"name": "Bundesliga", "country": "Germany"}}\n',
    "league registry"
)
replace_once(
    'path = EPL_MODEL_PATH if code == "PL" else LALIGA_MODEL_PATH if code == "PD" else None\n    if path is None: raise HTTPException(status_code=400, detail="Unsupported league. Use PL or PD.")',
    'path = EPL_MODEL_PATH if code == "PL" else LALIGA_MODEL_PATH if code == "PD" else BUNDESLIGA_MODEL_PATH if code == "BL" else None\n    if path is None: raise HTTPException(status_code=400, detail="Unsupported league. Use PL, PD or BL.")',
    "get_bundle league routing"
)
replace_once(
    '    if league.upper() == "PD":\n        info = {"model_name": "LaLiga Random Forest", "league": "LaLiga", "matches_used": bundle.get("matches_used"), "train_matches": bundle.get("train_matches"), "test_matches": bundle.get("test_matches"), "trained_at": bundle.get("trained_at"), "features": bundle.get("features", [])}\n    return info\n',
    '    if league.upper() == "PD":\n        info = {"model_name": "LaLiga Random Forest", "league": "LaLiga", "matches_used": bundle.get("matches_used"), "train_matches": bundle.get("train_matches"), "test_matches": bundle.get("test_matches"), "trained_at": bundle.get("trained_at"), "features": bundle.get("features", [])}\n    elif league.upper() == "BL":\n        info = {"model_name": bundle.get("info", {}).get("model_name", "Bundesliga model"), "league": "Bundesliga", "matches_used": bundle.get("matches_used"), "train_matches": bundle.get("train_matches"), "test_matches": bundle.get("test_matches"), "trained_at": bundle.get("trained_at"), "features": bundle.get("features", [])}\n    return info\n',
    "model-info Bundesliga"
)
replace_once(
    '    if code == "PL":\n        names = sorted(bundle["state"].played.keys())\n    else:\n        from core import laliga_teams\n        try: names = laliga_teams()\n        except Exception as exc: raise HTTPException(status_code=503, detail=f"Could not load LaLiga team history: {exc}")\n',
    '    if code == "PL":\n        names = sorted(bundle["state"].played.keys())\n    elif code == "PD":\n        from core import laliga_teams\n        try: names = laliga_teams()\n        except Exception as exc: raise HTTPException(status_code=503, detail=f"Could not load LaLiga team history: {exc}")\n    else:\n        from core import bundesliga_teams\n        try: names = bundesliga_teams()\n        except Exception as exc: raise HTTPException(status_code=503, detail=f"Could not load Bundesliga team history: {exc}")\n',
    "teams endpoint"
)
replace_once(
    '        else:\n            from core import predict_laliga_match\n            probabilities = predict_laliga_match(bundle, home, away, request.match_date.isoformat())\n',
    '        elif code == "PD":\n            from core import predict_laliga_match\n            probabilities = predict_laliga_match(bundle, home, away, request.match_date.isoformat())\n        else:\n            from core import predict_bundesliga_match\n            probabilities = predict_bundesliga_match(bundle, home, away, request.match_date.isoformat())\n',
    "predict endpoint league branch"
)
replace_once(
    '"LaLiga Random Forest" if code=="PD" else "EPL model"',
    '"LaLiga Random Forest" if code=="PD" else "Bundesliga model" if code=="BL" else "EPL model"',
    "prediction model label"
)
replace_once(
    'Available leagues: PL, PD.',
    'Available leagues: PL, PD, BL.',
    "fixtures error message"
)
replace_once(
    'known_names = bundle["state"].played.keys() if code == "PL" else __import__("core").laliga_teams()\n',
    'if code == "PL":\n        known_names = bundle["state"].played.keys()\n    elif code == "PD":\n        known_names = __import__("core").laliga_teams()\n    else:\n        known_names = __import__("core").bundesliga_teams()\n',
    "fixtures known team list"
)
replace_once(
    '                else:\n                    from core import predict_laliga_match\n                    result = predict_laliga_match(bundle, h_model, a_model, match_date.isoformat())\n',
    '                elif code == "PD":\n                    from core import predict_laliga_match\n                    result = predict_laliga_match(bundle, h_model, a_model, match_date.isoformat())\n                else:\n                    from core import predict_bundesliga_match\n                    result = predict_bundesliga_match(bundle, h_model, a_model, match_date.isoformat())\n',
    "fixtures prediction branch"
)

app_path.write_text(source, encoding="utf-8")
print("Bundesliga integration patch applied.")
print(f"Backup created/kept at: {backup}")
print("Next run: py -m py_compile app.py")
