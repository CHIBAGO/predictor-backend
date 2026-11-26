
"""Predictor API - EPL, LaLiga, Bundesliga, Ligue 1 and UEFA Champions League."""

from datetime import date, timedelta
from pathlib import Path
from time import monotonic
import os
import csv

import joblib
import requests
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent

EPL_MODEL_PATH = BASE_DIR / "predictor_model.joblib"
LALIGA_MODEL_PATH = BASE_DIR.parent / "models" / "laliga_model.joblib"
BUNDESLIGA_MODEL_PATH = BASE_DIR / "models" / "bundesliga_model.joblib"
LIGUE1_MODEL_PATH = BASE_DIR / "models" / "ligue1_model.joblib"
UCL_MODEL_PATH = BASE_DIR / "ucl" / "models" / "ucl_model.joblib"


UCL_FIXTURES_PATH = (
    BASE_DIR / "ucl" / "data" / "raw" /
    "open_matchday_2627" / "fixtures_2627.csv"
)


# football-data.org crest IDs for UCL clubs.
# Unknown clubs safely return None instead of receiving an incorrect logo.
UCL_TEAM_CRESTS = {
    "Arsenal": "57",
    "Arsenal FC": "57",
    "Aston Villa": "58",
    "Aston Villa FC": "58",
    "Atletico Madrid": "78",
    "Atlético Madrid": "78",
    "Barcelona": "81",
    "FC Barcelona": "81",
    "Bayern Munich": "5",
    "FC Bayern München": "5",
    "Benfica": "1903",
    "SL Benfica": "1903",
    "Borussia Dortmund": "4",
    "Dortmund": "4",
    "Chelsea": "61",
    "Chelsea FC": "61",
    "Inter": "108",
    "Inter Milan": "108",
    "Juventus": "109",
    "Liverpool": "64",
    "Liverpool FC": "64",
    "Manchester City": "65",
    "Manchester City FC": "65",
    "AC Milan": "98",
    "Milan": "98",
    "Napoli": "113",
    "Paris Saint-Germain": "524",
    "Paris Saint-Germain FC": "524",
    "PSG": "524",
    "Real Madrid": "86",
    "Real Madrid CF": "86",
    "Sporting CP": "498",
    "Sporting": "498",
    "Ajax": "678",
    "AFC Ajax": "678",
    "Bayer Leverkusen": "3",
    "Bayer 04 Leverkusen": "3",
    "Porto": "503",
    "FC Porto": "503",
    "Manchester United": "66",
    "Manchester United FC": "66",
}

load_dotenv(BASE_DIR / ".env")

FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_API_TOKEN")
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

LEAGUES = {
    "PL": {"name": "Premier League", "country": "England"},
    "PD": {"name": "LaLiga", "country": "Spain"},
    "BL": {"name": "Bundesliga", "country": "Germany"},
    "L1": {"name": "Ligue 1", "country": "France"},
    "UCL": {"name": "UEFA Champions League", "country": "Europe"},
    "CAF": {"name": "CAF Champions League", "country": "Africa"},
}

_FIXTURE_CACHE_SECONDS = 300

EPL_TEAM_MAPPING = {
    "Arsenal FC": "Arsenal", "Arsenal": "Arsenal",
    "Aston Villa FC": "Aston Villa", "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth", "Bournemouth FC": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford", "Brentford": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Brighton and Hove Albion FC": "Brighton", "Brighton": "Brighton",
    "Burnley FC": "Burnley", "Burnley": "Burnley",
    "Cardiff City FC": "Cardiff", "Cardiff City": "Cardiff",
    "Cardiff": "Cardiff",
    "Chelsea FC": "Chelsea", "Chelsea": "Chelsea",
    "Coventry City FC": "Coventry", "Coventry City": "Coventry",
    "Coventry": "Coventry",
    "Crystal Palace FC": "Crystal Palace", "Crystal Palace": "Crystal Palace",
    "Everton FC": "Everton", "Everton": "Everton",
    "Fulham FC": "Fulham", "Fulham": "Fulham",
    "Huddersfield Town FC": "Huddersfield",
    "Huddersfield Town": "Huddersfield", "Huddersfield": "Huddersfield",
    "Hull City FC": "Hull", "Hull City": "Hull", "Hull": "Hull",
    "Ipswich Town FC": "Ipswich", "Ipswich Town": "Ipswich",
    "Ipswich": "Ipswich",
    "Leeds United FC": "Leeds", "Leeds United": "Leeds",
    "Leeds": "Leeds",
    "Leicester City FC": "Leicester", "Leicester City": "Leicester",
    "Leicester": "Leicester",
    "Liverpool FC": "Liverpool", "Liverpool": "Liverpool",
    "Luton Town FC": "Luton", "Luton Town": "Luton", "Luton": "Luton",
    "Manchester City FC": "Man City", "Manchester City": "Man City",
    "Man City": "Man City",
    "Manchester United FC": "Man United",
    "Manchester United": "Man United", "Man United": "Man United",
    "Middlesbrough FC": "Middlesbrough",
    "Middlesbrough": "Middlesbrough",
    "Newcastle United FC": "Newcastle",
    "Newcastle United": "Newcastle", "Newcastle": "Newcastle",
    "Norwich City FC": "Norwich", "Norwich City": "Norwich",
    "Norwich": "Norwich",
    "Nottingham Forest FC": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest", "Nott'm Forest": "Nott'm Forest",
    "Sheffield United FC": "Sheffield United",
    "Sheffield United": "Sheffield United",
    "Southampton FC": "Southampton", "Southampton": "Southampton",
    "Stoke City FC": "Stoke", "Stoke City": "Stoke", "Stoke": "Stoke",
    "Sunderland AFC": "Sunderland", "Sunderland": "Sunderland",
    "Swansea City AFC": "Swansea", "Swansea City FC": "Swansea",
    "Swansea City": "Swansea", "Swansea": "Swansea",
    "Tottenham Hotspur FC": "Tottenham",
    "Tottenham Hotspur": "Tottenham", "Tottenham": "Tottenham",
    "Watford FC": "Watford", "Watford": "Watford",
    "West Bromwich Albion FC": "West Brom",
    "West Bromwich Albion": "West Brom", "West Brom": "West Brom",
    "West Ham United FC": "West Ham", "West Ham United": "West Ham",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Wolverhampton Wanderers": "Wolves", "Wolves": "Wolves",
}

LALIGA_TEAM_MAPPING = {
    "Real Madrid CF": "Real Madrid", "Real Madrid": "Real Madrid",
    "FC Barcelona": "Barcelona", "Barcelona": "Barcelona",
    "Atletico Madrid": "Ath Madrid",
    "Atlético de Madrid": "Ath Madrid",
    "Atlético Madrid": "Ath Madrid",
    "Club Atlético de Madrid": "Ath Madrid",
    "Club Atletico de Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao", "Athletic Bilbao": "Ath Bilbao",
    "Real Betis": "Betis", "Real Betis Balompie": "Betis",
    "Real Betis Balompié": "Betis",
    "Real Sociedad": "Sociedad",
    "Real Sociedad de Futbol": "Sociedad",
    "Real Sociedad de Fútbol": "Sociedad",
    "Sevilla FC": "Sevilla", "Sevilla": "Sevilla",
    "Valencia CF": "Valencia", "Valencia": "Valencia",
    "Villarreal CF": "Villarreal", "Villarreal": "Villarreal",
    "Girona FC": "Girona", "Girona": "Girona",
    "Getafe CF": "Getafe", "Getafe": "Getafe",
    "CA Osasuna": "Osasuna", "Osasuna": "Osasuna",
    "RCD Mallorca": "Mallorca", "Mallorca": "Mallorca",
    "Rayo Vallecano": "Vallecano",
    "Rayo Vallecano de Madrid": "Vallecano",
    "RC Celta de Vigo": "Celta", "Celta Vigo": "Celta",
    "Celta": "Celta",
    "Deportivo Alaves": "Alaves", "Deportivo Alavés": "Alaves",
    "Alaves": "Alaves",
    "UD Las Palmas": "Las Palmas", "Las Palmas": "Las Palmas",
    "CD Leganes": "Leganes", "Leganes": "Leganes",
    "Racing Santander": "Santander",
    "Racing de Santander": "Santander",
    "Real Racing Club de Santander": "Santander",
    "Real Racing Club Santander": "Santander",
    "Racing Club de Santander": "Santander",
    "Real Racing Santander": "Santander",
    "Espanyol": "Espanol", "RCD Espanyol": "Espanol",
    "Espanyol Barcelona": "Espanol",
    "Espanyol de Barcelona": "Espanol",
    "RCD Espanyol de Barcelona": "Espanol",
    "RCD Espanyol Barcelona": "Espanol",
    "Reial Club Deportiu Espanyol": "Espanol",
    "Real Valladolid CF": "Valladolid",
    "Real Valladolid": "Valladolid",
    "Elche CF": "Elche", "Elche": "Elche",
    "Granada CF": "Granada", "Granada": "Granada",
    "Cadiz CF": "Cadiz", "Cadiz": "Cadiz",
    "Almeria": "Almeria", "UD Almeria": "Almeria",
    "Real Oviedo": "Real Oviedo",
    "Levante UD": "Levante", "Levante": "Levante",
    "FC Andorra": "FC Andorra",
    "Málaga CF": "Malaga", "Malaga": "Malaga",
    "CD Tenerife": "Tenerife", "Tenerife": "Tenerife",
    "Burgos CF": "Burgos", "Burgos": "Burgos",
    "RC Deportivo de A Coruna": "La Coruna",
    "RC Deportivo La Coruña": "La Coruna",
    "RC Deportivo La Coruna": "La Coruna",
    "RC Deportivo de La Coruña": "La Coruna",
    "Deportivo La Coruna": "La Coruna",
    "Deportivo de La Coruña": "La Coruna",
    "Deportivo La Coruña": "La Coruna",
}

BUNDESLIGA_TEAM_MAPPING = {
    "1. FC Köln": "FC Koln",
    "1. FC Cologne": "FC Koln",
    "FC Cologne": "FC Koln",
    "Cologne": "FC Koln",
    "1. FSV Mainz 05": "Mainz",
    "FSV Mainz 05": "Mainz",
    "FSV Mainz": "Mainz",
    "Mainz 05": "Mainz",
    "Mainz": "Mainz",
    "1. FC Union Berlin": "Union Berlin",
    "Union Berlin": "Union Berlin",
    "SC Paderborn 07": "Paderborn",
    "Paderborn": "Paderborn",
    "FC Schalke 04": "Schalke 04",
    "Schalke 04": "Schalke 04",
    "Hamburger SV": "Hamburg",
    "Hamburg": "Hamburg",
    "FC Bayern München": "Bayern Munich",
    "FC Bayern Munich": "Bayern Munich",
    "Bayern Munich": "Bayern Munich",
    "VfB Stuttgart": "Stuttgart",
    "Stuttgart": "Stuttgart",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Frankfurt": "Ein Frankfurt",
    "SC Freiburg": "Freiburg",
    "Freiburg": "Freiburg",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "TSG Hoffenheim": "Hoffenheim",
    "Hoffenheim": "Hoffenheim",
    "FC Augsburg": "Augsburg",
    "Augsburg": "Augsburg",
    "Borussia Dortmund": "Dortmund",
    "Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Bayer Leverkusen": "Leverkusen",
    "Leverkusen": "Leverkusen",
    "Borussia Mönchengladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "Borussia Moenchengladbach": "M'gladbach",
    "Mönchengladbach": "M'gladbach",
    "Monchengladbach": "M'gladbach",
    "Moenchengladbach": "M'gladbach",
    "M'gladbach": "M'gladbach",
    "SV Werder Bremen": "Werder Bremen",
    "Werder Bremen": "Werder Bremen",
    "Werder": "Werder Bremen",
    "Bremen": "Werder Bremen",
    "RB Leipzig": "RB Leipzig",
    "Leipzig": "RB Leipzig",
}

LIGUE1_TEAM_MAPPING = {
    "Paris Saint-Germain": "Paris SG",
    "Paris Saint-Germain FC": "Paris SG",
    "Paris SG": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "Olympique Lyon": "Lyon",
    "AS Monaco FC": "Monaco",
    "AS Monaco": "Monaco",
    "LOSC Lille": "Lille",
    "Lille OSC": "Lille",
    "Stade Rennais FC": "Rennes",
    "Stade Rennais": "Rennes",
    "Stade Rennais FC 1901": "Rennes",
    "Stade Brestois 29": "Brest",
    "Stade Brest 29": "Brest",
    "Stade Brestois": "Brest",
    "RC Strasbourg Alsace": "Strasbourg",
    "Strasbourg Alsace": "Strasbourg",
    "RC Lens": "Lens",
    "Racing Club de Lens": "Lens",
    "FC Nantes": "Nantes",
    "Toulouse FC": "Toulouse",
    "FC Toulouse": "Toulouse",
    "OGC Nice": "Nice",
    "Montpellier HSC": "Montpellier",
    "Stade de Reims": "Reims",
    "Angers SCO": "Angers",
    "FC Lorient": "Lorient",
    "Clermont Foot 63": "Clermont",
    "Le Havre AC": "Le Havre",
    "AJ Auxerre": "Auxerre",
    "FC Metz": "Metz",
    "ESTAC Troyes": "Troyes",
    "ES Troyes AC": "Troyes",
    "Girondins de Bordeaux": "Bordeaux",
    "AS Saint-Etienne": "Saint-Etienne",
    "AS Saint-Étienne": "Saint-Etienne",
    "Le Mans FC": "Le Mans",
    "Le Mans": "Le Mans",
    "Le Mans UC 72": "Le Mans",
    "Le Mans UC": "Le Mans",
    "MUC 72": "Le Mans",
    "Paris FC": "Paris FC",
    "Paris Football Club": "Paris FC",
}


app = FastAPI(
    title="Predictor API",
    description=(
        "Multi-league football match probabilities for statistical analysis. "
        "Not betting advice."
    ),
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:80",
        "http://127.0.0.1:80",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_bundles = {}
_fixture_cache = {}
_ucl_crest_cache = {}


class PredictRequest(BaseModel):
    home_team: str = Field(min_length=1, max_length=100)
    away_team: str = Field(min_length=1, max_length=100)
    match_date: date = Field(default_factory=date.today)
    league: str = Field(default="PL", min_length=2, max_length=3)
    stage: str = Field(default="Group Stage", max_length=30)


def get_bundle(league="PL"):
    code = league.upper()

    model_paths = {
        "PL": EPL_MODEL_PATH,
        "PD": LALIGA_MODEL_PATH,
        "BL": BUNDESLIGA_MODEL_PATH,
        "L1": LIGUE1_MODEL_PATH,
        "UCL": UCL_MODEL_PATH,
	"CAF": CAF_MODEL_PATH,
    }

    path = model_paths.get(code)

    if path is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported league. Use PL, PD, BL, L1, UCL or CAF.",
        )

    if code not in _bundles:
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"{code} model file not found: {path}",
            )

        try:
            _bundles[code] = joblib.load(path)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Could not load {code} model: {exc}",
            )

    return _bundles[code]


def _fetch_fixtures(match_date: date, league_code: str):
    key = f"{league_code}_{match_date.isoformat()}"

    cached = _fixture_cache.get(key)
    if cached and monotonic() - cached["saved_at"] < _FIXTURE_CACHE_SECONDS:
        return cached["matches"]

    if league_code == "UCL":
        raise HTTPException(
            status_code=500,
            detail="UCL fixtures must be loaded from the local fixtures file.",
        )

    if not FOOTBALL_DATA_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="API token missing. Check FOOTBALL_DATA_API_TOKEN in .env.",
        )

    api_league_code = {
        "PL": "PL",
        "PD": "PD",
        "BL": "BL1",
        "L1": "FL1",
    }.get(league_code)

    if not api_league_code:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Football Data competition: {league_code}",
        )

    try:
        response = requests.get(
            f"{FOOTBALL_DATA_BASE}/competitions/{api_league_code}/matches",
            headers={"X-Auth-Token": FOOTBALL_DATA_TOKEN},
            params={
                "dateFrom": match_date.isoformat(),
                "dateTo": (match_date + timedelta(days=1)).isoformat(),
            },
            timeout=20,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Football Data API. Check internet and try again.",
        )

    if response.status_code == 401:
        raise HTTPException(
            status_code=502,
            detail="Football Data API rejected the token (401). Check your .env token.",
        )

    if response.status_code == 403:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Football Data API denied access to {api_league_code} (403). "
                "API plan may not support this competition."
            ),
        )

    if response.status_code == 429:
        raise HTTPException(
            status_code=503,
            detail="Football Data API rate limit reached. Wait and retry.",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Football Data API returned HTTP {response.status_code}.",
        )

    try:
        matches = response.json().get("matches", [])
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Football Data API returned invalid JSON.",
        )

    _fixture_cache[key] = {
        "saved_at": monotonic(),
        "matches": matches,
    }

    return matches


def _fetch_ucl_fixtures(match_date: date):
    """Read UCL fixtures from the locally downloaded Open Matchday CSV."""
    key = f"UCL_{match_date.isoformat()}"

    cached = _fixture_cache.get(key)
    if cached and monotonic() - cached["saved_at"] < _FIXTURE_CACHE_SECONDS:
        return cached["matches"]

    if not UCL_FIXTURES_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"UCL fixtures file not found: {UCL_FIXTURES_PATH}",
        )

    matches = []

    try:
        with UCL_FIXTURES_PATH.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                fixture_date = (
                    row.get("date")
                    or row.get("Date")
                    or row.get("match_date")
                    or ""
                ).strip()

                if not fixture_date:
                    continue

                try:
                    parsed_date = date.fromisoformat(fixture_date[:10])
                except ValueError:
                    continue

                if parsed_date != match_date:
                    continue

                home = (
                    row.get("home")
                    or row.get("HomeTeam")
                    or row.get("home_team")
                    or ""
                ).strip()

                away = (
                    row.get("away")
                    or row.get("AwayTeam")
                    or row.get("away_team")
                    or ""
                ).strip()

                if not home or not away:
                    continue

                kickoff = (
                    row.get("time")
                    or row.get("Time")
                    or ""
                ).strip()

                matches.append({
                    "id": row.get("id") or row.get("gw"),
                    "utcDate": (
                        f"{fixture_date}T{kickoff}:00Z"
                        if kickoff else f"{fixture_date}T00:00:00Z"
                    ),
                    "status": "SCHEDULED",
                    "matchday": row.get("gw"),
                    "homeTeam": {"name": home, "crest": get_ucl_crest(home)},
                    "awayTeam": {"name": away, "crest": get_ucl_crest(away)},
                    "score": None,
                })

    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not read UCL fixtures file: {exc}",
        )

    _fixture_cache[key] = {
        "saved_at": monotonic(),
        "matches": matches,
    }

    return matches


def _team_key(name):
    """Conservative name key: ignore accents, punctuation and common suffixes."""
    import unicodedata

    if not name:
        return ""

    value = unicodedata.normalize("NFKD", str(name))
    value = "".join(
        ch for ch in value if not unicodedata.combining(ch)
    ).casefold()

    value = value.replace("&", " and ")

    for token in (
        " football club", " fc", " cf", " afc", " cfc",
        " sc", " ac", " cd", " ud", " rcd", " rc", " ca", " sd",
    ):
        value = value.replace(token, " ")

    return " ".join(
        value.replace(".", " ").replace("-", " ").split()
    )


def get_ucl_crest(team_name):
    """Resolve known UCL crests, then try a cached read-only team search."""
    if not team_name:
        return None

    name = str(team_name).strip()
    cache_key = _team_key(name)
    if cache_key in _ucl_crest_cache:
        return _ucl_crest_cache[cache_key]

    aliases = {
        "B. Dortmund": "Borussia Dortmund",
        "Man City": "Manchester City",
        "Man Utd": "Manchester United",
        "Atleti": "Atletico Madrid",
        "Atletico": "Atletico Madrid",
        "Bayern": "Bayern Munich",
        "Bayern München": "Bayern Munich",
        "Paris": "Paris Saint-Germain",
        "Leipzig": "RB Leipzig",
        "Lens": "RC Lens",
        "Fenerbahçe": "Fenerbahce",
        "Feyenoord": "Feyenoord Rotterdam",
        "PSV": "PSV Eindhoven",
        "Roma": "AS Roma",
        "Slavia Praha": "Slavia Prague",
        "S. Bratislava": "Slovan Bratislava",
        "Shakhtar": "Shakhtar Donetsk",
        "Sabah": "Sabah FK",
        "Bodø/Glimt": "Bodo/Glimt",
        "LASK": "LASK Linz",
        "Stuttgart": "VfB Stuttgart",
        "Lille": "LOSC Lille",
        "Viking": "Viking FK",
    }

    candidates = [name]
    canonical = aliases.get(name)
    if canonical and canonical.casefold() != name.casefold():
        candidates.append(canonical)

    normalized_crests = {
        _team_key(club): crest_id
        for club, crest_id in UCL_TEAM_CRESTS.items()
    }
    team_id = UCL_TEAM_CRESTS.get(name)

    if not team_id:
        for candidate in candidates:
            team_id = normalized_crests.get(_team_key(candidate))
            if team_id:
                break

    if team_id:
        crest_url = f"https://crests.football-data.org/{team_id}.png"
        _ucl_crest_cache[cache_key] = crest_url
        return crest_url

    # Fallback for clubs without a static football-data.org crest ID.
    for candidate in candidates:
        try:
            response = requests.get(
                "https://www.thesportsdb.com/api/v1/json/3/searchteams.php",
                params={"t": candidate},
                timeout=5,
            )
            if response.status_code != 200:
                continue

            teams = (response.json() or {}).get("teams") or []
            for team in teams:
                result_name = team.get("strTeam") or ""
                badge = team.get("strBadge")
                if badge and _team_key(result_name) == _team_key(candidate):
                    _ucl_crest_cache[cache_key] = badge
                    return badge
        except (requests.RequestException, ValueError, TypeError):
            continue

    _ucl_crest_cache[cache_key] = None
    return None


def _match_model_team(api_name, known_names, league_code):
    """Match API names to the model's known team names."""
    if not api_name:
        return None

    known = list(known_names)
    name_map = {name.casefold(): name for name in known}

    direct = name_map.get(api_name.strip().casefold())
    if direct:
        return direct

    mappings = {
        "PL": EPL_TEAM_MAPPING,
        "PD": LALIGA_TEAM_MAPPING,
        "BL": BUNDESLIGA_TEAM_MAPPING,
        "L1": LIGUE1_TEAM_MAPPING,
    }

    mapping = mappings.get(league_code, {})
    mapped = mapping.get(api_name.strip())

    if mapped:
        found = name_map.get(mapped.casefold())
        if found:
            return found

    known_normalized = {
        _team_key(name): name
        for name in known
        if _team_key(name)
    }

    normalized_api = _team_key(api_name)
    direct_normalized = known_normalized.get(normalized_api)

    if direct_normalized:
        return direct_normalized

    for alias, target in mapping.items():
        if _team_key(alias) == normalized_api:
            found = (
                name_map.get(target.casefold())
                or known_normalized.get(_team_key(target))
            )
            if found:
                return found

    return None


def _probability_shape(result):
    if not result:
        return None

    return {
        "home": round(float(result.get("home", 0)), 4),
        "draw": round(float(result.get("draw", 0)), 4),
        "away": round(float(result.get("away", 0)), 4),
    }


def _get_team_names(code, bundle):
    if code == "PL":
        return sorted(bundle["state"].played.keys())
	if code == "CAF":
    if not CAF_DATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"CAF dataset not found: {CAF_DATA_PATH}",
        )

    try:
        df = pd.read_csv(CAF_DATA_PATH)

        teams = set(df["home_team"].dropna())
        teams.update(df["away_team"].dropna())

        return sorted(teams)

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load CAF teams: {exc}",
        )

    from core import (
        laliga_teams,
        bundesliga_teams,
        ligue1_teams,
        ucl_teams,
    )

    functions = {
        "PD": laliga_teams,
        "BL": bundesliga_teams,
        "L1": ligue1_teams,
        "UCL": ucl_teams,
    }

    try:
        return functions[code]()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load {code} team history: {exc}",
        )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "predictor-api",
        "models_loaded": list(_bundles.keys()),
        "available_leagues": list(LEAGUES.keys()),
    }


@app.get("/api/model-info")
def model_info(league: str = Query(default="PL")):
    code = league.upper()
    bundle = get_bundle(code)

    if code == "PL":
        return bundle.get("info", {})

    names = {
        "PD": "LaLiga Random Forest",
        "BL": "Bundesliga Random Forest",
        "L1": "Ligue 1 Random Forest",
        "UCL": "UEFA Champions League Random Forest",
    }

    return {
        "model_name": names.get(code, "Football Prediction Model"),
        "league": LEAGUES[code]["name"],
        "matches_used": bundle.get("matches_used"),
        "train_matches": bundle.get("train_matches"),
        "test_matches": bundle.get("test_matches"),
        "trained_at": bundle.get("trained_at"),
        "features": bundle.get("features", []),
        "classes": bundle.get("classes"),
    }


@app.get("/api/teams")
def teams(league: str = Query(default="PL")):
    code = league.upper()
    bundle = get_bundle(code)
    names = _get_team_names(code, bundle)

    return {
        "league": code,
        "count": len(names),
        "teams": names,
    }


@app.post("/api/predict")
def predict(request: PredictRequest):
    code = request.league.upper()
    bundle = get_bundle(code)

    home = request.home_team.strip()
    away = request.away_team.strip()

    if not home or not away:
        raise HTTPException(
            status_code=422,
            detail="Both team names are required.",
        )

    if home.casefold() == away.casefold():
        raise HTTPException(
            status_code=422,
            detail="Home and away teams must be different.",
        )

    try:
        if code == "PL":
            state = bundle["state"]
            known = {
                name.casefold(): name
                for name in state.played.keys()
            }

            if home.casefold() not in known:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown home team: {home}",
                )

            if away.casefold() not in known:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown away team: {away}",
                )

            home = known[home.casefold()]
            away = known[away.casefold()]

            from core import predict_match

            probabilities = predict_match(
                bundle, home, away,
                request.match_date.isoformat(),
            )

        elif code == "PD":
            from core import predict_laliga_match

            probabilities = predict_laliga_match(
                bundle, home, away,
                request.match_date.isoformat(),
            )

        elif code == "BL":
            from core import predict_bundesliga_match

            probabilities = predict_bundesliga_match(
                bundle, home, away,
                request.match_date.isoformat(),
            )

        elif code == "L1":
            from core import predict_ligue1_match

            probabilities = predict_ligue1_match(
                bundle, home, away,
                request.match_date.isoformat(),
            )

        elif code == "UCL":
            from core import predict_ucl_match

            probabilities = predict_ucl_match(
                bundle, home, away,
                request.match_date.isoformat(),
            )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction error: {exc}",
        )

    if probabilities is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "The model does not have enough historical information "
                "for one or both teams."
            ),
        )

    outcome = max(probabilities, key=probabilities.get)

    model_names = {
        "PL": "EPL model",
        "PD": "LaLiga Random Forest",
        "BL": "Bundesliga Random Forest",
        "L1": "Ligue 1 Random Forest",
        "UCL": "UEFA Champions League Random Forest",
    }

    model_name = bundle.get("info", {}).get(
        "model_name",
        model_names[code],
    )

    return {
        "match": {
            "home_team": home,
            "away_team": away,
            "date": request.match_date.isoformat(),
            "league": code,
        },
        "prediction": {
            "most_likely": outcome,
            "probabilities": _probability_shape(probabilities),
        },
        "model": model_name,
        "notice": (
            "Probabilities are statistical estimates, "
            "not certainties or betting advice."
        ),
    }


@app.get("/api/fixtures")
def fixtures(
    match_date: date = Query(default_factory=date.today, alias="date"),
    league: str = Query(default="PL"),
):
    code = league.upper()

    if code not in LEAGUES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported league '{league}'. "
                "Available leagues: PL, PD, BL, L1, UCL."
            ),
        )

    info = LEAGUES[code]

    if code == "UCL":
        raw = _fetch_ucl_fixtures(match_date)
    else:
        raw = _fetch_fixtures(match_date, code)

    bundle = get_bundle(code)
    known_names = _get_team_names(code, bundle)

    output = []

    for match in raw:
        home_data = match.get("homeTeam") or {}
        away_data = match.get("awayTeam") or {}

        home_api = home_data.get("name") or "Unknown"
        away_api = away_data.get("name") or "Unknown"

        if code == "UCL":
            # UCL local fixtures use Open Matchday names.
            # Match them against the names used by the UCL model.
            home_model = _match_model_team(
                home_api, known_names, code
            )
            away_model = _match_model_team(
                away_api, known_names, code
            )

            # If an exact UCL name exists, use it directly.
            if home_api in known_names:
                home_model = home_api
            if away_api in known_names:
                away_model = away_api

        else:
            home_model = _match_model_team(
                home_api, known_names, code
            )
            away_model = _match_model_team(
                away_api, known_names, code
            )

        probabilities = None
        note = None

        if (
            home_model
            and away_model
            and home_model != away_model
        ):
            try:
                if code == "PL":
                    from core import predict_match

                    result = predict_match(
                        bundle, home_model, away_model,
                        match_date.isoformat(),
                    )

                elif code == "PD":
                    from core import predict_laliga_match

                    result = predict_laliga_match(
                        bundle, home_model, away_model,
                        match_date.isoformat(),
                    )

                elif code == "BL":
                    from core import predict_bundesliga_match

                    result = predict_bundesliga_match(
                        bundle, home_model, away_model,
                        match_date.isoformat(),
                    )

                elif code == "L1":
                    from core import predict_ligue1_match

                    result = predict_ligue1_match(
                        bundle, home_model, away_model,
                        match_date.isoformat(),
                    )

                elif code == "UCL":
                    from core import predict_ucl_match

                    result = predict_ucl_match(
                        bundle, home_model, away_model,
                        match_date.isoformat(),
                    )

                probabilities = _probability_shape(result)

                if probabilities is None:
                    note = (
                        "The model does not have enough historical "
                        "information for this fixture."
                    )

            except Exception as exc:
                note = f"Prediction unavailable: {exc}"

        else:
            missing = []

            if not home_model:
                missing.append(
                    f"home team not recognized: {home_api}"
                )

            if not away_model:
                missing.append(
                    f"away team not recognized: {away_api}"
                )

            if (
                home_model
                and away_model
                and home_model == away_model
            ):
                missing.append(
                    "both API teams matched the same model team"
                )

            note = " | ".join(missing)

        output.append({
            "match_id": match.get("id"),
            "date_utc": match.get("utcDate"),
            "status": match.get("status"),
            "matchday": match.get("matchday"),
            "home_team": home_api,
            "away_team": away_api,
            "home_model_name": home_model,
            "away_model_name": away_model,
            "home_crest": home_data.get("crest") or (get_ucl_crest(home_api) if code == "UCL" else None),
            "away_crest": away_data.get("crest") or (get_ucl_crest(away_api) if code == "UCL" else None),
            "score": match.get("score"),
            "prediction": probabilities,
            "most_likely": (
                max(probabilities, key=probabilities.get)
                if probabilities else None
            ),
            "prediction_note": note,
        })

    return {
        "date": match_date.isoformat(),
        "competition": info["name"],
        "competition_code": code,
        "country": info["country"],
        "count": len(output),
        "matches": output,
        "source": (
            "Open Matchday local CSV"
            if code == "UCL"
            else "football-data.org"
        ),
        "notice": (
            "Predictions are statistical estimates, "
            "not certainties or betting advice."
        ),
    }


@app.get("/api/leagues")
def leagues():
    return {
        "count": len(LEAGUES),
        "leagues": [
            {
                "code": code,
                "name": info["name"],
                "country": info["country"],
            }
            for code, info in LEAGUES.items()
        ],
    }
