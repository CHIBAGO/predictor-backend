"""Download isolated UEFA Champions League datasets for Predictor.

Run from the Predictor project root:
    py ucl/scripts/download_ucl_data.py

Historical seasons (2011/12-2025/26):
    openfootball/champions-league (football.txt format, CC0-1.0)

Current season (2026/27):
    open-matchday CSV results and fixtures.

This script writes only inside the ucl/ directory. It does not modify
other league data or model files.
"""
from pathlib import Path
import io
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[2]
UCL_DIR = ROOT / "ucl"
RAW_DIR = UCL_DIR / "data" / "raw"
HIST_DIR = RAW_DIR / "openfootball_historical"
CURRENT_DIR = RAW_DIR / "open_matchday_2627"

HISTORICAL_ZIP_URL = (
    "https://codeload.github.com/openfootball/champions-league/zip/refs/heads/master"
)
CURRENT_BASE_URL = (
    "https://raw.githubusercontent.com/oladeeayo/open-matchday/master/"
    "data/UEFA_1_champions-league"
)
CURRENT_SEASON = "2627"
FILES = [
    f"results_{CURRENT_SEASON}.csv",
    f"fixtures_{CURRENT_SEASON}.csv",
    f"goals_{CURRENT_SEASON}.csv",
]

def make_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "Predictor-UCL-research-project"})
    return session

def download_historical(session):
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    marker = HIST_DIR / "DOWNLOAD_COMPLETE.txt"
    if marker.exists():
        print("Historical archive already downloaded. Keeping existing files.")
        return

    print("Downloading OpenFootball historical Champions League archive...")
    response = session.get(HISTORICAL_ZIP_URL, timeout=(30, 180))
    response.raise_for_status()

    saved = 0
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in archive.infolist():
            # Keep only season folders and text data; ignore git metadata.
            parts = Path(member.filename).parts
            if len(parts) < 3:
                continue
            season_folder = parts[1]
            if not (
                len(season_folder) == 7
                and season_folder[:4].isdigit()
                and season_folder[4] == "-"
                and season_folder[5:].isdigit()
            ):
                continue
            if member.is_dir():
                continue
            if not member.filename.lower().endswith((".txt", ".md")):
                continue

            relative = Path(*parts[1:])
            destination = HIST_DIR / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
            saved += 1

    marker.write_text(
        f"Downloaded from openfootball/champions-league\nFiles saved: {saved}\n",
        encoding="utf-8",
    )
    print(f"Historical files saved: {saved}")
    if saved == 0:
        raise RuntimeError("No historical season files found in the archive.")

def download_current(session):
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nDownloading 2026/27 current-season UCL CSV files...")
    for filename in FILES:
        url = f"{CURRENT_BASE_URL}/{filename}"
        destination = CURRENT_DIR / filename
        try:
            response = session.get(url, timeout=(20, 90))
            if response.status_code == 404:
                print(f"  NOT AVAILABLE: {filename} (source has not published it)")
                continue
            response.raise_for_status()
            if len(response.content) < 20:
                print(f"  EMPTY/TOO SMALL: {filename}")
                continue
            destination.write_bytes(response.content)
            print(f"  OK {filename}: {len(response.content):,} bytes")
        except requests.RequestException as exc:
            print(f"  FAILED {filename}: {exc}")

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    download_historical(session)
    time.sleep(1)
    download_current(session)
    print("\nDownload step finished.")
    print(f"Raw data location: {RAW_DIR}")
    print("Next: inspect the downloaded file formats before training the UCL model.")

if __name__ == "__main__":
    main()
