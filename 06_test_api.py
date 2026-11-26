import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("FOOTBALL_DATA_API_TOKEN")

if not TOKEN:
    print("ERROR: API token haijapatikana kwenye .env")
    exit()

url = "https://api.football-data.org/v4/competitions/PL"

headers = {
    "X-Auth-Token": TOKEN
}

print("Testing Football Data API...")

try:
    response = requests.get(url, headers=headers, timeout=20)

    if response.status_code == 200:
        data = response.json()
        print("SUCCESS: API connection works!")
        print("Competition:", data.get("name"))
        print("Code:", data.get("code"))
        print("Area:", data.get("area", {}).get("name"))

    elif response.status_code == 401:
        print("ERROR: Token is invalid or unauthorized.")

    elif response.status_code == 403:
        print("ERROR: Your API plan does not allow this request.")

    else:
        print("HTTP Status:", response.status_code)
        print(response.text[:300])

except requests.exceptions.Timeout:
    print("ERROR: Connection timed out.")

except requests.exceptions.RequestException as e:
    print("CONNECTION ERROR:", e)