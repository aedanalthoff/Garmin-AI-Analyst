import os

import requests
from dotenv import load_dotenv

BASE_URL = "https://personal.connectapi.garmin.com"
TOKEN_URL = f"{BASE_URL}/personalapitoken-service/personal-api/personal-api-key/token"


def get_token():
    """Exchange the API key in .env for a 24-hour access token. Never prints the token."""
    load_dotenv()
    key = os.environ.get("GARMIN_API_KEY", "").strip()
    if not key:
        raise SystemExit("GARMIN_API_KEY not found. Check that .env is in this folder and has the line GARMIN_API_KEY=...")
    resp = requests.post(TOKEN_URL, headers={"Authorization": f"ApiKey {key}"}, timeout=20)
    resp.raise_for_status()  # stops with a clear error if the key was rejected
    text = resp.text.strip()
    if text.startswith("{"):  # the body is either the bare token or JSON containing it
        return resp.json()["access_token"]
    return text


def api_get(path, token, params=None):
    """Make an authenticated GET request to the Personal API and return the response."""
    return requests.get(
        f"{BASE_URL}{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
    )