import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from garmin_auth import api_get, get_token

DB = "data/training.db"
FIRST_DAY = date(2019, 1, 1)  # your first run is from early 2019
CHUNK_DAYS = 28               # days per request; the script halves the range if Garmin rejects it
PAUSE = 0.4                   # seconds between requests, to be polite
OVERLAP_DAYS = 3              # on reruns, re-fetch the last few days (recent days get revised)


def g(d, *keys):
    """Safely read nested fields: g(x, 'heartRate', 'restingValue') returns None if anything is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def hours(seconds):
    return round(seconds / 3600, 2) if seconds else None


def sleep_row(x):
    return {
        "date": x.get("calendarDate"),
        "sleep_hours": hours(x.get("sleepTimeSeconds")),
        "deep_hours": hours(x.get("deepSleepSeconds")),
        "rem_hours": hours(x.get("remSleepSeconds")),
        "light_hours": hours(x.get("lightSleepSeconds")),
        "awake_hours": hours(x.get("awakeSleepSeconds")),
        "awake_count": x.get("awakeCount"),
        "sleep_score": g(x, "sleepScores", "overall", "value"),
        "sleep_stress": x.get("avgSleepStress"),
        "sleep_spo2_avg": x.get("averageSpO2Value"),
        "sleep_spo2_low": x.get("lowestSpO2Value"),
        "sleep_respiration": x.get("averageRespirationValue"),
        "sleep_hr": x.get("avgHeartRate"),  # only present for newer nights
    }


def daily_row(x):
    return {
        "date": x.get("calendarDate"),
        "resting_hr": g(x, "heartRate", "restingValue"),
        "stress_avg": g(x, "stress", "avgLevel"),
        "bb_max": g(x, "bodyBattery", "maxValue"),
        "bb_min": g(x, "bodyBattery", "minValue"),
        "bb_charged": g(x, "bodyBattery", "chargedValue"),
        "bb_drained": g(x, "bodyBattery", "drainedValue"),
        "respiration_avg": g(x, "respiration", "avgValue"),
        "pulse_ox_avg": g(x, "pulseOx", "avgValue"),
        "intensity_moderate": g(x, "intensityMinutes", "moderate"),
        "intensity_vigorous": g(x, "intensityMinutes", "vigorous"),
        "active_calories": g(x, "calories", "burnedActive"),
    }


def hrv_row(x):
    return {
        "date": x.get("calendarDate"),
        "hrv_last_night": x.get("lastNightAvg"),
        "hrv_weekly_avg": x.get("weeklyAvg"),
        "hrv_5min_high": x.get("lastNight5MinHigh"),
        "hrv_status": x.get("status"),
    }


def request(kind, s, e, token):
    """One API call. Returns a list of day records, or None if Garmin refused the request."""
    if kind == "sleep":
        resp = api_get("/sleep-service/sleep/dailySleepSummariesByDate", token, {"startDate": s, "endDate": e})
    elif kind == "daily":
        resp = api_get("/usersummary-service/v2/dailysummaries", token, {"startDate": s, "endDate": e})
    else:
        resp = api_get(f"/hrv-service/hrv/daily/{s}/{e}", token)
    time.sleep(PAUSE)
    if resp.status_code == 429:
        print("  Rate limited; waiting 60 seconds...")
        time.sleep(60)
        return request(kind, s, e, token)
    if not resp.ok:
        return None
    try:
        data = resp.json()
    except ValueError:  # no content, which is normal for dates before the feature existed
        return []
    if isinstance(data, dict):
        data = data.get("hrvSummaries") or data.get("data") or []
    return data or []


def fetch_range(kind, s, e, token):
    """Fetch a date range, splitting it in half if Garmin rejects it. A single day that fails is skipped."""
    out = request(kind, s, e, token)
    if out is not None:
        return out
    if s >= e:
        print(f"  Skipped {kind} for {s} (request failed)")
        return []
    mid = s + (e - s) // 2
    return fetch_range(kind, s, mid, token) + fetch_range(kind, mid + timedelta(days=1), e, token)


def main():
    Path("data").mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    existing = pd.DataFrame()
    start = FIRST_DAY
    try:
        existing = pd.read_sql("SELECT * FROM daily_health", con)
        start = max(FIRST_DAY, date.fromisoformat(existing.date.max()) - timedelta(days=OVERLAP_DAYS))
        print(f"daily_health has {len(existing)} days; refreshing from {start}")
    except Exception:
        print(f"Starting fresh from {start}")

    token = get_token()
    end = date.today()
    rows = {"sleep": [], "daily": [], "hrv": []}
    s = start
    while s <= end:
        e = min(s + timedelta(days=CHUNK_DAYS - 1), end)
        counts = []
        for kind, make in [("sleep", sleep_row), ("daily", daily_row), ("hrv", hrv_row)]:
            got = [make(x) for x in fetch_range(kind, s, e, token)]
            rows[kind] += [r for r in got if r["date"]]
            counts.append(f"{kind} {len(got)}")
        print(f"{s} to {e}: " + " | ".join(counts))
        s = e + timedelta(days=1)

    frames = [pd.DataFrame(r).drop_duplicates("date") for r in rows.values() if r]
    if not frames:
        raise SystemExit("No health data came back. Check the token and try python python/probe_health.py.")
    new = frames[0]
    for f in frames[1:]:
        new = new.merge(f, on="date", how="outer")

    combined = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    combined = combined.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    combined.to_sql("daily_health", con, if_exists="replace", index=False)
    con.close()

    print(f"\nSaved {len(combined)} days to daily_health ({combined.date.min()} to {combined.date.max()})")
    print("\nFirst date with data, and share of days with data, for each measurement:")
    for col in combined.columns.drop("date"):
        has = combined[combined[col].notna()]
        first = has.date.min() if len(has) else "never"
        print(f"  {col:<20} first: {first}   coverage: {len(has) / len(combined):.0%}")


if __name__ == "__main__":
    main()