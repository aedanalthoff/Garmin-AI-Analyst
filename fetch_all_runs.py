import time
from collections import Counter
from pathlib import Path

import pandas as pd

from garmin_auth import api_get, get_token

PATH = "/activitylist-service/activities/search/activities"
PAGE_SIZE = 50
M_PER_MILE = 1609.344  # meters in a mile
FT_PER_M = 3.28084      # feet in a meter
MAX_PAGES = 100  # safety stop so a bug can never loop forever

# To debug a week that looks short, set these dates; the script lists every activity in that range
DEBUG_START, DEBUG_END = "2026-10-12", "2026-10-18"

# Only the fields we need for modeling. No locations, names, or image URLs.
KEEP = [
    "activityId", "startTimeLocal", "distance", "duration", "movingDuration", "averageSpeed",
    "averageHR", "maxHR", "elevationGain", "averageRunningCadenceInStepsPerMinute",
    "avgStrideLength", "avgGroundContactTime", "avgVerticalOscillation",
    "activityTrainingLoad", "aerobicTrainingEffect", "calories",
]

token = get_token()
activities, start = [], 0

for page in range(MAX_PAGES):
    resp = api_get(PATH, token, params={"start": start, "limit": PAGE_SIZE})
    if resp.status_code == 429:
        raise SystemExit("Rate limited by Garmin. Wait a few minutes and run it again.")
    resp.raise_for_status()
    batch = resp.json()
    if not batch:  # an empty page means we have everything
        break
    activities.extend(batch)
    print(f"Page {page + 1}: got {len(batch)} (total {len(activities)})")
    start += PAGE_SIZE
    time.sleep(0.5)  # be polite to the API

def type_key(a):
    return (a.get("activityType") or {}).get("typeKey") or "unknown"


# Garmin has several kinds of running: running, trail_running, treadmill_running, track_running, ...
print("\nActivity types found:")
for key, n in Counter(type_key(a) for a in activities).most_common():
    print(f"  {key}: {n}")

runs = [a for a in activities if "run" in type_key(a)]  # any type with "run" in its name
print("\nIncluded as runs:", sorted({type_key(a) for a in runs}))
print(f"Total activities: {len(activities)}  |  runs: {len(runs)}")
if not runs:
    raise SystemExit("No runs found.")

print(f"\nEvery activity from {DEBUG_START} to {DEBUG_END}:")
for a in sorted(activities, key=lambda a: a["startTimeLocal"]):
    if DEBUG_START <= a["startTimeLocal"][:10] <= DEBUG_END:
        miles = (a.get("distance") or 0) / M_PER_MILE
        print(f"  {a['startTimeLocal']}  {type_key(a):<20} {miles:5.2f} mi")

df = pd.DataFrame([{k: a.get(k) for k in KEEP} for a in runs])
df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"])
df = df.sort_values("startTimeLocal").reset_index(drop=True)

# Garmin sends distance in meters, durations in seconds, speed in meters per second, elevation in meters
df["distance_mi"] = df["distance"] / M_PER_MILE
df["duration_min"] = df["duration"] / 60
df["pace_min_per_mi"] = M_PER_MILE / df["averageSpeed"] / 60
df["elev_gain_ft"] = df["elevationGain"] * FT_PER_M

print(f"Earliest run: {df.startTimeLocal.min()}")
print(f"Latest run:   {df.startTimeLocal.max()}")
print("\nSanity check, your most recent run (compare to your watch):")
last = df.iloc[-1]
print(last[["startTimeLocal", "distance_mi", "duration_min", "pace_min_per_mi", "averageHR"]])
m, s = divmod(round(last.pace_min_per_mi * 60), 60)
print(f"That pace is {m}:{s:02d} per mile")

Path("data").mkdir(exist_ok=True)
df.to_csv("data/runs_api.csv", index=False)
print("\nSaved data/runs_api.csv")
print("\nMissing values per column (share of runs with no data):")
print(df.isna().mean().round(2).sort_values(ascending=False).head(6))