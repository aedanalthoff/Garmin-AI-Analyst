import json
import time
from datetime import date, timedelta

from garmin_auth import api_get, get_token

END = date.today() - timedelta(days=1)  # yesterday
EARLIEST = date(2019, 1, 1)             # your first run is from early 2019


def fetch(name, s, e, token):
    """Return decoded JSON for one endpoint and date range, or None if there is nothing usable."""
    paths = {
        "sleep": ("/sleep-service/sleep/dailySleepSummariesByDate", {"startDate": s, "endDate": e}),
        "body battery": ("/wellness-service/wellness/bodyBattery/reports/daily", {"startDate": s, "endDate": e}),
        "stress": (f"/usersummary-service/stats/stress/daily/{s}/{e}", None),
        "HRV": (f"/hrv-service/hrv/daily/{s}/{e}", None),
        "daily summaries": ("/usersummary-service/v2/dailysummaries", {"startDate": s, "endDate": e}),
    }
    path, params = paths[name]
    resp = api_get(path, token, params=params)
    if not resp.ok:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def items(name, data):
    """The list of daily items inside a response."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("hrvSummaries") or data.get("data") or []
    return []


def shape(x, depth=0, max_depth=2):
    """Describe nested structure: field names and value TYPES, never the values themselves."""
    if isinstance(x, dict):
        if depth >= max_depth:
            return f"object with {len(x)} fields"
        return {k: shape(v, depth + 1, max_depth) for k, v in x.items()}
    if isinstance(x, list):
        return "empty list" if not x else [f"list of {len(x)}", shape(x[0], depth + 1, max_depth)]
    return type(x).__name__


token = get_token()

print("===== PART A: structure for yesterday (types only) =====")
for name in ["sleep", "stress", "HRV", "daily summaries", "body battery"]:
    data = fetch(name, END, END, token)
    day = items(name, data)
    print(f"\n--- {name}: {len(day)} item(s) ---")
    if not day:
        continue
    first = day[0]
    if name == "sleep":
        # the top-level fields were listed last time, so look inside the nested ones
        for key in ["sleepScores", "sleepNeed"]:
            print(f"{key}:", json.dumps(shape(first.get(key)), indent=1))
    elif name == "body battery":
        print("value descriptors:", json.dumps(shape(first.get("bodyBatteryValueDescriptorDTOList")), indent=1))
        print("values array:", json.dumps(shape(first.get("bodyBatteryValuesArray")), indent=1))
    else:
        print(json.dumps(shape(first, max_depth=2), indent=1))

print("\n===== PART B: how many days of data exist back through time (one sample week each) =====")
names = ["sleep", "body battery", "stress", "HRV", "daily summaries"]
print(f"{'week ending':<13}" + "".join(f"{n:<17}" for n in names))
ref = END
while ref >= EARLIEST:
    counts = []
    for n in names:
        data = fetch(n, ref - timedelta(days=6), ref, token)
        counts.append(str(len(items(n, data))))
        time.sleep(0.3)  # be polite to the API
    print(f"{ref!s:<13}" + "".join(f"{c:<17}" for c in counts))
    ref -= timedelta(days=182)