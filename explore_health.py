from datetime import date, timedelta

from garmin_auth import api_get, get_token

end = date.today() - timedelta(days=1)  # yesterday
start = end - timedelta(days=6)
old_end = end - timedelta(days=365)
old_start = start - timedelta(days=365)

# (name, path, params). Dates go in the path for some endpoints and in the query string for others.
def endpoints(s, e):
    return [
        ("sleep summaries", "/sleep-service/sleep/dailySleepSummariesByDate", {"startDate": s, "endDate": e}),
        ("HRV summaries", f"/hrv-service/hrv/daily/{s}/{e}", None),
        ("body battery", "/wellness-service/wellness/bodyBattery/reports/daily", {"startDate": s, "endDate": e}),
        ("stress", f"/usersummary-service/stats/stress/daily/{s}/{e}", None),
        ("daily summaries", "/usersummary-service/v2/dailysummaries", {"startDate": s, "endDate": e}),
        ("SpO2 (one day)", f"/wellness-service/wellness/daily/spo2acclimation/{e}", None),
    ]


def describe(data):
    """Describe the shape of a JSON response without showing any values."""
    if data is None:
        return "no content"
    if isinstance(data, list):
        first = data[0] if data else None
        keys = sorted(first.keys()) if isinstance(first, dict) else type(first).__name__
        return f"list of {len(data)} items; fields of the first item: {keys}"
    if isinstance(data, dict):
        return f"object with fields: {sorted(data.keys())}"
    return type(data).__name__


token = get_token()
for label, s, e in [("RECENT WEEK", start, end), ("ONE YEAR AGO", old_start, old_end)]:
    print(f"\n===== {label}: {s} to {e} =====")
    for name, path, params in endpoints(s, e):
        resp = api_get(path, token, params=params)
        if not resp.ok:
            print(f"{name}: HTTP {resp.status_code} {resp.text[:80]!r}")
            continue
        try:
            print(f"{name}: OK, {describe(resp.json())}")
        except ValueError:
            print(f"{name}: OK, but the response was not JSON")