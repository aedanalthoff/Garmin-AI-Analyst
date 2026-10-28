from garmin_auth import api_get, get_token

token = get_token()
resp = api_get(
    "/activitylist-service/activities/search/activities",
    token,
    params={"start": 0, "limit": 5},
)
print("Status:", resp.status_code)
if not resp.ok:
    raise SystemExit(resp.text[:200])

activities = resp.json()
print("Activities returned:", len(activities))

if activities:
    print("\nField names in the first activity:")
    print(sorted(activities[0].keys()))
    print("\nid, start time, type:")
    for a in activities:
        kind = (a.get("activityType") or {}).get("typeKey")
        print(a.get("activityId"), a.get("startTimeLocal"), kind)
