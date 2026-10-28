import sqlite3
from pathlib import Path

import pandas as pd

from watches import watch_for

RAW = "data/runs_api.csv"
DB = "data/training.db"
LOG = "data/cleaning_log.txt"

df = pd.read_csv(RAW, parse_dates=["startTimeLocal"])
df = df.rename(columns={
    "activityId": "run_id",
    "startTimeLocal": "start_time",
    "averageHR": "avg_hr",
    "maxHR": "max_hr",
    "elevationGain": "elev_gain_m",
    "averageRunningCadenceInStepsPerMinute": "cadence_spm",
    "avgStrideLength": "stride_length",
    "avgGroundContactTime": "ground_contact_time",
    "avgVerticalOscillation": "vertical_oscillation",
    "activityTrainingLoad": "training_load",
    "aerobicTrainingEffect": "aerobic_te",
})
log = [f"Started with {len(df)} runs"]


def flag(mask, why, reason):
    """Mark runs that fail a rule. They stay in the database (and in weekly mileage), but the model skips them."""
    global df
    m = mask & df.exclude_reason.isna()
    df.loc[m, "exclude_reason"] = reason
    log.append(f"Flagged {int(m.sum())} runs: {why}")


before = len(df)
df = df.drop_duplicates(subset="run_id").copy()
log.append(f"Removed {before - len(df)} runs: duplicate activity ids")

df["exclude_reason"] = None
flag(~(df.distance_mi > 0.62), "shorter than 0.62 miles (1 km)", "short_distance")
flag(~(df.duration_min > 5), "shorter than 5 minutes", "short_duration")
flag(~df.pace_min_per_mi.between(4.0, 19.0), "pace outside 4:00 to 19:00 min/mile (GPS errors, walks)", "pace_outlier")
log.append("Flagged runs stay in weekly mileage totals but are left out of the heart rate model")

bad_hr = (df.avg_hr < 40) | (df.avg_hr > 220)
log.append(f"Set {int(bad_hr.sum())} heart rate values to missing: outside 40 to 220 bpm (sensor dropouts)")
df.loc[bad_hr, "avg_hr"] = None

df = df.sort_values("start_time").reset_index(drop=True)
df["watch"] = df.start_time.dt.strftime("%Y-%m-%d").apply(watch_for)  # which watch recorded the run
log.append(f"Finished with {len(df)} runs ({df.start_time.min().date()} to {df.start_time.max().date()}), "
           f"{int(df.exclude_reason.isna().sum())} usable for the model")

Path("data").mkdir(exist_ok=True)
con = sqlite3.connect(DB)
out = df.copy()
out["start_time"] = out.start_time.dt.strftime("%Y-%m-%d %H:%M:%S")
out.to_sql("runs", con, if_exists="replace", index=False)

# A view that groups runs by week (weeks start on Monday); it counts ALL runs, so it matches Garmin Connect
con.execute("DROP VIEW IF EXISTS weekly")
con.execute(
    """
    CREATE VIEW weekly AS
    SELECT date(start_time, 'weekday 0', '-6 days') AS week_start,
           COUNT(*)                         AS runs,
           ROUND(SUM(distance_mi), 1)       AS total_mi,
           -- weighted by distance, so a 0.7 mile stride session doesn't count as much as a 16 mile long run
           ROUND(SUM(pace_min_per_mi * distance_mi)
                 / SUM(CASE WHEN pace_min_per_mi IS NOT NULL THEN distance_mi END), 2) AS avg_pace_min_mi,
           ROUND(SUM(avg_hr * duration_min)
                 / SUM(CASE WHEN avg_hr IS NOT NULL THEN duration_min END), 1)         AS avg_hr
    FROM runs
    GROUP BY week_start
    ORDER BY week_start
    """
)
con.commit()
con.close()

Path(LOG).write_text("\n".join(log))
print("\n".join(log))
print(f"\nSaved to {DB}. Check the weekly view with:")
print(f"  sqlite3 {DB} 'SELECT * FROM weekly ORDER BY week_start DESC LIMIT 10;'")