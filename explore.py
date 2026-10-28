import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

con = sqlite3.connect("data/training.db")
weekly = pd.read_sql("SELECT * FROM weekly", con, parse_dates=["week_start"])
runs = pd.read_sql("SELECT * FROM runs", con, parse_dates=["start_time"])
runs["year"] = runs.start_time.dt.year
Path("charts").mkdir(exist_ok=True)


def mmss(minutes):
    """Turn decimal minutes (7.2) into a runner-friendly string (7:12)."""
    m, s = divmod(round(minutes * 60), 60)
    return f"{m}:{s:02d}"


# 1. Weekly distance, with a 4-week average to smooth out the noise
weekly["mi_4wk_avg"] = weekly.total_mi.rolling(4, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(weekly.week_start, weekly.total_mi, width=5, alpha=0.5, label="weekly miles")
ax.plot(weekly.week_start, weekly.mi_4wk_avg, color="black", label="4-week average")
ax.set_title("Weekly running distance")
ax.set_ylabel("miles")
ax.legend()
fig.tight_layout()
fig.savefig("charts/weekly_miles.png", dpi=150)

# 2. Pace vs heart rate, colored by year (are you fitter now than in 2019?)
ok = runs[runs.exclude_reason.isna()]  # runs that passed the cleaning rules
hr = ok.dropna(subset=["avg_hr"])
fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(hr.pace_min_per_mi, hr.avg_hr, c=hr.year, cmap="viridis", s=10, alpha=0.6)
fig.colorbar(sc, label="year")
ax.set_xlabel("pace (min/mile, lower is faster)")
ax.set_ylabel("average heart rate (bpm)")
ax.set_title("Pace vs heart rate")
fig.tight_layout()
fig.savefig("charts/pace_vs_hr.png", dpi=150)

# 3. Distance of each run over time (spot short workouts vs long runs)
fig, ax = plt.subplots(figsize=(11, 4))
ax.scatter(runs.start_time, runs.distance_mi, s=8, alpha=0.5)
ax.set_title("Distance of every run")
ax.set_ylabel("miles")
fig.tight_layout()
fig.savefig("charts/run_distances.png", dpi=150)

# Yearly summary table
summary = runs.groupby("year").agg(
    runs=("run_id", "count"),
    total_mi=("distance_mi", "sum"),
    median_mi=("distance_mi", "median"),
)
# Pace and heart rate medians use only runs that passed the cleaning rules
summary["median_pace"] = ok.groupby("year").pace_min_per_mi.median()
summary["median_hr"] = ok.groupby("year").avg_hr.median()
summary["hr_missing"] = ok.groupby("year").avg_hr.apply(lambda s: s.isna().mean())
summary = summary.round(2)
summary["median_pace_mmss"] = summary.median_pace.apply(mmss)  # min:sec per mile
print(summary)
print("\nCharts saved in charts/: weekly_miles.png, pace_vs_hr.png, run_distances.png")