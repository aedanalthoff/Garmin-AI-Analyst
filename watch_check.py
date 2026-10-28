import sqlite3

import numpy as np
import pandas as pd

from watches import switch_dates

WINDOWS = [56, 112, 168]  # compare this many days before with the same number after each switch
MIN_MI = 2        # same minimum run length as the model

con = sqlite3.connect("data/training.db")
runs = pd.read_sql("SELECT * FROM runs", con, parse_dates=["start_time"])
runs = runs[runs.exclude_reason.isna() & (runs.distance_mi >= MIN_MI)].dropna(subset=["avg_hr", "pace_min_per_mi"])

print("Runs per watch (used for heart rate analysis):")
print(runs.groupby("watch").agg(runs=("run_id", "count"), first=("start_time", "min"),
                                median_hr=("avg_hr", "median"), median_pace=("pace_min_per_mi", "median")).round({"median_hr": 1, "median_pace": 2}))

print("\nChange in heart rate at the same pace and distance, before vs after each watch switch.")
print("Each row also allows a steady fitness trend, so slow improvement isn't mistaken for a sensor change.")
for watch, day in switch_dates():
    switch = pd.Timestamp(day)
    print(f"\n{watch} (switched {day}):")
    for window in WINDOWS:
        near = runs[(runs.start_time >= switch - pd.Timedelta(days=window)) &
                    (runs.start_time < switch + pd.Timedelta(days=window))].copy()
        near["after"] = (near.start_time >= switch).astype(int)
        n_before, n_after = int((near.after == 0).sum()), int((near.after == 1).sum())
        if min(n_before, n_after) < 12:
            print(f"  +/-{window:>3} days: too few runs ({n_before} before, {n_after} after)")
            continue
        days = (near.start_time - switch).dt.days.to_numpy()  # steady trend in time, centered on the switch
        X = np.column_stack([np.ones(len(near)), near.pace_min_per_mi, near.distance_mi, days, near.after])
        y = near.avg_hr.to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = resid @ resid / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
        effect, err = beta[-1], se[-1]
        verdict = "clear shift" if abs(effect) > 2 * err else "no clear shift"
        print(f"  +/-{window:>3} days: {effect:+5.1f} bpm (give or take {err:.1f})   "
              f"[{n_before} runs before, {n_after} after]  -> {verdict}")

print("\nHow to read this: 'give or take' is the standard error. A shift larger than about twice that is unlikely to be")
print("chance. If the three windows agree, the estimate is more trustworthy. A wide 'give or take' means the test")
print("cannot rule out a shift of a few bpm, which is different from proving there is none.")