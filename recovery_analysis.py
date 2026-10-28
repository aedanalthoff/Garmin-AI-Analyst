import math
import sqlite3

import numpy as np
import pandas as pd

MIN_MI = 2     # same minimum run length as the model
WINDOW = 30    # how many past runs define "recent typical heart rate"
MIN_DAYS = 40  # don't report a correlation built on fewer days than this

MEASURES = {
    "sleep_hours": "hours slept",
    "deep_hours": "deep sleep hours",
    "rem_hours": "REM sleep hours",
    "sleep_score": "sleep score",
    "sleep_stress": "stress during sleep",
    "resting_hr": "resting heart rate",
    "hrv_last_night": "HRV last night",
    "prev_day_stress": "yesterday's average stress",
}

con = sqlite3.connect("data/training.db")
runs = pd.read_sql("SELECT * FROM runs ORDER BY start_time", con, parse_dates=["start_time"])
health = pd.read_sql("SELECT * FROM daily_health", con)

# ---- Step 1: heart rate gap ----
runs["load_7d"] = [  # miles in the 7 days before each run (every run counts toward load)
    runs.loc[(runs.start_time < s) & (runs.start_time >= s - pd.Timedelta(days=7)), "distance_mi"].sum()
    for s in runs.start_time
]
eligible = runs[runs.exclude_reason.isna() & (runs.distance_mi >= MIN_MI)].copy()
eligible["recent_hr"] = eligible.avg_hr.shift(1).rolling(WINDOW, min_periods=max(5, WINDOW // 3)).median()
features = [f for f in ["pace_min_per_mi", "distance_mi", "elev_gain_ft", "load_7d", "recent_hr"]
            if eligible[f].notna().mean() > 0.7]
d = eligible.dropna(subset=features + ["avg_hr"]).copy()
watch_levels = pd.get_dummies(d.watch, drop_first=True, dtype=float)  # lets each watch have its own baseline
X = np.column_stack([np.ones(len(d))] + [d[f].to_numpy() for f in features] + [watch_levels[c].to_numpy() for c in watch_levels])
beta, *_ = np.linalg.lstsq(X, d.avg_hr.to_numpy(), rcond=None)
d["hr_gap"] = d.avg_hr - X @ beta
print(f"{len(d)} runs; the model explains heart rate with {features} plus a separate baseline for each watch")
print(f"Typical size of the gap: {d.hr_gap.std():.1f} bpm\n")

# ---- Step 2: one row per day ----
d["date"] = d.start_time.dt.strftime("%Y-%m-%d")
daily = (d.groupby(["date", "watch"])
          .apply(lambda g: pd.Series({"hr_gap": np.average(g.hr_gap, weights=g.distance_mi)}), include_groups=False)
          .reset_index())

# ---- Step 3: attach last night's sleep and yesterday's stress ----
prev = health[["date", "stress_avg"]].copy()
prev["date"] = (pd.to_datetime(prev.date) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")  # yesterday's value, filed under today
prev = prev.rename(columns={"stress_avg": "prev_day_stress"})
daily = daily.merge(health, on="date", how="left").merge(prev, on="date", how="left")


def p_value(r, n):
    t = abs(r) * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    try:
        from scipy import stats
        return float(2 * stats.t.sf(t, n - 2))
    except ImportError:  # normal approximation is fine for n in the dozens
        return math.erfc(t / math.sqrt(2))


daily["month"] = daily.date.str[5:7]
rows, tests = [], 0
for watch, g in daily.groupby("watch"):
    for col, label in MEASURES.items():
        sub = g.dropna(subset=["hr_gap", col])
        n = len(sub)
        if watch == "FR 35" or n < MIN_DAYS or sub[col].nunique() < 3:  # the FR 35 era is too small and has little data
            continue
        r_spear = sub.hr_gap.rank().corr(sub[col].rank())
        gap_adj = sub.hr_gap - sub.groupby("month").hr_gap.transform("mean")  # compare within the same month only
        col_adj = sub[col] - sub.groupby("month")[col].transform("mean")
        r_adj = gap_adj.rank().corr(col_adj.rank())
        bpm_per_sd = sub.hr_gap.corr(sub[col]) * sub.hr_gap.std()  # change in the gap for +1 standard deviation
        rows.append({"watch": watch, "measure": label, "days": n, "spearman_r": r_spear,
                     "bpm_per_sd": bpm_per_sd, "p": p_value(r_spear, n),
                     "r_same_month": r_adj, "p_same_month": p_value(r_adj, n)})
        tests += 1

if not rows:
    raise SystemExit("Not enough overlapping days. Run python python/fetch_health.py and clean_load.py first.")

res = pd.DataFrame(rows)
res["flag"] = np.where(res.p < 0.05 / tests, "**", np.where(res.p < 0.05, "*", ""))
res = res.sort_values(["watch", "p"])
pd.set_option("display.width", 140)
print(res.assign(spearman_r=res.spearman_r.round(2), bpm_per_sd=res.bpm_per_sd.round(2), p=res.p.round(3),
                 r_same_month=res.r_same_month.round(2), p_same_month=res.p_same_month.round(3))
         .to_string(index=False))

print("\nDoes the direction repeat across watches? (Spearman r; a real effect should keep its sign)")
print(res.pivot(index="measure", columns="watch", values="spearman_r").round(2).to_string())

print(f"\n{tests} tests were run. At p < 0.05 you should expect about {tests * 0.05:.0f} 'significant' results by pure luck.")
print("'*' means p < 0.05. '**' means it also survives a Bonferroni correction (p < 0.05 divided by the number of tests).")
print("bpm_per_sd: how much the heart rate gap changes when the measure rises by one standard deviation.")
print("r_same_month / p_same_month: the same test comparing days within the same calendar month only. Heat and season raise")
print("both resting heart rate and running heart rate, so a result that vanishes here may just be the weather.")
print("Correlation is not causation: hard weeks lower HRV and also change your runs. Report the null results too.")