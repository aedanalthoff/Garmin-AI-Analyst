import sqlite3

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

MIN_MI = 2  # short runs have unreliable average heart rate (it lags behind effort)
WINDOW = 30  # how many past runs define your "typical recent heart rate"

con = sqlite3.connect("data/training.db")
runs = pd.read_sql("SELECT * FROM runs ORDER BY start_time", con, parse_dates=["start_time"])

# Features built only from the PAST, so they never leak information from the run being predicted
runs["load_7d"] = [  # miles run in the 7 days before this run (every run counts toward training load)
    runs.loc[(runs.start_time < s) & (runs.start_time >= s - pd.Timedelta(days=7)), "distance_mi"].sum()
    for s in runs.start_time
]

# Only runs that passed the cleaning rules and are long enough are used for heart rate modeling
eligible = runs[runs.exclude_reason.isna() & (runs.distance_mi >= MIN_MI)].copy()
eligible["prev_hr"] = eligible.avg_hr.shift(1)  # heart rate of the previous eligible run
eligible["recent_hr"] = eligible.avg_hr.shift(1).rolling(WINDOW, min_periods=max(5, WINDOW // 3)).median()

BASE = ["pace_min_per_mi", "distance_mi", "elev_gain_ft", "cadence_spm", "load_7d"]
BASE = [c for c in BASE if eligible[c].notna().mean() > 0.7]  # skip columns that are mostly empty
WITH_RECENT = BASE + ["recent_hr"]
print("Base features:", BASE)

df = eligible.dropna(subset=WITH_RECENT + ["avg_hr", "prev_hr"]).reset_index(drop=True)
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
print(f"{len(train)} training runs ({train.start_time.min().date()} to {train.start_time.max().date()})")
print(f"{len(test)} test runs ({test.start_time.min().date()} to {test.start_time.max().date()})\n")

results = {
    "Baseline: previous run's HR": mean_absolute_error(test.avg_hr, test.prev_hr),
    f"Baseline: median HR of last {WINDOW} runs": mean_absolute_error(test.avg_hr, test.recent_hr),
    "Baseline: average HR of training runs": mean_absolute_error(test.avg_hr, [train.avg_hr.mean()] * len(test)),
}
models = {
    "Linear regression": (LinearRegression(), BASE),
    "Random forest": (RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0), BASE),
    "Linear regression + recent HR": (LinearRegression(), WITH_RECENT),
    "Random forest + recent HR": (RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0), WITH_RECENT),
}
for name, (model, feats) in models.items():
    model.fit(train[feats], train.avg_hr)
    results[name] = mean_absolute_error(test.avg_hr, model.predict(test[feats]))

lines = ["Mean absolute error in heart rate (bpm), lower is better:"]
lines += [f"  {k}: {v:.2f}" for k, v in results.items()]
best_baseline = min(v for k, v in results.items() if k.startswith("Baseline"))
best_model = min(v for k, v in results.items() if not k.startswith("Baseline"))
if best_baseline > 0:
    lines.append(f"\nBest model vs best baseline: {100 * (best_baseline - best_model) / best_baseline:+.1f}% change in error")
text = "\n".join(lines)
print(text)
open("data/model_results.txt", "w").write(text)