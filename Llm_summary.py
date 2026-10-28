import json
import os
import re
import sqlite3
import sys
from datetime import datetime

import pandas as pd

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("LLM_MODEL", "your-model-name")
N_WEEKS = int(os.environ.get("N_WEEKS", "12"))  # how many recent weeks to summarize
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4000"))  # reasoning models spend many tokens thinking before they answer
DRY_RUN = "--dry-run" in sys.argv
VERSION = "v6-richer-facts"  # printed at the start, so you can tell which copy of the script you are running
# Experimental: ask the server to turn off the model's thinking. Try:  NO_THINKING=1 python llm_summary.py
NO_THINKING = os.environ.get("NO_THINKING") == "1"
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}} if NO_THINKING else None

SYSTEM_PROMPT = (
    "Rewrite the facts in the JSON as one short, friendly paragraph addressed to 'you'. Use every fact once. "
    "Keep every number exactly as written. Do not add, remove, or repeat any fact, and add nothing else: "
    "no comments about consistency, effort, or how the week felt, and no advice. "
    "Reply with only the paragraph, with no introduction."
)

INTERPRETATION_WORDS = ["slower", "faster", "improv", "declin", "better", "worse", "tired", "fatigue",
                        "impressive", "great", "strong", "indicat", "suggest", "consistent", "steady",
                        "fluctuat", "intensity", "effort", "maintain", "throughout", "solid"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def mmss(minutes):
    """Turn decimal minutes (7.2) into a runner-friendly string (7:12)."""
    m, s = divmod(round(minutes * 60), 60)
    return f"{m}:{s:02d}"


def extra_week_numbers(con):
    """Per-week numbers that the weekly view doesn't have: longest run, run days, hours, climbing."""
    runs = pd.read_sql("SELECT start_time, distance_mi, duration_min, elev_gain_ft FROM runs", con,
                       parse_dates=["start_time"])
    monday = runs.start_time.dt.normalize() - pd.to_timedelta(runs.start_time.dt.weekday, unit="D")
    runs["week_start"] = monday.dt.strftime("%Y-%m-%d")
    g = runs.groupby("week_start")
    return pd.DataFrame({
        "longest_run_mi": g.distance_mi.max().round(1),
        "run_days": g.start_time.apply(lambda s: s.dt.date.nunique()),
        "total_hours": (g.duration_min.sum() / 60).round(1),
        "climb_ft": g.elev_gain_ft.sum().round(0),
    }).reset_index()


def weekly_stats(n_weeks):
    """Read the weekly view from the database and return the last n weeks as plain dictionaries."""
    con = sqlite3.connect("data/training.db")
    w = pd.read_sql("SELECT * FROM weekly ORDER BY week_start", con)
    w = w.merge(extra_week_numbers(con), on="week_start", how="left")
    prev4 = w.total_mi.shift(1).rolling(4).mean()  # average of the 4 weeks BEFORE this one
    w["avg_window_weeks"] = 4  # a number the facts mention, so the checker knows it is real
    w["avg_prev_4_weeks_mi"] = prev4.round(1)
    gap = (w.total_mi - prev4).round(1)
    w["vs_4_week_avg_mi"] = gap.abs()
    w["direction_vs_4_week_avg"] = gap.apply(lambda c: None if pd.isna(c) else ("more" if c > 0 else "fewer" if c < 0 else "same"))
    change = w.total_mi.diff().round(1)
    w["miles_vs_last_week"] = change.abs()  # positive number, so the model never writes "decreased by -5"
    w["direction_vs_last_week"] = change.apply(
        lambda c: None if pd.isna(c) else ("more" if c > 0 else "fewer" if c < 0 else "same")
    )
    w["avg_pace_per_mile"] = w.avg_pace_min_mi.apply(mmss)

    def sentence(c):  # the wording is decided here in code, so the model only has to copy it
        if pd.isna(c):
            return None
        if c == 0:
            return "the same mileage as the previous week"
        return f"{abs(c)} miles {'more' if c > 0 else 'fewer'} than the previous week"

    w["change_sentence"] = change.apply(sentence)

    def facts(row):  # every fact is written here in code, so the model has nothing to invent
        out = [f"You ran {int(row.runs)} time{'s' if row.runs != 1 else ''} for {row.total_mi} miles."]
        pace = f"Your average pace was {row.avg_pace_per_mile} per mile"
        out.append(pace + (f" and your average heart rate was {row.avg_hr} bpm." if pd.notna(row.avg_hr) else "."))
        out.append(f"Your longest run was {row.longest_run_mi} miles, spread over {int(row.run_days)} running days "
                   f"and {row.total_hours} hours.")
        if pd.notna(row.climb_ft) and row.climb_ft > 0:
            out.append(f"You climbed {int(row.climb_ft)} feet.")
        if isinstance(row.change_sentence, str):
            out.append(f"That is {row.change_sentence}.")
        if pd.notna(row.avg_prev_4_weeks_mi) and row.direction_vs_4_week_avg in ("more", "fewer"):
            out.append(f"Your average over the {int(row.avg_window_weeks)} weeks before was {row.avg_prev_4_weeks_mi} miles, "
                       f"so this week was {row.vs_4_week_avg_mi} miles {row.direction_vs_4_week_avg} than that average.")
        return out

    w["facts"] = w.apply(facts, axis=1)
    # Skip the current week: it isn't over, so its totals would look like a sudden drop in training
    finished = pd.to_datetime(w.week_start) + pd.Timedelta(days=7) <= pd.Timestamp.today().normalize()
    w = w[finished]
    return json.loads(w.tail(n_weeks).to_json(orient="records"))  # NaN becomes null


def generate_summary(stats):
    if DRY_RUN:  # the facts joined together, plus one deliberately wrong number to test the checker
        return " ".join(stats["facts"]) + f" You also gained {round((stats['total_mi'] or 0) + 3.7, 1)} feet."
    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key="lm-studio")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            max_tokens=MAX_TOKENS,
            extra_body=EXTRA_BODY,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"facts": stats["facts"]})},
            ],
        )
    except Exception as e:
        raise SystemExit(
            f"Could not get a response from LM Studio at {BASE_URL} with model '{MODEL}'.\n"
            f"Check that the server is running and LLM_MODEL matches the identifier LM Studio shows.\n{e}"
        )
    choice = resp.choices[0]
    text = choice.message.content or ""
    if choice.finish_reason == "length" and not text.strip():
        print(f"  The model ran out of tokens while thinking. Try a bigger limit: MAX_TOKENS={MAX_TOKENS * 2}")
    return clean(text)


def clean(text):
    """Drop visible reasoning tags and a chatty first line like 'Here is your summary:'."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    lines = text.splitlines()
    if lines and re.match(r"(here|sure|certainly|okay)", lines[0].strip(), re.I) and lines[0].strip().endswith(":"):
        text = "\n".join(lines[1:]).strip()
    return text


def remove_week_start(summary, week_start):
    """Writing the week's own start date in words ('October 12th') is fine, so don't count it as an extra number."""
    d = datetime.strptime(week_start, "%Y-%m-%d")
    name = MONTHS[d.month - 1]
    return re.sub(rf"\b(?:{name}|{name[:3]})\.?\s+{d.day}(?:st|nd|rd|th)?\b", " ", summary, flags=re.I)


def numbers_in(text):
    """Pull every number out of the text. Paces like 7:12 become 7.2 minutes."""
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)  # ignore ISO dates
    values = [int(m) + int(s) / 60 for m, s in re.findall(r"(\d+):(\d{2})", text)]
    text = re.sub(r"\d+:\d{2}", " ", text)
    words = r"(\d+)\s*(?:minutes?|mins?)(?:\s*and)?\s*(\d+)\s*(?:seconds?|secs?)"  # "7 minutes and 6 seconds"
    values += [int(m) + int(s) / 60 for m, s in re.findall(words, text, flags=re.I)]
    text = re.sub(words, " ", text, flags=re.I)
    values += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    return values


def direction_ok(summary, direction):
    """In sentences comparing with last week, the direction words must match the real direction."""
    if direction not in ("more", "fewer"):
        return True
    wrong = {
        "more": r"\b(fewer|less|decrease\w*|drop\w*|fell|down)\b",
        "fewer": r"\b(more|increase\w*|rose|up)\b",
    }[direction]
    for sentence in re.split(r"(?<=[.!?])\s+", summary):
        about_last_week = re.search(r"previous week|last week", sentence, re.I)
        about_average = re.search(r"average|\b4\b|four", sentence, re.I)
        if about_last_week and not about_average and re.search(wrong, sentence, re.I):
            return False
    return True


def check(summary, stats):
    """Return (accuracy, problems, direction_ok).
    Every number must match a value in stats, and the only date allowed is the week_start itself."""
    if not summary.strip():  # an empty answer must never count as a perfect score
        return 0.0, ["EMPTY SUMMARY"], False
    truth = [abs(float(v)) for v in stats.values() if isinstance(v, (int, float))]
    nums = numbers_in(remove_week_start(summary, stats["week_start"]))
    bad = [n for n in nums if not any(abs(n - t) <= max(0.06, 0.01 * t) for t in truth)]
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", summary)
    bad_dates = [d for d in dates if d != stats["week_start"]]
    claims = len(nums) + len(dates)
    accuracy = 1.0 if not claims else (claims - len(bad) - len(bad_dates)) / claims
    problems = bad + bad_dates
    if re.search(r"\bX\b", summary):  # an unfilled placeholder copied from the prompt
        problems.append("placeholder X")
        accuracy = min(accuracy, 0.5)
    return accuracy, problems, direction_ok(summary, stats.get("direction_vs_last_week"))


def interpretation(summary):
    """Words that make claims the data can't support (e.g. 'slower pace'). Reported as a warning, not scored."""
    return [w for w in INTERPRETATION_WORDS if w in summary.lower()]


def unsupported_sentences(summary):
    """A summary should only state facts from the data, so every sentence needs a number.
    A sentence with no number (like 'You kept a consistent pace') is a claim nothing in the data backs up."""
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    return [t for t in sentences if t and not re.search(r"\d", t)]


def is_perfect(r):
    return r["accuracy"] == 1.0 and r["direction_ok"] and not r["interpretation"] and not r["sentences_without_data"]


def main():
    print(f"llm_summary.py {VERSION} | model: {MODEL}")
    weeks = weekly_stats(N_WEEKS)
    if not weeks:
        raise SystemExit("No finished weeks found in data/training.db. Run clean_load.py first.")
    results = []
    for stats in weeks:
        summary = generate_summary(stats)
        acc, bad, dir_ok = check(summary, stats)
        interp = interpretation(summary)
        loose = unsupported_sentences(summary)
        results.append({"week_start": stats["week_start"], "stats": stats, "summary": summary,
                        "accuracy": acc, "unmatched": bad, "direction_ok": dir_ok, "interpretation": interp,
                        "sentences_without_data": loose})
        flags = (f"  unmatched: {bad}" if bad else "") + ("" if dir_ok else "  WRONG DIRECTION")
        flags += f"  unsupported claims: {interp}" if interp else ""
        flags += f"  SENTENCE WITH NO DATA: {loose[0][:60]!r}" if loose else ""
        print(f"{stats['week_start']}: accuracy {acc:.0%}{flags}")

    mean_acc = sum(r["accuracy"] for r in results) / len(results)
    perfect = sum(is_perfect(r) for r in results)
    print(f"\nAverage accuracy (numbers and dates): {mean_acc:.1%}  |  fully correct summaries: {perfect}/{len(results)}")
    flagged = [r for r in results if not is_perfect(r)]
    if flagged:
        print(f"\nRead the flagged summaries yourself ({len(flagged)} flagged, showing up to 4):")
        for r in flagged[:4]:
            print(f"\n{r['week_start']}: {r['summary']}")
    print("\nExample summary:\n" + results[-1]["summary"])
    with open("data/summaries.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved data/summaries.json")


if __name__ == "__main__":
    main()