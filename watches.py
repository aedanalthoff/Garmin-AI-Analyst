ERAS = [
    ("FR 35", "2019-01-01"),
    ("FR 245", "2019-12-25"),
    ("FR 945", "2023-09-14"),
    ("FR 955 Solar", "2026-03-10"),
]


def watch_for(day):
    """day is a 'YYYY-MM-DD' string. Returns the name of the watch in use that day."""
    name = ERAS[0][0]
    for era, start in ERAS:
        if day >= start:
            name = era
    return name


def switch_dates():
    """The dates when a new watch started, as (watch name, 'YYYY-MM-DD')."""
    return ERAS[1:]