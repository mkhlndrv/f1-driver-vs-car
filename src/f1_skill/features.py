"""The modelling table: one row per driver-session, measured in Q1.

Q1 because it is the only segment every driver runs. Q2 and Q3 entry is decided by pace, so
comparing drivers there selects on the thing being measured, and the track gains about 0.46s
between Q1 and Q2 as rubber goes down.

The target is the percentage gap to the session's fastest Q1 lap, which puts Monaco and Spa on one
scale.
"""

from f1_skill.cleaning import rain_fraction, valid_segment_laps

SESSION_KEYS = ["season", "gp"]
DRIVER_SESSION_KEYS = ["season", "gp", "team", "drv"]

# F1's own rule: fail to lap within 107% of the fastest Q1 time and you do not qualify. Applied
# here as a flag rather than a filter so the variants stay runnable as sensitivity checks.
PERCENT_OVER_FASTEST = 7

STAGE_A_COLUMNS = [
    "season",
    "gp",
    "team",
    "drv",
    "best_lap",
    "n_valid_laps",
    "sesT",
    "gap_to_fastest",
    "rain_fraction",
    "within_107",
]


def q1_pace(laps):
    q1 = valid_segment_laps(laps, "Q1")
    driver_session = (
        q1.groupby(DRIVER_SESSION_KEYS)["time"]
        .agg(best_lap="min", n_valid_laps="size")
        .reset_index()
    )

    # When in the session the best lap was set. Carried through unused: Q1 runs about eighteen
    # minutes and the track rubbers in, so this could be a covariate once the format is checked.
    fastest_lap_rows = q1.loc[
        q1.groupby(DRIVER_SESSION_KEYS)["time"].idxmin(), DRIVER_SESSION_KEYS + ["sesT"]
    ]
    driver_session = driver_session.merge(fastest_lap_rows, on=DRIVER_SESSION_KEYS, how="left")

    session_fastest = driver_session.groupby(SESSION_KEYS)["best_lap"].transform("min")
    driver_session["gap_to_fastest"] = (
        (driver_session["best_lap"] - session_fastest) / session_fastest * 100
    )
    driver_session = driver_session.join(
        rain_fraction(laps).rename("rain_fraction"), on=SESSION_KEYS
    )
    driver_session["within_107"] = driver_session["gap_to_fastest"] <= PERCENT_OVER_FASTEST
    return driver_session[STAGE_A_COLUMNS]
