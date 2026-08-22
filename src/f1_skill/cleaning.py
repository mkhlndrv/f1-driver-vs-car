"""Which laps count, and which sessions were wet.

The validity rule is the load-bearing part. Without the deleted-lap half, `min(time)` picks up
aborted and cooldown laps: 625 deleted laps exist across the dataset, and the rule reproduces
FastF1's own verdict on Verstappen's aborted Miami 2023 Q3 lap without being told the answer.
"""

import pandas as pd

QUALIFYING_SEGMENTS = ["Q1", "Q2", "Q3"]


def is_valid(laps):
    return laps["iacc"].eq(True) & laps["del"].eq(False)


def numeric_times(laps):
    """`time` arrives as text: a lap with no time is the string 'None', not a null.

    Left as it comes, the column is object dtype, every derived gap is object dtype too, and the
    first `np.sqrt` downstream fails. Writing to CSV and reading back hides this by coercing on the
    way through, so the pipeline only works if it round-trips through disk.
    """
    return laps.assign(time=pd.to_numeric(laps["time"], errors="coerce"))


def valid_segment_laps(laps, segment):
    kept = laps[is_valid(laps) & laps["qs"].eq(segment)]
    return numeric_times(kept)


def rain_fraction(laps):
    """Share of a session's laps run in rain. 0 means dry throughout."""
    return laps.groupby(["season", "gp"])["wR"].apply(lambda wet: wet.eq(True).mean())
