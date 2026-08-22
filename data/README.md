# data/

**Nothing in here is committed except this file.** The datasets are large and reproducible from code,
so the loader ships and the data does not.

## What lives here

```
data/
  raw/        as fetched, unmodified
  processed/  one file per pipeline stage, so each step is inspectable
```

## Where it comes from

**The TracingInsights GitHub mirror**, one repository per season for 2018 to 2025, Apache 2.0. Per
qualifying session it serves `session_laptimes.json` (lap level, 43 fields including the deleted-lap
flag, the Q1/Q2/Q3 segment and per-lap rainfall), `weather.json`, `rcm.json` for race control, and
`drivers.json`. `TracingInsights/RaceData` mirrors F1DB/Ergast for race results, grid and standings,
which the baseline model uses.

**Not FastF1 directly, and the reason matters.** F1's live timing CDN returns 403 to datacenter IPs,
so on Kaggle `session.load()` degrades silently: the results table loads while weather, lap timing,
telemetry and race control all fail. The mirror serves the same bytes over `raw.githubusercontent.com`,
which Kaggle can reach. Verified byte-identical against a local FastF1 fetch, and the schema was
checked across all eight seasons.

`raw/fastf1_cache/` is a local FastF1 cache from that verification. It is not needed to reproduce
anything.

## Stages

| file | what it is |
|---|---|
| `processed/q1_pace.csv` | one row per driver-session: best Q1 lap, percentage gap to the session's fastest, valid-lap count, plus `within_107` and `rain_fraction` flags so the wet and 107% variants stay runnable as sensitivity checks rather than being baked in |
| `processed/model_data.csv` | `q1_pace.csv` after the 107% rule, which is what the models are fitted on |

3,029 driver-sessions before the 107% rule and 3,008 after, from 156 usable sessions, 43 drivers,
19 team names (10 constructors) and 8 seasons.

## Regenerating

Once the pipeline is ported to `src/f1_skill`, `make train` rebuilds everything from the mirror. The
notebooks in `notebooks/` currently fetch it themselves and run on Kaggle, where the data arrives as
an attached dataset rather than from this directory.
