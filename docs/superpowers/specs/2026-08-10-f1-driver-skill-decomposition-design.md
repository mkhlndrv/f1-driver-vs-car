# Design: F1 Driver-vs-Car Skill Decomposition

**Date:** 2026-08-10
**Status:** Approved by user, pending implementation plan

## The Question

*How much of a Formula 1 result is the driver, and how much is the car?*

This is the entire project. Every component exists to serve this question or to
support communicating its answer. Nothing is included because it is a common or
expected thing to have in an F1 project.

## Why This Framing

An earlier draft scoped this as a four-component "F1 Race Intelligence System"
(race predictor, driver-skill decomposition, pit-strategy optimizer, race
simulator), with all four weighted roughly equally. That draft was rejected for
this project's actual goal — a junior data-science portfolio piece intended to
read as clearly above-junior-level to anyone who reads it carefully.

The reasoning: four medium-effort components read as "did several standard ML
tasks," each individually unremarkable next to someone who spent all their time
on just one. Race prediction alone is a common project type (the F1 equivalent
of Titanic). Depth on one hard, genuinely contested question — handled with
visible statistical rigor, honest treatment of confounders, and a clear written
finding — is a rarer and stronger signal at the junior level than breadth.

No project design "guarantees" standout results; hiring outcomes depend on who
reads a portfolio and how carefully. This design maximizes the odds conditional
on being read, and its value is realized only through careful execution, not by
the plan existing on paper.

## Explicitly Out of Scope (and why)

| Cut | Reason |
|---|---|
| Pit-strategy optimizer | Different problem domain (operations research); doing it well or poorly says nothing about driver vs. car. Dilutes focus without adding rigor to the core question. Also a common project type on its own. |
| Race simulator | High effort relative to relevance — mostly a visualization/demo exercise, not a statistical contribution to the core question. Effort here would compete directly with time needed to learn Bayesian methods properly. |
| Full 1950–present coverage | Two problems: (1) FastF1 data quality/reliability is poor before ~2018 (sparse, manually-timed); (2) comparing driver skill across eras conflates skill with era-specific confounds (regulations, competitive field, car category) — an additional confound stacked on the one already being fought. A tighter, higher-quality window also gives denser teammate-pair connectivity for the model. |

These are cuts, not deferrals. Any of them could be added later as an explicit
stretch goal, clearly marked as secondary to the core question — not folded
back in as a co-equal component.

## Data Pipeline

**Source:** FastF1 (Python library), local caching to avoid repeated API calls.

**Coverage:** 2018–2025 seasons, qualifying and race sessions.

**Primary metric:** qualifying pace expressed as percentage gap to the
session's fastest lap — more robust across tracks/eras than raw seconds, since
lap length varies by circuit.

**Messy cases, handled explicitly rather than silently dropped:**
- **DNFs / no representative quali lap** — excluded from pace calculation for
  that session, but tracked as a separate flag/count rather than discarded.
- **Wet/mixed sessions** — flagged via FastF1 track-status/weather data, not
  excluded by default; carried as a covariate so the sensitivity analysis can
  test the effect of dropping them.
- **Sprint-weekend shootout sessions (2021+)** — treated as a distinct session
  type from regular qualifying; inclusion/exclusion is a sensitivity-analysis
  variant, not a default assumption.
- **Mid-season driver swaps** — handled naturally by the model structure (see
  Methodology): driver and constructor are separate per-row random effects,
  not season-level assignments, so a driver with 3 races for a team still
  contributes those 3 data points correctly.
- **Rookies / low-race-count drivers** — no manual filtering; partial pooling
  in the hierarchical model pulls sparse-data drivers toward the group mean
  with wide uncertainty, rather than either excluding them or overweighting a
  tiny sample.

**Pipeline:**
```
src/f1_skill/data.py       # FastF1 fetch -> data/raw/ (gitignored cache)
src/f1_skill/cleaning.py   # DNF/wet/sprint handling -> clean session table
src/f1_skill/features.py   # pace-gap computation, covariates -> data/processed/
```
The processed dataset (thousands of rows: driver × session) is small enough to
regenerate quickly from raw FastF1 data via one command; it is not committed
(per the never-commit-data convention), only the loader/pipeline code is.

## Core Methodology

### Primary model: Bayesian hierarchical (crossed random-effects)

```python
model = bmb.Model(
    "quali_pace_gap ~ 1 + (1|driver) + (1|constructor) + (1|season)",
    data=quali_df, family="gaussian",
)
```

Implemented via Bambi (formula-based interface on top of PyMC). Chosen over a
hand-rolled Bayesian Elo/TrueSkill network as the *primary* method for three
reasons: it is grounded in published methodology for this exact question
(Bell, Smith, Sabel & Jones, 2016, "Formula for success: multilevel modelling
of Formula One driver and constructor performance, 1950–2014," *Journal of
Quantitative Analysis in Sports*); it produces driver skill, car strength, and
variance decomposition from one coherent model rather than two stitched-together
ones; and it is less implementation-risk than hand-rolling a custom Bayesian
pairwise-update algorithm from scratch, which matters given the user is
learning Bayesian methods from zero for this project.

Diagnostics are a required part of the deliverable, not an afterthought:
R-hat convergence, effective sample size, trace plots, and a posterior
predictive check. Skipping these would be exactly the kind of hand-waving that
fails scrutiny from an F1-literate or stats-literate reviewer.

**Variance decomposition:** from the fitted model's random-effect variance
components (driver / constructor / residual), compute each as a share of total
variance (standard multilevel-model variance partitioning). This is the direct,
quantified answer to the headline question — e.g., "constructor effects
account for ~65% of pace variance, driver effects ~25%, residual ~10%"
(illustrative, not a predicted result).

### Secondary model: Bayesian Elo/TrueSkill network (robustness check)

Built from teammate qualifying-gap comparisons only (same car, same session —
the cleanest possible pairwise signal), propagated across seasons through
shared teammates. Its role is narrow and explicit: check whether an
independent methodology produces a driver ranking consistent with the primary
model's, reported via rank correlation (Spearman). Agreement strengthens the
finding; disagreement for specific drivers is reported honestly, not smoothed
over — this triangulation, not a second primary method, is the reason it
earns a place in scope.

### Sensitivity analysis

Reruns the primary model under variant assumptions and compares the resulting
driver ranking (rank correlation) against the baseline run:
1. Qualifying-only pace vs. blended qualifying + race pace
2. Wet/mixed sessions included vs. excluded
3. Sprint-weekend shootouts included vs. excluded
4. Alternative DNF handling (exclude vs. down-weight)

This mirrors the sensitivity-analysis approach already validated in the user's
prior churn-prediction portfolio project, and is what separates "built a
model" from "interrogated a model."

## Supporting Component: Baseline Race-Outcome Model

Exists for two specific reasons, not as filler: it demonstrates conventional
supervised-ML competency (some reviewers specifically look for this and the
decomposition work doesn't show it), and it gives the writeup a contrast point
— a standard predictive model treats car and driver as inseparable, unlike the
core analysis.

- **Target:** podium finish or points finish (binary classification), not raw
  finishing-position regression — position is discrete, bounded, and
  drivers' positions aren't independent within a race, which raw regression
  misrepresents.
- **Features:** grid position, constructor, driver, season-to-date form,
  circuit characteristics — all pre-race-known. Explicitly excludes any
  feature derived from the hierarchical model's ratings, so it remains an
  independent, non-circular baseline to contrast against.
- **Model:** gradient-boosted trees (LightGBM or XGBoost) — standard,
  defensible choice for mixed categorical/numeric tabular data, handles
  categoricals natively (per the project's own data-science-gotchas
  conventions), and is fast enough not to compete for time with the Bayesian
  work, which is the actual point of the project.
- **Deliberately not included:** deep hyperparameter tuning, ensembling, or
  feature-engineering sophistication. This component is intentionally
  "competent, not exceptional" — over-investing here would undercut the whole
  premise of the redesign.

## Dashboard (Streamlit)

Not a duplicate of the writeup — it exists because letting a visitor
manipulate the findings interactively (pick a driver, change a season) is more
memorable than static prose alone, and Streamlit was chosen specifically
because it is Python-native and plays to the user's existing skills rather
than requiring web-development work that would compete for project time.

**Views**, each tied to a specific finding:
1. **Headline finding** — variance decomposition with credible intervals,
   stated plainly.
2. **Driver skill trajectory** — per-driver skill estimate over time with
   uncertainty bands (visibly widening in low-race-count seasons).
3. **Driver-vs-car scatter** — driver skill vs. constructor strength for a
   chosen season.
4. **Model agreement view** — primary vs. secondary model rankings side by
   side with their rank correlation.
5. **Sensitivity view** — ranking stability across the variant runs.

**Data contract:** the dashboard reads only pre-computed artifacts (posterior
summaries, decomposition results, sensitivity outputs — committed contents of
`models/` and `reports/`), never refits MCMC live. Live refitting would be slow
on free hosting and is unnecessary, since the analysis artifacts are static
once produced.

**Deployment:** Streamlit Community Cloud — free, deploys directly from the
GitHub repo, standard expected hosting for a Streamlit portfolio piece.

## Deliverables

1. **Technical writeup** (README, primary deliverable) — methodology,
   findings, confounders addressed and not fully resolved, what can and can't
   be concluded. Most junior candidates don't write; this is the intended
   differentiator.
2. **Ranking with uncertainty** — car-adjusted driver skill ratings with
   credible intervals.
3. **Streamlit dashboard** — interactive companion to the writeup.
4. **GitHub repository**, structured per this user's CLAUDE.md conventions
   (see below).

## Success Criteria

This is an inferential/decomposition project, not a standard prediction
problem, so success is multi-part rather than a single metric to beat:
1. Primary model MCMC convergence (R-hat ≈ 1.0, adequate effective sample
   size) and a passing posterior predictive check.
2. Meaningful positive rank correlation between the primary and secondary
   (Elo) model rankings, reported honestly even if imperfect.
3. Sensitivity analysis showing the ranking is reasonably stable under
   plausible variable choices — or, if it isn't, an honest account of where
   and why it breaks down.
4. The baseline race-outcome model reaches a conventional, reported metric
   (e.g., precision/recall or log-loss on podium/points prediction) — no
   specific target number is fixed in advance; it exists to demonstrate
   competency, not to be optimized.

## Repository Structure & Tooling

Follows the user's existing CLAUDE.md template conventions, extended for
multiple models and the dashboard (the template's default assumes a single
model pipeline):

```
f1_portfolio_project/
├── .github/workflows/ci.yml
├── notebooks/                      # 01_eda, 02_hierarchical_model, 03_decomposition,
│                                    # 04_elo_robustness, 05_sensitivity, 06_baseline
├── src/f1_skill/
│   ├── data.py                     # FastF1 fetch (raw, cached)
│   ├── cleaning.py                 # DNF/wet/sprint handling -> clean session table
│   ├── features.py                 # pace-gap computation, covariates
│   ├── models/
│   │   ├── hierarchical.py         # primary Bambi model + diagnostics
│   │   ├── decomposition.py        # variance partitioning
│   │   ├── elo_network.py          # secondary robustness-check model
│   │   ├── sensitivity.py          # variant runs + rank-correlation comparisons
│   │   └── baseline.py             # supporting race-outcome model
│   ├── plots.py                    # Agg-backend evaluation + skill/uncertainty figures
│   └── __main__.py                 # full pipeline: fetch -> clean -> features -> fit all -> save
├── app/
│   └── streamlit_app.py            # dashboard, imports src.f1_skill (trained artifacts only)
├── models/                         # committed: posterior traces (arviz .nc), baseline model, metrics.json
├── reports/figures/                # committed: skill trajectories, decomposition chart, etc.
├── data/                           # gitignored except data/README.md
│   ├── raw/  └── processed/
├── tests/ · Dockerfile · Makefile · pyproject.toml · uv.lock · .python-version
├── .env.example
├── README.md                       # the writeup - the primary deliverable
└── CLAUDE.md                       # gitignored, not committed
```

Tooling per CLAUDE.md: `uv` for env/deps, `ruff` for lint/format, `pytest`
mirroring `src/`, one `Dockerfile`, `Makefile` as the control panel,
pre-commit with ruff (notebook outputs kept, not stripped — they're
deliverables). Solo project: git flow is `main` ← `feature/*`, no `develop`.
Rules for Claude (never commit to main directly, never push without explicit
permission, no AI attribution in commits, never commit `data/`/`.env`/
`CLAUDE.md`, verify before claiming done) govern implementation work going
forward and are not repeated in full here — see `CLAUDE.md`.

## Open Risks

- The core methodology (Bayesian hierarchical modeling) is genuinely harder
  than the user's prior project, and they are learning it from zero as part of
  this project. Budget real time for getting the model wrong before getting
  it right.
- Confounders may limit how strong a conclusion is possible — e.g., the
  decomposition might end up with wide credible intervals for some driver/
  constructor combinations. That is an acceptable, honestly-reported outcome,
  not a failure of the project.
- An F1-knowledgeable or stats-knowledgeable reviewer may probe the
  methodology directly. The design is built to survive that scrutiny (real
  diagnostics, explicit confounder handling, cited prior methodology); it
  will not survive if any of that is executed superficially.
