"""The whole pipeline: fetch, clean, fit, decompose, check, save.

    python -m f1_skill                 the shipped result
    python -m f1_skill --sensitivity   also refit the segment variants, several minutes more

Each stage writes its output to data/processed/ so the pipeline is inspectable rather than a black
box, and so a rerun does not refetch 173 sessions to change one number downstream.

The sensitivity refits are off by default on purpose. They are robustness checks, and making the
command that shows the headline take five minutes to answer a question it answered in the first
thirty seconds is the wrong trade.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from f1_skill import cleaning, data, features, plots
from f1_skill.models import baseline, decomposition, elo_network, hierarchical

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

LAPS = PROCESSED / "qualifying_laps.csv"
PACE = PROCESSED / "q1_pace.csv"
MODEL_TABLE = PROCESSED / "model_data.csv"
FIT = MODELS / "model_sqrt.nc"
RESULTS = MODELS / "decomposition_results.json"
BASELINE_RESULTS = MODELS / "baseline_metrics.json"
AGREEMENT = MODELS / "model_agreement.json"


def _label(path):
    """For the progress lines only. `relative_to` raises outside the repo, and a tidy message is
    not worth taking a stage down for."""
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def _cached(path, build, reader=pd.read_csv, **reader_kwargs):
    """Stage outputs are reused. Delete the file to force that stage to run again."""
    if path.exists():
        print(f"  reusing {_label(path)}")
        return reader(path, **reader_kwargs)
    frame = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  wrote {_label(path)}  {frame.shape}")
    return frame


def run(sensitivity=False):
    print("fetching qualifying laps")
    laps = _cached(LAPS, data.qualifying_laps, low_memory=False)

    print("building the Q1 pace table")
    pace = _cached(PACE, lambda: features.q1_pace(laps))

    print("applying the 107% rule")
    table = _cached(MODEL_TABLE, lambda: hierarchical.model_table(pace))

    print("fitting the hierarchical model")
    if FIT.exists():
        import arviz as az

        print(f"  reusing {FIT.relative_to(ROOT)}")
        idata = az.from_netcdf(FIT)
    else:
        _, idata = hierarchical.fit(table)
        MODELS.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(FIT)
        print(f"  wrote {FIT.relative_to(ROOT)}")

    convergence = hierarchical.convergence(idata)
    print(
        f"  divergences {convergence['divergences']}  "
        f"worst R-hat {convergence['worst_rhat']:.4f}  "
        f"lowest ESS {convergence['lowest_ess']:.0f}"
    )
    if not convergence["converged"]:
        print("  WARNING: R-hat above the threshold, the numbers below are not trustworthy")

    print("decomposing the variance")
    summary = decomposition.summary(idata)
    observed = decomposition.observation_weighted(idata, table)
    results = {
        "shares_of_all_variation": (summary["shares"] * 100).round(1).to_dict(),
        "shares_of_explained_variation": (summary["explained"] * 100).round(1).to_dict(),
        "car_to_driver_ratio": round(summary["ratio_median"], 3),
        "car_to_driver_89": [round(bound, 2) for bound in summary["ratio_89"]],
        "observation_weighted_ratio": round(
            float((observed["var_car"] / observed["var_driver"]).median()), 3
        ),
        "driver_car_correlation": round(float(observed["correlation"].median()), 3),
        # numpy scalars are not JSON-serialisable: np.float64 subclasses float and slips
        # through, np.bool_ does not.
        "convergence": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in convergence.items()
        },
    }
    print(
        f"  car : driver {results['car_to_driver_ratio']}x "
        f"{results['car_to_driver_89']}  |  observation-weighted "
        f"{results['observation_weighted_ratio']}x"
    )

    print("checking the ranking a second way")
    # One download, used twice: the ranking needs the round numbers and the baseline needs the
    # results, and they arrive in the same zip.
    archive = data.race_archive()
    rounds = archive["races"][["year", "name", "round"]].rename(
        columns={"year": "season", "name": "gp"}
    )
    pairs = elo_network.head_to_head(table.merge(rounds, on=["season", "gp"], how="inner"))
    ranking = decomposition.driver_ranking(idata)
    agreement = {
        "teammate win rate": elo_network.rank_agreement(elo_network.win_rate(pairs), ranking),
        "Elo": elo_network.rank_agreement(elo_network.run_elo(pairs)[0], ranking),
        "Bradley-Terry": elo_network.rank_agreement(elo_network.bradley_terry(pairs), ranking),
    }
    for name, value in agreement.items():
        print(f"  {name:20} rho {value:+.3f}")

    print("fitting the baseline")
    frame = baseline.model_frame(baseline.race_results(archive))
    metrics = baseline.evaluate(frame)
    print(metrics.round(4).to_string())

    print("writing artifacts")
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    AGREEMENT.write_text(
        json.dumps({k: round(v, 3) for k, v in agreement.items()}, indent=2) + "\n"
    )
    BASELINE_RESULTS.write_text(json.dumps(metrics.round(4).to_dict(), indent=2) + "\n")

    ratings = decomposition.effects_in_seconds(idata, table, "1|drv", "drv__factor_dim")
    ratings["teams"] = table.groupby("drv")["team"].nunique().reindex(ratings.index)

    FIGURES.mkdir(parents=True, exist_ok=True)
    plots.variance_split(summary["shares"]).savefig(FIGURES / "variance_split.png", dpi=150)
    plots.ranking_agreement(agreement).savefig(FIGURES / "ranking_agreement.png", dpi=150)
    plots.driver_ratings(ratings, table["best_lap"].median()).savefig(
        FIGURES / "driver_ratings.png", dpi=150
    )
    print(f"  wrote {RESULTS.relative_to(ROOT)}, {AGREEMENT.relative_to(ROOT)}, 3 figures")

    if sensitivity:
        run_sensitivity(laps)
    return results


def run_sensitivity(laps):
    from f1_skill.models import sensitivity

    print("refitting the segment variants, this takes several minutes")
    tables = {name: sensitivity.segment_table(laps, name) for name in cleaning.QUALIFYING_SEGMENTS}
    fits = {
        name: sensitivity.fit_variant(table[table["within_107"]]) for name, table in tables.items()
    }
    print(sensitivity.compare(fits).round(3).to_string())


def main():
    parser = argparse.ArgumentParser(prog="f1_skill", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="also refit the Q1/Q2/Q3 variants (several minutes)",
    )
    run(sensitivity=parser.parse_args().sensitivity)


if __name__ == "__main__":
    main()
