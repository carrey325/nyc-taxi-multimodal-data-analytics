"""Estimate the displayed Knicks game-event specifications.

The repository does not distribute trip-level data. This script documents the
final model layer and accepts a prepared, game-tagged Parquet file supplied by
the analyst. No machine-specific paths or credentials are embedded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


CORE_COLUMNS = {
    "tip_pct",
    "game_id",
    "season",
    "after",
    "game",
    "W",
    "theta_W",
    "theta_noW",
}

RIDE_CONTROLS = (
    "fare_amount",
    "trip_distance",
    "trip_duration_minutes",
    "passenger_count",
)

FULL_CONTROLS = RIDE_CONTROLS + (
    "pickup_hour",
    "pickup_weekday",
    "PULocationID",
    "DOLocationID",
)

UPSET_TERMS = (
    "pred_loss_cl_win",
    "pred_loss_md_win",
    "pred_loss_lg_win",
    "pred_win_cl_loss",
    "pred_win_md_loss",
    "pred_win_lg_loss",
    "pred_close_win",
)


def available_terms(frame: pd.DataFrame, names: tuple[str, ...]) -> list[str]:
    """Return numeric controls that are present in the prepared sample."""
    return [name for name in names if name in frame.columns]


def control_formula(frame: pd.DataFrame, full: bool) -> str:
    controls = available_terms(frame, FULL_CONTROLS if full else RIDE_CONTROLS)
    terms: list[str] = []
    for name in controls:
        if name in {"pickup_hour", "pickup_weekday", "PULocationID", "DOLocationID"}:
            terms.append(f"C({name})")
        else:
            terms.append(name)
    return " + ".join(terms)


def fit_clustered(frame: pd.DataFrame, formula: str):
    """Fit OLS with standard errors clustered by game."""
    model = smf.ols(formula, data=frame, missing="drop")
    model_rows = model.data.row_labels
    groups = frame.loc[model_rows, "game_id"]
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def coefficient_frame(result, model_name: str, sample_name: str) -> pd.DataFrame:
    confidence = result.conf_int()
    return pd.DataFrame(
        {
            "sample": sample_name,
            "model": model_name,
            "term": result.params.index,
            "estimate": result.params.values,
            "std_error": result.bse.values,
            "p_value": result.pvalues.values,
            "conf_low": confidence.iloc[:, 0].values,
            "conf_high": confidence.iloc[:, 1].values,
            "nobs": int(result.nobs),
            "r_squared": float(result.rsquared),
        }
    )


def estimate_slice(frame: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []

    ride = control_formula(frame, full=False)
    full = control_formula(frame, full=True)

    did_rhs = "game + after + game:after"
    if ride:
        did_rhs += f" + {ride}"
    did = fit_clustered(frame, f"tip_pct ~ {did_rhs}")
    outputs.append(coefficient_frame(did, "event_did_ride_controls", sample_name))

    surprise_rhs = "W + theta_W + theta_noW"
    if ride:
        surprise_rhs += f" + {ride}"
    surprise = fit_clustered(frame, f"tip_pct ~ {surprise_rhs}")
    outputs.append(coefficient_frame(surprise, "surprise_ride_controls", sample_name))

    upset_terms = available_terms(frame, UPSET_TERMS)
    if upset_terms:
        upset_rhs = " + ".join(upset_terms)
        if full:
            upset_rhs += f" + {full}"
        upset = fit_clustered(frame, f"tip_pct ~ {upset_rhs}")
        outputs.append(coefficient_frame(upset, "upset_full_controls", sample_name))

    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tagged_parquet", type=Path)
    parser.add_argument("--output", type=Path, default=Path("knicks_model_coefficients.csv"))
    args = parser.parse_args()

    frame = pd.read_parquet(args.tagged_parquet).reset_index(drop=True)
    missing = sorted(CORE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Prepared sample is missing columns: {missing}")

    estimates = [estimate_slice(frame, "pooled")]
    for season, season_frame in frame.groupby("season", observed=True):
        estimates.append(estimate_slice(season_frame.copy(), str(season)))

    output = pd.concat(estimates, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output):,} coefficient rows to {args.output}")


if __name__ == "__main__":
    main()
