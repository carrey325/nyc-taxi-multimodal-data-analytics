"""Estimate nested taxi-tip models with an incremental weather block.

The input is a prepared trip-weather Parquet sample. Raw TLC files and the
large integrated warehouse are intentionally excluded from this course
showcase. Paths are supplied at runtime rather than embedded in the script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_TERMS = (
    "fare_amount + trip_distance + fare_per_mile + trip_duration_minutes + "
    "average_speed_mph + passenger_count + tolls_amount + extra + mta_tax + "
    "improvement_surcharge + is_airport_trip + C(RatecodeID)"
)
VENDOR_TERMS = "C(VendorID)"
TIME_TERMS = "C(hour_of_day) + C(pickup_weekday) + C(pickup_month) + is_holiday"
LOCATION_TERMS = "C(PULocationID_top) + C(DOLocationID_top) + C(route_pair_top)"
WEATHER_TERMS = (
    "temperature + precipitation + snowfall + snow_depth + wind_speed + "
    "humidity + cloud_cover + is_raining + is_snowing + is_extreme_temp"
)


def collapse_categories(train: pd.DataFrame, frames: list[pd.DataFrame]) -> None:
    """Create training-defined top-zone and top-route categories."""
    for frame in frames:
        frame["route_pair"] = (
            frame["PULocationID"].astype(str) + "->" + frame["DOLocationID"].astype(str)
        )

    definitions = {
        "PULocationID_top": ("PULocationID", 30),
        "DOLocationID_top": ("DOLocationID", 30),
        "route_pair_top": ("route_pair", 20),
    }
    for output_name, (source_name, count) in definitions.items():
        keep = set(train[source_name].astype(str).value_counts().head(count).index)
        for frame in frames:
            values = frame[source_name].astype(str)
            frame[output_name] = values.where(values.isin(keep), "OTHER")


def split_sample(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the stored split bucket when available, otherwise make a fixed split."""
    if "split_bucket" not in frame.columns:
        rng = np.random.default_rng(42)
        frame = frame.copy()
        frame["split_bucket"] = rng.integers(0, 100, len(frame))
    train = frame.loc[frame["split_bucket"] < 70].copy()
    valid = frame.loc[frame["split_bucket"].between(70, 84)].copy()
    test = frame.loc[frame["split_bucket"] >= 85].copy()
    return train, valid, test


def build_formulas() -> dict[str, str]:
    common = f"{BASE_TERMS} + {VENDOR_TERMS} + {TIME_TERMS} + {LOCATION_TERMS}"
    return {
        "Spec_A_trip": f"tip_rate_fare ~ {BASE_TERMS}",
        "Spec_B_vendor": f"tip_rate_fare ~ {BASE_TERMS} + {VENDOR_TERMS}",
        "Spec_C_time": f"tip_rate_fare ~ {BASE_TERMS} + {VENDOR_TERMS} + {TIME_TERMS}",
        "Spec_D_location": f"tip_rate_fare ~ {common}",
        "Spec_E_weather": f"tip_rate_fare ~ {common} + {WEATHER_TERMS}",
        "Spec_F_weather_interactions": (
            f"tip_rate_fare ~ {common} + {WEATHER_TERMS} + "
            "precipitation:is_late_night + snowfall:is_late_night + "
            "temperature:precipitation"
        ),
        "Spec_G_comparison_interaction": (
            f"tip_rate_fare ~ {common} + {WEATHER_TERMS} + "
            "precipitation:is_extreme_temp"
        ),
        "Spec_H_time_interactions": (
            f"tip_rate_fare ~ {common} + {WEATHER_TERMS} + "
            "precipitation:is_extreme_temp + is_raining:is_late_night + "
            "is_snowing:is_late_night"
        ),
    }


def evaluate(model, frame: pd.DataFrame, split: str) -> dict[str, float | str]:
    prediction = model.predict(frame)
    return {
        "split": split,
        "rmse": float(np.sqrt(mean_squared_error(frame["tip_rate_fare"], prediction))),
        "mae": float(mean_absolute_error(frame["tip_rate_fare"], prediction)),
        "r_squared": float(r2_score(frame["tip_rate_fare"], prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_parquet", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("weather_model_results"))
    parser.add_argument("--max-train-rows", type=int, default=180_000)
    args = parser.parse_args()

    frame = pd.read_parquet(args.analysis_parquet)
    frame["is_extreme_temp"] = (
        (frame["apparent_temp"] >= 26.7) | (frame["temperature"] <= 0.0)
    ).astype(int)

    train, valid, test = split_sample(frame)
    train = train.sample(n=min(len(train), args.max_train_rows), random_state=42)
    collapse_categories(train, [train, valid, test])

    summary_rows: list[dict[str, float | int | str]] = []
    coefficient_rows: list[pd.DataFrame] = []
    for spec_name, formula in build_formulas().items():
        model = smf.ols(formula, data=train).fit(cov_type="HC3")
        summary_rows.append(
            {
                "specification": spec_name,
                "split": "train",
                "nobs": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adjusted_r_squared": float(model.rsquared_adj),
            }
        )
        for split_name, split_frame in (("validation", valid), ("test", test)):
            row = {"specification": spec_name, "nobs": len(split_frame)}
            row.update(evaluate(model, split_frame, split_name))
            summary_rows.append(row)

        confidence = model.conf_int()
        coefficient_rows.append(
            pd.DataFrame(
                {
                    "specification": spec_name,
                    "term": model.params.index,
                    "estimate": model.params.values,
                    "std_error": model.bse.values,
                    "p_value": model.pvalues.values,
                    "conf_low": confidence.iloc[:, 0].values,
                    "conf_high": confidence.iloc[:, 1].values,
                }
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "model_summary.csv", index=False)
    pd.concat(coefficient_rows, ignore_index=True).to_csv(
        args.output_dir / "model_coefficients.csv", index=False
    )
    print(f"Saved model summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
