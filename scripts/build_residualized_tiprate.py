from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from research_config import (
    DROPOFF_LONGRUN_PATH,
    DROPOFF_PANEL_PATH,
    PICKUP_LONGRUN_PATH,
    PICKUP_PANEL_PATH,
    RESIDUAL_BOOL_COLUMNS,
    RESIDUAL_CONTROL_COLUMNS,
    RESIDUAL_TREE_OUTPUT_DIR,
    TYPOLOGY_OUTPUT_DIR,
    ensure_workspace_dirs,
    residual_formula,
)


def load_side_inputs(side: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if side == "pickup":
        return pd.read_parquet(PICKUP_PANEL_PATH), pd.read_parquet(PICKUP_LONGRUN_PATH)
    if side == "dropoff":
        return pd.read_parquet(DROPOFF_PANEL_PATH), pd.read_parquet(DROPOFF_LONGRUN_PATH)
    raise ValueError(f"Unsupported side: {side}")


def build_zone_residuals(side: str, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    panel, longrun = load_side_inputs(side)
    formula = residual_formula()

    panel = panel.copy()
    panel["year_month"] = panel["year_month"].astype(str)
    for column in RESIDUAL_BOOL_COLUMNS:
        panel[column] = panel[column].fillna(False).astype(int)

    required_columns = ["avg_tip_rate", "valid_tip_rate_trip_count"] + RESIDUAL_CONTROL_COLUMNS + ["year_month", "Borough", "service_zone"]
    model_df = panel.dropna(subset=required_columns).copy()
    model = smf.wls(formula, data=model_df, weights=model_df["valid_tip_rate_trip_count"]).fit()

    model_df["fitted_tip_rate"] = model.fittedvalues
    model_df["residualized_tip_rate_month"] = model.resid
    model_df["weighted_residual"] = model_df["residualized_tip_rate_month"] * model_df["valid_tip_rate_trip_count"]

    zone_residual = model_df.groupby("LocationID", as_index=False).agg(
        model_ready_month_rows=("year_month", "size"),
        model_ready_weight_sum=("valid_tip_rate_trip_count", "sum"),
        residual_numerator=("weighted_residual", "sum"),
    )
    zone_residual["residualized_tip_rate"] = zone_residual["residual_numerator"] / zone_residual["model_ready_weight_sum"]
    zone_residual = zone_residual.drop(columns=["residual_numerator"])

    longrun_zone = longrun[["LocationID", "avg_tip_rate"]].rename(columns={"avg_tip_rate": "raw_longrun_tip_rate"})
    zone_output = labels.merge(zone_residual, on="LocationID", how="left").merge(longrun_zone, on="LocationID", how="left")
    zone_output["target_side"] = side
    zone_output["model_ready_zone"] = zone_output["model_ready_month_rows"].notna()

    summary = {
        "target_side": side,
        "formula": formula,
        "panel_rows_total": int(len(panel)),
        "panel_rows_model_ready": int(len(model_df)),
        "panel_rows_dropped": int(len(panel) - len(model_df)),
        "zones_total": int(panel["LocationID"].nunique()),
        "zones_model_ready": int(model_df["LocationID"].nunique()),
        "zones_without_model_ready_rows": int(zone_output["model_ready_month_rows"].isna().sum()),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "nobs": float(model.nobs),
    }
    return zone_output, summary


def main() -> None:
    ensure_workspace_dirs()
    labels = pd.read_csv(TYPOLOGY_OUTPUT_DIR / "zone_typology_labels.csv")

    rows = []
    for side in ["pickup", "dropoff"]:
        zone_output, summary = build_zone_residuals(side, labels)
        zone_output.to_csv(RESIDUAL_TREE_OUTPUT_DIR / f"{side}_zone_residualized_tiprate.csv", index=False)
        rows.append(summary)

    pd.DataFrame(rows).to_csv(RESIDUAL_TREE_OUTPUT_DIR / "residual_model_summary.csv", index=False)
    print("Saved residualized tiprate outputs.")


if __name__ == "__main__":
    main()
