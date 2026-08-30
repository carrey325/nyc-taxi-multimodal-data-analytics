from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
OUTPUT_ROOT = WORKSPACE_ROOT / "results"
TYPOLOGY_OUTPUT_DIR = OUTPUT_ROOT / "typology"
RESIDUAL_TREE_OUTPUT_DIR = OUTPUT_ROOT / "residual_trees"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = WORKSPACE_ROOT / "reports"

# Large source and processed datasets are intentionally excluded from GitHub.
# Set NYC_TAXI_DATA_ROOT to the root of the full local data workspace. The
# expected layout below matches the research workspace used to build this
# release package.
DATA_ROOT = Path(os.environ.get("NYC_TAXI_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
DATASET_DIR = DATA_ROOT / "dataset"
DATA_SCRIPT_DIR = DATA_ROOT / "data script"
PARQUET_DIR = DATA_SCRIPT_DIR / "parquet"
SPATIAL_DIR = DATA_SCRIPT_DIR / "spatial"

COMMONPLACE_PATH = DATASET_DIR / "CommonPlace_20260426.csv"
ZONE_FEATURES_GROUPED_PATH = PARQUET_DIR / "zone_features_grouped.parquet"
ZONE_FEATURES_ACS_PATH = PARQUET_DIR / "zone_features_acs.parquet"
ZONE_FEATURES_PLUTO_PATH = PARQUET_DIR / "zone_features_pluto.parquet"
ZONE_FEATURES_MTA_PATH = PARQUET_DIR / "zone_features_mta.parquet"
PICKUP_PANEL_PATH = PARQUET_DIR / "pickup_zone_month_panel_cc_enriched.parquet"
DROPOFF_PANEL_PATH = PARQUET_DIR / "dropoff_zone_month_panel_cc_enriched.parquet"
PICKUP_LONGRUN_PATH = PARQUET_DIR / "pickup_zone_longrun_summary_cc.parquet"
DROPOFF_LONGRUN_PATH = PARQUET_DIR / "dropoff_zone_longrun_summary_cc.parquet"
TAXI_ZONE_GEOMETRY_PATH = SPATIAL_DIR / "taxi_zone_geometry.geoparquet"
TAXI_ZONE_TO_TRACT_PATH = PARQUET_DIR / "taxi_zone_to_tract.parquet"
TAXI_ZONE_TO_NTA_PATH = PARQUET_DIR / "taxi_zone_to_nta.parquet"
TAXI_ZONE_TO_CDTA_PATH = PARQUET_DIR / "taxi_zone_to_cdta.parquet"

SELECTED_FEATURES = [
    "acs_mean_travel_time_min",
    "acs_median_household_income",
    "acs_poverty_rate",
    "acs_rent_burden_30plus_share",
    "commonplace_count_per_sqmi",
    "mta_nearest_cbd_complex_dist_ft",
    "mta_station_density_sqmi",
    "pluto_landuse_share_04_mixed_residential_commercial",
    "pluto_landuse_share_06_industrial_manufacturing",
    "pluto_landuse_share_09_open_space_recreation",
    "pluto_office_area_share_of_bldg",
    "pluto_units_res_per_acre",
]

AIRPORT_LOCATION_IDS = {1, 132, 138}
AIRPORT_CLUSTER_ID = 7
AIRPORT_CLUSTER_NAME = "Airport special zone"
CLUSTER_COUNT = 6
CLUSTER_K_RANGE = range(4, 9)
CLUSTER_ALGORITHMS = ("KMeans", "Agglomerative")

RESIDUAL_BOOL_COLUMNS = [
    "is_partial_coverage_zone",
    "is_water_park_special_proxy",
    "is_source_missing_special_area",
]

RESIDUAL_FACTOR_COLUMNS = [
    "year_month",
    "Borough",
    "service_zone",
]

RESIDUAL_CONTROL_COLUMNS = [
    "tract_coverage_share",
    "is_partial_coverage_zone",
    "is_water_park_special_proxy",
    "is_source_missing_special_area",
    "avg_trip_distance",
    "avg_fare_per_mile",
    "avg_passenger_count",
    "share_airport_fee",
    "share_airport_ratecode",
    "share_negotiated_fare",
    "share_special_ratecode",
    "share_peak_hour_pickup",
    "share_night_trip",
    "share_weekend_trip",
    "share_commute_hour_trip",
]

TREE_MAX_DEPTHS = [2, 3, 4]
TREE_MIN_SAMPLES_LEAF = [10, 15, 20, 25]
TREE_MIN_LEAVES = 5
TREE_MAX_LEAVES = 7


def ensure_workspace_dirs() -> None:
    for path in [
        WORKSPACE_ROOT,
        SCRIPTS_DIR,
        OUTPUT_ROOT,
        TYPOLOGY_OUTPUT_DIR,
        RESIDUAL_TREE_OUTPUT_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def residual_formula() -> str:
    factor_terms = " + ".join(f"C({column})" for column in RESIDUAL_FACTOR_COLUMNS)
    control_terms = " + ".join(RESIDUAL_CONTROL_COLUMNS)
    return f"avg_tip_rate ~ {factor_terms} + {control_terms}"


SOURCE_MANIFEST_SPECS = [
    ("commonplace_raw", COMMONPLACE_PATH, "CommonPlace", "Raw point-level civic-place inventory."),
    ("taxi_zone_geometry", TAXI_ZONE_GEOMETRY_PATH, "Geometry", "Strict Taxi Zone polygon geometry."),
    ("taxi_zone_to_tract", TAXI_ZONE_TO_TRACT_PATH, "Crosswalk", "Taxi Zone to tract area-weighted crosswalk."),
    ("taxi_zone_to_nta", TAXI_ZONE_TO_NTA_PATH, "Crosswalk", "Taxi Zone to NTA area-weighted crosswalk."),
    ("taxi_zone_to_cdta", TAXI_ZONE_TO_CDTA_PATH, "Crosswalk", "Taxi Zone to CDTA area-weighted crosswalk."),
    ("zone_features_grouped", ZONE_FEATURES_GROUPED_PATH, "Feature table", "Grouped static zone feature stack."),
    ("zone_features_acs", ZONE_FEATURES_ACS_PATH, "ACS", "Zone-level ACS-derived contextual features."),
    ("zone_features_pluto", ZONE_FEATURES_PLUTO_PATH, "PLUTO", "Zone-level PLUTO built-environment features."),
    ("zone_features_mta", ZONE_FEATURES_MTA_PATH, "MTA", "Zone-level MTA accessibility features."),
    ("pickup_zone_month_panel", PICKUP_PANEL_PATH, "Taxi panel", "Pickup zone-month credit-card outcome panel."),
    ("dropoff_zone_month_panel", DROPOFF_PANEL_PATH, "Taxi panel", "Dropoff zone-month credit-card outcome panel."),
    ("pickup_longrun_summary", PICKUP_LONGRUN_PATH, "Taxi summary", "Pickup long-run zone summary."),
    ("dropoff_longrun_summary", DROPOFF_LONGRUN_PATH, "Taxi summary", "Dropoff long-run zone summary."),
]
