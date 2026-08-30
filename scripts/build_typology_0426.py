from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from research_config import (
    AIRPORT_CLUSTER_ID,
    AIRPORT_CLUSTER_NAME,
    AIRPORT_LOCATION_IDS,
    CLUSTER_ALGORITHMS,
    CLUSTER_COUNT,
    CLUSTER_K_RANGE,
    COMMONPLACE_PATH,
    DROPOFF_LONGRUN_PATH,
    FIGURES_DIR,
    OUTPUT_ROOT,
    PICKUP_LONGRUN_PATH,
    SELECTED_FEATURES,
    TAXI_ZONE_GEOMETRY_PATH,
    TYPOLOGY_OUTPUT_DIR,
    ZONE_FEATURES_GROUPED_PATH,
    ensure_workspace_dirs,
)
from research_utils import (
    entropy_from_series,
    load_taxi_zones,
    parse_wkt_points,
    top_share_from_series,
    winsorize_series,
    zscore_series,
)

TIPRATE_COLOR_RAMP = [
    "#24110D",
    "#6F2617",
    "#96381F",
    "#B8512A",
    "#D07A4A",
    "#E6A47A",
    "#F2D3C0",
]
AIRPORT_MAP_COLOR = "#5F7F98"


def compute_commonplace_zone_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = pd.read_csv(COMMONPLACE_PATH)
    zones = load_taxi_zones()
    zone_lookup = zones[["LocationID", "Zone", "Borough", "service_zone", "zone_area_sqft", "zone_area_sqmi", "geometry"]].copy()

    common["_point_id"] = np.arange(len(common))
    common_geom = parse_wkt_points(common["the_geom"])
    gdf = gpd.GeoDataFrame(common.copy(), geometry=common_geom, crs="EPSG:4326").to_crs(zones.crs)

    within_join = gpd.sjoin(gdf, zone_lookup, how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    matched_within_ids = set(within_join.loc[within_join["LocationID"].notna(), "_point_id"])

    unmatched = gdf.loc[~gdf["_point_id"].isin(matched_within_ids)].copy()
    if unmatched.empty:
        rescue_join = unmatched.copy()
    else:
        rescue_candidates = gpd.sjoin(unmatched, zone_lookup, how="left", predicate="intersects").drop(columns=["index_right"], errors="ignore")
        rescue_join = rescue_candidates.sort_values(["_point_id", "LocationID"]).drop_duplicates("_point_id", keep="first").copy()

    joined = pd.concat(
        [
            within_join.loc[within_join["LocationID"].notna()].copy(),
            rescue_join.loc[rescue_join["LocationID"].notna()].copy(),
        ],
        ignore_index=True,
    )
    matched_point_ids = set(joined["_point_id"])
    unmatched_points = gdf.loc[~gdf["_point_id"].isin(matched_point_ids)].copy()

    grouped = joined.groupby("LocationID", dropna=False)
    zone_features = pd.DataFrame({"LocationID": zones["LocationID"]}).merge(
        zones.drop(columns="geometry"),
        on="LocationID",
        how="left",
    )
    agg = grouped.apply(
        lambda part: pd.Series(
            {
                "commonplace_count": len(part),
                "commonplace_domain_entropy": entropy_from_series(part["FACILITY DOMAINS"]),
                "commonplace_top_domain_share": top_share_from_series(part["FACILITY DOMAINS"]),
                "commonplace_source_entropy": entropy_from_series(part["SOURCE"]),
                "commonplace_top_source_share": top_share_from_series(part["SOURCE"]),
                "commonplace_type_entropy": entropy_from_series(part["FACILITY TYPE"]),
            }
        ),
        include_groups=False,
    ).reset_index()

    zone_features = zone_features.merge(agg, on="LocationID", how="left")
    fill_zero_cols = [
        "commonplace_count",
        "commonplace_domain_entropy",
        "commonplace_top_domain_share",
        "commonplace_source_entropy",
        "commonplace_top_source_share",
        "commonplace_type_entropy",
    ]
    for column in fill_zero_cols:
        zone_features[column] = zone_features[column].fillna(0.0)
    zone_features["commonplace_count"] = zone_features["commonplace_count"].astype(int)
    zone_features["commonplace_count_per_sqmi"] = np.where(
        zone_features["zone_area_sqmi"] > 0,
        zone_features["commonplace_count"] / zone_features["zone_area_sqmi"],
        np.nan,
    )
    zone_features["commonplace_has_points"] = zone_features["commonplace_count"] > 0

    join_audit = pd.DataFrame(
        [
            {
                "total_points": int(len(gdf)),
                "matched_points": int(len(matched_point_ids)),
                "unmatched_points": int(len(unmatched_points)),
                "matched_zone_count": int(zone_features["commonplace_has_points"].sum()),
                "within_matched_points": int(len(matched_within_ids)),
                "rescue_matched_points": int(len(matched_point_ids) - len(matched_within_ids)),
            }
        ]
    )
    source_summary = (
        common["SOURCE"].fillna("MISSING").astype(str).value_counts(dropna=False).rename_axis("SOURCE").reset_index(name="count")
    )
    domain_summary = (
        common["FACILITY DOMAINS"].fillna("MISSING").astype(str).value_counts(dropna=False).rename_axis("FACILITY_DOMAINS").reset_index(name="count")
    )
    type_summary = (
        common["FACILITY TYPE"].fillna("MISSING").astype(str).value_counts(dropna=False).rename_axis("FACILITY_TYPE").reset_index(name="count")
    )

    return zone_features, join_audit, source_summary, domain_summary, type_summary


def build_typology_feature_matrix(grouped: pd.DataFrame, commonplace: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "LocationID",
        "commonplace_count",
        "commonplace_count_per_sqmi",
        "commonplace_domain_entropy",
        "commonplace_top_domain_share",
        "commonplace_source_entropy",
        "commonplace_top_source_share",
        "commonplace_type_entropy",
        "commonplace_has_points",
    ]
    master = grouped.merge(commonplace[keep_cols], on="LocationID", how="left")
    for column in keep_cols[1:-1]:
        master[column] = master[column].fillna(0.0)
    master["commonplace_has_points"] = master["commonplace_has_points"].fillna(False).astype(bool)
    return master.sort_values("LocationID").reset_index(drop=True)


def build_standardized_matrix(master: pd.DataFrame, selected_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = master.loc[:, ["LocationID", "Zone", "Borough", "service_zone", "is_airport_zone"] + selected_features].copy()
    processed["cluster_eligible"] = (~processed["is_airport_zone"]) & processed[selected_features].notna().all(axis=1)
    eligible_subset = processed.loc[processed["cluster_eligible"], selected_features].copy().reset_index(drop=True)

    standardized = pd.DataFrame(index=eligible_subset.index)
    bounds_rows = []
    for feature in selected_features:
        wins = winsorize_series(eligible_subset[feature])
        standardized[feature] = zscore_series(wins)
        bounds_rows.append(
            {
                "feature_name": feature,
                "lower_cap": float(eligible_subset[feature].quantile(0.01)),
                "upper_cap": float(eligible_subset[feature].quantile(0.99)),
            }
        )
    bounds = pd.DataFrame(bounds_rows)
    matrix = processed.loc[processed["cluster_eligible"], ["LocationID", "Zone", "Borough", "service_zone"]].reset_index(drop=True)
    matrix = pd.concat([matrix, standardized], axis=1)
    return matrix, bounds


def fit_and_score(matrix: pd.DataFrame, algorithm: str, k: int) -> dict[str, object]:
    X = matrix.to_numpy()
    if algorithm == "KMeans":
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = model.fit_predict(X)
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)

    label_counts = pd.Series(labels).value_counts()
    standardized_means = pd.DataFrame(X, columns=matrix.columns).assign(cluster_id=labels).groupby("cluster_id").mean()
    return {
        "algorithm": algorithm,
        "k": int(k),
        "labels": labels,
        "model": model,
        "silhouette_score": float(silhouette_score(X, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
        "min_cluster_size": int(label_counts.min()),
        "max_cluster_share": float(label_counts.max() / label_counts.sum()),
        "average_cluster_profile_contrast": float(
            standardized_means.abs().apply(lambda row: row.nlargest(min(3, len(row))).mean(), axis=1).mean()
        ),
    }


def rename_clusters(standardized_means: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    for cluster_id, row in standardized_means.iterrows():
        office_high = row.get("pluto_office_area_share_of_bldg", 0) >= 0.75
        cbd_near = row.get("mta_nearest_cbd_complex_dist_ft", 0) <= -0.75
        cbd_far = row.get("mta_nearest_cbd_complex_dist_ft", 0) >= 0.75
        transit_dense = row.get("mta_station_density_sqmi", 0) >= 0.75
        transit_sparse = row.get("mta_station_density_sqmi", 0) <= -0.75
        industrial_high = row.get("pluto_landuse_share_06_industrial_manufacturing", 0) >= 0.75
        open_space_high = row.get("pluto_landuse_share_09_open_space_recreation", 0) >= 0.75
        residential_density_high = row.get("pluto_units_res_per_acre", 0) >= 0.75
        mixed_use_high = row.get("pluto_landuse_share_04_mixed_residential_commercial", 0) >= 0.50
        rent_burden_high = row.get("acs_rent_burden_30plus_share", 0) >= 0.75
        poverty_high = row.get("acs_poverty_rate", 0) >= 0.75
        commute_time_high = row.get("acs_mean_travel_time_min", 0) >= 0.50
        commonplace_high = row.get("commonplace_count_per_sqmi", 0) >= 0.75

        if office_high and cbd_near and transit_dense:
            name = "CBD-accessible business core"
        elif industrial_high:
            name = "Industrial / logistics zone"
        elif open_space_high:
            name = "Recreation / open-space zone"
        elif residential_density_high and mixed_use_high:
            name = "Dense mixed-use neighborhood"
        elif rent_burden_high and (poverty_high or commute_time_high):
            name = "Housing-pressure residential zone"
        elif transit_dense and cbd_near:
            name = "Transit-rich neighborhood center"
        elif cbd_far and transit_sparse:
            name = "Low-access residential zone"
        elif commonplace_high:
            name = "High-activity community place zone"
        else:
            name = "Outer-borough residential zone"
        names[int(cluster_id)] = name

    used: dict[str, int] = {}
    for cluster_id, name in list(names.items()):
        used[name] = used.get(name, 0) + 1
        if used[name] > 1:
            names[cluster_id] = f"{name} ({cluster_id})"
    return names


def build_cluster_summary(
    matrix: pd.DataFrame,
    standardized_means: pd.DataFrame,
    raw_means: pd.DataFrame,
    labels_df: pd.DataFrame,
    cluster_name_map: dict[int, str],
    cluster_tip_summary: pd.DataFrame,
) -> pd.DataFrame:
    representative_rows = []
    for cluster_id in sorted(cluster_name_map):
        cluster_rows = matrix.loc[labels_df.loc[labels_df["cluster_status"] == "clustered", "cluster_id"].reset_index(drop=True) == cluster_id, SELECTED_FEATURES]
        centroid = cluster_rows.mean(axis=0).to_numpy()
        distances = np.sqrt(((cluster_rows.to_numpy() - centroid) ** 2).sum(axis=1))
        cluster_meta = labels_df.loc[labels_df["cluster_id"] == cluster_id, ["LocationID", "Zone"]].reset_index(drop=True)
        nearest = cluster_meta.loc[pd.Series(distances).sort_values().head(3).index, "Zone"].tolist()
        strongest_features = standardized_means.loc[cluster_id].abs().sort_values(ascending=False).head(5).index.tolist()
        representative_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name_map[cluster_id],
                "zone_count": int((labels_df["cluster_id"] == cluster_id).sum()),
                "representative_zones": "; ".join(nearest),
                "defining_features": ", ".join(strongest_features),
            }
        )

    airport_rows = labels_df.loc[labels_df["cluster_status"] == "airport_special", "Zone"].tolist()
    representative_rows.append(
        {
            "cluster_id": AIRPORT_CLUSTER_ID,
            "cluster_name": AIRPORT_CLUSTER_NAME,
            "zone_count": int((labels_df["cluster_status"] == "airport_special").sum()),
            "representative_zones": "; ".join(airport_rows),
            "defining_features": "rule-based airport special handling",
        }
    )
    summary = pd.DataFrame(representative_rows).merge(
        cluster_tip_summary[
            [
                "cluster_id",
                "pickup_mean_raw_tip_rate",
                "dropoff_mean_raw_tip_rate",
                "combined_mean_tip_rate",
                "tiprate_rank",
            ]
        ],
        on="cluster_id",
        how="left",
    )
    return summary.sort_values("cluster_id").reset_index(drop=True)


def compute_cluster_tiprate_summary(labels_df: pd.DataFrame) -> pd.DataFrame:
    pickup = pd.read_parquet(PICKUP_LONGRUN_PATH)[["LocationID", "avg_tip_rate"]].rename(
        columns={"avg_tip_rate": "pickup_raw_longrun_tip_rate"}
    )
    dropoff = pd.read_parquet(DROPOFF_LONGRUN_PATH)[["LocationID", "avg_tip_rate"]].rename(
        columns={"avg_tip_rate": "dropoff_raw_longrun_tip_rate"}
    )
    cluster_tip = (
        labels_df.merge(pickup, on="LocationID", how="left")
        .merge(dropoff, on="LocationID", how="left")
        .groupby(["cluster_id", "cluster_name", "cluster_status"], dropna=False)
        .agg(
            pickup_mean_raw_tip_rate=("pickup_raw_longrun_tip_rate", "mean"),
            dropoff_mean_raw_tip_rate=("dropoff_raw_longrun_tip_rate", "mean"),
        )
        .reset_index()
    )
    cluster_tip["combined_mean_tip_rate"] = cluster_tip[
        ["pickup_mean_raw_tip_rate", "dropoff_mean_raw_tip_rate"]
    ].mean(axis=1)
    cluster_tip = cluster_tip.sort_values(
        ["combined_mean_tip_rate", "cluster_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    cluster_tip["tiprate_rank"] = np.arange(1, len(cluster_tip) + 1)
    return cluster_tip


def save_typology_map(labels_df: pd.DataFrame, cluster_tip_summary: pd.DataFrame) -> None:
    zones = gpd.read_parquet(TAXI_ZONE_GEOMETRY_PATH)
    map_df = zones.merge(labels_df[["LocationID", "cluster_id", "cluster_name"]], on="LocationID", how="left").merge(
        cluster_tip_summary[["cluster_id", "combined_mean_tip_rate", "tiprate_rank"]],
        on="cluster_id",
        how="left",
    )
    ordered_summary = cluster_tip_summary.sort_values("tiprate_rank").reset_index(drop=True)
    color_map = {}
    non_airport_rows = ordered_summary.loc[ordered_summary["cluster_name"] != AIRPORT_CLUSTER_NAME].reset_index(drop=True)
    non_airport_ramp = TIPRATE_COLOR_RAMP[: len(non_airport_rows)]
    for idx, row in non_airport_rows.iterrows():
        color_map[row["cluster_name"]] = non_airport_ramp[idx]
    color_map[AIRPORT_CLUSTER_NAME] = AIRPORT_MAP_COLOR

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor="#dfd9d2")
    ax.set_facecolor("#dfd9d2")
    zones.plot(ax=ax, color="#d8d2cb", edgecolor="#bfb8b0", linewidth=0.2, zorder=0)
    for cluster_name, color in color_map.items():
        subset = map_df.loc[map_df["cluster_name"] == cluster_name]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="#fbf7f2", linewidth=0.35, zorder=2)

    ax.set_axis_off()
    legend_handles = []
    for row in ordered_summary.itertuples(index=False):
        if row.cluster_name == AIRPORT_CLUSTER_NAME:
            label = f"{int(row.tiprate_rank)}. {row.cluster_name} ({row.combined_mean_tip_rate:.1%}, blue highlight)"
        else:
            label = f"{int(row.tiprate_rank)}. {row.cluster_name} ({row.combined_mean_tip_rate:.1%})"
        legend_handles.append(Patch(facecolor=color_map[row.cluster_name], edgecolor="#fbf7f2", label=label))
    legend = ax.legend(
        handles=legend_handles,
        title="Typology classes\nranked by combined tip rate",
        loc="upper left",
        bbox_to_anchor=(0.03, 0.99),
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        facecolor="#f1ece6",
        edgecolor="#b9aea2",
        fontsize=15.2,
        title_fontsize=17.5,
        borderpad=1.15,
        labelspacing=0.62,
        handlelength=1.9,
        handleheight=1.2,
        handletextpad=0.8,
    )
    ax.add_artist(legend)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "typology_map.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_workspace_dirs()

    grouped = pd.read_parquet(ZONE_FEATURES_GROUPED_PATH).copy()
    airport_ids = set(grouped.loc[grouped["is_airport_zone"], "LocationID"])
    if airport_ids != AIRPORT_LOCATION_IDS:
        raise ValueError(f"Airport ID validation failed. Expected {sorted(AIRPORT_LOCATION_IDS)}, got {sorted(airport_ids)}")

    commonplace, join_audit, source_summary, domain_summary, type_summary = compute_commonplace_zone_features()
    master = build_typology_feature_matrix(grouped, commonplace)
    master.to_parquet(TYPOLOGY_OUTPUT_DIR / "typology_feature_matrix.parquet", index=False)

    matrix, winsor_bounds = build_standardized_matrix(master, SELECTED_FEATURES)
    if len(matrix) != 257:
        raise ValueError(f"Expected 257 non-airport cluster-eligible zones, got {len(matrix)}")

    diagnostics_rows = []
    fitted_results: dict[tuple[str, int], dict[str, object]] = {}
    for algorithm in CLUSTER_ALGORITHMS:
        for k in CLUSTER_K_RANGE:
            result = fit_and_score(matrix[SELECTED_FEATURES], algorithm=algorithm, k=k)
            diagnostics_rows.append(
                {
                    "algorithm": result["algorithm"],
                    "k": result["k"],
                    "silhouette_score": result["silhouette_score"],
                    "davies_bouldin_score": result["davies_bouldin_score"],
                    "min_cluster_size": result["min_cluster_size"],
                    "max_cluster_share": result["max_cluster_share"],
                    "average_cluster_profile_contrast": result["average_cluster_profile_contrast"],
                    "chosen_for_0426_typology": algorithm == "KMeans" and k == CLUSTER_COUNT,
                }
            )
            fitted_results[(algorithm, k)] = result
    diagnostics = pd.DataFrame(diagnostics_rows).sort_values(["algorithm", "k"]).reset_index(drop=True)

    chosen = fitted_results[("KMeans", CLUSTER_COUNT)]
    labels = pd.Series(chosen["labels"], name="cluster_id_raw")
    label_map = {old: idx + 1 for idx, old in enumerate(sorted(labels.unique()))}
    cluster_ids = labels.map(label_map).astype(int)

    standardized_with_labels = matrix[["LocationID"] + SELECTED_FEATURES].copy()
    standardized_with_labels["cluster_id"] = cluster_ids.values
    standardized_means = standardized_with_labels.groupby("cluster_id")[SELECTED_FEATURES].mean()
    cluster_name_map = rename_clusters(standardized_means)

    labels_df = master[["LocationID", "Zone", "Borough", "service_zone", "is_airport_zone"]].copy()
    labels_df["cluster_status"] = np.where(labels_df["is_airport_zone"], "airport_special", "clustered")
    labels_df["cluster_id"] = np.nan
    labels_df.loc[~labels_df["is_airport_zone"], "cluster_id"] = cluster_ids.values
    labels_df.loc[labels_df["is_airport_zone"], "cluster_id"] = AIRPORT_CLUSTER_ID
    labels_df["cluster_id"] = labels_df["cluster_id"].astype(int)
    labels_df["cluster_name"] = labels_df["cluster_id"].map(cluster_name_map).fillna(AIRPORT_CLUSTER_NAME)
    labels_df["typology_version"] = "0426_kmeans6_airport_special"
    cluster_tip_summary = compute_cluster_tiprate_summary(labels_df)

    cluster_summary = build_cluster_summary(
        matrix=matrix,
        standardized_means=standardized_means,
        raw_means=master.loc[~master["is_airport_zone"], ["LocationID"] + SELECTED_FEATURES].assign(cluster_id=cluster_ids.values).groupby("cluster_id")[SELECTED_FEATURES].mean(),
        labels_df=labels_df,
        cluster_name_map=cluster_name_map,
        cluster_tip_summary=cluster_tip_summary,
    )

    standardized_output = standardized_means.reset_index()
    standardized_output["cluster_name"] = standardized_output["cluster_id"].map(cluster_name_map)

    labels_df.to_csv(TYPOLOGY_OUTPUT_DIR / "zone_typology_labels.csv", index=False)
    diagnostics.to_csv(TYPOLOGY_OUTPUT_DIR / "typology_diagnostics.csv", index=False)
    winsor_bounds.to_csv(TYPOLOGY_OUTPUT_DIR / "typology_winsor_bounds.csv", index=False)
    standardized_output.to_csv(TYPOLOGY_OUTPUT_DIR / "typology_standardized_means.csv", index=False)
    cluster_summary.to_csv(TYPOLOGY_OUTPUT_DIR / "typology_cluster_summary.csv", index=False)
    commonplace.to_parquet(TYPOLOGY_OUTPUT_DIR / "commonplace_zone_features.parquet", index=False)
    join_audit.to_csv(TYPOLOGY_OUTPUT_DIR / "commonplace_join_audit.csv", index=False)
    source_summary.to_csv(TYPOLOGY_OUTPUT_DIR / "commonplace_source_summary.csv", index=False)
    domain_summary.to_csv(TYPOLOGY_OUTPUT_DIR / "commonplace_domain_code_summary.csv", index=False)
    type_summary.to_csv(TYPOLOGY_OUTPUT_DIR / "commonplace_type_code_summary.csv", index=False)
    save_typology_map(labels_df, cluster_tip_summary)

    print("Saved 0426 typology outputs.")
    print(f"Non-airport clustered zones: {len(matrix)}")
    print(f"Airport special zones: {(labels_df['cluster_status'] == 'airport_special').sum()}")


if __name__ == "__main__":
    main()
