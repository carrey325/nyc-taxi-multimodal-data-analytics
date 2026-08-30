from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from sklearn.tree import _tree

from research_config import TAXI_ZONE_GEOMETRY_PATH


def load_taxi_zones() -> gpd.GeoDataFrame:
    zones = gpd.read_parquet(TAXI_ZONE_GEOMETRY_PATH).copy()
    zones["zone_area_sqmi"] = zones["zone_area_sqft"] / 27_878_400.0
    return zones


def winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    lower = numeric.quantile(lower_q)
    upper = numeric.quantile(upper_q)
    return numeric.clip(lower=lower, upper=upper)


def zscore_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - numeric.mean()) / std


def entropy_from_series(values: pd.Series) -> float:
    valid = values.dropna().astype(str)
    if valid.empty:
        return 0.0
    counts = valid.value_counts(dropna=False)
    shares = counts / counts.sum()
    return float(-(shares * np.log(shares)).sum())


def top_share_from_series(values: pd.Series) -> float:
    valid = values.dropna().astype(str)
    if valid.empty:
        return 0.0
    counts = valid.value_counts(dropna=False)
    return float(counts.iloc[0] / counts.sum())


def parse_wkt_points(series: pd.Series) -> gpd.GeoSeries:
    return gpd.GeoSeries(series.astype(str).map(wkt.loads), crs="EPSG:4326")


def df_to_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows to display._"

    display_df = df.copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("\n", " ")
        )

    headers = [str(column) for column in display_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_df.values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def describe_missingness(df: pd.DataFrame, top_n: int = 8) -> str:
    ratios = df.isna().mean().sort_values(ascending=False)
    ratios = ratios[ratios > 0]
    if ratios.empty:
        return "No missing values."
    return "; ".join(f"{column}: {ratio:.1%}" for column, ratio in ratios.head(top_n).items())


def infer_row_count(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return len(pd.read_csv(path))
    if suffix == ".parquet":
        return len(pd.read_parquet(path))
    if suffix in {".geoparquet", ".gpkg", ".shp"}:
        return len(gpd.read_parquet(path))
    raise ValueError(f"Unsupported manifest row-count file type: {path}")


def extract_tree_leaf_rules(model, feature_names: list[str]) -> list[dict[str, object]]:
    tree = model.tree_
    results: list[dict[str, object]] = []

    def walk(node_id: int, conditions: list[str]) -> None:
        feature_index = tree.feature[node_id]
        if feature_index == _tree.TREE_UNDEFINED:
            results.append(
                {
                    "node_id": int(node_id),
                    "rule": " and ".join(conditions) if conditions else "all zones",
                    "prediction": float(tree.value[node_id][0][0]),
                    "sample_count": int(tree.n_node_samples[node_id]),
                }
            )
            return

        feature_name = feature_names[feature_index]
        threshold = float(tree.threshold[node_id])
        walk(tree.children_left[node_id], conditions + [f"{feature_name} <= {threshold:.4f}"])
        walk(tree.children_right[node_id], conditions + [f"{feature_name} > {threshold:.4f}"])

    walk(0, [])
    return results
