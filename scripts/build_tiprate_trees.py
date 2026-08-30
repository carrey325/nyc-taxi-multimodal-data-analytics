from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

from research_config import (
    AIRPORT_CLUSTER_NAME,
    FIGURES_DIR,
    RESIDUAL_TREE_OUTPUT_DIR,
    SELECTED_FEATURES,
    TREE_MAX_DEPTHS,
    TREE_MAX_LEAVES,
    TREE_MIN_LEAVES,
    TREE_MIN_SAMPLES_LEAF,
    TYPOLOGY_OUTPUT_DIR,
    ensure_workspace_dirs,
)
from research_utils import df_to_markdown_table, extract_tree_leaf_rules, winsorize_series

BALANCED_TREE_SHOWCASE_SIDES = {"pickup", "dropoff"}
BALANCED_TREE_PALETTE = [
    "#24110D",
    "#6F2617",
    "#96381F",
    "#B8512A",
    "#D07A4A",
    "#E6A47A",
    "#F2D3C0",
]
INTERNAL_NODE_FILL = "#F1ECE6"
INTERNAL_NODE_EDGE = "#8A786A"
INTERNAL_NODE_TEXT = "#2F2620"
FEATURE_LABELS = {
    "mta_nearest_cbd_complex_dist_ft": "Nearest CBD\nsubway hub distance",
    "acs_poverty_rate": "Poverty rate",
    "pluto_landuse_share_09_open_space_recreation": "Open-space\nland-use share",
    "commonplace_count_per_sqmi": "CommonPlace density",
    "acs_mean_travel_time_min": "Mean commute time",
    "pluto_landuse_share_04_mixed_residential_commercial": "Mixed-use\nland-use share",
    "acs_rent_burden_30plus_share": "Rent burden 30%+",
    "pluto_units_res_per_acre": "Residential units\nper acre",
    "pluto_office_area_share_of_bldg": "Office area share",
    "mta_station_density_sqmi": "Subway stop density",
    "acs_median_household_income": "Median household income",
    "pluto_landuse_share_06_industrial_manufacturing": "Industrial\nland-use share",
}


def select_tree_model(training_df: pd.DataFrame, target_column: str) -> tuple[DecisionTreeRegressor, pd.DataFrame]:
    X = training_df[SELECTED_FEATURES]
    y = training_df[target_column]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    candidates = []
    for depth in TREE_MAX_DEPTHS:
        for min_leaf in TREE_MIN_SAMPLES_LEAF:
            model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=42)
            model.fit(X, y)
            leaf_count = model.get_n_leaves()
            if not (TREE_MIN_LEAVES <= leaf_count <= TREE_MAX_LEAVES):
                continue
            scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            candidates.append(
                {
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                    "leaf_count": int(leaf_count),
                    "cv_r2_mean": float(scores.mean()),
                    "cv_r2_std": float(scores.std()),
                    "train_r2": float(model.score(X, y)),
                    "chosen": False,
                }
            )

    if not candidates:
        raise ValueError("No tree candidate satisfied the leaf-count constraint.")

    candidate_df = pd.DataFrame(candidates).sort_values(
        ["cv_r2_mean", "leaf_count", "max_depth", "min_samples_leaf"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    best_row = candidate_df.iloc[0].to_dict()
    candidate_df.loc[0, "chosen"] = True
    best_model = DecisionTreeRegressor(
        max_depth=int(best_row["max_depth"]),
        min_samples_leaf=int(best_row["min_samples_leaf"]),
        random_state=42,
    )
    best_model.fit(X, y)
    return best_model, candidate_df


def build_leaf_assignment(
    model: DecisionTreeRegressor,
    assignment_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoring_df = assignment_df.copy()
    scoring_df["tree_node_id"] = model.apply(scoring_df[SELECTED_FEATURES])

    training_leaf_order = (
        scoring_df.loc[scoring_df[target_column].notna()]
        .groupby("tree_node_id", as_index=False)[target_column]
        .mean()
        .sort_values(target_column)
        .reset_index(drop=True)
    )
    training_leaf_order["tree_leaf_id"] = np.arange(1, len(training_leaf_order) + 1)
    leaf_map = dict(zip(training_leaf_order["tree_node_id"], training_leaf_order["tree_leaf_id"]))
    scoring_df["tree_leaf_id"] = scoring_df["tree_node_id"].map(leaf_map).astype("Int64")

    leaf_summary = (
        scoring_df.groupby(["tree_leaf_id", "tree_node_id"], dropna=False)
        .agg(
            zone_count=("LocationID", "nunique"),
            zones_with_target=(target_column, lambda s: int(s.notna().sum())),
            mean_target=(target_column, "mean"),
            mean_raw_longrun_tip_rate=("raw_longrun_tip_rate", "mean"),
            representative_zones=("Zone", lambda s: "; ".join(s.head(3).tolist())),
        )
        .reset_index()
        .sort_values("tree_leaf_id")
    )
    return scoring_df, leaf_summary


def build_cluster_leaf_table(assignment_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    grouped = (
        assignment_df.loc[assignment_df["cluster_name"] != AIRPORT_CLUSTER_NAME]
        .groupby(["cluster_id", "cluster_name", "tree_leaf_id"], dropna=False)
        .agg(
            zone_count=("LocationID", "nunique"),
            zones_with_target=(target_column, lambda s: int(s.notna().sum())),
            mean_residualized_tip_rate=(target_column, "mean"),
            mean_raw_longrun_tip_rate=("raw_longrun_tip_rate", "mean"),
            representative_zones=("Zone", lambda s: "; ".join(s.head(3).tolist())),
        )
        .reset_index()
        .sort_values(["cluster_id", "tree_leaf_id"])
    )
    return grouped


def save_tree_rule_report(side: str, model: DecisionTreeRegressor, candidate_df: pd.DataFrame, leaf_summary: pd.DataFrame) -> None:
    rules = pd.DataFrame(extract_tree_leaf_rules(model, SELECTED_FEATURES))
    selected = candidate_df.loc[
        candidate_df["chosen"],
        ["max_depth", "min_samples_leaf", "leaf_count", "cv_r2_mean", "cv_r2_std", "train_r2"],
    ]
    report_lines = [
        f"# {side.upper()} residualized tiprate tree rules",
        "",
        "## Selected hyperparameters",
        df_to_markdown_table(selected),
        "",
        "## Leaf rules",
        df_to_markdown_table(rules),
        "",
        "## Leaf summary",
        df_to_markdown_table(leaf_summary),
    ]
    (RESIDUAL_TREE_OUTPUT_DIR / f"{side}_tree_rules.md").write_text("\n".join(report_lines), encoding="utf-8")


def save_tree_plot(side: str, model: DecisionTreeRegressor) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(model, feature_names=SELECTED_FEATURES, filled=True, rounded=True, impurity=False, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{side}_tree.png", dpi=200)
    plt.close(fig)


def sample_palette(palette: list[str], n: int) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return [palette[0]]
    idxs = np.linspace(0, len(palette) - 1, n).round().astype(int)
    return [palette[idx] for idx in idxs]


def format_threshold(feature: str, value: float) -> str:
    if feature.endswith("_dist_ft"):
        return f"{value:,.0f} ft"
    if feature == "acs_mean_travel_time_min":
        return f"{value:.1f} min"
    if feature in {"acs_poverty_rate", "acs_rent_burden_30plus_share"}:
        return f"{value:.1f}%"
    if feature in {
        "pluto_landuse_share_04_mixed_residential_commercial",
        "pluto_landuse_share_06_industrial_manufacturing",
        "pluto_landuse_share_09_open_space_recreation",
        "pluto_office_area_share_of_bldg",
    }:
        return f"{value * 100:.1f}%"
    if feature == "commonplace_count_per_sqmi":
        return f"{value:,.0f} / sq mi"
    if feature == "pluto_units_res_per_acre":
        return f"{value:.1f} / acre"
    if feature == "mta_station_density_sqmi":
        return f"{value:.1f} / sq mi"
    if feature == "acs_median_household_income":
        return f"${value:,.0f}"
    return f"{value:.2f}"


def compute_tree_positions(model: DecisionTreeRegressor) -> dict[int, tuple[float, float]]:
    tree = model.tree_

    def is_leaf(node_id: int) -> bool:
        return tree.children_left[node_id] == tree.children_right[node_id]

    leaf_order: list[int] = []

    def collect_leaves(node_id: int) -> None:
        if is_leaf(node_id):
            leaf_order.append(node_id)
            return
        collect_leaves(tree.children_left[node_id])
        collect_leaves(tree.children_right[node_id])

    collect_leaves(0)
    leaf_x = {node_id: idx for idx, node_id in enumerate(leaf_order)}
    raw_positions: dict[int, tuple[float, int]] = {}
    max_depth = 0

    def assign(node_id: int, depth: int) -> float:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        if is_leaf(node_id):
            x_pos = float(leaf_x[node_id])
        else:
            left_x = assign(tree.children_left[node_id], depth + 1)
            right_x = assign(tree.children_right[node_id], depth + 1)
            x_pos = (left_x + right_x) / 2
        raw_positions[node_id] = (x_pos, depth)
        return x_pos

    assign(0, 0)

    x_max = max(1, len(leaf_order) - 1)
    y_max = max(1, max_depth)
    positions = {}
    for node_id, (x_raw, depth) in raw_positions.items():
        x = 0.08 + (x_raw / x_max) * 0.84
        y = 0.90 - (depth / y_max) * 0.72
        positions[node_id] = (x, y)
    return positions


def save_balanced_tree_diagram(side: str, model: DecisionTreeRegressor, leaf_summary: pd.DataFrame) -> None:
    tree = model.tree_
    positions = compute_tree_positions(model)
    leaf_rows = leaf_summary.dropna(subset=["tree_node_id"]).copy()
    leaf_rows["tree_node_id"] = leaf_rows["tree_node_id"].astype(int)
    leaf_rows = leaf_rows.sort_values("mean_target").reset_index(drop=True)

    sampled_palette = sample_palette(BALANCED_TREE_PALETTE, len(leaf_rows))
    low_to_high_colors = list(reversed(sampled_palette))
    leaf_color_map = {
        int(row.tree_node_id): low_to_high_colors[idx]
        for idx, row in leaf_rows.iterrows()
    }
    leaf_meta_map = {
        int(row.tree_node_id): row
        for row in leaf_rows.itertuples(index=False)
    }

    def is_leaf(node_id: int) -> bool:
        return tree.children_left[node_id] == tree.children_right[node_id]

    fig, ax = plt.subplots(figsize=(20, 11.5), facecolor="#DFD9D2")
    ax.set_facecolor("#DFD9D2")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for node_id, (x0, y0) in positions.items():
        if is_leaf(node_id):
            continue
        for child_id in [tree.children_left[node_id], tree.children_right[node_id]]:
            x1, y1 = positions[child_id]
            ax.plot([x0, x1], [y0 - 0.035, y1 + 0.045], color="#7D6B5E", linewidth=2.7, solid_capstyle="round", zorder=1)

    for node_id, (x_pos, y_pos) in positions.items():
        if is_leaf(node_id):
            meta = leaf_meta_map[node_id]
            color = leaf_color_map[node_id]
            text_color = "white" if color in BALANCED_TREE_PALETTE[:3] else "#2F2620"
            text = "\n".join(
                [
                    f"Leaf {int(meta.tree_leaf_id)}",
                    f"Residual {meta.mean_target * 100:+.2f} pp",
                    f"Raw tip {meta.mean_raw_longrun_tip_rate:.1%}",
                    f"{int(meta.zone_count)} zones",
                ]
            )
            bbox = dict(boxstyle="round,pad=0.75,rounding_size=0.22", facecolor=color, edgecolor="#F7F1EA", linewidth=2.0)
            ax.text(
                x_pos,
                y_pos,
                text,
                ha="center",
                va="center",
                fontsize=15.5,
                fontweight="bold",
                color=text_color,
                linespacing=1.35,
                bbox=bbox,
                zorder=3,
            )
        else:
            feature = SELECTED_FEATURES[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
            text = "\n".join(
                [
                    label,
                    f"<= {format_threshold(feature, threshold)}",
                    f"n = {int(tree.n_node_samples[node_id])}",
                ]
            )
            bbox = dict(boxstyle="round,pad=0.68,rounding_size=0.18", facecolor=INTERNAL_NODE_FILL, edgecolor=INTERNAL_NODE_EDGE, linewidth=1.8)
            ax.text(
                x_pos,
                y_pos,
                text,
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=INTERNAL_NODE_TEXT,
                linespacing=1.28,
                bbox=bbox,
                zorder=2,
            )

    ax.text(
        0.02,
        0.035,
        "Darker leaves indicate higher residualized tiprate after controls.",
        fontsize=13.5,
        color="#5A493E",
        ha="left",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{side}_balanced_tree.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_side(side: str, labels: pd.DataFrame, master_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    residuals = pd.read_csv(RESIDUAL_TREE_OUTPUT_DIR / f"{side}_zone_residualized_tiprate.csv")
    zone_df = labels.merge(
        residuals[["LocationID", "raw_longrun_tip_rate", "residualized_tip_rate", "model_ready_zone"]],
        on="LocationID",
        how="left",
        suffixes=("", "_resid"),
    ).merge(master_features[["LocationID"] + SELECTED_FEATURES], on="LocationID", how="left")

    non_airport = zone_df.loc[zone_df["cluster_name"] != AIRPORT_CLUSTER_NAME].copy()
    training_df = non_airport.loc[non_airport["residualized_tip_rate"].notna()].copy()

    bounds_rows = []
    for feature in SELECTED_FEATURES:
        wins = winsorize_series(training_df[feature])
        lower = float(training_df[feature].quantile(0.01))
        upper = float(training_df[feature].quantile(0.99))
        training_df[feature] = wins
        non_airport[feature] = pd.to_numeric(non_airport[feature], errors="coerce").clip(lower, upper)
        bounds_rows.append({"feature_name": feature, "lower_cap": lower, "upper_cap": upper})
    pd.DataFrame(bounds_rows).to_csv(RESIDUAL_TREE_OUTPUT_DIR / f"{side}_tree_feature_bounds.csv", index=False)

    model, candidate_df = select_tree_model(training_df, "residualized_tip_rate")
    assignment_non_airport, leaf_summary = build_leaf_assignment(model, non_airport, "residualized_tip_rate")

    airport_df = zone_df.loc[zone_df["cluster_name"] == AIRPORT_CLUSTER_NAME].copy()
    airport_df["tree_node_id"] = pd.NA
    airport_df["tree_leaf_id"] = pd.NA
    full_assignment = pd.concat([assignment_non_airport, airport_df], ignore_index=True, sort=False)
    full_assignment["target_side"] = side

    cluster_leaf = build_cluster_leaf_table(assignment_non_airport, "residualized_tip_rate")
    candidate_df.to_csv(RESIDUAL_TREE_OUTPUT_DIR / f"{side}_tree_model_summary.csv", index=False)
    full_assignment.to_csv(RESIDUAL_TREE_OUTPUT_DIR / f"{side}_tree_leaf_assignments.csv", index=False)
    cluster_leaf.to_csv(RESIDUAL_TREE_OUTPUT_DIR / f"{side}_cluster_x_leaf.csv", index=False)
    save_tree_rule_report(side, model, candidate_df, leaf_summary)
    save_tree_plot(side, model)
    if side in BALANCED_TREE_SHOWCASE_SIDES:
        save_balanced_tree_diagram(side, model, leaf_summary)
    return full_assignment, candidate_df


def main() -> None:
    ensure_workspace_dirs()
    labels = pd.read_csv(TYPOLOGY_OUTPUT_DIR / "zone_typology_labels.csv")
    master_features = pd.read_parquet(TYPOLOGY_OUTPUT_DIR / "typology_feature_matrix.parquet")

    assignments = {}
    for side in ["pickup", "dropoff"]:
        assignment_df, _ = run_side(side, labels, master_features)
        assignments[side] = assignment_df

    bridge = labels.copy()
    for side in ["pickup", "dropoff"]:
        residuals = pd.read_csv(RESIDUAL_TREE_OUTPUT_DIR / f"{side}_zone_residualized_tiprate.csv")
        leaves = assignments[side][["LocationID", "tree_leaf_id"]].rename(columns={"tree_leaf_id": f"{side}_tree_leaf_id"})
        bridge = bridge.merge(
            residuals[["LocationID", "raw_longrun_tip_rate", "residualized_tip_rate"]].rename(
                columns={
                    "raw_longrun_tip_rate": f"{side}_raw_longrun_tip_rate",
                    "residualized_tip_rate": f"{side}_residualized_tip_rate",
                }
            ),
            on="LocationID",
            how="left",
        ).merge(leaves, on="LocationID", how="left")

    bridge.to_csv(RESIDUAL_TREE_OUTPUT_DIR / "zone_typology_tree_bridge.csv", index=False)
    print("Saved residual tree outputs and bridge table.")


if __name__ == "__main__":
    main()
