# NYC Taxi Multimodal Data Analytics

This repository studies how NYC taxi tipping patterns relate to trip composition and urban context. It combines taxi zone-month outcomes with socioeconomic, built-environment, transit-accessibility, and place-inventory features derived from ACS, MapPLUTO, MTA, and CommonPlace data.

“Multimodal” refers to the integration of multiple data sources and analytical methods: spatial joins, weighted regression, unsupervised clustering, and interpretable regression trees.

![NYC taxi zone typology](results/figures/typology_map.png)

## Research questions

1. What types of urban environments characterize NYC Taxi Zones?
2. How do raw tip rates differ across these zone types?
3. After controlling for month, geography, coverage, and trip composition, which spatial thresholds are associated with residual tipping differences?

## Main findings

- A frozen 12-feature specification groups non-airport Taxi Zones into six KMeans classes; Newark, JFK, and LaGuardia are handled as a separate airport class.
- CBD-accessible business cores and dense mixed-use neighborhoods have the highest raw average tip rates. Housing-pressure and outer-borough residential zones have the lowest.
- Weighted residual models explain most zone-month variation after month, geographic, coverage, and trip-composition controls (`R² = 0.966` for pickup and `0.943` for dropoff).
- The remaining tree-based spatial signal is weak out of sample. The trees are therefore interpreted as exploratory heterogeneity summaries, not high-accuracy prediction models.
- Results describe associations and do not identify causal effects.

## Repository contents

```text
scripts/                  Reproducible analysis pipeline
reports/                  English and Chinese methodology/findings reports
results/figures/          Final maps and tree diagrams
results/typology/         Cluster summaries, diagnostics, and zone labels
results/residual_trees/   Residual-model, tree, and zone-level result tables
```

The original raw data, processed parquet warehouse, temporary files, logs, exploratory reports, and superseded iterations are intentionally excluded.

## Key outputs

- [English methodology and findings](reports/methodology_and_findings_en.md)
- [中文方法与结果报告](reports/methodology_and_findings_zh.md)
- [Taxi Zone typology summary](results/typology/typology_cluster_summary.csv)
- [Zone-level typology and tree bridge](results/residual_trees/zone_typology_tree_bridge.csv)
- [Pickup tree rules](results/residual_trees/pickup_tree_rules.md)
- [Dropoff tree rules](results/residual_trees/dropoff_tree_rules.md)

## Reproducing the analysis

Python 3.13 was used for the final run.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:NYC_TAXI_DATA_ROOT = "D:\path\to\full\research\workspace"
python scripts\build_source_manifest.py
python scripts\build_typology_0426.py
python scripts\build_residualized_tiprate.py
python scripts\build_tiprate_trees.py
python scripts\generate_0426_reports.py
```

`NYC_TAXI_DATA_ROOT` should contain the following upstream structure:

```text
dataset/
  CommonPlace_20260426.csv
data script/
  parquet/
    zone_features_grouped.parquet
    zone_features_acs.parquet
    zone_features_pluto.parquet
    zone_features_mta.parquet
    pickup_zone_month_panel_cc_enriched.parquet
    dropoff_zone_month_panel_cc_enriched.parquet
    pickup_zone_longrun_summary_cc.parquet
    dropoff_zone_longrun_summary_cc.parquet
    taxi_zone_to_tract.parquet
    taxi_zone_to_nta.parquet
    taxi_zone_to_cdta.parquet
  spatial/
    taxi_zone_geometry.geoparquet
```

The data are not committed because the complete taxi store and spatial sources are large and may have separate distribution terms.

## Study scope

- Taxi panel: January 2023–December 2025
- Geographic unit: 260 strict Taxi Zone polygons
- Main payment sample: credit-card trips
- Panel size: 9,360 pickup rows and 9,360 dropoff rows
- Typology: KMeans `k=6` plus three airport special zones

