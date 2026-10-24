# Data preparation: evidence and public demo

**Reconstructed public demo; not a reproduction of the historical sample.**
All distributed input records are synthetic. The demonstration makes cleaning,
joining, filtering and sampling decisions inspectable, using the existing
Python stack. The four research lenses and their historical results remain unchanged.

## Recovery record

The inspected release baseline is commit
`1ad2d599e6e2b5174c265e26d660ad1f332c77a4`. This local clone has complete
available history (three commits), not the shallow history seen by the planning
audit. No earlier production trip-ingestion pipeline was found in that Git
history. The selected local research workspace and the user-provided project
archive did contain relevant source code. These sources were read without
running their large-data workflows. Paths below are logical project paths,
not personal filesystem paths.

| Source ID | Logical source | SHA256 |
| --- | --- | --- |
| baseline-source | `new0419/run_taxi_tip_baseline.py` | `091ad9af67628d1196e2ce8e68fc1d416f356dfe88ccbc19b6c3c7fad920b930` |
| baseline-utils | `new0419/taxi_tip_utils.py` | `59dc47c571d4913542b6c61ca0fdcd7ce9b4c9a46510aadae1cd57ef9fe8871a` |
| weather-07 | Archive `Project/weather/07_sample_and_features.ipynb` | `a8a1c25261b03dbf495d9ba2c8ba5280dd2d42114897b2f73bb2cf1b542c9041` |
| weather-09 | Archive `Project/weather/09_ols_layer1.ipynb` | `94a8b92444c4238109920abdc1147c9c494d7a036160feb3fa8460c3862d0783` |
| crosswalk-source | `data script/build_taxi_zone_crosswalks.py` | `93a6c43be05653e4d6b52552f5945ff672f9c93d855069c8e4536d52fa41406c` |
| panel-source | `data script/build_zone_time_behavior_dataset.py` | `fe17943f71e0919b909b0d9dae941b1589ad2ef1052ae0a0076e79799395adeb` |

## Policy provenance and conflicts

`recovered` means source code was inspected; `documented` means a report or
interface states it; `demo_assumption` means a deliberate new demonstration
decision. These labels do not certify the execution of the historical sample.

| Rule | Evidence and historical meaning | Public-demo decision | Status |
| --- | --- | --- | --- |
| Card code and target | Baseline and weather-07 use payment_type=1; tip/fare is a fraction | Explicit card_values=[1]; cash tips are not treated as true zeros | recovered |
| Sample policy | Baseline main outcome is positive-tip card; weather-07 saves all eligible cards then defines a positive-tip subset | all_card_nonnegative main; positive_tip comparison | recovered |
| Validity order | Positive fare; nonnegative tip; positive distance; duration (0,240]; fare<=500, tip<=200, distance<=100, speed<=80; card; tip/fare in [0,2] | Positive finite fare/distance/duration and finite nonnegative tip; optional caps are null by default | recovered thresholds; demo_assumption defaults |
| Date window | Baseline defaults to 2024, but saved output is January 2024; weather-07 filters 2023–2025 | Synthetic dates only; no recovered historical configuration claimed | recovered |
| Time interpretation | Historical SQL/Spark handles timestamps without a demonstrated DST-resolution contract | Explicit naive/aware setting; New York local features; ambiguous/nonexistent times quarantined | demo_assumption |
| Identity | Historical models hash selected fields, not a natural TLC trip ID | Versioned file-SHA256 plus original record ordinal; separate content fingerprint | demo_assumption |
| Zone retention | Weather-07 excludes Unknown/N/A pickup borough, unlike the baseline | Many-to-one left joins; retain unmatched rows unless explicitly configured | recovered conflict; demo_assumption retention |
| Weather | Weather-07 performs a left hourly-string join for lag features; original complete trip/weather integration and publication-time semantics remain unrecovered | Nearest observation, 60 minutes, synthetic station mapping, earlier tie; descriptive only | recovered partial interface; demo_assumption join |
| Split and sample | Baseline DuckDB HASH uses eight trip fields, weather-09 uses pickup time/PU/DO; <70 / 70–84 / >=85, hash rank for subsamples | SHA256 group split and trip-ID rank, seed 42; small per-stratum caps | recovered boundaries; demo_assumption algorithm |
| Learning | Weather-09 and current category helper fit top categories on training; notebook winsorizes at training .5%/99.5% | No new learned transformation | recovered |

Differences are intentional: the demo does not silently import every historical
cutoff. It also does not reproduce DuckDB's hash or claim that saved model
samples can be reconstructed from the archive. Valid existing buckets are
preserved with their external policy metadata; conflicting duplicate groups
are rejected.

## Source interfaces outside this demo

The spatial configuration still resolves `NYC_TAXI_DATA_ROOT` and its
`dataset/`, `data script/parquet/`, and `data script/spatial/` inputs.
The new demo does not replace these independent prepared inputs.

| Family | Existing interface | Recovery boundary |
| --- | --- | --- |
| ACS | zone_features_acs.parquet; taxi_zone_to_tract.parquet | Feature/crosswalk source code available locally; full production rebuild deferred |
| PLUTO | zone_features_pluto.parquet | Local feature builder available; not executed |
| MTA | zone_features_mta.parquet | Local accessibility builder available; not executed |
| CommonPlace | CommonPlace_20260426.csv | Existing typology builder remains unchanged |
| Zone panels | pickup/dropoff_zone_month_panel_cc_enriched.parquet | Panel source available locally; not replaced by trip demo |
| Knicks | Prepared game-tagged Parquet; archive game/sample builders | Some sibling regression functions and raw odds input are absent; event identification unchanged |

Recovered crosswalk code distinguishes `area_share=intersection/zone_area`,
`covered_area_share=intersection/matched_area`, and tract-area share.
Partial coverage must not silently become complete coverage. Totals and ratios
require separate aggregation rules. No crosswalk is consumed by this demo,
so crosswalk validation/aggregation is not implemented or claimed.

## Scope of verification

The new tests, CLI and notebook verify synthetic inputs only. Historical
132-file, approximately 780M-row, 89.6M-card-trip and model R-squared statements
are quoted research outcomes, not verified by this demo. The old OLS/LightGBM
same-sample comparison, model re-estimation, full source builders and large-data
throughput benchmarks are deferred.
