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

## Data dictionary and identities

Raw TLC-style columns are retained unchanged. Additional normalized columns
use the following contract:

| Columns | Type/unit | Meaning |
| --- | --- | --- |
| source_id, source_file_name, source_file_sha256 | string | Logical source and exact file bytes; no absolute path |
| source_row_number | nonnegative integer | Zero-based original data-record ordinal before filtering; CSV header is excluded, a quoted multiline record counts once |
| trip_id | SHA256 hex | Canonical JSON `["source-row-v1", file_sha256, ordinal]`, encoded as UTF-8 |
| record_fingerprint | SHA256 hex | Canonical JSON `["record-v1", pairs]`; pairs are sorted canonical keys from the configured eight raw columns, with null/empty as null and other values as strings |
| fare_usd, tip_usd | numeric USD | Parsed copies, preserving original raw columns |
| distance_miles | numeric miles | Missing or invalid values remain missing |
| payment_code, pu_location_id, do_location_id | nullable integer | Noninteger/invalid identifiers become missing |
| pickup_time, dropoff_time | raw timestamp representation | Original timezone evidence |
| pickup_utc, dropoff_utc, pickup_local | aware timestamp | UTC for elapsed time, configured local timezone for features |
| trip_duration_minutes, average_speed_mph | minutes, miles/hour | Derived from UTC duration; invalid inputs are not filled with zero |
| pickup_hour, pickup_weekday | local hour, English day name | Local calendar features |
| tip_rate_fare | fraction | tip/fare for nonnegative card tips and positive finite fare; .2 means 20% |
| tip_or_not | nullable 0/1 | Positive tip indicator for eligible card recording; non-card is missing |
| pickup/dropoff_zone, pickup/dropoff_borough | string | Many-to-one Zone labels |
| pickup/dropoff_zone_matched | bool | Unmatched IDs remain in the left table |
| weather_station_id, weather_record_id | nullable string | Configured station and selected source observation |
| weather_observed_at, weather_available_at | aware UTC or missing | Observation time and separately declared publication time |
| temperature_c, precipitation_mm | degrees Celsius, mm | Two fixture weather measurements only |
| weather_lag_minutes | signed minutes | Observation minus pickup; positive means future observation |
| weather_matched, weather_descriptive_only | bool | Match flag and explicit descriptive-only label |
| weather_observation_after_pickup, weather_available_after_pickup | nullable bool | Missing when no observation or availability information exists |
| weather_mode | string | nearest_observation; tie selects earlier observation |
| split_label, split_bucket | string, integer 0–99 | train <70, validation 70–84, test >=85 |
| split_policy, split_version, split_seed | string | Original split metadata retained if supplied; otherwise explicit new policy |
| sample_hash, sample_rank, sample_version, sample_seed | hex, integer, string, integer | Stable ranking and sampling provenance |

The content fingerprint excludes `fixture_case`, source name, source ID and
record ordinal. It detects identical values in this explicit field set,
not every semantic duplicate: a differently formatted numeric/time value may
have a different fingerprint. Identical-looking physical records are retained.
Confirmed duplicate file imports and duplicate trip IDs are rejected.

Renaming a byte-identical file preserves IDs. Re-encoding or changing line
endings changes the file digest and IDs by design. Fixture CSV files are
checked out with LF endings to keep the published example consistent.
Chunking must occur after original ordinals are attached by the source reader.
This implementation targets in-memory samples, not a 780M-row execution engine.

## Audit and sampling decisions

The main sample includes nonnegative card tips. Selecting `positive_tip`
excludes zero tips and changes the estimand; the notebook shows that difference.
Each exclusion ledger row carries a reason. The first reason per trip is primary
and sums to total exclusions; overlapping reasons are separately counted.
Sampling omissions are reported as sampling removals, not economic exclusions.

The default sample flow is:

| Stage | Input | Output | Removed |
| --- | ---: | ---: | ---: |
| Source records | 33 | 33 | 0 |
| Timestamp normalization | 33 | 30 | 3 |
| Card/economic filter | 30 | 23 | 7 |
| Zone left join | 23 | 23 | 0 |
| Descriptive weather join | 23 | 23 | 0 |
| Optional match filter | 23 | 23 | 0 |
| Group split | 23 | 23 | 0 |
| Fixed-cap sample | 23 | 14 | 9 |

One pickup zone, one dropoff zone and two weather observations are unmatched
before sampling. The identical-content pair is retained before sampling and
assigned to one split. Sampling caps apply separately to each split and each
tip_or_not stratum; empty strata appear with zero counts.

New split buckets are SHA256 of canonical JSON
`["sha256-group-v1", seed, record_fingerprint]` modulo 100.
Sampling sorts SHA256 of `["sha256-rank-v1", seed, trip_id]`, then trip_id,
and takes min(cap, eligible count). ID audits hash
`["sorted-trip-ids-v1", sorted_ids]`. All JSON hash encodings use sorted
object keys, compact separators, UTF-8 and no nonfinite JSON numbers.
These are within-period partitions, not a forward forecasting evaluation.

If an input already has buckets, they must all be valid integers in 0–99.
Valid buckets are preserved and checked against any supplied labels.
Duplicate-content groups crossing existing splits cause an error, not reassignment.
No winsorization or category fitting is introduced in the demo; the existing
weather helper's training-only category behavior is covered by a test.

Nearest weather is explicitly descriptive: it may select an observation
after pickup, and the lag/availability flags reveal that choice. Missing
available_at stays unknown. The fixture's available_at is an invented release
time, not an alias for observed_at. `asof_available` is unimplemented and
refused, even if the caller supplies an available_at column.

## Consumer boundaries

The existing weather model requires a richer prepared input. Its full input
contract, before its internal category preparation and extreme-temperature
calculation, includes:

- Trip/control fields: fare_amount, trip_distance, fare_per_mile,
  trip_duration_minutes, average_speed_mph, passenger_count, tolls_amount,
  extra, mta_tax, improvement_surcharge, is_airport_trip, RatecodeID, VendorID,
  hour_of_day, pickup_weekday, pickup_month, is_holiday, PULocationID,
  DOLocationID, tip_rate_fare, is_late_night.
- Weather fields: temperature, apparent_temp, precipitation, snowfall,
  snow_depth, wind_speed, humidity, cloud_cover, is_raining, is_snowing.
- Optional split_bucket; the new demo supplies it explicitly.

The synthetic output supplies only a subset. It lacks passenger/vendor/rate-code,
fare component and scenario controls; it uses pickup_hour instead of hour_of_day
and temperature_c/precipitation_mm instead of the historical weather names.
It lacks the richer weather variables. Raw CSV fields remain strings, so an
explicit typed adapter would also be needed. No model is run as an acceptance
check, and this sample is not advertised as a drop-in weather-model input.

The spatial panels and feature matrices remain separate prepared inputs.
The Knicks event-tagging contract is likewise not rebuilt by this trip demo.
