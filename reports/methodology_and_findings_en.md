# NYC Taxi Multimodal Data Analytics: Methodology and Findings

## 1. Research objective
The final research package separates the research problem into two layers:
1. Build a spatial typology with `KMeans k=6 + airport special` without using tiprate in clustering.
2. Construct residualized tiprate for `pickup` and `dropoff` separately after controlling for month and trip composition, then fit shallow regression trees for rule-based segmentation.

The typology answers “what kinds of urban zones exist,” while the trees answer “what threshold rules still explain tiprate differences after controls.”

## 2. Execution order
1. `build_source_manifest.py`
2. `build_typology_0426.py`
3. `build_residualized_tiprate.py`
4. `build_tiprate_trees.py`
5. `generate_0426_reports.py`

## 3. External data dependencies
The release package does not copy data. It reads the existing analysis data from `dataset/`, `data script/parquet/`, and `data script/spatial/`. The tracked source manifest is:

| source_id | source_family | file_name | row_count | purpose |
| --- | --- | --- | --- | --- |
| zone_features_acs | ACS | zone_features_acs.parquet | 260 | Zone-level ACS-derived contextual features. |
| commonplace_raw | CommonPlace | CommonPlace_20260426.csv | 20523 | Raw point-level civic-place inventory. |
| taxi_zone_to_cdta | Crosswalk | taxi_zone_to_cdta.parquet | 739 | Taxi Zone to CDTA area-weighted crosswalk. |
| taxi_zone_to_nta | Crosswalk | taxi_zone_to_nta.parquet | 1383 | Taxi Zone to NTA area-weighted crosswalk. |
| taxi_zone_to_tract | Crosswalk | taxi_zone_to_tract.parquet | 5174 | Taxi Zone to tract area-weighted crosswalk. |
| zone_features_grouped | Feature table | zone_features_grouped.parquet | 260 | Grouped static zone feature stack. |
| taxi_zone_geometry | Geometry | taxi_zone_geometry.geoparquet | 260 | Strict Taxi Zone polygon geometry. |
| zone_features_mta | MTA | zone_features_mta.parquet | 260 | Zone-level MTA accessibility features. |
| zone_features_pluto | PLUTO | zone_features_pluto.parquet | 260 | Zone-level PLUTO built-environment features. |
| dropoff_zone_month_panel | Taxi panel | dropoff_zone_month_panel_cc_enriched.parquet | 9360 | Dropoff zone-month credit-card outcome panel. |
| pickup_zone_month_panel | Taxi panel | pickup_zone_month_panel_cc_enriched.parquet | 9360 | Pickup zone-month credit-card outcome panel. |
| dropoff_longrun_summary | Taxi summary | dropoff_zone_longrun_summary_cc.parquet | 260 | Dropoff long-run zone summary. |
| pickup_longrun_summary | Taxi summary | pickup_zone_longrun_summary_cc.parquet | 260 | Pickup long-run zone summary. |

## 4. Typology methodology
### 4.1 Frozen feature set
The typology uses the following 12 features:
- `acs_mean_travel_time_min`
- `acs_median_household_income`
- `acs_poverty_rate`
- `acs_rent_burden_30plus_share`
- `commonplace_count_per_sqmi`
- `mta_nearest_cbd_complex_dist_ft`
- `mta_station_density_sqmi`
- `pluto_landuse_share_04_mixed_residential_commercial`
- `pluto_landuse_share_06_industrial_manufacturing`
- `pluto_landuse_share_09_open_space_recreation`
- `pluto_office_area_share_of_bldg`
- `pluto_units_res_per_acre`

### 4.2 Special-zone handling
- Airport zones with `LocationID in {1, 132, 138}` are excluded from the main clustering and grouped into `Airport special zone`.
- The non-airport clustering universe is expected to contain 257 zones.

### 4.3 Clustering rule
- The 12 features are winsorized at `1%/99%` and standardized with `z-score` for non-airport zones.
- Diagnostics are still computed for `k=4..8` and both `KMeans` / `Agglomerative`, but the official typology is fixed at `KMeans(n_clusters=6, random_state=42, n_init=50)`.

### 4.4 Typology results
| cluster_id | cluster_name | zone_count | representative_zones | defining_features | pickup_mean_raw_tip_rate | dropoff_mean_raw_tip_rate | combined_mean_tip_rate | tiprate_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Recreation / open-space zone | 16 | Jamaica Bay; Astoria Park; Fresh Meadows | pluto_landuse_share_09_open_space_recreation, pluto_units_res_per_acre, pluto_landuse_share_04_mixed_residential_commercial, mta_station_density_sqmi, mta_nearest_cbd_complex_dist_ft | 0.1195468856678815 | 0.1513147540127005 | 0.135430819840291 | 5 |
| 2 | Dense mixed-use neighborhood | 50 | Roosevelt Island; Boerum Hill; Williamsburg (South Side) | pluto_units_res_per_acre, acs_mean_travel_time_min, pluto_landuse_share_04_mixed_residential_commercial, mta_nearest_cbd_complex_dist_ft, acs_rent_burden_30plus_share | 0.2229635273082005 | 0.2359997678195249 | 0.2294816475638627 | 2 |
| 3 | Housing-pressure residential zone | 51 | Highbridge; West Concourse; Longwood | acs_poverty_rate, acs_median_household_income, acs_rent_burden_30plus_share, acs_mean_travel_time_min, pluto_office_area_share_of_bldg | 0.0362044572060607 | 0.102902402102025 | 0.0695534296540429 | 7 |
| 4 | CBD-accessible business core | 18 | Union Sq; Midtown South; Penn Station/Madison Sq West | mta_station_density_sqmi, pluto_office_area_share_of_bldg, acs_mean_travel_time_min, pluto_landuse_share_04_mixed_residential_commercial, acs_median_household_income | 0.2441773337772089 | 0.2446329163927812 | 0.2444051250849951 | 1 |
| 5 | Outer-borough residential zone | 107 | Kew Gardens Hills; Sheepshead Bay; Richmond Hill | mta_nearest_cbd_complex_dist_ft, commonplace_count_per_sqmi, pluto_landuse_share_04_mixed_residential_commercial, acs_mean_travel_time_min, pluto_units_res_per_acre | 0.0533738295140572 | 0.1373682260694652 | 0.0953710277917612 | 6 |
| 6 | Industrial / logistics zone | 15 | Sunset Park West; Sunnyside; Long Island City/Queens Plaza | pluto_landuse_share_06_industrial_manufacturing, mta_nearest_cbd_complex_dist_ft, pluto_units_res_per_acre, commonplace_count_per_sqmi, acs_rent_burden_30plus_share | 0.1297872725785465 | 0.1856476335322932 | 0.1577174530554199 | 4 |
| 7 | Airport special zone | 3 | Newark Airport; JFK Airport; LaGuardia Airport | rule-based airport special handling | 0.1833951397038673 | 0.2085061492884769 | 0.1959506444961721 | 3 |

### 4.5 Clustering QA
| algorithm | k | silhouette_score | davies_bouldin_score | min_cluster_size | max_cluster_share | average_cluster_profile_contrast | chosen_for_0426_typology |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agglomerative | 4 | 0.2700358493737088 | 1.2394284152809891 | 17 | 0.4591439688715953 | 1.1287891843070903 | False |
| Agglomerative | 5 | 0.2894360671823157 | 1.1518420547787642 | 14 | 0.4046692607003891 | 1.280701096926753 | False |
| Agglomerative | 6 | 0.2875711278801667 | 1.1489432619136164 | 14 | 0.4046692607003891 | 1.4843180415639816 | False |
| Agglomerative | 7 | 0.2730135063507071 | 1.1911620991994576 | 14 | 0.4046692607003891 | 1.442625050013108 | False |
| Agglomerative | 8 | 0.264241105992201 | 1.2437651168617538 | 14 | 0.4046692607003891 | 1.472261741265099 | False |
| KMeans | 4 | 0.2671715605627409 | 1.3584538108280246 | 18 | 0.4708171206225681 | 1.3592046654411978 | False |
| KMeans | 5 | 0.2856656537172629 | 1.173117462313802 | 15 | 0.4669260700389105 | 1.4379357529244652 | False |
| KMeans | 6 | 0.3078981972832068 | 1.0950039897335662 | 15 | 0.4163424124513619 | 1.5043813706384477 | True |
| KMeans | 7 | 0.2917129018151482 | 1.245571461543856 | 15 | 0.4046692607003891 | 1.493194743788912 | False |
| KMeans | 8 | 0.2546217124628623 | 1.1707112173499894 | 14 | 0.3463035019455253 | 1.4037522503194666 | False |

## 5. Residualized tiprate methodology
### 5.1 Modeling rule
- Unified WLS formula: `avg_tip_rate ~ C(year_month) + C(Borough) + C(service_zone) + tract_coverage_share + is_partial_coverage_zone + is_water_park_special_proxy + is_source_missing_special_area + avg_trip_distance + avg_fare_per_mile + avg_passenger_count + share_airport_fee + share_airport_ratecode + share_negotiated_fare + share_special_ratecode + share_peak_hour_pickup + share_night_trip + share_weekend_trip + share_commute_hour_trip`
- Outcome: `avg_tip_rate`
- Weight: `valid_tip_rate_trip_count`
- `pickup` and `dropoff` are modeled separately.
- Monthly residuals are aggregated back to zone level using `valid_tip_rate_trip_count` as the weight.

### 5.2 Residual model summary
| target_side | formula | panel_rows_total | panel_rows_model_ready | panel_rows_dropped | zones_total | zones_model_ready | zones_without_model_ready_rows | r_squared | adj_r_squared | nobs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pickup | avg_tip_rate ~ C(year_month) + C(Borough) + C(service_zone) + tract_coverage_share + is_partial_coverage_zone + is_water_park_special_proxy + is_source_missing_special_area + avg_trip_distance + avg_fare_per_mile + avg_passenger_count + share_airport_fee + share_airport_ratecode + share_negotiated_fare + share_special_ratecode + share_peak_hour_pickup + share_night_trip + share_weekend_trip + share_commute_hour_trip | 9360 | 8644 | 716 | 260 | 259 | 1 | 0.9661345826039642 | 0.965913729759644 | 8644.0 |
| dropoff | avg_tip_rate ~ C(year_month) + C(Borough) + C(service_zone) + tract_coverage_share + is_partial_coverage_zone + is_water_park_special_proxy + is_source_missing_special_area + avg_trip_distance + avg_fare_per_mile + avg_passenger_count + share_airport_fee + share_airport_ratecode + share_negotiated_fare + share_special_ratecode + share_peak_hour_pickup + share_night_trip + share_weekend_trip + share_commute_hour_trip | 9360 | 8989 | 371 | 260 | 259 | 1 | 0.9425742337344571 | 0.9422141975823222 | 8989.0 |

## 6. Tree segmentation methodology
- Trees are trained on non-airport zones only.
- Targets are `pickup_residualized_tip_rate` and `dropoff_residualized_tip_rate` separately.
- Predictors remain the same 12 zone-level features.
- Tree inputs are winsorized but not z-scored so that thresholds remain interpretable.
- Hyperparameter grid: `max_depth in {2,3,4}` and `min_samples_leaf in {10,15,20,25}`.
- Only trees with `5-7` terminal leaves are eligible, and the final choice is based on `5-fold CV R²`.

### 6.1 Pickup tree selection
| max_depth | min_samples_leaf | leaf_count | cv_r2_mean | cv_r2_std | train_r2 | chosen |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 20 | 5 | -0.0117707319865933 | 0.2363465591354648 | 0.2324301107542773 | True |
| 4 | 20 | 6 | -0.0342814148603919 | 0.2359369167148829 | 0.2487066624015387 | False |
| 3 | 15 | 5 | -0.0352568489662527 | 0.2566114362431863 | 0.2378282841346239 | False |
| 4 | 25 | 5 | -0.0653352683562584 | 0.2715649864563016 | 0.2251977095918641 | False |
| 4 | 15 | 6 | -0.0771526412537105 | 0.2606656908652742 | 0.2624887508890331 | False |
| 3 | 10 | 7 | -0.0873961512388272 | 0.3108060127721387 | 0.2992474443602858 | False |

### 6.2 Dropoff tree selection
| max_depth | min_samples_leaf | leaf_count | cv_r2_mean | cv_r2_std | train_r2 | chosen |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 25 | 5 | -0.0683681253762641 | 0.1572598974320904 | 0.1382401482011549 | True |
| 4 | 20 | 5 | -0.1056235691938681 | 0.1778778832435738 | 0.1396043910915594 | False |
| 4 | 10 | 7 | -0.1323901215318162 | 0.09918183462655 | 0.2397065870848297 | False |
| 3 | 10 | 5 | -0.1560898797962445 | 0.0987123431283833 | 0.1876283202328033 | False |
| 4 | 15 | 6 | -0.2481830814145222 | 0.1445519723355427 | 0.2204272609292247 | False |

## 7. Main findings
### 7.1 Typology-layer findings
At the typology level, the highest raw tiprate clusters remain `CBD-accessible business core` and `Dense mixed-use neighborhood`, while the lowest clusters are mainly `Housing-pressure residential zone` and `Outer-borough residential zone`.

| cluster_id | cluster_name | zone_count | pickup_mean_raw_tiprate | pickup_mean_residualized_tiprate | dropoff_mean_raw_tiprate | dropoff_mean_residualized_tiprate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Recreation / open-space zone | 16 | 0.11954688566788153 | 0.016755044436897782 | 0.1513147540127005 | 0.017365417279616414 |
| 2 | Dense mixed-use neighborhood | 50 | 0.22296352730820054 | -0.0005070600144209279 | 0.23599976781952492 | -0.0008903031751265194 |
| 3 | Housing-pressure residential zone | 51 | 0.03620445720606073 | -0.0010095038626372545 | 0.10290240210202503 | -0.004786386357316596 |
| 4 | CBD-accessible business core | 18 | 0.24417733377720893 | -0.0032896867424235 | 0.24463291639278129 | -0.0015439577729319026 |
| 5 | Outer-borough residential zone | 107 | 0.0533738295140572 | 0.0035362881710135586 | 0.13736822606946517 | 0.0019350987002196196 |
| 6 | Industrial / logistics zone | 15 | 0.12978727257854647 | -0.007705166380539905 | 0.18564763353229322 | -0.002250851946290587 |
| 7 | Airport special zone | 3 | 0.18339513970386737 | 2.386138264131785e-05 | 0.20850614928847686 | -3.648765173511039e-05 |

### 7.2 Tree-layer findings
The trees do not redefine spatial types. Instead, they split zones within and across typology classes using threshold rules that explain residualized tiprate differences after controls.
If one cluster maps into multiple leaves, that indicates residual heterogeneity inside the same spatial type.
It is also important that once month fixed effects, borough/service-zone structure, coverage, and trip composition are controlled, the shallow-tree 5-fold CV R² drops materially.
That means the remaining spatial signal in residualized tiprate is much weaker than in raw long-run tiprate, so the final trees should be read as rule-based heterogeneity probes rather than high-accuracy predictors.

#### Pickup cluster × leaf
| cluster_id | cluster_name | tree_leaf_id | zone_count | zones_with_target | mean_residualized_tip_rate | mean_raw_longrun_tip_rate | representative_zones |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Recreation / open-space zone | 1 | 6 | 6 | -0.0012353857775923 | 0.152296049263179 | Astoria Park; Battery Park; Crotona Park |
| 1 | Recreation / open-space zone | 2 | 2 | 2 | 0.0193325302248694 | 0.1065426303154349 | Highbridge Park; Inwood Hill Park |
| 1 | Recreation / open-space zone | 3 | 1 | 1 | 0.0058498435092819 | 0.0834179250296447 | Bronx Park |
| 1 | Recreation / open-space zone | 4 | 1 | 1 | 0.0120500800568573 | 0.0906430635412059 | Freshkills Park |
| 1 | Recreation / open-space zone | 5 | 6 | 5 | 0.0404345994406285 | 0.0984561480474856 | Jamaica Bay; Breezy Point/Fort Tilden/Riis Beach; Fresh Meadows |
| 2 | Dense mixed-use neighborhood | 1 | 50 | 50 | -0.0005070600144209 | 0.2229635273082005 | Alphabet City; Battery Park City; Bloomingdale |
| 3 | Housing-pressure residential zone | 1 | 31 | 31 | -0.0062261637818999 | 0.0468143571078641 | Bedford; Belmont; Borough Park |
| 3 | Housing-pressure residential zone | 2 | 10 | 10 | 0.0023234924431295 | 0.0265558792710489 | Brownsville; East New York; East New York/Pennsylvania Avenue |
| 3 | Housing-pressure residential zone | 3 | 8 | 8 | 0.0129927213413021 | 0.0072557989916554 | Bedford Park; Brighton Beach; Bronxdale |
| 3 | Housing-pressure residential zone | 4 | 1 | 1 | 0.0111968007700198 | 0.0027110796259442 | Williamsbridge/Olinville |
| 3 | Housing-pressure residential zone | 5 | 1 | 1 | 0.0031528843126668 | 0.0688659828956332 | Pelham Bay Park |
| 4 | CBD-accessible business core | 1 | 18 | 18 | -0.0032896867424235 | 0.2441773337772089 | Downtown Brooklyn/MetroTech; Financial District North; Financial District South |
| 5 | Outer-borough residential zone | 1 | 39 | 39 | -0.0082233885244815 | 0.0806996382008978 | Arrochar/Fort Wadsworth; Astoria; Bay Ridge |
| 5 | Outer-borough residential zone | 2 | 13 | 13 | 0.0008753765427732 | 0.0187132553577654 | Bath Beach; Bensonhurst East; Canarsie |
| 5 | Outer-borough residential zone | 3 | 19 | 19 | 0.0030313441008412 | 0.0399061728488406 | Allerton/Pelham Gardens; Auburndale; Briarwood/Jamaica Hills |
| 5 | Outer-borough residential zone | 4 | 20 | 20 | 0.0123040175976639 | 0.0394980762505515 | Baisley Park; Bayside; Bloomfield/Emerson Hill |
| 5 | Outer-borough residential zone | 5 | 16 | 16 | 0.0240024501142446 | 0.0482664212111967 | Arden Heights; Bay Terrace/Fort Totten; Bellerose |
| 6 | Industrial / logistics zone | 1 | 15 | 15 | -0.0077051663805399 | 0.1297872725785464 | East Williamsburg; Gowanus; Greenpoint |

#### Dropoff cluster × leaf
| cluster_id | cluster_name | tree_leaf_id | zone_count | zones_with_target | mean_residualized_tip_rate | mean_raw_longrun_tip_rate | representative_zones |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Recreation / open-space zone | 1 | 1 | 1 | -0.0193369650668054 | 0.0954783518172735 | Crotona Park |
| 1 | Recreation / open-space zone | 2 | 11 | 11 | -0.0003225982399811 | 0.1734705629646119 | Astoria Park; Battery Park; Bronx Park |
| 1 | Recreation / open-space zone | 5 | 4 | 4 | 0.075183055545115 | 0.1043453799438008 | Jamaica Bay; Breezy Point/Fort Tilden/Riis Beach; Freshkills Park |
| 2 | Dense mixed-use neighborhood | 2 | 5 | 5 | -0.0080240295034519 | 0.2227548589726301 | Alphabet City; DUMBO/Vinegar Hill; Fort Greene |
| 2 | Dense mixed-use neighborhood | 3 | 35 | 35 | -0.0013439149826244 | 0.2435896246540549 | Battery Park City; Bloomingdale; Boerum Hill |
| 2 | Dense mixed-use neighborhood | 4 | 10 | 10 | 0.004264201315279 | 0.2160577233221171 | Carroll Gardens; Cobble Hill; Columbia Street |
| 3 | Housing-pressure residential zone | 1 | 25 | 25 | -0.0109110113344045 | 0.1027413650957039 | Bedford; Belmont; Borough Park |
| 3 | Housing-pressure residential zone | 2 | 2 | 2 | -0.0034681894413526 | 0.0536522682266624 | East New York; Soundview/Castle Hill |
| 3 | Housing-pressure residential zone | 3 | 4 | 4 | -0.0072767958775913 | 0.1612678936910196 | Central Harlem North; Hamilton Heights; Ocean Hill |
| 3 | Housing-pressure residential zone | 4 | 19 | 19 | 0.0030605855462728 | 0.0975456117110641 | Bedford Park; Brighton Beach; Bronxdale |
| 3 | Housing-pressure residential zone | 5 | 1 | 1 | 0.0065620161508552 | 0.0737456460830553 | Pelham Bay Park |
| 4 | CBD-accessible business core | 2 | 1 | 1 | -0.0032392677378215 | 0.1932723298148834 | Downtown Brooklyn/MetroTech |
| 4 | CBD-accessible business core | 3 | 16 | 16 | -0.002368817019334 | 0.2463774702074471 | Financial District North; Financial District South; Garment District |
| 4 | CBD-accessible business core | 4 | 1 | 1 | 0.0133491001343913 | 0.2680806419360244 | Flatiron |
| 5 | Outer-borough residential zone | 2 | 10 | 10 | -0.0089344502592739 | 0.1300732297913495 | Canarsie; Cypress Hills; Dyker Heights |
| 5 | Outer-borough residential zone | 4 | 70 | 69 | 0.0012085195856238 | 0.1440761809543273 | Allerton/Pelham Gardens; Arrochar/Fort Wadsworth; Astoria |
| 5 | Outer-borough residential zone | 5 | 27 | 27 | 0.0078176708669621 | 0.1229275251704157 | Arden Heights; Baisley Park; Bellerose |
| 6 | Industrial / logistics zone | 1 | 2 | 2 | -0.013478210806506 | 0.1092771416037511 | Hunts Point; Mott Haven/Port Morris |
| 6 | Industrial / logistics zone | 2 | 2 | 2 | -0.0333418403106376 | 0.1580416224986929 | Queensbridge/Ravenswood; Saint Michaels Cemetery/Woodside |
| 6 | Industrial / logistics zone | 4 | 11 | 11 | 0.0054433930036298 | 0.2045524522526827 | East Williamsburg; Gowanus; Greenpoint |

## 8. Interpretation boundaries
- The typology is descriptive, not causal.
- Residualized tiprate controls for time and trip composition, but it is still not a causal treatment effect.
- The trees are intended for threshold-based interpretation rather than black-box prediction.
- Zones such as `Great Kills Park` may lack enough model-ready panel observations for a residual target, but can still be assigned to a tree leaf using static features.

## 9. Key figures
![Typology map](../outputs/figures/typology_map.png)

![Pickup tree](../outputs/figures/pickup_tree.png)

![Dropoff tree](../outputs/figures/dropoff_tree.png)
