# NYC Taxi Multimodal Data Analytics：方法与结果

## 1. 研究目标
最终研究包把问题分成两层：
1. 用 `KMeans k=6 + airport special` 固定一个不看 tiprate 的空间 typology。
2. 在控制月度和行程结构后，对 `pickup` / `dropoff` 分别构造 residualized tiprate，并用浅层 regression tree 做 rule segmentation。

这意味着 typology 回答“区域本身分成哪些城市类型”，而 tree 回答“控制后还有哪些阈值规则会系统性抬高或压低 tiprate”。

## 2. 运行顺序
1. `build_source_manifest.py`
2. `build_typology_0426.py`
3. `build_residualized_tiprate.py`
4. `build_tiprate_trees.py`
5. `generate_0426_reports.py`

## 3. 外部数据依赖
发布包不复制数据，所有上游输入继续引用外部数据目录中的 `dataset/`、`data script/parquet/`、`data script/spatial/`。下面是本轮实际追踪的 source manifest：

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

## 4. Typology 方法
### 4.1 冻结特征
本轮 typology 只保留 12 个变量：
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

### 4.2 特殊区处理
- `LocationID in {1, 132, 138}` 的 airport zones 不进入主聚类，单独标记为 `Airport special zone`。
- 非 airport universe 预期为 257 个 zone。

### 4.3 聚类口径
- 非 airport 样本对 12 个特征先做 `1%/99% winsorize`，再做 `z-score`。
- diagnostics 继续计算 `k=4..8` 和 `KMeans/Agglomerative`，但正式 typology 固定为 `KMeans(n_clusters=6, random_state=42, n_init=50)`。

### 4.4 Typology 结果
| cluster_id | cluster_name | zone_count | representative_zones | defining_features | pickup_mean_raw_tip_rate | dropoff_mean_raw_tip_rate | combined_mean_tip_rate | tiprate_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Recreation / open-space zone | 16 | Jamaica Bay; Astoria Park; Fresh Meadows | pluto_landuse_share_09_open_space_recreation, pluto_units_res_per_acre, pluto_landuse_share_04_mixed_residential_commercial, mta_station_density_sqmi, mta_nearest_cbd_complex_dist_ft | 0.1195468856678815 | 0.1513147540127005 | 0.135430819840291 | 5 |
| 2 | Dense mixed-use neighborhood | 50 | Roosevelt Island; Boerum Hill; Williamsburg (South Side) | pluto_units_res_per_acre, acs_mean_travel_time_min, pluto_landuse_share_04_mixed_residential_commercial, mta_nearest_cbd_complex_dist_ft, acs_rent_burden_30plus_share | 0.2229635273082005 | 0.2359997678195249 | 0.2294816475638627 | 2 |
| 3 | Housing-pressure residential zone | 51 | Highbridge; West Concourse; Longwood | acs_poverty_rate, acs_median_household_income, acs_rent_burden_30plus_share, acs_mean_travel_time_min, pluto_office_area_share_of_bldg | 0.0362044572060607 | 0.102902402102025 | 0.0695534296540429 | 7 |
| 4 | CBD-accessible business core | 18 | Union Sq; Midtown South; Penn Station/Madison Sq West | mta_station_density_sqmi, pluto_office_area_share_of_bldg, acs_mean_travel_time_min, pluto_landuse_share_04_mixed_residential_commercial, acs_median_household_income | 0.2441773337772089 | 0.2446329163927812 | 0.2444051250849951 | 1 |
| 5 | Outer-borough residential zone | 107 | Kew Gardens Hills; Sheepshead Bay; Richmond Hill | mta_nearest_cbd_complex_dist_ft, commonplace_count_per_sqmi, pluto_landuse_share_04_mixed_residential_commercial, acs_mean_travel_time_min, pluto_units_res_per_acre | 0.0533738295140572 | 0.1373682260694652 | 0.0953710277917612 | 6 |
| 6 | Industrial / logistics zone | 15 | Sunset Park West; Sunnyside; Long Island City/Queens Plaza | pluto_landuse_share_06_industrial_manufacturing, mta_nearest_cbd_complex_dist_ft, pluto_units_res_per_acre, commonplace_count_per_sqmi, acs_rent_burden_30plus_share | 0.1297872725785465 | 0.1856476335322932 | 0.1577174530554199 | 4 |
| 7 | Airport special zone | 3 | Newark Airport; JFK Airport; LaGuardia Airport | rule-based airport special handling | 0.1833951397038673 | 0.2085061492884769 | 0.1959506444961721 | 3 |

### 4.5 聚类 QA
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

## 5. Residualized tiprate 方法
### 5.1 建模口径
- 统一 WLS 公式：`avg_tip_rate ~ C(year_month) + C(Borough) + C(service_zone) + tract_coverage_share + is_partial_coverage_zone + is_water_park_special_proxy + is_source_missing_special_area + avg_trip_distance + avg_fare_per_mile + avg_passenger_count + share_airport_fee + share_airport_ratecode + share_negotiated_fare + share_special_ratecode + share_peak_hour_pickup + share_night_trip + share_weekend_trip + share_commute_hour_trip`
- 因变量：`avg_tip_rate`
- 权重：`valid_tip_rate_trip_count`
- `pickup` 与 `dropoff` 分开拟合，不合并目标。
- 月度残差生成后，再按 zone 用 `valid_tip_rate_trip_count` 加权聚合成长期结构性偏差。

### 5.2 Residual 模型摘要
| target_side | formula | panel_rows_total | panel_rows_model_ready | panel_rows_dropped | zones_total | zones_model_ready | zones_without_model_ready_rows | r_squared | adj_r_squared | nobs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pickup | avg_tip_rate ~ C(year_month) + C(Borough) + C(service_zone) + tract_coverage_share + is_partial_coverage_zone + is_water_park_special_proxy + is_source_missing_special_area + avg_trip_distance + avg_fare_per_mile + avg_passenger_count + share_airport_fee + share_airport_ratecode + share_negotiated_fare + share_special_ratecode + share_peak_hour_pickup + share_night_trip + share_weekend_trip + share_commute_hour_trip | 9360 | 8644 | 716 | 260 | 259 | 1 | 0.9661345826039642 | 0.965913729759644 | 8644.0 |
| dropoff | avg_tip_rate ~ C(year_month) + C(Borough) + C(service_zone) + tract_coverage_share + is_partial_coverage_zone + is_water_park_special_proxy + is_source_missing_special_area + avg_trip_distance + avg_fare_per_mile + avg_passenger_count + share_airport_fee + share_airport_ratecode + share_negotiated_fare + share_special_ratecode + share_peak_hour_pickup + share_night_trip + share_weekend_trip + share_commute_hour_trip | 9360 | 8989 | 371 | 260 | 259 | 1 | 0.9425742337344571 | 0.9422141975823222 | 8989.0 |

## 6. Tree segmentation 方法
- tree 只在非 airport zones 上训练。
- 目标分别是 `pickup_residualized_tip_rate` 和 `dropoff_residualized_tip_rate`。
- 训练特征仍是同一组 12 个 zone-level 特征。
- tree 输入特征只做 `1%/99% winsorize`，不做 z-score，保证阈值可解释。
- 超参数网格固定：`max_depth in {2,3,4}`，`min_samples_leaf in {10,15,20,25}`。
- 只接受最终 leaf 数在 `5-7` 的树，再按 `5-fold CV R²` 选择最优模型。

### 6.1 Pickup tree 选择结果
| max_depth | min_samples_leaf | leaf_count | cv_r2_mean | cv_r2_std | train_r2 | chosen |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 20 | 5 | -0.0117707319865933 | 0.2363465591354648 | 0.2324301107542773 | True |
| 4 | 20 | 6 | -0.0342814148603919 | 0.2359369167148829 | 0.2487066624015387 | False |
| 3 | 15 | 5 | -0.0352568489662527 | 0.2566114362431863 | 0.2378282841346239 | False |
| 4 | 25 | 5 | -0.0653352683562584 | 0.2715649864563016 | 0.2251977095918641 | False |
| 4 | 15 | 6 | -0.0771526412537105 | 0.2606656908652742 | 0.2624887508890331 | False |
| 3 | 10 | 7 | -0.0873961512388272 | 0.3108060127721387 | 0.2992474443602858 | False |

### 6.2 Dropoff tree 选择结果
| max_depth | min_samples_leaf | leaf_count | cv_r2_mean | cv_r2_std | train_r2 | chosen |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 25 | 5 | -0.0683681253762641 | 0.1572598974320904 | 0.1382401482011549 | True |
| 4 | 20 | 5 | -0.1056235691938681 | 0.1778778832435738 | 0.1396043910915594 | False |
| 4 | 10 | 7 | -0.1323901215318162 | 0.09918183462655 | 0.2397065870848297 | False |
| 3 | 10 | 5 | -0.1560898797962445 | 0.0987123431283833 | 0.1876283202328033 | False |
| 4 | 15 | 6 | -0.2481830814145222 | 0.1445519723355427 | 0.2204272609292247 | False |

## 7. 主结论
### 7.1 Typology 层结论
Typology 层面，高 tiprate cluster 仍然集中在 `CBD-accessible business core` 和 `Dense mixed-use neighborhood`，而低 tiprate cluster 主要是 `Housing-pressure residential zone` 与 `Outer-borough residential zone`。

| cluster_id | cluster_name | zone_count | pickup_mean_raw_tiprate | pickup_mean_residualized_tiprate | dropoff_mean_raw_tiprate | dropoff_mean_residualized_tiprate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Recreation / open-space zone | 16 | 0.11954688566788153 | 0.016755044436897782 | 0.1513147540127005 | 0.017365417279616414 |
| 2 | Dense mixed-use neighborhood | 50 | 0.22296352730820054 | -0.0005070600144209279 | 0.23599976781952492 | -0.0008903031751265194 |
| 3 | Housing-pressure residential zone | 51 | 0.03620445720606073 | -0.0010095038626372545 | 0.10290240210202503 | -0.004786386357316596 |
| 4 | CBD-accessible business core | 18 | 0.24417733377720893 | -0.0032896867424235 | 0.24463291639278129 | -0.0015439577729319026 |
| 5 | Outer-borough residential zone | 107 | 0.0533738295140572 | 0.0035362881710135586 | 0.13736822606946517 | 0.0019350987002196196 |
| 6 | Industrial / logistics zone | 15 | 0.12978727257854647 | -0.007705166380539905 | 0.18564763353229322 | -0.002250851946290587 |
| 7 | Airport special zone | 3 | 0.18339513970386737 | 2.386138264131785e-05 | 0.20850614928847686 | -3.648765173511039e-05 |

### 7.2 Tree 层结论
Tree 层面，模型不是在“重新定义空间类型”，而是在现有 typology 之上继续切出解释 residualized tiprate 的阈值规则。
如果某个 cluster 内部被分到多个 leaf，说明同类空间结构内部仍然存在残余的 tiprate 分层。
同时也要注意：在控制 month FE、borough/service zone、coverage 和 trip composition 之后，当前浅层树的 5-fold CV R² 已经明显下降，说明 residualized tiprate 的剩余空间信号比 raw long-run tiprate 弱得多。
因此最终 tree 更适合被解释为“规则化分段与异质性探查工具”，而不是高精度预测器。

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

## 8. 结果解读边界
- Typology 是描述性的空间分型，不是因果识别。
- Residualized tiprate 已经控制了一部分月度和 trip composition，但仍然不是严格因果效应。
- Tree 的价值是提供阈值规则解释，不是追求黑箱预测极限。
- `Great Kills Park` 这类 zone 如果没有足够 panel 观测，可能没有 residual target，但仍然可以基于静态特征被映射到某个 tree leaf。

## 9. 关键图
![Typology map](../outputs/figures/typology_map.png)

![Pickup tree](../outputs/figures/pickup_tree.png)

![Dropoff tree](../outputs/figures/dropoff_tree.png)
