# PICKUP residualized tiprate tree rules

## Selected hyperparameters
| max_depth | min_samples_leaf | leaf_count | cv_r2_mean | cv_r2_std | train_r2 |
| --- | --- | --- | --- | --- | --- |
| 3 | 20 | 5 | -0.011770731986593331 | 0.23634655913546485 | 0.23243011075427733 |

## Leaf rules
| node_id | rule | prediction | sample_count |
| --- | --- | --- | --- |
| 3 | mta_nearest_cbd_complex_dist_ft <= 50170.5625 and mta_nearest_cbd_complex_dist_ft <= 43392.4922 and acs_mean_travel_time_min <= 44.7827 | -0.0045363547305157246 | 159 |
| 4 | mta_nearest_cbd_complex_dist_ft <= 50170.5625 and mta_nearest_cbd_complex_dist_ft <= 43392.4922 and acs_mean_travel_time_min > 44.7827 | 0.0029311951974834403 | 25 |
| 5 | mta_nearest_cbd_complex_dist_ft <= 50170.5625 and mta_nearest_cbd_complex_dist_ft > 43392.4922 | 0.005978112576988686 | 28 |
| 7 | mta_nearest_cbd_complex_dist_ft > 50170.5625 and pluto_landuse_share_04_mixed_residential_commercial <= 0.0089 | 0.02678932197016926 | 22 |
| 8 | mta_nearest_cbd_complex_dist_ft > 50170.5625 and pluto_landuse_share_04_mixed_residential_commercial > 0.0089 | 0.012242146944552557 | 22 |

## Leaf summary
| tree_leaf_id | tree_node_id | zone_count | zones_with_target | mean_target | mean_raw_longrun_tip_rate | representative_zones |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 159 | 159 | -0.004536354730515722 | 0.14466971520651487 | Alphabet City; Arrochar/Fort Wadsworth; Astoria |
| 2 | 4 | 25 | 25 | 0.002931195197483441 | 0.028876654919692397 | Bath Beach; Bensonhurst East; Brownsville |
| 3 | 5 | 28 | 28 | 0.005978112576988686 | 0.032131485753245 | Allerton/Pelham Gardens; Auburndale; Bedford Park |
| 4 | 8 | 22 | 22 | 0.012242146944552558 | 0.040150712189917344 | Baisley Park; Bayside; Bloomfield/Emerson Hill |
| 5 | 7 | 23 | 22 | 0.02678932197016926 | 0.06060952102328224 | Jamaica Bay; Arden Heights; Bay Terrace/Fort Totten |