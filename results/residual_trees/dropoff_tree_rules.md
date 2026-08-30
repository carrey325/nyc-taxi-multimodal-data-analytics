# DROPOFF residualized tiprate tree rules

## Selected hyperparameters
| max_depth | min_samples_leaf | leaf_count | cv_r2_mean | cv_r2_std | train_r2 |
| --- | --- | --- | --- | --- | --- |
| 4 | 25 | 5 | -0.06836812537626417 | 0.15725989743209048 | 0.13824014820115493 |

## Leaf rules
| node_id | rule | prediction | sample_count |
| --- | --- | --- | --- |
| 4 | mta_nearest_cbd_complex_dist_ft <= 55160.7109 and acs_poverty_rate <= 28.1696 and pluto_landuse_share_09_open_space_recreation <= 0.1081 and commonplace_count_per_sqmi <= 153.8850 | 0.00234006739194851 | 110 |
| 5 | mta_nearest_cbd_complex_dist_ft <= 55160.7109 and acs_poverty_rate <= 28.1696 and pluto_landuse_share_09_open_space_recreation <= 0.1081 and commonplace_count_per_sqmi > 153.8850 | -0.0020735505493012 | 55 |
| 6 | mta_nearest_cbd_complex_dist_ft <= 55160.7109 and acs_poverty_rate <= 28.1696 and pluto_landuse_share_09_open_space_recreation > 0.1081 | -0.006770082515857857 | 31 |
| 7 | mta_nearest_cbd_complex_dist_ft <= 55160.7109 and acs_poverty_rate > 28.1696 | -0.011395309644283284 | 28 |
| 8 | mta_nearest_cbd_complex_dist_ft > 55160.7109 | 0.016199104741852884 | 32 |

## Leaf summary
| tree_leaf_id | tree_node_id | zone_count | zones_with_target | mean_target | mean_raw_longrun_tip_rate | representative_zones |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7 | 28 | 28 | -0.011395309644283286 | 0.10294881294347766 | Bedford; Belmont; Borough Park |
| 2 | 6 | 31 | 31 | -0.006770082515857858 | 0.1593336418275152 | Alphabet City; Astoria Park; Battery Park |
| 3 | 5 | 55 | 55 | -0.0020735505493011995 | 0.2384135992904574 | Battery Park City; Bloomingdale; Boerum Hill |
| 4 | 4 | 111 | 110 | 0.00234006739194851 | 0.1497577996208683 | Allerton/Pelham Gardens; Arrochar/Fort Wadsworth; Astoria |
| 5 | 8 | 32 | 32 | 0.016199104741852884 | 0.11906782329560887 | Jamaica Bay; Arden Heights; Baisley Park |