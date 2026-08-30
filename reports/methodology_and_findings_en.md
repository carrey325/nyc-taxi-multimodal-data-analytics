# NYC Taxi Multimodal Data Analytics: Methodology and Findings

## 1. Project objective

This course project studies NYC taxi tipping through four parallel lenses:

1. taxi-record baseline relationships;
2. urban spatial context;
3. Knicks home-game events; and
4. weather context.

Each lens uses taxi tipping as the common outcome but asks a different question. The project therefore combines trip-level prediction, spatial classification, event-study logic, and weather interactions without treating any one module as a causal explanation for the others.

## 2. Analytical framework

| Lens | Unit of analysis | Outcome | Core comparison |
| --- | --- | --- | --- |
| Taxi-record baseline | Individual Yellow Taxi trip | Tip rate and whether a tip is recorded | Incremental predictive contribution of fare, trip, vendor, time, and route variables |
| Urban spatial context | Taxi Zone and zone-month panel | Raw and residualized average tip rate | Zone typology and threshold heterogeneity after month and trip-composition controls |
| Knicks game events | MSG-area trip around a home game | Tip percentage | Pre/post timing, win/loss outcome, and result surprise relative to market expectations |
| Weather context | Trip joined to hourly weather | Tip rate, tip probability, and tip amount | Incremental weather block after trip, vendor, time, and location controls |

The modules use different samples and estimands. Comparisons across modules are interpretive rather than direct coefficient comparisons.

## 3. Lens I: Taxi-record baseline

### 3.1 Question and data

The baseline asks how much tipping variation can be described using TLC trip records alone. The saved benchmark output covers January 2024 and starts from 2,824,462 trips. After economic and payment filters, the main positive-tip card sample contains 2,188,575 trips before fixed modeling subsamples are selected.

The primary outcome is:

`tip_rate_fare = tip_amount / fare_amount`

The baseline distinguishes the conditional positive-tip rate from the broader `tip_or_not` outcome because cash tips and card tips are recorded differently.

### 3.2 Method

Nested OLS specifications add four predictor blocks in order:

1. fare and trip characteristics;
2. vendor indicators;
3. hour, weekday, month, weekend, rush-hour, late-night, and holiday indicators; and
4. pickup zone, dropoff zone, and common route controls.

The linear analysis uses robust standard errors. A LightGBM model provides a nonlinear benchmark, and grouped permutation importance summarizes which predictor blocks carry the most held-out information.

### 3.3 Findings

| Model | Test R-squared | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: |
| Full OLS | 0.1500 | 0.0927 | 0.0606 |
| LightGBM | 0.2132 | 0.0828 | 0.0589 |

Fare and trip characteristics account for about 91.3% of positive grouped permutation loss, followed by location at 4.3%, time at 2.5%, and vendor at 1.9%. The nonlinear model performs better than OLS, indicating interactions and nonlinearities, but most individual-level variation remains unexplained.

The baseline also shows why payment handling matters: almost all recorded positive tips occur among card-paid trips. The broader extensive-margin model therefore partly reflects the recording system rather than only latent willingness to tip.

![Baseline feature importance](../results/figures/baseline_feature_importance.png)

### 3.4 Boundaries

The baseline describes prediction and association. Vendor indicators may proxy for terminal, fleet, route, or customer differences, while raw location IDs do not identify neighborhood mechanisms.

## 4. Lens II: Urban spatial context

### 4.1 Question and data integration

The spatial module asks what types of urban environments characterize NYC Taxi Zones and whether static urban features explain residual tipping differences after conventional controls.

The 260-zone feature stack integrates:

- ACS socioeconomic measures;
- MapPLUTO land-use and building characteristics;
- MTA accessibility;
- CommonPlace civic-place inventory; and
- TLC pickup and dropoff zone-month panels for January 2023-December 2025.

### 4.2 Zone typology

The official typology uses 12 frozen features. Features are winsorized at the 1st and 99th percentiles and standardized across 257 non-airport zones. KMeans with `k=6`, `random_state=42`, and `n_init=50` defines the main clusters. Newark, JFK, and LaGuardia form a rule-based airport class.

| Class | Zones | Combined raw tip rate | Interpretation |
| --- | ---: | ---: | --- |
| CBD-accessible business core | 18 | 0.2444 | Highest raw tip-rate environment |
| Dense mixed-use neighborhood | 50 | 0.2295 | High-density residential and commercial mix |
| Airport special zone | 3 | 0.1960 | Distinct airport pricing and trip structure |
| Industrial / logistics zone | 15 | 0.1577 | Production and logistics land use |
| Recreation / open-space zone | 16 | 0.1354 | Parks and low-density special areas |
| Outer-borough residential zone | 107 | 0.0954 | Large and internally diverse residential class |
| Housing-pressure residential zone | 51 | 0.0696 | Highest poverty and rent-pressure profile |

KMeans `k=6` has the best tested silhouette score (`0.308`) and Davies-Bouldin score (`1.095`) among the candidate configurations.

![NYC Taxi Zone typology](../results/figures/typology_map.png)

### 4.3 Residual models and trees

Pickup and dropoff weighted least-squares models control for month, borough, service zone, spatial coverage, trip distance, fare per mile, passenger count, airport and rate-code composition, peak-hour share, night share, weekend share, and commute-hour share.

| Side | Model-ready rows | R-squared | Adjusted R-squared |
| --- | ---: | ---: | ---: |
| Pickup | 8,644 | 0.9661 | 0.9659 |
| Dropoff | 8,989 | 0.9426 | 0.9422 |

Monthly residuals are aggregated to zone-level targets and modeled with shallow trees using the same 12 static urban features. The selected pickup tree has five leaves and mean five-fold cross-validation R-squared of `-0.0118`; the selected dropoff tree also has five leaves and mean cross-validation R-squared of `-0.0684`.

The high residual-model R-squared and weak tree cross-validation tell a consistent story: month, geography, coverage, and trip composition absorb most systematic variation. The remaining static-feature rules are useful for exploratory segmentation, not accurate prediction.

### 4.4 Boundaries

The clusters are descriptive types, and the tree thresholds are exploratory. Neither identifies a neighborhood treatment effect. Some special or low-volume zones also have limited model-ready outcome coverage.

## 5. Lens III: Knicks home-game events

### 5.1 Question and event design

The event module tests whether sports-related mood and reference-point surprise are associated with tips on card-paid taxi trips originating near Madison Square Garden. It covers 144 Knicks home games across 2022-23, 2023-24, and 2024-25 and uses pickup zones 164 and 186.

Trips are tagged into windows around game start and game end. A previous-day, same-time window provides a control comparison. Market-implied win probability is derived from the point spread, and surprise is defined as realized win status minus expected win probability.

The model sequence includes:

1. pre/post and difference-in-differences comparisons;
2. win and continuous surprise terms;
3. predicted-outcome by realized-outcome interactions; and
4. a default-tip choice model.

### 5.2 Findings

| Sample and specification | Term | Estimate | p-value |
| --- | --- | ---: | ---: |
| 2023-24, ride controls | Win | +0.847 percentage points | 0.0067 |
| 2023-24, ride controls | Loss surprise | -1.670 | 0.0002 |
| 2023-24, full controls | Unexpected moderate win | +1.015 percentage points | 0.0001 |
| Pooled, ride controls | Win | +0.281 percentage points | 0.4063 |
| Pooled, ride controls | Loss surprise | -0.147 | 0.7732 |
| Pooled, full controls | Unexpected moderate win | +0.123 percentage points | 0.6844 |

The 2023-24 season produces the clearest positive estimates. However, neither the broad win term nor the continuous loss-surprise term remains significant in the three-season pooled sample. The 2023-24 upset-win association also disappears under pooled full controls.

The defensible conclusion is therefore season-specific: game outcomes may be associated with short-lived tipping changes near MSG, but the effect is not stable enough to support a general claim that Knicks wins raise tips.

![Knicks 2023-24 tip-rate trajectory](../results/figures/knicks_2023_24_trajectory.png)

### 5.3 Boundaries

Game start times are approximated at 19:30 and game duration at 2.5 hours, which creates timing error for matinees and unusual broadcasts. Some missing moneylines are synthesized from model-implied win probability. These approximations, limited geographic coverage, and cross-season instability constrain interpretation.

## 6. Lens IV: Weather context

### 6.1 Question and sample

The weather module asks whether hourly conditions add information after conventional taxi controls. The broader data workflow integrates 132 TLC monthly files with hourly NYC weather and filters the source warehouse from approximately 780 million rows to 89.6 million card-paid trips. Estimation uses fixed training, validation, and test subsamples rather than loading the full warehouse into each statistical model.

Weather fields include temperature, apparent temperature, precipitation, snowfall, snow depth, wind speed, humidity, cloud cover, rain and snow indicators, and an extreme-temperature indicator.

### 6.2 Method

Nested OLS specifications proceed from fare and trip variables to vendor, time, location, weather main effects, and weather interactions. The comparison interaction is:

`precipitation × extreme temperature`

Additional Logit and tip-amount models check whether conclusions depend on the outcome definition. Subsample models test time-of-day, borough, airport, and severity heterogeneity.

### 6.3 Findings

| Result | Estimate | Interpretation |
| --- | ---: | --- |
| Incremental R-squared from weather | 0.0002 | Almost no additional explanatory power |
| Precipitation × extreme temperature, tip rate | -0.000592, `p=0.40` | Not significant |
| Same interaction, tip probability | OR 1.0211, `p=0.56` | Not significant |
| Same interaction, tip amount | -$0.0119, `p=0.43` | Not significant |
| Wind speed | -0.014 percentage points per km/h, `p<0.001` | Stable but economically small |
| Extreme temperature, Manhattan | +0.26 percentage points, `p<0.05` | Localized subgroup association |
| Rain, 00:00-05:00 | +0.725 percentage points, `p<0.05` | Localized late-night association |

The central precipitation-by-extreme-temperature result from the comparison study is not replicated across three outcome definitions. Wind speed is the most stable weather coefficient, but its magnitude is small. Manhattan extreme-temperature and late-night rain estimates suggest heterogeneity, not a broad weather effect.

![Weather findings summary](../results/figures/weather_results_summary.png)

### 6.4 Boundaries

The analysis covers Yellow Taxi and is shaped by card-tip recording. Severe weather is rare, the highest-severity sample is small, and driver fixed effects are unavailable. Weather may affect trip selection or taxi demand before it affects observed tipping, so these coefficients should not be interpreted as isolated causal effects.

## 7. Cross-module synthesis

The four lenses point to a common hierarchy without collapsing them into one model:

- Trip economics and trip structure provide the largest predictive signal at the individual level.
- Urban context strongly organizes raw geographic differences, but conventional controls absorb most zone-month variation.
- Knicks events can coincide with localized, season-specific changes, but the estimates are specification-sensitive.
- Weather contributes the least incremental explanatory power overall, despite a few meaningful subgroup patterns.

Together, the results show why NYC tipping is best understood as a layered behavioral outcome. Stable trip and spatial structure coexist with weaker, short-lived event and environmental effects.

## 8. Release contents

The public repository contains final scripts, compact result tables, and selected figures. It excludes raw TLC records, weather warehouses, trip-level MSG samples, notebooks, logs, cached bytecode, machine-specific paths, and draft documents.

Key result files:

- `results/baseline/key_metrics.csv`
- `results/typology/typology_cluster_summary.csv`
- `results/residual_trees/residual_model_summary.csv`
- `results/knicks_event_study/key_estimates.csv`
- `results/weather_effects/key_estimates.csv`
