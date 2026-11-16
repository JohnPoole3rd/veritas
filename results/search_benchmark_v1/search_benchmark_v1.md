Search benchmark v1: 170,454 squares in 50 held-out regions; 532 (query, region) pairs; relevance = mapped in OpenStreetMap; 95% CI by region bootstrap.

| Model | All queries P@10 [95% CI] | Open-vocabulary P@10 | Land-cover P@10 | Chance (all) |
|---|---|---|---|---|
| eurosat | 0.302 [0.274, 0.330] | 0.142 | 0.480 | 0.105 |
| osm_91k | 0.309 [0.284, 0.335] | 0.149 | 0.485 | 0.105 |
| osm_1m | 0.340 [0.315, 0.367] | 0.202 | 0.493 | 0.105 |
| identity (no rotation) | 0.148 [0.125, 0.171] | 0.060 | 0.245 | 0.105 |
| random rotation | 0.112 [0.092, 0.135] | 0.049 | 0.182 | 0.105 |

| Query | eurosat | osm_91k | osm_1m | identity (no rotation) | random rotation | chance |
|---|---|---|---|---|---|---|
| parking lots (open, 30 regions) | 0.29 | 0.36 | 0.43 | 0.05 | 0.09 | 0.06 |
| a golf course (open, 6 regions) | 0.05 | 0.18 | 0.23 | 0.00 | 0.05 | 0.01 |
| sports fields (open, 31 regions) | 0.20 | 0.20 | 0.31 | 0.09 | 0.03 | 0.06 |
| a railway line (open, 31 regions) | 0.11 | 0.06 | 0.18 | 0.04 | 0.07 | 0.05 |
| a quarry (open, 20 regions) | 0.01 | 0.15 | 0.18 | 0.02 | 0.03 | 0.01 |
| a cemetery (open, 26 regions) | 0.05 | 0.02 | 0.05 | 0.02 | 0.03 | 0.02 |
| an airport (open, 15 regions) | 0.05 | 0.10 | 0.21 | 0.02 | 0.00 | 0.01 |
| a construction site (open, 20 regions) | 0.14 | 0.13 | 0.15 | 0.06 | 0.02 | 0.04 |
| an orchard (open, 22 regions) | 0.06 | 0.06 | 0.06 | 0.04 | 0.03 | 0.03 |
| a school (open, 36 regions) | 0.30 | 0.16 | 0.17 | 0.15 | 0.10 | 0.07 |
| a sandy beach (open, 16 regions) | 0.04 | 0.23 | 0.16 | 0.05 | 0.01 | 0.04 |
| wetland or marsh (open, 26 regions) | 0.11 | 0.13 | 0.20 | 0.07 | 0.05 | 0.06 |
| dense forest (vocab, 43 regions) | 0.50 | 0.55 | 0.45 | 0.27 | 0.28 | 0.34 |
| a river (vocab, 45 regions) | 0.40 | 0.36 | 0.48 | 0.21 | 0.19 | 0.10 |
| farm fields (vocab, 38 regions) | 0.47 | 0.45 | 0.39 | 0.27 | 0.10 | 0.19 |
| houses and residential streets (vocab, 44 regions) | 0.71 | 0.63 | 0.71 | 0.35 | 0.30 | 0.17 |
| an industrial area (vocab, 38 regions) | 0.28 | 0.48 | 0.49 | 0.09 | 0.03 | 0.05 |
| a lake or open water (vocab, 45 regions) | 0.49 | 0.45 | 0.44 | 0.26 | 0.17 | 0.17 |

Paired comparison (same query and same place; 95% CI by region bootstrap):

| Comparison | All queries | Open-vocabulary | Land cover |
|---|---|---|---|
| osm_1m - eurosat | +0.038 [+0.019, +0.055] | +0.060 [+0.038, +0.085] | +0.013 [-0.015, +0.041] |
| osm_1m - osm_91k | +0.031 [+0.018, +0.045] | +0.053 [+0.035, +0.071] | +0.007 [-0.013, +0.027] |

Chance by group: all 0.105, open-vocabulary 0.043, land cover 0.174.
Caveat: relevance = mapped in OpenStreetMap (mapped is not always visible); the OSM rotations learned from OSM captions, so this proxy favours them over the EuroSAT rotation.
