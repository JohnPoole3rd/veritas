# Platonic Earth website — HTTP API

Owner: D. Served by `web/server.py` (FastAPI). The static site in `web/static/` talks only to these routes.
Every number comes from `artifacts/metrics/` or from a live computation; nothing is typed in by hand.

All errors are JSON: `{"error": "plain-language message"}` with a 4xx/5xx status.

## Pages

| Route | What |
|---|---|
| `GET /` | the site (`web/static/index.html`) |
| `GET /static/...` | CSS, JS, fonts, baked images |
| `/classic` | the original Gradio demo (T14), unchanged |

## `GET /api/meta`

Everything the page needs at load. Fields are `null` when the source file is missing.

```jsonc
{
  "mode": "REAL",                       // or "MOCK" (placeholder tiles, fake scores)
  "disclaimer": "Prototype, not a land-use determination. ...",
  "device": "mps",                      // "cuda" | "mps" | "cpu" | null in MOCK
  "classes": [                          // fixed order = EUROSAT_CLASSES
    {"name": "AnnualCrop", "label": "Annual crop", "color": "#E3B23C", "ipcc": "Cropland", "group": "Cropland"}
  ],
  "galleries": [{"key": "local", "label": "Upper Arlington", "n": 50},
                {"key": "eurosat", "label": "Europe (EuroSAT test)", "n": 4050}],
  "examples": ["dense forest", "residential neighborhood", "park or grassland", "commercial area", "river corridor"],
  "metrics": {
    "majority": {"accuracy": 0.1111, "macro_f1": 0.02, "test_n": 4050, "most_common_class": "AnnualCrop"},
    "resnet18": {"accuracy": 0.9817, "macro_f1": 0.9812, "test_n": 4050},
    "caption_classification": {            // image -> class, artifacts/metrics/caption_classification_test.csv
      "procrustes": {"accuracy": 0.8121, "top5": 0.9901, "macro_f1": 0.7948, "n_images": 4050},
      "random_rotation": {"accuracy": 0.1551, "top5": 0.5958, "macro_f1": 0.1392, "n_images": 4050},
      "unaligned": {"accuracy": 0.0827, "top5": 0.5326, "macro_f1": 0.072, "n_images": 4050}
    },
    "retrieval": {                         // text -> image, retrieval.csv
      "procrustes": {"top1": 0.96, "top5": 1.0, "n_captions": 50, "n_images": 4050},
      "random_rotation": {"top1": 0.152, "top5": 0.28, "n_captions": 50, "n_images": 4050},
      "unaligned": {"top1": 0.1, "top5": 0.34, "n_captions": 50, "n_images": 4050}
    },
    "local": {                             // local_summary.csv (+ oracle row from local.json)
      "procrustes": {"accuracy": 0.64, "n": 50, "accuracy_unambiguous": 0.8108, "n_unambiguous": 37},
      "resnet18": {"accuracy": 0.64, "n": 50, "accuracy_unambiguous": 0.8108, "n_unambiguous": 37},
      "majority_train": {"accuracy": 0.0, "n": 50, "accuracy_unambiguous": 0.0, "n_unambiguous": 37},
      "oracle_majority_local": {"accuracy": 0.62, "n": 50, "accuracy_unambiguous": 0.7838, "n_unambiguous": 37, "label": "Residential"}
    },
    "local_by_category": [{"local_category": "water_river", "n": 10, "accuracy": 0.0, "resnet18_acc": 0.1,
                           "majority_train_acc": 0.0, "oracle_majority_local_acc": 0.0}],
    "local_label_counts": {"Residential": 31, "River": 10},      // from the manifest
    "per_class": [{"class": "River", "recall": 0.4613, "resnet18_recall": 0.9733}],
    "confusion": {"classes": ["AnnualCrop", "..."], "matrix": [[414, 2, 0]]},   // rows = true, cols = predicted
    "top_confusions": [{"true": "River", "predicted": "Highway", "count": 122, "share_of_true": 0.325}],
    "failure_taxonomy": [{"failure_type": "...", "example": "...", "count": "...", "likely_cause": "...", "fix_idea": "..."}]
  },
  "local_tiles": [{"tile_id": "ua_001", "label": "Residential", "local_category": "residential",
                   "lat": 40.02502, "lon": -83.05117, "date": "2025-08-01", "notes": "...",
                   "pred": "Residential", "score": 0.3834, "correct": true,
                   "top3": ["Residential", "AnnualCrop", "PermanentCrop"],
                   "resnet18_pred": "Residential", "resnet18_prob": 0.9933, "resnet18_correct": true,
                   "image": "/api/tiles/local/ua_001.png"}],
  "local_examples": [{"tile_id": "ua_011", "kind": "Success", "why": "Big-box retail."},
                     {"tile_id": "ua_049", "kind": "Failure", "why": "The Scioto River channel. ..."}],
  "probe": {"query": "climate change", "why": "...", "hits": [{"tile_id": "ua_0xx", "label": "...", "score": 0.05}]},
  "limits": {"max_tiles": 400, "tile_m": 640, "tile_px": 64},
  "presets": [{"key": "upper-arlington", "name": "Upper Arlington, Ohio", "note": "Our test site",
               "bbox": [-83.110, 39.985, -83.037, 40.070], "cached": true}]
}
```

## Search and labelling (precomputed galleries)

| Route | Returns |
|---|---|
| `GET /api/search?q=river+corridor&gallery=local&k=6` | `{"query","gallery","k","hits":[{"tile_id","label","score","image","lat","lon"}],"note"}` — `score` is cosine similarity; `lat`/`lon` only for local tiles; `k` 1–24 |
| `GET /api/tiles/{gallery}/{tile_id}.png` | the original 64×64 tile (`gallery` = `local` or `eurosat`) |
| `POST /api/classify` (multipart `file`) | `{"top":[{"label","score"}×3],"disclaimer"}` |
| `GET /api/classify/local/{tile_id}` | same, for one of the local tiles |

## Region explorer (live Sentinel-2)

| Route | Returns |
|---|---|
| `POST /api/region/estimate` `{"bbox":[w,s,e,n]}` | `{"rows","cols","n_tiles","width_km","height_km","max_tiles","ok","eta_s","message"}` — no network |
| `POST /api/region` `{"bbox":[w,s,e,n],"when":null}` | `{"job": id}`; `when` may be `"YYYY"` or `"YYYY-MM"` |
| `GET /api/jobs/{id}` | `{"id","kind","status":"running"|"done"|"error","stages":[{"stage","detail","t"}],"error","result"}` |
| `GET /api/region/{rid}` | the region result (below) |
| `GET /api/region/{rid}/image.png` | true-colour mosaic |
| `GET /api/region/{rid}/tile/{i}.png` | one 64×64 chip |
| `GET /api/region/{rid}/search?q=parking+lots&k=8` | `{"query","scores":[cosine × n_tiles],"top":[{"i","score"}],"min","max"}` |
| `GET /api/region/{rid}/similar?i=12&k=8` | same shape; image→image cosine in the rotated space |
| `POST /api/region/{rid}/climate` | `{"job": id}`; result below |
| `GET /api/region/{rid}/layer/{ndvi\|lst}.png` | RGBA overlay for the map (same footprint as `image.png`) |
| `GET /api/region/{rid}/export.csv` · `export.geojson` | per-tile table with scene ids, for GIS / notebooks |
| `GET /api/geocode?q=Phoenix` | `[{"name","lat","lon","bbox":[w,s,e,n],"type"}]` (OpenStreetMap Nominatim proxy; `[]` offline) |

Region result:

```jsonc
{
  "id": "a1b2c3d4", "bbox": [-83.11, 39.985, -83.037, 40.07], "mode": "REAL",
  "scene": {"id": "S2C_17TLE_20250801_0_L2A", "datetime": "2025-08-01T16:32:38Z", "cloud_cover": 0.24,
            "platform": "sentinel-2c", "epsg": 32617, "source": "Copernicus Sentinel-2 L2A via Element84 Earth Search"},
  "rows": 15, "cols": 10, "n_tiles": 150, "tile_m": 640, "area_km2": 61.4,
  "corners": [[lon, lat], [lon, lat], [lon, lat], [lon, lat]],          // TL, TR, BR, BL
  "image": "/api/region/a1b2c3d4/image.png",
  "tiles": [{"i": 0, "row": 0, "col": 0, "corners": [[lon, lat]], "center": [lon, lat],
             "nodata_frac": 0.0, "cloud_frac": 0.0,
             "cls": "Residential", "score": 0.38, "top3": [{"label": "Residential", "score": 0.38}]}],
  "composition": [{"cls": "Residential", "n": 96, "share": 0.64, "km2": 39.3}],      // sorted, largest first
  "ipcc": [{"category": "Settlements", "n": 110, "share": 0.73, "km2": 45.1}],
  "elapsed_s": 7.9, "from_cache": false
}
```

Climate job result (measured layers + statistics):

```jsonc
{
  "ndvi": {"tile": [0.61, null], "image": "/api/region/a1b2c3d4/layer/ndvi.png", "vmin": 0.0, "vmax": 0.9},
  "water": {"tile": [0.0, 0.42]},                      // fraction of the tile with NDWI > 0
  "lst": {"tile": [34.2, null], "image": "/api/region/a1b2c3d4/layer/lst.png", "vmin": 27.0, "vmax": 44.0,
          "scene": {"id": "LC08_L2SP_019032_20250723_02_T1", "datetime": "...", "cloud_cover": 0.03, "platform": "landsat-8",
                    "source": "USGS Landsat Collection 2 Level-2 ST via Microsoft Planetary Computer"},
          "days_from_s2": -9},                         // null (with "lst_error") when no usable pass exists
  "lst_error": null,
  "stats": {
    "n_land": 140, "n_water": 10,
    "lst_mean": 35.1, "lst_min": 27.9, "lst_max": 43.6, "ndvi_mean": 0.58,
    "regression": {"slope_per_0p1_ndvi": -1.3, "intercept": 42.6, "r": -0.81, "n": 140},     // land tiles only
    "tercile": {"hot_minus_green_c": 4.2, "least_green_mean": 37.5, "greenest_mean": 33.3, "n_each": 46},
    "by_class": [{"cls": "Industrial", "n": 9, "lst_mean": 40.2, "ndvi_mean": 0.31}],
    "by_group": [{"group": "Built", "n": 110, "lst_mean": 36.0, "ndvi_mean": 0.52}],
    "built_minus_vegetated_c": 3.1,
    "priority": [{"i": 57, "lst": 43.6, "ndvi": 0.22, "center": [lon, lat], "cls": "Industrial"}],   // hottest + least green
    "refuges": [{"i": 3, "lst": 28.4, "ndvi": 0.83, "center": [lon, lat], "cls": "Forest"}]
  }
}
```

Notes for readers of the numbers:

- Land-cover classes are **model estimates** (one rotation fit on European tiles). Surface temperature, NDVI and
  NDWI are **measurements** from Landsat 8/9 and Sentinel-2.
- Surface temperature is the skin temperature of roofs, roads and leaves at the satellite overpass
  (mid-to-late morning local time), not air temperature.
- Tiles that are mostly water (NDWI > 0 on more than half the tile) are left out of the greenness–temperature
  regression and the tercile contrast.
