# EV Charging Station Site Analysis — Multispectral Campus Mapper + ML Predictor

A Python tool that ingests any OpenStreetMap `.osm` file and produces
multispectral classification maps, EV suitability heatmaps, 3D visualisations,
and a ranked JSON report of candidate charging station locations.

Includes an **ML pipeline** that trains a Random Forest classifier on India-wide
EV charging station data and predicts optimal placement on any user-supplied OSM map.

Built as a prototype on **BHU (Banaras Hindu University), Varanasi** campus.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Outputs](#outputs)
3. [Classification System](#classification-system)
4. [EV Suitability Scoring](#ev-suitability-scoring)
5. [Installation](#installation)
6. [Usage](#usage)
7. [How to Get an OSM File](#how-to-get-an-osm-file)
8. [Project Structure](#project-structure)
9. [Known Limitations & Data Notes](#known-limitations--data-notes)
10. [Extending the Tool](#extending-the-tool)

---

## What It Does

```
.osm file  ──▶  Parse nodes & ways  ──▶  Classify each feature
                                              │
               ┌──────────────────────────────┼──────────────────────┐
               ▼                              ▼                      ▼
      2D Multispectral Map        EV Suitability Heatmap     3D Extruded Map
      (roads + buildings +        (scored + candidate         (buildings by
       land use, all colours)      zone markers)               type + roads)
               │                              │                      │
               └──────────────────────────────┴──────────────────────┘
                                              │
                                   Legend Panel + JSON Report
```

---

## Outputs

### Phase 1 — Multispectral Classification (`ev_campus_analyzer.py`)

| File | Description |
|------|-------------|
| `01_multispectral_map.png` | 2D top-down map with every feature colour-coded by type |
| `02_ev_suitability_heatmap.png` | Thermal heatmap of EV suitability scores across the campus |
| `03_3d_map.png` | 3D perspective — buildings extruded by type, land layers, road network |
| `04_legend_stats.png` | Full colour legend + feature count bar chart |
| `05_ev_analysis_report.json` | Ranked list of top EV candidate zones with lat/lon and scores |

### Phase 2 — ML Prediction (`ev_ml_predictor.py`)

| File | Description |
|------|-------------|
| `06_ml_prediction_map.png` | Colour-coded predicted probability map from Random Forest |
| `07_ml_feature_importance.png` | Feature importance bar chart (Mean Decrease Impurity) |
| `08_ml_ev_report.json` | Ranked ML candidate locations with scores and tiers |
| `09_cv_auc_scores.png` | 5-fold cross-validation AUC scores |

---

## Classification System

### Roads — classified by hierarchy (proxy for width)

| Colour | Type | Approx. Width |
|--------|------|---------------|
| 🔴 Red-Orange | Primary Road | > 12 m |
| 🟠 Amber | Secondary Road | 8–12 m |
| 🟡 Yellow | Tertiary Road | 6–8 m |
| 🔵 Periwinkle | Residential Road | < 6 m |
| ⬜ Light Grey | Service Road / Alley | narrow |
| 🟢 Mint | Footway / Pedestrian | — |
| 🩵 Turquoise | Cycleway | — |
| 🟤 Tan | Path / Track (Unpaved) | — |
| 🟤 Dark Gold | Dirt Track | — |

> **Note:** Explicit `width=` tags are rare in OSM.  
> When present, the script uses them; otherwise width is inferred from road hierarchy.  
> If you need precise widths, they must be added to OSM or supplied via a supplementary dataset.

### Buildings — classified by function

| Colour | Category | Identified by |
|--------|----------|---------------|
| 🟣 Violet | Academic / Dept / Lab | `building=university/college`, dept names |
| 🔵 Dodger Blue | Hostel / Student Housing | `guest_house`, hostel name keywords |
| 🟦 Steel Blue | Residential (Staff/Faculty) | `building=house/residential`, colony names |
| 🔴 Red | Hospital / Medical | `amenity=hospital/clinic`, `healthcare`, medical name keywords |
| 🟠 Dark Orange | Temple / Place of Worship | `amenity=place_of_worship`, mandir/temple names |
| 🟡 Golden | Sports / Pool / Gymnasium | `leisure=swimming_pool/sports_centre`, sport tags |
| 🩵 Dark Teal | Library / Museum / Auditorium | `amenity=library`, `tourism=museum`, Bhavan names |
| 🟢 Teal | Admin / Office / Amenity | `amenity=community_centre/bank/parking`, etc. |
| 🩷 Pink | Canteen / Cafe / Restaurant | `amenity=restaurant/cafe/fast_food`, mess names |
| ⬛ Grey | Building (Unclassified) | `building=yes` with no further tags |
| ⬜ Light Grey | Gate / Entrance | `building=gate`, historic=city_gate |

### Land Use & Natural Features

| Colour | Category |
|--------|----------|
| 🌲 Dark Green | Forest / Woodland |
| 🌿 Grass Green | Grassland / Lawn / Scrub |
| 🍃 Pale Green | Garden / Park / Flowerbed |
| 🌾 Khaki | Farmland / Agricultural / Nursery |
| 🟤 Sandy Brown | Barren / Open / Greenfield Land |
| 🍑 Peach | Sports Ground / Recreation / Playground |
| 🔵 Sky Blue | Water Body / River / Pond |
| 🩵 Pale Cyan | Wetland / Marsh |
| 🌊 Dark Teal | Aquaculture |

### Campus Boundary
The outer `amenity=university` polygon (the entire BHU boundary) is rendered
as a **dashed blue outline only** — it is never filled. All features inside it
are classified individually by their own tags.

---

## EV Suitability Scoring

Each way is scored 0–1 based on weighted criteria:

| Criterion | Weight |
|-----------|--------|
| Primary / secondary road access | +0.40 |
| Parking area | +0.35 |
| University / college amenity | +0.30 |
| Hospital / library nearby | +0.25 |
| Conference / community centre | +0.22 |
| Education land use | +0.20 |
| Name keywords (gate, parking, main entry) | +0.20 |
| Restaurant / cafe (footfall) | +0.12 |
| Recreation ground | +0.12 |
| Water body / waterway | −0.20 to −0.30 |

**Priority tiers:**
- ★ Green `≥ 0.55` → HIGH priority  
- ★ Gold `0.40–0.54` → MEDIUM priority  
- ★ Red `< 0.40` → LOWER priority (still a candidate)

---

## Installation

### Requirements
- Python **3.8+**
- pip packages listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

Key dependencies:
- `numpy`, `matplotlib` — core mapping and visualisation
- `scikit-learn`, `pandas` — ML pipeline and data handling
- `lxml` — faster XML parsing for large OSM files (optional but recommended)

No GPU, no GIS software, no geospatial databases needed.

### Verify
```bash
python -c "import matplotlib, numpy, sklearn, pandas; print('Ready')"
```

---

## Usage

### Unified Pipeline (recommended)
```bash
python main.py --osm map.osm --csv ev-charging-stations-india.csv --output ./output
```

This runs both phases in sequence and produces all 9 output files.

### Run phases individually
```bash
# Phase 1 only — multispectral classification
python main.py --osm map.osm --skip-ml

# Phase 2 only — ML prediction
python main.py --osm map.osm --csv ev-charging-stations-india.csv --skip-multispectral

# Or run scripts directly:
python ev_campus_analyzer.py --osm map.osm --output ./output
python ev_ml_predictor.py --osm map.osm --csv ev-charging-stations-india.csv --output ./output
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--osm` | `map.osm` | Path to input `.osm` file |
| `--csv` | `ev-charging-stations-india.csv` | Path to EV stations CSV (for ML) |
| `--output` | `./output` | Directory for all output files (created if missing) |
| `--skip-ml` | — | Skip Phase 2 (ML prediction) |
| `--skip-multispectral` | — | Skip Phase 1 (classification) |

### Utility: Inject EV stations into OSM
```bash
python identity.py --osm map.osm --csv ev-charging-stations-india.csv --output map_with_ev.osm
```

### Expected runtime
| Campus size | Approximate time |
|-------------|-----------------|
| Small (< 1k ways) | 10–20 seconds |
| Medium (1k–5k ways, e.g. BHU) | 30–90 seconds |
| Large (5k–20k ways) | 2–5 minutes |

---

## How to Get an OSM File

### Option 1 — OpenStreetMap Export (small areas)
1. Go to [openstreetmap.org](https://www.openstreetmap.org)
2. Navigate to your campus
3. Click **Export** → **Manually select a different area**
4. Draw a bounding box around the campus
5. Click **Export** → downloads `map.osm`

> Limit: OSM direct export is capped at ~50,000 nodes.  
> For larger areas use Option 2.

### Option 2 — Overpass API (recommended for campuses)
```
https://overpass-api.de/api/map?bbox=MIN_LON,MIN_LAT,MAX_LON,MAX_LAT
```

Example for BHU:
```
https://overpass-api.de/api/map?bbox=82.9667,25.2582,83.0176,25.2831
```

Save the response as `campus.osm`.

### Option 3 — JOSM / osmium (large downloads)
Use [Geofabrik](https://download.geofabrik.de/) to download a regional extract,
then clip to your bounding box with `osmium`:
```bash
osmium extract --bbox=MIN_LON,MIN_LAT,MAX_LON,MAX_LAT region.osm -o campus.osm
```

---

## Project Structure

```
Project Explo/
├── main.py                           # Unified pipeline runner
├── ev_campus_analyzer.py             # Phase 1: Multispectral OSM classification
├── ev_ml_predictor.py                # Phase 2: ML-based EV station prediction
├── identity.py                       # Utility: Inject EV stations into OSM
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── map.osm                           # BHU campus OSM data
├── ev-charging-stations-india.csv    # India EV station dataset
└── output/
    ├── 01_multispectral_map.png      # 2D classification map
    ├── 02_ev_suitability_heatmap.png # EV suitability heatmap
    ├── 03_3d_map.png                 # 3D extruded map
    ├── 04_legend_stats.png           # Legend & statistics
    ├── 05_ev_analysis_report.json    # Heuristic EV zone report
    ├── 06_ml_prediction_map.png      # ML prediction map
    ├── 07_ml_feature_importance.png  # Feature importance chart
    ├── 08_ml_ev_report.json          # ML candidate locations
    └── 09_cv_auc_scores.png          # Cross-validation AUC
```

---

## Known Limitations & Data Notes

### OSM Data Quality Issues (specific to BHU)

| Issue | Impact | Workaround |
|-------|--------|------------|
| Many buildings tagged `building=yes` only | Classified as "Unclassified" (grey) | Add `amenity`, `name`, or specific `building=` tags in OSM |
| `width=` tag present on only ~15 ways | Road width inferred from hierarchy | Add explicit width tags or use a survey dataset |
| Large `amenity=university` polygon covers entire campus | Would colour entire campus — **fixed** by rendering as outline only | Already handled |
| Some hostels tagged as `landuse=education` | May show as land patch instead of building | Add `building=dormitory` + `guest_house=hostel` tags |
| Agricultural fields partially tagged `landuse=grass` | Shown as grassland instead of farmland | Add `landuse=farmland` or `landuse=orchard` |
| Hospital buildings sometimes lack `amenity=hospital` | Classified as generic building | Add proper `amenity=hospital` + `healthcare=hospital` tags |
| Swimming pool in BHU lacks `leisure=swimming_pool` on building way | Classified by name keyword match | Works but fragile; add proper tags |

### General Limitations

- **No satellite imagery** — classification is purely from OSM vector tags.
  Spectral analysis in the title refers to the colour-band analogy, not
  actual multispectral remote sensing.
- **EV scores are heuristic** — they do not account for electrical grid
  proximity, transformer capacity, traffic counts, or parking regulations.
- **3D heights** are approximate — OSM `height=` tags are sparse; most
  buildings use category defaults.
- **No relation support** — OSM multipolygon relations (e.g., buildings with
  holes, complex land parcels) are not parsed. Only simple way polygons.

---

## Extending the Tool

### Add a new building category
In `classify_way()`, add a condition before the generic building fallback:

```python
# Example: identify research labs by name
if _name_match(name, {"research","lab","institute","iit"}):
    return ("bld_research", "#FF6B35", 0.8, 7, "Research Institute / Lab")
```

Then add the colour to the `C` dict and the height to `BLD_HEIGHTS`.

### Change EV scoring weights
Edit `ev_score()` — each criterion is a simple additive term.
You can add new signals such as:
- Distance to substation (`man_made=power_substation`)
- Proximity to parking (`amenity=parking`)
- Traffic volume proxy (road lane count)

### Add an interactive HTML map
Install `folium` and replace the matplotlib outputs with an interactive
Leaflet map:
```bash
pip install folium
```

### Use actual road widths
If you have a supplementary CSV of road widths, merge by way ID:
```python
width_df = pd.read_csv("road_widths.csv")  # columns: way_id, width_m
width_map = dict(zip(width_df.way_id, width_df.width_m))
# then in classify_way: w = width_map.get(way_id, default_for_highway_type)
```

---

## Colour Reference Card

```
ROADS          BUILDINGS          LAND USE
─────────────  ─────────────────  ──────────────────
█ Primary      █ Academic/Lab     █ Forest
█ Secondary    █ Hostel           █ Grass/Lawn
█ Tertiary     █ Staff Housing    █ Garden/Park
█ Residential  █ Hospital         █ Farmland
█ Service      █ Temple           █ Barren/Open
█ Footway      █ Sports/Pool      █ Recreation
█ Cycleway     █ Library/Museum   █ Water
█ Path/Track   █ Admin/Office     █ Wetland
               █ Canteen/Cafe     █ Aquaculture
               █ Unclassified

EV ZONES
──────────────
★ Green  ≥ 0.55  HIGH
★ Gold   0.40+   MEDIUM
★ Red    < 0.40  LOWER
```

---

## ML Model Details

The ML pipeline in `ev_ml_predictor.py` works as follows:

1. **Grid cells** — the OSM bounding box is divided into ~111m grid cells
2. **Feature extraction** — for each cell, 23 spatial features are computed:
   - Road access by type (primary/secondary/tertiary/residential/service)
   - Nearby amenities (parking, university, hospital, restaurants, banks, etc.)
   - Land use (education, commercial, residential, recreation)
   - Building density, node density, existing EV station proximity
3. **Training labels** — grid cells are labelled positive if a known EV station
   from the CSV falls within ~220m. If no stations exist in the map area,
   synthetic labels are generated from the top 15% of heuristic-scored cells
4. **Random Forest** — 300 trees, max depth 8, balanced class weights
5. **Cross-validation** — 5-fold stratified CV with ROC-AUC scoring
6. **Output** — probability map, feature importance chart, ranked candidates

---

*Developed as prototype for EV charging station suitability analysis.*  
*Data source: OpenStreetMap contributors, © ODbL.*  
*EV station data: ev-charging-stations-india.csv*
