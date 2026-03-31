# -*- coding: utf-8 -*-
"""
ev_ml_predictor.py
==================
ML-powered EV Charging Station Placement Predictor

Trains a Random Forest on REAL EV station locations across Indian cities
by fetching surrounding OSM features via the free Overpass API, then
predicts suitability on any user-supplied OSM map.

Two modes:
  --train   : Fetch features around real EV stations, train model, save to disk
  (default) : Load saved model and predict on a user's .osm file

Pipeline:
  TRAINING (one-time, needs internet):
    1. Sample EV stations from CSV (multiple cities)
    2. Query Overpass API for OSM features around each station (+negatives)
    3. Extract 23 spatial features per sample
    4. Train Random Forest → save ev_model.joblib + training_data.csv

  PREDICTION (fast, offline):
    1. Load ev_model.joblib
    2. Parse user's .osm → build grid → extract features
    3. Predict → output maps + reports

Usage:
    python ev_ml_predictor.py --train --csv ev-charging-stations-india.csv
    python ev_ml_predictor.py --osm map.osm --output ./output
    python ev_ml_predictor.py --train --csv ev-charging-stations-india.csv --osm map.osm --output ./output
"""

import argparse
import json
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report
    import pandas as pd
    import joblib
except ImportError:
    sys.exit("ERROR: scikit-learn, pandas, and joblib are required.\n"
             "  pip install scikit-learn pandas joblib")

try:
    import requests
except ImportError:
    sys.exit("ERROR: requests is required.\n  pip install requests")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
QUERY_RADIUS_M = 500          # metres around each station for Overpass query
NEGATIVE_OFFSET_DEG = 0.015   # ~1.5 km offset for generating negative samples
NEGATIVES_PER_POSITIVE = 2    # number of negative samples per positive
MAX_STATIONS_SAMPLE = 150     # max stations to sample for training
QUERY_DELAY_SEC = 2.0         # seconds between Overpass API calls (be polite)
GRID_RESOLUTION = 0.001       # ~111 m per grid cell for prediction
SEARCH_RADIUS_DEG = 0.003     # neighbour search radius for prediction features

MODEL_FILE = "ev_model.joblib"
TRAINING_DATA_FILE = "training_data.csv"

FEATURE_NAMES = [
    "road_primary", "road_secondary", "road_tertiary",
    "road_residential", "road_service", "road_any", "road_count_norm",
    "parking_nearby",
    "amenity_university", "amenity_hospital", "amenity_restaurant_cafe",
    "amenity_bank", "amenity_school", "amenity_conference",
    "amenity_any",
    "landuse_education", "landuse_commercial", "landuse_residential",
    "landuse_recreation",
    "building_count_norm", "node_density",
    "has_existing_ev", "poi_density",
]

# ─────────────────────────────────────────────────────────────────────────────
# OVERPASS API HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def overpass_query(lat, lon, radius_m=QUERY_RADIUS_M, retries=3):
    """Query Overpass API for all ways and nodes around (lat, lon)."""
    query = f"""
    [out:json][timeout:30];
    (
      way(around:{radius_m},{lat},{lon})["highway"];
      way(around:{radius_m},{lat},{lon})["building"];
      way(around:{radius_m},{lat},{lon})["amenity"];
      way(around:{radius_m},{lat},{lon})["landuse"];
      way(around:{radius_m},{lat},{lon})["leisure"];
      node(around:{radius_m},{lat},{lon})["amenity"];
      node(around:{radius_m},{lat},{lon})["shop"];
    );
    out tags;
    """
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=45)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                print(f"      Rate limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            if resp.status_code == 504:  # gateway timeout
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"      WARNING: Overpass query failed for ({lat:.4f}, {lon:.4f}): {e}")
                return []
    return []


def extract_features_from_overpass(elements):
    """Extract the 23-feature vector from Overpass API response elements."""
    road_types = {"primary": 0, "secondary": 0, "tertiary": 0,
                  "residential": 0, "service": 0}
    road_count = 0
    parking = 0
    amenity_counts = defaultdict(int)
    landuse_types = set()
    building_count = 0
    node_count = len(elements)
    has_ev = 0
    poi_count = 0

    for el in elements:
        tags = el.get("tags", {})
        hw = tags.get("highway", "")
        amen = tags.get("amenity", "")
        lu = tags.get("landuse", "")
        bld = tags.get("building", "")

        # Roads
        if hw:
            road_count += 1
            for rt in road_types:
                if hw == rt or hw == f"{rt}_link":
                    road_types[rt] = 1

        # Parking
        if amen == "parking" or tags.get("parking"):
            parking = 1

        # Amenities
        if amen:
            amenity_counts[amen] += 1
            poi_count += 1

        # Land use
        if lu:
            landuse_types.add(lu)

        # Buildings
        if bld:
            building_count += 1

        # Existing EV
        if amen == "charging_station" or tags.get("ev:charging") or \
           "charging" in tags.get("name", "").lower():
            has_ev = 1

        # Shops count as POIs
        if tags.get("shop"):
            poi_count += 1

    def has_amenity(value_set):
        return int(any(a in value_set for a in amenity_counts))

    def has_landuse(value_set):
        return int(bool(landuse_types & value_set))

    vec = [
        road_types["primary"],
        road_types["secondary"],
        road_types["tertiary"],
        road_types["residential"],
        road_types["service"],
        int(road_count > 0),
        min(road_count / 10.0, 1.0),
        parking,
        has_amenity({"university", "college"}),
        has_amenity({"hospital", "clinic", "healthcare", "doctors"}),
        has_amenity({"restaurant", "cafe", "fast_food", "food_court"}),
        has_amenity({"bank", "atm"}),
        has_amenity({"school", "kindergarten"}),
        has_amenity({"conference_centre", "community_centre", "events_venue"}),
        int(len(amenity_counts) > 0),
        has_landuse({"education", "university", "school"}),
        has_landuse({"commercial", "retail", "industrial"}),
        has_landuse({"residential"}),
        has_landuse({"recreation_ground", "park", "leisure"}),
        min(building_count / 20.0, 1.0),
        min(node_count / 100.0, 1.0),
        has_ev,
        min(poi_count / 15.0, 1.0),
    ]
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def sample_stations(ev_df, max_n=MAX_STATIONS_SAMPLE):
    """Sample stations from diverse cities for balanced training."""
    # Normalise city names
    ev_df = ev_df.copy()
    ev_df["city_clean"] = ev_df["city"].str.strip().str.lower()

    # Get top cities by count
    city_counts = ev_df["city_clean"].value_counts()
    top_cities = city_counts.head(20).index.tolist()

    sampled = []
    per_city = max(max_n // len(top_cities), 3)

    for city in top_cities:
        city_df = ev_df[ev_df["city_clean"] == city]
        n = min(per_city, len(city_df))
        sampled.append(city_df.sample(n=n, random_state=42))

    result = pd.concat(sampled).drop_duplicates(subset=["latitude", "longitude"])

    # If under budget, add more from remaining cities
    if len(result) < max_n:
        remaining = ev_df[~ev_df.index.isin(result.index)]
        extra = min(max_n - len(result), len(remaining))
        if extra > 0:
            result = pd.concat([result, remaining.sample(n=extra, random_state=42)])

    return result.head(max_n).reset_index(drop=True)


def generate_negative_point(lat, lon, ev_lats, ev_lons):
    """Generate a negative sample point offset from (lat, lon) that's far from any EV station."""
    for _ in range(10):
        angle = random.uniform(0, 2 * math.pi)
        offset = random.uniform(NEGATIVE_OFFSET_DEG * 0.7, NEGATIVE_OFFSET_DEG * 1.5)
        nlat = lat + offset * math.sin(angle)
        nlon = lon + offset * math.cos(angle)

        # Ensure it's not too close to any known EV station
        min_dist = min(
            max(abs(nlat - elat), abs(nlon - elon))
            for elat, elon in zip(ev_lats, ev_lons)
        ) if len(ev_lats) > 0 else 999

        if min_dist > 0.005:  # at least ~500m from any known station
            return nlat, nlon

    # Fallback
    return lat + NEGATIVE_OFFSET_DEG, lon + NEGATIVE_OFFSET_DEG


def build_training_data(ev_df, output_dir):
    """Build training dataset by querying Overpass API around real EV stations."""
    cache_path = os.path.join(output_dir, TRAINING_DATA_FILE)

    # Check for cached data
    if os.path.exists(cache_path):
        print(f"   Found cached training data: {cache_path}")
        cached_df = pd.read_csv(cache_path)
        X = cached_df[FEATURE_NAMES].values.astype(np.float32)
        y = cached_df["label"].values.astype(np.int32)
        print(f"   Loaded {len(X)} samples ({y.sum()} positive, {(y==0).sum()} negative)")
        return X, y

    print("[2/5] Building training data from real EV station locations ...")

    # Normalise lat column
    ev_df = ev_df.copy()
    ev_df.columns = [c.strip().lower() for c in ev_df.columns]
    if "lattitude" in ev_df.columns:
        ev_df.rename(columns={"lattitude": "latitude"}, inplace=True)
    ev_df = ev_df.dropna(subset=["latitude", "longitude"])
    ev_df["latitude"] = pd.to_numeric(ev_df["latitude"], errors="coerce")
    ev_df["longitude"] = pd.to_numeric(ev_df["longitude"], errors="coerce")
    ev_df = ev_df.dropna(subset=["latitude", "longitude"])

    all_ev_lats = ev_df["latitude"].tolist()
    all_ev_lons = ev_df["longitude"].tolist()

    # Sample stations
    sampled = sample_stations(ev_df)
    print(f"   Sampled {len(sampled)} stations from {sampled['city_clean'].nunique()} cities")

    features = []
    labels = []
    total = len(sampled)

    for idx, row in sampled.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        progress = len(features) // (1 + NEGATIVES_PER_POSITIVE) + 1
        print(f"   [{progress}/{total}] Querying ({lat:.4f}, {lon:.4f}) "
              f"- {row.get('city', 'unknown')} ...", end="")

        # ── Positive: features around real EV station ──
        elements = overpass_query(lat, lon)
        if not elements:
            print(" skipped (no data)")
            continue
        fv = extract_features_from_overpass(elements)
        features.append(fv)
        labels.append(1)
        print(f" [OK] {len(elements)} elements")

        # ── Negatives: random offsets with no EV station ──
        for _ in range(NEGATIVES_PER_POSITIVE):
            nlat, nlon = generate_negative_point(lat, lon, all_ev_lats, all_ev_lons)
            neg_elements = overpass_query(nlat, nlon)
            neg_fv = extract_features_from_overpass(neg_elements)
            features.append(neg_fv)
            labels.append(0)

        # Rate limiting: be polite to Overpass API
        time.sleep(QUERY_DELAY_SEC)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Cache to disk
    df_out = pd.DataFrame(X, columns=FEATURE_NAMES)
    df_out["label"] = y
    os.makedirs(output_dir, exist_ok=True)
    df_out.to_csv(cache_path, index=False)
    print(f"   Cached training data to: {cache_path}")
    print(f"   Total samples: {len(X)} ({y.sum()} positive, {(y==0).sum()} negative)")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X, y, output_dir):
    """Train Random Forest and save to disk."""
    print("[3/5] Training Random Forest model ...")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"   CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    train_pred = clf.predict(X)
    print("   Training classification report:")
    print(classification_report(y, train_pred, target_names=["Non-EV", "EV Site"],
                                zero_division=0))

    # Save model
    model_path = os.path.join(output_dir, MODEL_FILE)
    joblib.dump(clf, model_path)
    print(f"   Model saved to: {model_path}")

    return clf, cv_scores


def load_model(output_dir):
    """Load a previously saved model."""
    model_path = os.path.join(output_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        return None
    print(f"   Loading saved model: {model_path}")
    return joblib.load(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# OSM PARSER (for prediction on user's map)
# ─────────────────────────────────────────────────────────────────────────────

def parse_osm(osm_path):
    """Parse OSM file and return nodes dict and ways list."""
    print(f"[1/5] Parsing OSM: {osm_path}")
    try:
        tree = ET.parse(osm_path)
    except ET.ParseError as e:
        sys.exit(f"XML parse error: {e}")

    root = tree.getroot()

    nodes = {}
    for n in root.findall("node"):
        nid = n.attrib["id"]
        try:
            lat = float(n.attrib["lat"])
            lon = float(n.attrib["lon"])
        except (KeyError, ValueError):
            continue
        tags = {t.attrib["k"]: t.attrib["v"] for t in n.findall("tag")}
        nodes[nid] = (lat, lon, tags)

    ways = []
    for w in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in w.findall("tag")}
        nd_refs = [nd.attrib["ref"] for nd in w.findall("nd")]
        coords = [(nodes[r][0], nodes[r][1]) for r in nd_refs if r in nodes]
        if not coords:
            continue
        clat = sum(c[0] for c in coords) / len(coords)
        clon = sum(c[1] for c in coords) / len(coords)
        ways.append({
            "id": w.attrib["id"],
            "lat": clat, "lon": clon,
            "tags": tags,
            "node_count": len(coords),
        })

    bb = root.find("bounds")
    if bb is not None:
        bbox = (float(bb.attrib["minlat"]), float(bb.attrib["minlon"]),
                float(bb.attrib["maxlat"]), float(bb.attrib["maxlon"]))
    else:
        lats = [v[0] for v in nodes.values()]
        lons = [v[1] for v in nodes.values()]
        bbox = (min(lats), min(lons), max(lats), max(lons))

    print(f"   Nodes: {len(nodes):,}   Ways: {len(ways):,}")
    print(f"   Bounding box: lat {bbox[0]:.4f}–{bbox[2]:.4f}, "
          f"lon {bbox[1]:.4f}–{bbox[3]:.4f}")
    return nodes, ways, bbox


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL INDEX (for prediction)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialIndex:
    def __init__(self, items, bucket_size=0.01):
        self.bucket_size = bucket_size
        self.buckets = defaultdict(list)
        for item in items:
            bk = (int(item["lat"] / bucket_size), int(item["lon"] / bucket_size))
            self.buckets[bk].append(item)

    def nearby(self, lat, lon, radius):
        results = []
        steps = int(radius / self.bucket_size) + 1
        bk = (int(lat / self.bucket_size), int(lon / self.bucket_size))
        for di in range(-steps, steps + 1):
            for dj in range(-steps, steps + 1):
                results.extend(self.buckets.get((bk[0]+di, bk[1]+dj), []))
        return [r for r in results
                if abs(r["lat"] - lat) <= radius and abs(r["lon"] - lon) <= radius]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION FROM LOCAL OSM (for prediction)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector_local(clat, clon, way_index, node_index,
                                radius=SEARCH_RADIUS_DEG):
    """Extract the same 23 features from a parsed OSM file (local, no API)."""
    nearby_ways = way_index.nearby(clat, clon, radius)
    nearby_nodes = node_index.nearby(clat, clon, radius)

    road_types = {"primary": 0, "secondary": 0, "tertiary": 0,
                  "residential": 0, "service": 0}
    road_count = 0
    parking = 0
    amenity_counts = defaultdict(int)
    landuse_types = set()
    building_count = 0
    has_ev = 0
    poi_count = 0

    for w in nearby_ways:
        tags = w["tags"]
        hw = tags.get("highway", "")
        amen = tags.get("amenity", "")
        lu = tags.get("landuse", "")
        bld = tags.get("building", "")

        if hw:
            road_count += 1
            for rt in road_types:
                if hw == rt or hw == f"{rt}_link":
                    road_types[rt] = 1
        if amen == "parking" or tags.get("parking"):
            parking = 1
        if amen:
            amenity_counts[amen] += 1
            poi_count += 1
        if lu:
            landuse_types.add(lu)
        if bld:
            building_count += 1
        if amen == "charging_station":
            has_ev = 1
        if tags.get("shop"):
            poi_count += 1

    # Also check node-level amenities
    for n in nearby_nodes:
        tags = n.get("tags", {})
        amen = tags.get("amenity", "")
        if amen:
            amenity_counts[amen] += 1
            poi_count += 1
        if amen == "charging_station":
            has_ev = 1
        if tags.get("shop"):
            poi_count += 1

    def has_amenity(value_set):
        return int(any(a in value_set for a in amenity_counts))

    def has_landuse(value_set):
        return int(bool(landuse_types & value_set))

    vec = [
        road_types["primary"],
        road_types["secondary"],
        road_types["tertiary"],
        road_types["residential"],
        road_types["service"],
        int(road_count > 0),
        min(road_count / 10.0, 1.0),
        parking,
        has_amenity({"university", "college"}),
        has_amenity({"hospital", "clinic", "healthcare", "doctors"}),
        has_amenity({"restaurant", "cafe", "fast_food", "food_court"}),
        has_amenity({"bank", "atm"}),
        has_amenity({"school", "kindergarten"}),
        has_amenity({"conference_centre", "community_centre", "events_venue"}),
        int(len(amenity_counts) > 0),
        has_landuse({"education", "university", "school"}),
        has_landuse({"commercial", "retail", "industrial"}),
        has_landuse({"residential"}),
        has_landuse({"recreation_ground", "park", "leisure"}),
        min(building_count / 20.0, 1.0),
        min(len(nearby_nodes) / 100.0, 1.0),
        has_ev,
        min(poi_count / 15.0, 1.0),
    ]
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ON MAP GRID
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(bbox, resolution=GRID_RESOLUTION):
    minlat, minlon, maxlat, maxlon = bbox
    lats = np.arange(minlat, maxlat, resolution)
    lons = np.arange(minlon, maxlon, resolution)
    cells = [(lat + resolution / 2, lon + resolution / 2)
             for lat in lats for lon in lons]
    return cells


def predict_map(clf, cells, way_index, node_index):
    """Predict EV suitability for each grid cell."""
    print("[4/5] Predicting EV suitability across map ...")
    X_pred = []
    for clat, clon in cells:
        fv = build_feature_vector_local(clat, clon, way_index, node_index)
        X_pred.append(fv)
    X_pred = np.array(X_pred, dtype=np.float32)
    probs = clf.predict_proba(X_pred)[:, 1]
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction_map(cells, probs, bbox, ways, output_dir, ev_df=None):
    """Render the probability map with road overlay and EV station markers."""
    print("[5/5] Generating output files ...")
    os.makedirs(output_dir, exist_ok=True)
    minlat, minlon, maxlat, maxlon = bbox

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Build 2D grid for imshow
    lats = np.array([c[0] for c in cells])
    lons = np.array([c[1] for c in cells])
    unique_lats = sorted(set(round(l, 6) for l in lats), reverse=True)
    unique_lons = sorted(set(round(l, 6) for l in lons))
    lat_idx = {v: i for i, v in enumerate(unique_lats)}
    lon_idx = {v: i for i, v in enumerate(unique_lons)}

    grid = np.zeros((len(unique_lats), len(unique_lons)))
    for (clat, clon), p in zip(cells, probs):
        ri = lat_idx.get(round(clat, 6))
        ci = lon_idx.get(round(clon, 6))
        if ri is not None and ci is not None:
            grid[ri, ci] = p

    im = ax.imshow(grid, extent=[minlon, maxlon, minlat, maxlat],
                   origin="lower", cmap="plasma", alpha=0.85,
                   aspect="auto", vmin=0, vmax=1)

    # Road overlay
    road_colors = {"primary": ("#FF4444", 1.8), "secondary": ("#FF8800", 1.4),
                   "tertiary": ("#FFDD00", 1.0), "residential": ("#8888FF", 0.6)}
    for w in ways:
        hw = w["tags"].get("highway", "")
        if hw in road_colors:
            col, lw = road_colors[hw]
            ax.plot(w["lon"], w["lat"], "o", color=col,
                    markersize=1.5, alpha=0.5, zorder=3)

    # Top predicted locations (deduped)
    sorted_idx = np.argsort(probs)[::-1]
    top_n = min(30, len(sorted_idx))
    top_cells = [(cells[i], probs[i]) for i in sorted_idx[:top_n]]
    dedup = []
    min_sep = GRID_RESOLUTION * 3
    for (clat, clon), p in top_cells:
        too_close = any(abs(clat - d[0][0]) < min_sep and abs(clon - d[0][1]) < min_sep
                        for d in dedup)
        if not too_close:
            dedup.append(((clat, clon), p))

    for rank, ((clat, clon), p) in enumerate(dedup[:15]):
        color = "#00FF88" if p >= 0.6 else "#FFD700" if p >= 0.40 else "#FF4466"
        ax.scatter(clon, clat, s=120, c=color, marker="*",
                   edgecolors="white", linewidths=0.5, zorder=6, alpha=0.95)
        ax.annotate(f"#{rank+1} {p:.2f}", xy=(clon, clat), xytext=(4, 4),
                    textcoords="offset points", fontsize=6.5, color=color,
                    fontweight="bold", zorder=7)

    # Existing EV stations in area
    if ev_df is not None and len(ev_df):
        lat_col = "latitude" if "latitude" in ev_df.columns else "lattitude"
        pad = 0.02
        local_ev = ev_df[
            (ev_df[lat_col].between(minlat - pad, maxlat + pad)) &
            (ev_df["longitude"].between(minlon - pad, maxlon + pad))
        ]
        if len(local_ev):
            ax.scatter(local_ev["longitude"], local_ev[lat_col],
                       s=80, c="cyan", marker="^", edgecolors="white",
                       linewidths=0.5, zorder=8, label="Known EV Station", alpha=0.9)

    # Styling
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("ML Suitability Score", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)
    ax.set_title("ML-Predicted EV Charging Station Suitability\n"
                 "(Trained on real EV station locations across India)",
                 color="white", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude", color="white")
    ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    legend_elems = [
        mpatches.Patch(facecolor="#00FF88", label="HIGH priority (≥0.60)"),
        mpatches.Patch(facecolor="#FFD700", label="MEDIUM priority (≥0.40)"),
        mpatches.Patch(facecolor="#FF4466", label="LOWER priority (<0.40)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="cyan",
                   markersize=8, label="Known EV Station", linewidth=0),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="white", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "06_ml_prediction_map.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Saved: {out_path}")
    return dedup


def save_feature_importance(clf, output_dir):
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(FEATURE_NAMES)))
    ax.barh([FEATURE_NAMES[i] for i in idx[::-1]],
            importances[idx[::-1]], color=colors, edgecolor="#333")
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", color="white")
    ax.set_title("Random Forest — Feature Importances\n"
                 "(Trained on real multi-city EV station data)",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "07_ml_feature_importance.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Saved: {out_path}")


def save_json_report(dedup, output_dir):
    candidates = []
    for rank, ((clat, clon), p) in enumerate(dedup):
        tier = "HIGH" if p >= 0.60 else "MEDIUM" if p >= 0.40 else "LOWER"
        candidates.append({
            "rank": rank + 1,
            "latitude": round(clat, 6),
            "longitude": round(clon, 6),
            "ml_score": round(float(p), 4),
            "priority_tier": tier,
        })
    report = {
        "model": "RandomForestClassifier (trained on real multi-city data)",
        "description": "ML-predicted EV charging station candidate locations",
        "training_source": "Overpass API features around real EV stations from India CSV",
        "top_candidates": candidates,
    }
    out_path = os.path.join(output_dir, "08_ml_ev_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   [OK] Saved: {out_path}")
    return candidates


def save_cv_scores(cv_scores, output_dir):
    """Plot cross-validation AUC scores."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars = ax.bar(folds, cv_scores,
                  color=plt.cm.plasma(np.linspace(0.3, 0.9, len(cv_scores))),
                  edgecolor="#333")
    ax.axhline(cv_scores.mean(), color="cyan", linestyle="--", linewidth=1.5,
               label=f"Mean AUC = {cv_scores.mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC Score", color="white")
    ax.set_title("5-Fold Cross-Validation AUC\n"
                 "(Multi-city training data)",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    for bar, s in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{s:.3f}", ha="center", va="bottom", color="white", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "09_cv_auc_scores.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ML-powered EV Charging Station Placement Predictor")
    parser.add_argument("--osm", default="map.osm",
                        help="Path to input .osm file for prediction (default: map.osm)")
    parser.add_argument("--csv", default="ev-charging-stations-india.csv",
                        help="Path to EV stations CSV for training (default: ev-charging-stations-india.csv)")
    parser.add_argument("--output", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--train", action="store_true",
                        help="Train model from CSV + Overpass API (requires internet)")
    args = parser.parse_args()

    print("=" * 64)
    print("  EV Charging Station ML Predictor")
    print("  Trained on real multi-city EV station data")
    print("=" * 64)

    os.makedirs(args.output, exist_ok=True)

    # ── TRAINING ──
    clf = None
    cv_scores = None

    if args.train:
        print(f"\n[1/5] Loading EV station CSV: {args.csv}")
        if not os.path.exists(args.csv):
            sys.exit(f"ERROR: CSV file not found: {args.csv}")
        ev_df = pd.read_csv(args.csv)

        X, y = build_training_data(ev_df, args.output)
        clf, cv_scores = train_model(X, y, args.output)

        # Save CV scores plot
        save_cv_scores(cv_scores, args.output)

        if not os.path.exists(args.osm):
            print(f"\n  Training complete! Model saved to {args.output}/{MODEL_FILE}")
            print(f"  To predict on a map, run:")
            print(f"    python ev_ml_predictor.py --osm <your_map.osm> --output {args.output}")
            return

    # ── PREDICTION ──
    if not os.path.exists(args.osm):
        sys.exit(f"ERROR: OSM file not found: {args.osm}\n"
                 f"  To train a model first: python ev_ml_predictor.py --train --csv {args.csv}")

    if clf is None:
        clf = load_model(args.output)
        if clf is None:
            sys.exit(f"ERROR: No trained model found at {args.output}/{MODEL_FILE}\n"
                     f"  Train first: python ev_ml_predictor.py --train --csv {args.csv}")

    # Parse OSM
    nodes_dict, ways, bbox = parse_osm(args.osm)

    # Build spatial indices
    way_index = SpatialIndex(ways)
    node_items = [{"lat": v[0], "lon": v[1], "tags": v[2]}
                  for v in nodes_dict.values()]
    node_index = SpatialIndex(node_items)

    # Grid + predict
    cells = build_grid(bbox)
    print(f"   Grid cells: {len(cells):,}")
    probs = predict_map(clf, cells, way_index, node_index)

    # Load EV CSV for overlay (optional)
    ev_df = None
    if os.path.exists(args.csv):
        ev_df = pd.read_csv(args.csv)
        ev_df.columns = [c.strip().lower() for c in ev_df.columns]
        if "lattitude" in ev_df.columns:
            ev_df.rename(columns={"lattitude": "latitude"}, inplace=True)
        ev_df["latitude"] = pd.to_numeric(ev_df["latitude"], errors="coerce")
        ev_df["longitude"] = pd.to_numeric(ev_df["longitude"], errors="coerce")
        ev_df = ev_df.dropna(subset=["latitude", "longitude"])

    # Generate outputs
    dedup = save_prediction_map(cells, probs, bbox, ways, args.output, ev_df)
    save_feature_importance(clf, args.output)
    candidates = save_json_report(dedup, args.output)

    if cv_scores is not None:
        save_cv_scores(cv_scores, args.output)

    print(f"\n{'=' * 64}")
    print("  TOP 10 PREDICTED EV STATION LOCATIONS")
    print("=" * 64)
    for c in candidates[:10]:
        print(f"  #{c['rank']:2d}  [{c['priority_tier']:6s}]  "
              f"lat={c['latitude']:.5f}  lon={c['longitude']:.5f}  "
              f"score={c['ml_score']:.3f}")
    print(f"\n  All outputs saved to: {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
