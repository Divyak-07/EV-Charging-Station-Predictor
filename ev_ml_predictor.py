# -*- coding: utf-8 -*-
"""
ev_ml_predictor.py
==================
ML-powered EV Charging Station Placement Predictor

Pipeline:
  1. Parse OSM file  →  extract candidate grid cells with spatial features
  2. Load EV CSV     →  create positive training labels (known station locations)
  3. Engineer features per grid cell (road access, land-use, amenity density, etc.)
  4. Train Random Forest classifier
  5. Predict suitability on every cell of the user-supplied map
  6. Output:
       • 06_ml_prediction_map.png   — colour-coded predicted probability map
       • 07_ml_feature_importance.png — feature importance bar chart
       • 08_ml_ev_report.json       — ranked candidate locations with scores

Usage:
    python3 ev_ml_predictor.py
    python3 ev_ml_predictor.py --osm map.osm --csv ev-charging-stations-india.csv --output ./output
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.pipeline import Pipeline
    import pandas as pd
except ImportError:
    sys.exit("ERROR: scikit-learn and pandas are required.\n  pip install scikit-learn pandas")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GRID_RESOLUTION = 0.001          # ~111 m per degree → ~111 m grid cells
SEARCH_RADIUS_DEG = 0.003        # ~330 m neighbour search radius
POSITIVE_SNAP_RADIUS_DEG = 0.002 # snap EV CSV point → grid cell within this radius

# Feature names (must match order in build_feature_vector)
FEATURE_NAMES = [
    "road_primary",
    "road_secondary",
    "road_tertiary",
    "road_residential",
    "road_service",
    "road_any",
    "road_count",
    "parking_nearby",
    "amenity_university",
    "amenity_hospital",
    "amenity_restaurant_cafe",
    "amenity_bank",
    "amenity_school",
    "amenity_conference",
    "amenity_any",
    "landuse_education",
    "landuse_commercial",
    "landuse_residential",
    "landuse_recreation",
    "building_count",
    "node_density",
    "existing_ev_nearby",
    "dist_to_nearest_ev",    # in degrees (inverted later for ML: closer = better)
]

# ─────────────────────────────────────────────────────────────────────────────
# OSM PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_osm(osm_path):
    """Parse OSM file and return nodes dict and ways list."""
    print(f"[1/6] Parsing OSM: {osm_path}")
    try:
        tree = ET.parse(osm_path)
    except ET.ParseError as e:
        sys.exit(f"XML parse error: {e}")

    root = tree.getroot()

    # Collect all nodes: id -> (lat, lon, tags)
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

    # Collect ways with their centre point and tags
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
            "lat": clat,
            "lon": clon,
            "tags": tags,
            "node_count": len(coords),
        })

    # Bounding box
    bb = root.find("bounds")
    if bb is not None:
        bbox = (
            float(bb.attrib["minlat"]),
            float(bb.attrib["minlon"]),
            float(bb.attrib["maxlat"]),
            float(bb.attrib["maxlon"]),
        )
    else:
        lats = [v[0] for v in nodes.values()]
        lons = [v[1] for v in nodes.values()]
        bbox = (min(lats), min(lons), max(lats), max(lons))

    print(f"   Nodes: {len(nodes):,}   Ways: {len(ways):,}")
    print(f"   Bounding box: lat {bbox[0]:.4f}–{bbox[2]:.4f}, "
          f"lon {bbox[1]:.4f}–{bbox[3]:.4f}")
    return nodes, ways, bbox


# ─────────────────────────────────────────────────────────────────────────────
# EV CSV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_ev_stations(csv_path):
    """Load EV station CSV; handle typo 'lattitude'."""
    print(f"[2/6] Loading EV stations: {csv_path}")
    df = pd.read_csv(csv_path)
    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    if "lattitude" in df.columns:
        df.rename(columns={"lattitude": "latitude"}, inplace=True)
    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"   Loaded {len(df):,} valid EV stations across India")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GRID BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(bbox, resolution=GRID_RESOLUTION):
    """Generate (lat, lon) grid cell centres covering the bounding box."""
    minlat, minlon, maxlat, maxlon = bbox
    lats = np.arange(minlat, maxlat, resolution)
    lons = np.arange(minlon, maxlon, resolution)
    cells = [(lat + resolution / 2, lon + resolution / 2)
             for lat in lats for lon in lons]
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL INDEX (simple grid-based bucket)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialIndex:
    def __init__(self, items, bucket_size=0.01):
        self.bucket_size = bucket_size
        self.buckets = defaultdict(list)
        for item in items:
            bk = self._key(item["lat"], item["lon"])
            self.buckets[bk].append(item)

    def _key(self, lat, lon):
        return (int(lat / self.bucket_size), int(lon / self.bucket_size))

    def nearby(self, lat, lon, radius):
        results = []
        steps = int(radius / self.bucket_size) + 1
        bk = self._key(lat, lon)
        for di in range(-steps, steps + 1):
            for dj in range(-steps, steps + 1):
                results.extend(self.buckets.get((bk[0]+di, bk[1]+dj), []))
        return [
            r for r in results
            if abs(r["lat"] - lat) <= radius and abs(r["lon"] - lon) <= radius
        ]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(clat, clon, way_index, node_index, ev_lats, ev_lons,
                          radius=SEARCH_RADIUS_DEG):
    """Return a feature vector for a single grid cell (clat, clon)."""
    nearby_ways = way_index.nearby(clat, clon, radius)
    nearby_nodes = node_index.nearby(clat, clon, radius)

    # ── Road features ──────────────────────────────────────────────────
    road_types = {"primary": 0, "secondary": 0, "tertiary": 0,
                  "residential": 0, "service": 0}
    road_count = 0
    for w in nearby_ways:
        hw = w["tags"].get("highway", "")
        if hw:
            road_count += 1
            for rt in road_types:
                if hw == rt:
                    road_types[rt] = 1
    road_any = int(road_count > 0)

    # ── Parking ────────────────────────────────────────────────────────
    parking_nearby = int(any(
        w["tags"].get("amenity") == "parking" or
        w["tags"].get("parking") is not None
        for w in nearby_ways
    ))

    # ── Amenity features ───────────────────────────────────────────────
    def has_amenity(value_set):
        return int(any(w["tags"].get("amenity") in value_set for w in nearby_ways))

    amenity_university   = has_amenity({"university", "college"})
    amenity_hospital     = has_amenity({"hospital", "clinic", "healthcare"})
    amenity_restaurant   = has_amenity({"restaurant", "cafe", "fast_food", "food_court"})
    amenity_bank         = has_amenity({"bank", "atm"})
    amenity_school       = has_amenity({"school", "kindergarten"})
    amenity_conference   = has_amenity({"conference_centre", "community_centre", "events_venue"})
    amenity_any          = int(any(w["tags"].get("amenity") for w in nearby_ways))

    # ── Land use ───────────────────────────────────────────────────────
    def has_landuse(value_set):
        return int(any(w["tags"].get("landuse") in value_set for w in nearby_ways))

    landuse_education   = has_landuse({"education", "university", "school"})
    landuse_commercial  = has_landuse({"commercial", "retail", "industrial"})
    landuse_residential = has_landuse({"residential"})
    landuse_recreation  = has_landuse({"recreation_ground", "park", "leisure"})

    # ── Building density ──────────────────────────────────────────────
    building_count = sum(1 for w in nearby_ways if w["tags"].get("building"))

    # ── Node density (proxy for urban density) ─────────────────────────
    node_density = min(len(nearby_nodes) / 100.0, 1.0)

    # ── Existing EV proximity ──────────────────────────────────────────
    dists = [
        max(abs(elat - clat), abs(elon - clon))
        for elat, elon in zip(ev_lats, ev_lons)
    ]
    dist_nearest = min(dists) if dists else 999.0
    existing_ev_nearby = int(dist_nearest <= SEARCH_RADIUS_DEG * 2)

    vec = [
        road_types["primary"],
        road_types["secondary"],
        road_types["tertiary"],
        road_types["residential"],
        road_types["service"],
        road_any,
        min(road_count / 10.0, 1.0),   # normalised
        parking_nearby,
        amenity_university,
        amenity_hospital,
        amenity_restaurant,
        amenity_bank,
        amenity_school,
        amenity_conference,
        amenity_any,
        landuse_education,
        landuse_commercial,
        landuse_residential,
        landuse_recreation,
        min(building_count / 20.0, 1.0),
        node_density,
        existing_ev_nearby,
        1.0 / (1.0 + dist_nearest * 100),  # closer = higher value
    ]
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_training_data(ways, nodes_dict, ev_df, bbox):
    """
    Build training dataset from the EV CSV.
    Positive samples = grid cells that have a known EV station nearby.
    Negative samples = random cells in the same map with low-suitability features.
    For generalisation we also include all India EV station coords as positive
    anchors mapped onto the local feature space.
    """
    print("[3/6] Building training features …")
    minlat, minlon, maxlat, maxlon = bbox

    # Build spatial indices
    way_index = SpatialIndex(ways)
    node_items = [{"lat": v[0], "lon": v[1], "tags": v[2]}
                  for v in nodes_dict.values()]
    node_index = SpatialIndex(node_items)

    # All India EV coords for proximity feature
    all_ev_lats = ev_df["latitude"].tolist()
    all_ev_lons = ev_df["longitude"].tolist()

    # Filter EV stations to those inside (or near) our map bbox
    pad = 0.05
    local_ev = ev_df[
        (ev_df["latitude"].between(minlat - pad, maxlat + pad)) &
        (ev_df["longitude"].between(minlon - pad, maxlon + pad))
    ].copy()
    print(f"   EV stations in/near map area: {len(local_ev)}")

    # Grid over the map area
    cells = build_grid(bbox)
    print(f"   Grid cells: {len(cells):,}")

    # Label each cell
    labels = []
    features = []
    ev_coords = list(zip(local_ev["latitude"], local_ev["longitude"]))

    for clat, clon in cells:
        # Positive if any known EV station is within snap radius
        is_positive = any(
            abs(elat - clat) <= POSITIVE_SNAP_RADIUS_DEG and
            abs(elon - clon) <= POSITIVE_SNAP_RADIUS_DEG
            for elat, elon in ev_coords
        )
        fv = build_feature_vector(
            clat, clon, way_index, node_index,
            all_ev_lats, all_ev_lons
        )
        features.append(fv)
        labels.append(1 if is_positive else 0)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    pos = y.sum()
    neg = (y == 0).sum()
    print(f"   Positive cells: {pos}   Negative cells: {neg}")

    # If no local positives, generate synthetic ones from India-wide patterns
    if pos == 0:
        print("   ⚠  No local EV stations in map area — using heuristic scoring for labels")
        # Assign positives to high-scoring grid cells via heuristic
        scores = []
        for fv in features:
            s = (fv[0]*0.4 + fv[1]*0.3 + fv[2]*0.2 +  # road hierarchy
                 fv[7]*0.35 +                             # parking
                 fv[8]*0.3 + fv[9]*0.25 +                # university, hospital
                 fv[14]*0.2 + fv[19]*0.1 +               # amenity, buildings
                 fv[21]*0.15)                             # node density
            scores.append(s)
        threshold = np.percentile(scores, 85)
        y = np.array([1 if s >= threshold else 0 for s in scores], dtype=np.int32)
        pos = y.sum()
        print(f"   Synthetic positives (top 15% by heuristic): {pos}")

    return X, y, cells, way_index, node_index, all_ev_lats, all_ev_lons


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X, y):
    print("[4/6] Training Random Forest model …")
    # Class weight to handle imbalance
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
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
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ON MAP GRID
# ─────────────────────────────────────────────────────────────────────────────

def predict_map(clf, cells, way_index, node_index, all_ev_lats, all_ev_lons):
    print("[5/6] Predicting EV suitability across map …")
    X_pred = []
    for clat, clon in cells:
        fv = build_feature_vector(
            clat, clon, way_index, node_index,
            all_ev_lats, all_ev_lons
        )
        X_pred.append(fv)
    X_pred = np.array(X_pred, dtype=np.float32)
    probs = clf.predict_proba(X_pred)[:, 1]  # probability of class 1 (EV site)
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction_map(cells, probs, bbox, ways, output_dir, ev_df):
    """Render the probability map with road overlay and EV station markers."""
    print("[6/6] Generating output files …")
    os.makedirs(output_dir, exist_ok=True)
    minlat, minlon, maxlat, maxlon = bbox

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # ── Draw prediction heatmap ────────────────────────────────────────
    lats = np.array([c[0] for c in cells])
    lons = np.array([c[1] for c in cells])

    # Build 2D grid for imshow
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

    im = ax.imshow(
        grid,
        extent=[minlon, maxlon, minlat, maxlat],
        origin="lower",
        cmap="plasma",
        alpha=0.85,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    # ── Draw roads overlay ─────────────────────────────────────────────
    road_colors = {
        "primary":     ("#FF4444", 1.8),
        "secondary":   ("#FF8800", 1.4),
        "tertiary":    ("#FFDD00", 1.0),
        "residential": ("#8888FF", 0.6),
        "service":     ("#666688", 0.4),
    }
    for w in ways:
        hw = w["tags"].get("highway", "")
        if hw in road_colors:
            col, lw = road_colors[hw]
            ax.plot(w["lon"], w["lat"], "o", color=col,
                    markersize=1.5, alpha=0.5, zorder=3)

    # ── Top predicted locations ────────────────────────────────────────
    sorted_idx = np.argsort(probs)[::-1]
    top_n = min(30, len(sorted_idx))
    top_cells = [(cells[i], probs[i]) for i in sorted_idx[:top_n]]

    # Deduplicate by min distance
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
        ax.annotate(f"#{rank+1} {p:.2f}",
                    xy=(clon, clat), xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.5, color=color,
                    fontweight="bold", zorder=7)

    # ── Existing EV stations in area ──────────────────────────────────
    pad = 0.02
    local_ev = ev_df[
        (ev_df["latitude"].between(minlat - pad, maxlat + pad)) &
        (ev_df["longitude"].between(minlon - pad, maxlon + pad))
    ]
    if len(local_ev):
        ax.scatter(local_ev["longitude"], local_ev["latitude"],
                   s=80, c="cyan", marker="^", edgecolors="white",
                   linewidths=0.5, zorder=8, label="Known EV Station", alpha=0.9)

    # ── Styling ────────────────────────────────────────────────────────
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("ML Suitability Score", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)
    ax.set_title("ML-Predicted EV Charging Station Suitability", color="white",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude", color="white")
    ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Legend
    legend_elems = [
        mpatches.Patch(facecolor="#00FF88", label="HIGH priority (≥0.60)"),
        mpatches.Patch(facecolor="#FFD700", label="MEDIUM priority (≥0.40)"),
        mpatches.Patch(facecolor="#FF4466", label="LOWER priority (<0.40)"),
        plt.Line2D([0],[0], marker="^", color="w", markerfacecolor="cyan",
                   markersize=8, label="Known EV Station", linewidth=0),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="white", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "06_ml_prediction_map.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: {out_path}")
    return dedup


def save_feature_importance(clf, output_dir):
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(FEATURE_NAMES)))
    bars = ax.barh(
        [FEATURE_NAMES[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color=colors,
        edgecolor="#333",
    )
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", color="white")
    ax.set_title("Random Forest — Feature Importances", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "07_ml_feature_importance.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: {out_path}")


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
        "model": "RandomForestClassifier",
        "description": "ML-predicted EV charging station candidate locations",
        "top_candidates": candidates,
    }
    out_path = os.path.join(output_dir, "08_ml_ev_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   ✓ Saved: {out_path}")
    return candidates


def save_confusion_summary(clf, X, y, output_dir):
    """Quick cross-val score plot."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    bars = ax.bar(folds, scores, color=plt.cm.plasma(np.linspace(0.3, 0.9, len(scores))),
                  edgecolor="#333")
    ax.axhline(scores.mean(), color="cyan", linestyle="--", linewidth=1.5,
               label=f"Mean AUC = {scores.mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC Score", color="white")
    ax.set_title("5-Fold Cross-Validation AUC", color="white",
                 fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{s:.3f}", ha="center", va="bottom", color="white", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "09_cv_auc_scores.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ML-powered EV Charging Station Placement Predictor")
    parser.add_argument("--osm",    default="map.osm",
                        help="Path to input .osm file (default: map.osm)")
    parser.add_argument("--csv",    default="ev-charging-stations-india.csv",
                        help="Path to EV stations CSV (default: ev-charging-stations-india.csv)")
    parser.add_argument("--output", default="./output",
                        help="Output directory (default: ./output)")
    args = parser.parse_args()

    print("=" * 60)
    print("  EV Charging Station ML Predictor")
    print("=" * 60)

    # 1. Parse OSM
    nodes_dict, ways, bbox = parse_osm(args.osm)

    # 2. Load EV stations
    ev_df = load_ev_stations(args.csv)

    # 3. Build training data
    X, y, cells, way_index, node_index, all_ev_lats, all_ev_lons = \
        build_training_data(ways, nodes_dict, ev_df, bbox)

    # 4. Train model
    clf = train_model(X, y)

    # 5. Predict
    probs = predict_map(clf, cells, way_index, node_index, all_ev_lats, all_ev_lons)

    # 6. Save outputs
    dedup = save_prediction_map(cells, probs, bbox, ways, args.output, ev_df)
    save_feature_importance(clf, args.output)
    candidates = save_json_report(dedup, args.output)
    save_confusion_summary(clf, X, y, args.output)

    print("\n" + "=" * 60)
    print("  TOP 10 PREDICTED EV STATION LOCATIONS")
    print("=" * 60)
    for c in candidates[:10]:
        print(f"  #{c['rank']:2d}  [{c['priority_tier']:6s}]  "
              f"lat={c['latitude']:.5f}  lon={c['longitude']:.5f}  "
              f"score={c['ml_score']:.3f}")
    print(f"\n  All outputs saved to: {args.output}/")
    print("  Files: 06_ml_prediction_map.png, 07_ml_feature_importance.png,")
    print("         08_ml_ev_report.json, 09_cv_auc_scores.png")


if __name__ == "__main__":
    main()
