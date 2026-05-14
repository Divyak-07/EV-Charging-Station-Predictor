# -*- coding: utf-8 -*-
"""
web/app.py - Flask backend for EV Charging Station Dashboard
=============================================================
Serves an interactive Leaflet.js map and exposes REST API endpoints
for ML-based EV station suitability prediction.

Usage:
    python -m web.app                   # from project root
    python main.py --web                # via unified pipeline
"""

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

# Add project root to path so we can import ev_ml_predictor
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

import numpy as np

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Global model reference (loaded once at startup) ───────────────────────
_model = None
_model_info = {}


def get_model():
    """Load the ML model (singleton)."""
    global _model, _model_info
    if _model is not None:
        return _model

    try:
        import joblib
    except ImportError:
        raise RuntimeError("joblib is required: pip install joblib")

    # Try best model first, then fallback to standard
    output_dir = PROJECT_ROOT / "output"
    best_path = output_dir / "ev_model_best.joblib"
    std_path = output_dir / "ev_model.joblib"

    if best_path.exists():
        _model = joblib.load(best_path)
        _model_info["model_file"] = str(best_path.name)
    elif std_path.exists():
        _model = joblib.load(std_path)
        _model_info["model_file"] = str(std_path.name)
    else:
        raise FileNotFoundError(
            f"No trained model found in {output_dir}. "
            f"Run: python main.py --train --csv ev-charging-stations-india.csv"
        )

    _model_info["model_type"] = type(_model).__name__
    _model_info["n_features"] = _model.n_features_in_
    print(f"   Model loaded: {_model_info['model_file']} "
          f"({_model_info['model_type']}, {_model_info['n_features']} features)")
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.route("/api/model-info")
def model_info():
    """Return model metadata."""
    try:
        get_model()
        return jsonify({
            "status": "ok",
            **_model_info,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accept an .osm file upload, run the prediction pipeline, return GeoJSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a .osm file as 'file'."}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".osm"):
        return jsonify({"error": "File must have .osm extension."}), 400

    # Save uploaded file temporarily
    upload_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
    file.save(str(upload_path))

    try:
        clf = get_model()
        result = run_prediction(clf, str(upload_path))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up
        if upload_path.exists():
            upload_path.unlink()


@app.route("/api/predict-bbox", methods=["POST"])
def predict_bbox():
    """Accept bounding box coordinates, download OSM via Overpass, predict."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Send JSON with south, west, north, east."}), 400

    required = ["south", "west", "north", "east"]
    for key in required:
        if key not in data:
            return jsonify({"error": f"Missing required field: {key}"}), 400

    try:
        south = float(data["south"])
        west = float(data["west"])
        north = float(data["north"])
        east = float(data["east"])
    except (ValueError, TypeError):
        return jsonify({"error": "Coordinates must be numbers."}), 400

    # Validate bbox size (prevent massive downloads)
    lat_span = north - south
    lon_span = east - west
    if lat_span > 0.1 or lon_span > 0.1:
        return jsonify({
            "error": "Bounding box too large. Max span is ~0.1 degrees (~11 km). "
                     "Please draw a smaller area."
        }), 400

    if lat_span <= 0 or lon_span <= 0:
        return jsonify({"error": "Invalid bounding box (north must be > south, east > west)."}), 400

    try:
        # Download OSM data via Overpass
        osm_data = download_osm_bbox(south, west, north, east)
        if not osm_data:
            return jsonify({"error": "Failed to download OSM data for this area."}), 500

        # Save to temp file
        upload_id = str(uuid.uuid4())[:8]
        upload_path = UPLOAD_DIR / f"{upload_id}_bbox.osm"
        upload_path.write_text(osm_data, encoding="utf-8")

        clf = get_model()
        result = run_prediction(clf, str(upload_path))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if "upload_path" in locals() and upload_path.exists():
            upload_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction(clf, osm_path):
    """Run the ML prediction pipeline and return GeoJSON-formatted results."""
    from ev_ml_predictor import (
        parse_osm, build_grid, build_feature_vector_local,
        SpatialIndex, GRID_RESOLUTION
    )

    nodes_dict, ways, bbox = parse_osm(osm_path)
    minlat, minlon, maxlat, maxlon = bbox

    # Build spatial indices
    way_index = SpatialIndex(ways)
    node_items = [{"lat": v[0], "lon": v[1], "tags": v[2]}
                  for v in nodes_dict.values()]
    node_index = SpatialIndex(node_items)

    # Build grid and predict
    cells = build_grid(bbox)
    X_pred = []
    for clat, clon in cells:
        fv = build_feature_vector_local(clat, clon, way_index, node_index)
        X_pred.append(fv)
    X_pred = np.array(X_pred, dtype=np.float32)
    probs = clf.predict_proba(X_pred)[:, 1]

    # Build heatmap GeoJSON (grid cells as rectangles)
    half = GRID_RESOLUTION / 2
    heatmap_features = []
    for (clat, clon), prob in zip(cells, probs):
        if prob < 0.05:
            continue  # skip very low scores to reduce payload
        heatmap_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [clon - half, clat - half],
                    [clon + half, clat - half],
                    [clon + half, clat + half],
                    [clon - half, clat + half],
                    [clon - half, clat - half],
                ]],
            },
            "properties": {
                "score": round(float(prob), 4),
            },
        })

    # Top candidates (deduplicated)
    sorted_idx = np.argsort(probs)[::-1]
    top_n = min(30, len(sorted_idx))
    top_cells = [(cells[i], probs[i]) for i in sorted_idx[:top_n]]
    dedup = []
    min_sep = GRID_RESOLUTION * 3
    for (clat, clon), p in top_cells:
        too_close = any(
            abs(clat - d[0][0]) < min_sep and abs(clon - d[0][1]) < min_sep
            for d in dedup
        )
        if not too_close:
            dedup.append(((clat, clon), p))

    candidate_features = []
    for rank, ((clat, clon), p) in enumerate(dedup[:15]):
        tier = "HIGH" if p >= 0.60 else "MEDIUM" if p >= 0.40 else "LOW"
        candidate_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [clon, clat],
            },
            "properties": {
                "rank": rank + 1,
                "score": round(float(p), 4),
                "tier": tier,
            },
        })

    # Road features for overlay
    road_colors = {"primary", "secondary", "tertiary", "residential"}
    road_features = []
    for w in ways:
        hw = w["tags"].get("highway", "")
        if hw in road_colors:
            road_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [w["lon"], w["lat"]],
                },
                "properties": {
                    "highway": hw,
                },
            })

    return {
        "status": "ok",
        "bbox": {"south": minlat, "west": minlon, "north": maxlat, "east": maxlon},
        "stats": {
            "grid_cells": len(cells),
            "nodes": len(nodes_dict),
            "ways": len(ways),
            "candidates": len(candidate_features),
            "max_score": round(float(probs.max()), 4) if len(probs) > 0 else 0,
            "mean_score": round(float(probs.mean()), 4) if len(probs) > 0 else 0,
        },
        "heatmap": {
            "type": "FeatureCollection",
            "features": heatmap_features,
        },
        "candidates": {
            "type": "FeatureCollection",
            "features": candidate_features,
        },
        "roads": {
            "type": "FeatureCollection",
            "features": road_features[:500],  # limit to prevent huge payloads
        },
    }


def download_osm_bbox(south, west, north, east):
    """Download OSM data for a bounding box via Overpass API."""
    import requests

    query = f"""
    [out:xml][timeout:60];
    (
      node({south},{west},{north},{east});
      way({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """
    mirrors = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]
    for url in mirrors:
        try:
            resp = requests.post(url, data={"data": query}, timeout=90)
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            print(f"   Mirror {url.split('/')[2]} failed: {e}")
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def create_app(port=5000):
    """Initialize and return the Flask app."""
    print("=" * 64)
    print("  EV Charging Station Dashboard")
    print("=" * 64)
    try:
        get_model()
    except Exception as e:
        print(f"   WARNING: {e}")
        print("   Dashboard will start, but predictions will fail until model is available.")
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    create_app(args.port)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
