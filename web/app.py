# -*- coding: utf-8 -*-
"""
web/app.py - FastAPI backend for EV Charging Station Dashboard
=============================================================
Serves an interactive Leaflet.js map and exposes REST API endpoints
for ML-based EV station suitability prediction.

Usage:
    python -m uvicorn web.app:app --reload      # from project root
    python main.py --web                # via unified pipeline
"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any

# Add project root to path so we can import ev_ml_predictor
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np

app = FastAPI(title="EV Charging Station Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

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
    _model_info["n_features"] = getattr(_model, "n_features_in_", 23)
    print(f"   Model loaded: {_model_info['model_file']} "
          f"({_model_info['model_type']}, {_model_info['n_features']} features)")
    return _model

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": _model is not None}

@app.get("/api/model-info")
async def model_info():
    """Return model metadata."""
    try:
        get_model()
        return {
            "status": "ok",
            **_model_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an .osm file upload, run the prediction pipeline, return GeoJSON."""
    if not file.filename or not file.filename.lower().endswith(".osm"):
        raise HTTPException(status_code=400, detail="File must have .osm extension.")

    # Save uploaded file temporarily
    upload_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
    
    with open(upload_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    try:
        clf = get_model()
        result = run_prediction(clf, str(upload_path))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if upload_path.exists():
            upload_path.unlink()

class BboxRequest(BaseModel):
    south: float
    west: float
    north: float
    east: float

@app.post("/api/predict-bbox")
async def predict_bbox(data: BboxRequest):
    """Accept bounding box coordinates, download OSM via Overpass, predict."""
    lat_span = data.north - data.south
    lon_span = data.east - data.west
    
    if lat_span > 0.1 or lon_span > 0.1:
        raise HTTPException(
            status_code=400, 
            detail="Bounding box too large. Max span is ~0.1 degrees (~11 km). Please draw a smaller area."
        )

    if lat_span <= 0 or lon_span <= 0:
        raise HTTPException(status_code=400, detail="Invalid bounding box (north must be > south, east > west).")

    upload_path = None
    try:
        # Download OSM data via Overpass
        osm_data = download_osm_bbox(data.south, data.west, data.north, data.east)
        if not osm_data:
            raise HTTPException(status_code=500, detail="Failed to download OSM data for this area.")

        # Save to temp file
        upload_id = str(uuid.uuid4())[:8]
        upload_path = UPLOAD_DIR / f"{upload_id}_bbox.osm"
        upload_path.write_text(osm_data, encoding="utf-8")

        clf = get_model()
        result = run_prediction(clf, str(upload_path))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if upload_path and upload_path.exists():
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
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
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
    """Initialize and return the FastAPI app."""
    print("=" * 64)
    print("  EV Charging Station Dashboard (FastAPI)")
    print("=" * 64)
    try:
        get_model()
    except Exception as e:
        print(f"   WARNING: {e}")
        print("   Dashboard will start, but predictions will fail until model is available.")
    return app

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    create_app(args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
