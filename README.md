# EV Charging Station Suitability Predictor

> AI-powered pipeline for predicting optimal EV charging station locations using OpenStreetMap data, multi-model ML comparison, and an interactive web dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-green?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red?logo=xgboost&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)

## Overview

This project implements an end-to-end ML pipeline that analyzes geographic features from OpenStreetMap data to predict suitability scores for EV charging station placement. It combines:

- **Multispectral OSM Classification** — Categorizes roads, buildings, amenities, and land use from raw `.osm` files
- **23-Feature Spatial Analysis** — Extracts road density, POI proximity, parking availability, and infrastructure metrics within a 500m radius
- **Multi-Model Comparison** — Random Forest, XGBoost, and LightGBM with Optuna Bayesian hyperparameter optimization
- **Interactive Web Dashboard** — Flask + Leaflet.js interface with heatmap visualization, file upload, and draw-on-map analysis
- **Global Dataset** — 17,900+ real EV station locations from 20 countries via OpenChargeMap API

## Architecture

```
User Input (.osm file or bounding box)
        │
        ▼
┌─────────────────────┐
│  Phase 1: OSM Parse │ ← Road classification, building detection
│  & Feature Extract  │   POI density, amenity proximity
└────────┬────────────┘
         │ 23-feature vector per grid cell
         ▼
┌─────────────────────┐
│  Phase 2: ML Model  │ ← XGBoost (best), RF, LightGBM
│  Suitability Score  │   Optuna-tuned hyperparameters
└────────┬────────────┘
         │ Probability scores [0, 1]
         ▼
┌─────────────────────┐
│  Phase 3: Dashboard │ ← Leaflet.js heatmap
│  Visualization      │   Candidate markers, layer controls
└─────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/EV-Charging-Station-Predictor.git
cd EV-Charging-Station-Predictor
pip install -r requirements.txt
```

### Run the Dashboard
```bash
python main.py --web
# Open http://localhost:5000
```

### Run Model Comparison
```bash
python model_comparison.py --trials 50
```

### Train on Custom Data
```bash
python main.py --train --csv your-stations.csv --osm your-map.osm
```

## Docker

```bash
docker build -t ev-predictor .
docker run -p 5000:5000 ev-predictor
```

## Deploy to Render

1. Push to GitHub
2. Connect your repo on [render.com](https://render.com)
3. It will auto-detect `render.yaml` and deploy

## Features

| Feature | Description |
|---|---|
| `.osm` File Upload | Drag & drop OpenStreetMap files for instant analysis |
| Draw on Map | Select any area on the map for live prediction |
| Suitability Heatmap | Color-coded grid overlay showing prediction scores |
| Top Candidates | Ranked location markers with score tiers (HIGH/MEDIUM/LOW) |
| Layer Controls | Toggle heatmap, candidates, and road overlay independently |
| Score Filtering | Real-time slider to filter by minimum suitability score |

## Model Performance

| Model | Mean AUC | Std | Status |
|---|---|---|---|
| Random Forest | 0.6817 | ±0.0589 | Baseline |
| **XGBoost** | **0.6897** | **±0.0698** | **Best** |
| LightGBM | 0.6877 | ±0.0645 | Competitive |

*Trained on 432 samples from India. Performance expected to improve significantly with global dataset (17,900+ stations from 20 countries).*

## Project Structure

```
├── main.py                 # Unified CLI entry point
├── ev_ml_predictor.py      # ML pipeline (training + prediction)
├── ev_campus_analyzer.py   # Multispectral OSM classification
├── model_comparison.py     # XGBoost/LightGBM/Optuna comparison
├── data_fetcher.py         # OpenChargeMap API + Overpass extraction
├── web/
│   ├── app.py              # Flask backend + REST API
│   ├── templates/
│   │   └── index.html      # Dashboard SPA
│   └── static/
│       ├── css/style.css    # Dark theme UI
│       └── js/app.js        # Leaflet.js map logic
├── output/
│   ├── ev_model_best.joblib # Best trained model (XGBoost)
│   ├── training_data.csv    # Training dataset
│   └── *.png               # Generated charts
├── data/
│   └── global_ev_stations.csv  # 17,900+ stations from 20 countries
├── Dockerfile              # Multi-stage Docker build
├── render.yaml             # Render deployment blueprint
└── requirements.txt
```

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, Optuna
- **Backend**: Flask, gunicorn
- **Frontend**: Leaflet.js, Vanilla JS, CSS3
- **Data**: OpenChargeMap API, Overpass API, OpenStreetMap
- **Deployment**: Docker, Render

## License

MIT
