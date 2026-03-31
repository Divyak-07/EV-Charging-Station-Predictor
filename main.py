# -*- coding: utf-8 -*-
"""
main.py — Unified EV Charging Station Analysis Pipeline
========================================================
Runs both pipelines in sequence:
  1. Multispectral OSM classification  (ev_campus_analyzer.py)
  2. ML-based EV station prediction    (ev_ml_predictor.py)

Usage:
    python main.py --osm map.osm --csv ev-charging-stations-india.csv --output ./output
    python main.py --train --csv ev-charging-stations-india.csv    # train model only
    python main.py --osm map.osm --skip-ml                        # classification only
"""

import argparse
import os
import sys
import time


def run_multispectral(osm_path, output_dir):
    """Run the multispectral classification pipeline."""
    print("\n" + "=" * 64)
    print("  PHASE 1: Multispectral OSM Classification")
    print("=" * 64 + "\n")

    from ev_campus_analyzer import OSMParser, CoordNorm, draw_2d_map, \
        draw_ev_heatmap, draw_3d_map, draw_legend_panel, write_report

    parser = OSMParser(osm_path)
    norm = CoordNorm(parser.bounds)

    stats = draw_2d_map(parser, norm,
                        os.path.join(output_dir, "01_multispectral_map.png"))
    high_zones = draw_ev_heatmap(parser, norm,
                                 os.path.join(output_dir, "02_ev_suitability_heatmap.png"))
    draw_3d_map(parser, norm,
                os.path.join(output_dir, "03_3d_map.png"))
    draw_legend_panel(stats,
                      os.path.join(output_dir, "04_legend_stats.png"))
    report = write_report(parser, norm, high_zones,
                          os.path.join(output_dir, "05_ev_analysis_report.json"))

    print("\n✅  Phase 1 complete. Top EV zones (heuristic):")
    for z in report["high_ev_zones"][:5]:
        print(f"   [{z['priority']}]  score={z['score']}  "
              f"lat={z['lat']}  lon={z['lon']}  name='{z['name']}'")

    return report


def run_ml_predictor(osm_path, csv_path, output_dir, do_train=False):
    """Run the ML prediction pipeline."""
    print("\n" + "=" * 64)
    print("  PHASE 2: ML-Powered EV Station Prediction")
    if do_train:
        print("  MODE: Training on real multi-city data via Overpass API")
    else:
        print("  MODE: Predicting with saved model")
    print("=" * 64 + "\n")

    from ev_ml_predictor import (
        parse_osm, build_training_data, train_model, load_model,
        build_grid, predict_map, save_prediction_map,
        save_feature_importance, save_json_report, save_cv_scores,
        SpatialIndex, MODEL_FILE
    )
    import pandas as pd

    clf = None
    cv_scores = None

    # ── Training ──
    if do_train:
        if not os.path.exists(csv_path):
            print(f"ERROR: CSV not found: {csv_path}")
            return None
        print(f"[1/5] Loading EV station CSV: {csv_path}")
        ev_df = pd.read_csv(csv_path)
        X, y = build_training_data(ev_df, output_dir)
        clf, cv_scores = train_model(X, y, output_dir)
        save_cv_scores(cv_scores, output_dir)

    # ── Prediction ──
    if not os.path.exists(osm_path):
        if do_train:
            print(f"\n✅  Training complete! Model saved to {output_dir}/{MODEL_FILE}")
            print(f"  To predict: python main.py --osm <your_map.osm>")
            return None
        else:
            print(f"ERROR: OSM file not found: {osm_path}")
            return None

    if clf is None:
        clf = load_model(output_dir)
        if clf is None:
            print(f"ERROR: No trained model found. Train first with --train flag.")
            return None

    nodes_dict, ways, bbox = parse_osm(osm_path)
    way_index = SpatialIndex(ways)
    node_items = [{"lat": v[0], "lon": v[1], "tags": v[2]}
                  for v in nodes_dict.values()]
    node_index = SpatialIndex(node_items)

    cells = build_grid(bbox)
    print(f"   Grid cells: {len(cells):,}")
    probs = predict_map(clf, cells, way_index, node_index)

    # Load EV CSV for overlay
    ev_df = None
    if os.path.exists(csv_path):
        ev_df = pd.read_csv(csv_path)
        ev_df.columns = [c.strip().lower() for c in ev_df.columns]
        if "lattitude" in ev_df.columns:
            ev_df.rename(columns={"lattitude": "latitude"}, inplace=True)
        ev_df["latitude"] = pd.to_numeric(ev_df["latitude"], errors="coerce")
        ev_df["longitude"] = pd.to_numeric(ev_df["longitude"], errors="coerce")
        ev_df = ev_df.dropna(subset=["latitude", "longitude"])

    dedup = save_prediction_map(cells, probs, bbox, ways, output_dir, ev_df)
    save_feature_importance(clf, output_dir)
    candidates = save_json_report(dedup, output_dir)

    if cv_scores is not None:
        save_cv_scores(cv_scores, output_dir)

    print("\n✅  Phase 2 complete. Top ML-predicted locations:")
    for c in candidates[:5]:
        print(f"   #{c['rank']:2d}  [{c['priority_tier']:6s}]  "
              f"lat={c['latitude']:.5f}  lon={c['longitude']:.5f}  "
              f"score={c['ml_score']:.3f}")

    return candidates


def main():
    ap = argparse.ArgumentParser(
        description="EV Charging Station Analysis — Unified Pipeline")
    ap.add_argument("--osm", default="map.osm",
                    help="Path to input .osm file (default: map.osm)")
    ap.add_argument("--csv", default="ev-charging-stations-india.csv",
                    help="Path to EV stations CSV (default: ev-charging-stations-india.csv)")
    ap.add_argument("--output", default="./output",
                    help="Output directory (default: ./output)")
    ap.add_argument("--train", action="store_true",
                    help="Train ML model from real EV data via Overpass API (needs internet)")
    ap.add_argument("--skip-ml", action="store_true",
                    help="Skip the ML prediction phase (run only multispectral)")
    ap.add_argument("--skip-multispectral", action="store_true",
                    help="Skip the multispectral phase (run only ML)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   EV Charging Station Site Analysis — Complete Pipeline     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  OSM file : {args.osm}")
    print(f"  CSV file : {args.csv}")
    print(f"  Output   : {args.output}")
    if args.train:
        print(f"  Mode     : TRAINING + PREDICTION")

    # Phase 1
    if not args.skip_multispectral and os.path.exists(args.osm):
        run_multispectral(args.osm, args.output)
    elif args.skip_multispectral:
        print("\n⏭  Skipping Phase 1 (multispectral classification)")

    # Phase 2
    if not args.skip_ml:
        run_ml_predictor(args.osm, args.csv, args.output, do_train=args.train)
    else:
        print("\n⏭  Skipping Phase 2 (ML prediction)")

    elapsed = time.time() - start
    print(f"\n{'=' * 64}")
    print(f"  All done in {elapsed:.1f}s — outputs in: {os.path.abspath(args.output)}")
    print(f"{'=' * 64}")
    print("\n  Output files:")
    if not args.skip_multispectral:
        print("    01_multispectral_map.png       — 2D classification map")
        print("    02_ev_suitability_heatmap.png   — EV suitability heatmap")
        print("    03_3d_map.png                   — 3D extruded map")
        print("    04_legend_stats.png             — Legend & statistics")
        print("    05_ev_analysis_report.json      — Heuristic EV zone report")
    if not args.skip_ml:
        print("    06_ml_prediction_map.png        — ML prediction map")
        print("    07_ml_feature_importance.png     — Feature importance chart")
        print("    08_ml_ev_report.json             — ML candidate locations")
        print("    09_cv_auc_scores.png             — Cross-validation AUC")


if __name__ == "__main__":
    main()
