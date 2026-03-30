# -*- coding: utf-8 -*-
"""
main.py — Unified EV Charging Station Analysis Pipeline
========================================================
Runs both pipelines in sequence:
  1. Multispectral OSM classification  (ev_campus_analyzer.py)
  2. ML-based EV station prediction    (ev_ml_predictor.py)

Usage:
    python main.py
    python main.py --osm map.osm --csv ev-charging-stations-india.csv --output ./output
    python main.py --osm map.osm --output ./output --skip-ml    # skip the ML step
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


def run_ml_predictor(osm_path, csv_path, output_dir):
    """Run the ML prediction pipeline."""
    print("\n" + "=" * 64)
    print("  PHASE 2: ML-Powered EV Station Prediction")
    print("=" * 64 + "\n")

    from ev_ml_predictor import parse_osm, load_ev_stations, \
        build_training_data, train_model, predict_map, \
        save_prediction_map, save_feature_importance, \
        save_json_report, save_confusion_summary

    nodes_dict, ways, bbox = parse_osm(osm_path)
    ev_df = load_ev_stations(csv_path)

    X, y, cells, way_index, node_index, all_ev_lats, all_ev_lons = \
        build_training_data(ways, nodes_dict, ev_df, bbox)

    clf = train_model(X, y)
    probs = predict_map(clf, cells, way_index, node_index,
                        all_ev_lats, all_ev_lons)

    dedup = save_prediction_map(cells, probs, bbox, ways, output_dir, ev_df)
    save_feature_importance(clf, output_dir)
    candidates = save_json_report(dedup, output_dir)
    save_confusion_summary(clf, X, y, output_dir)

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
    ap.add_argument("--skip-ml", action="store_true",
                    help="Skip the ML prediction phase (run only multispectral)")
    ap.add_argument("--skip-multispectral", action="store_true",
                    help="Skip the multispectral phase (run only ML)")
    args = ap.parse_args()

    # Validate inputs
    if not os.path.exists(args.osm):
        sys.exit(f"ERROR: OSM file not found: {args.osm}")
    if not args.skip_ml and not os.path.exists(args.csv):
        sys.exit(f"ERROR: CSV file not found: {args.csv}")

    os.makedirs(args.output, exist_ok=True)
    start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   EV Charging Station Site Analysis — Complete Pipeline     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  OSM file : {args.osm}")
    print(f"  CSV file : {args.csv}")
    print(f"  Output   : {args.output}")

    # Phase 1
    if not args.skip_multispectral:
        run_multispectral(args.osm, args.output)
    else:
        print("\n⏭  Skipping Phase 1 (multispectral classification)")

    # Phase 2
    if not args.skip_ml:
        run_ml_predictor(args.osm, args.csv, args.output)
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
