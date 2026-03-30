# -*- coding: utf-8 -*-
"""
identity.py
===========
Integrate EV charging station data from a CSV into an OSM file.

Reads EV station locations from a CSV, filters those within the OSM file's
bounding box, and injects them as tagged nodes into the OSM XML.

Usage:
    python identity.py
    python identity.py --osm planet.osm --csv ev-charging-stations-india.csv --output output.osm

If the input is a .osm.gz file, it will be decompressed automatically.
"""

import xml.etree.ElementTree as ET
import csv
import math
import gzip
import os
import argparse
from collections import defaultdict


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculates simple Euclidean distance between two points."""
    return math.hypot(lat1 - lat2, lon1 - lon2)


class SpatialNodeIndex:
    """Simple grid-based spatial index for fast nearest-node lookups."""

    def __init__(self, nodes_data, bucket_size=0.005):
        self.bucket_size = bucket_size
        self.buckets = defaultdict(list)
        for node_id, (lat, lon, node_el) in nodes_data.items():
            bk = (int(lat / bucket_size), int(lon / bucket_size))
            self.buckets[bk].append((node_id, lat, lon, node_el))

    def find_nearest(self, lat, lon, search_steps=2):
        """Find the nearest node to (lat, lon) using bucket search."""
        bk = (int(lat / self.bucket_size), int(lon / self.bucket_size))
        best_id = None
        best_dist = float('inf')
        best_node = None

        for di in range(-search_steps, search_steps + 1):
            for dj in range(-search_steps, search_steps + 1):
                for node_id, n_lat, n_lon, node_el in self.buckets.get(
                        (bk[0] + di, bk[1] + dj), []):
                    dist = calculate_distance(lat, lon, n_lat, n_lon)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = node_id
                        best_node = node_el

        return best_id, best_dist, best_node


def integrate_ev_stations(osm_path, csv_path, output_path):
    """Main integration pipeline."""

    # ── Step 1: Handle gzip if needed ─────────────────────────────────
    if osm_path.endswith('.gz'):
        unzipped = osm_path[:-3]  # strip .gz
        if not os.path.exists(unzipped):
            print(f"1. Decompressing {osm_path} ...")
            with gzip.open(osm_path, 'rb') as f_in:
                with open(unzipped, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"   -> Decompressed to {unzipped}")
        else:
            print(f"1. Using existing decompressed file: {unzipped}")
        osm_path = unzipped
    else:
        print(f"1. Using OSM file: {osm_path}")

    # ── Step 2: Parse OSM and extract bounding box ────────────────────
    print("2. Parsing the OSM XML ...")
    tree = ET.parse(osm_path)
    root = tree.getroot()

    # Try to get bounds from the file
    bounds_el = root.find('bounds')
    if bounds_el is not None:
        min_lat = float(bounds_el.attrib.get('minlat', 0))
        max_lat = float(bounds_el.attrib.get('maxlat', 0))
        min_lon = float(bounds_el.attrib.get('minlon', 0))
        max_lon = float(bounds_el.attrib.get('maxlon', 0))
    else:
        # Compute from nodes
        min_lat = min_lon = float('inf')
        max_lat = max_lon = float('-inf')
        for node in root.findall('node'):
            try:
                lat = float(node.attrib['lat'])
                lon = float(node.attrib['lon'])
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)
            except (KeyError, ValueError):
                continue

    print(f"   Bounding box: lat {min_lat:.4f}–{max_lat:.4f}, "
          f"lon {min_lon:.4f}–{max_lon:.4f}")

    # ── Step 3: Index all existing nodes ──────────────────────────────
    print("3. Indexing all existing map nodes ...")
    nodes_data = {}
    for node in root.findall('node'):
        try:
            nid = node.attrib['id']
            lat = float(node.attrib['lat'])
            lon = float(node.attrib['lon'])
            nodes_data[nid] = (lat, lon, node)
        except (KeyError, ValueError):
            continue
    print(f"   -> Indexed {len(nodes_data):,} nodes.")

    # Build spatial index for fast nearest-neighbour lookups
    spatial_idx = SpatialNodeIndex(nodes_data)

    # ── Step 4: Filter EV stations from CSV ───────────────────────────
    print(f"4. Filtering EV stations from {csv_path} ...")
    ev_stations = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # Handle the common CSV typo 'lattitude'
                lat = float(row.get('lattitude', row.get('latitude', 0)))
                lon = float(row.get('longitude', 0))

                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    ev_stations.append({
                        'lat': lat,
                        'lon': lon,
                        'name': row.get('name', 'Unknown EV Station'),
                        'operator': row.get('operator', ''),
                        'type': row.get('type', '')
                    })
            except ValueError:
                continue

    print(f"   -> Found {len(ev_stations)} EV stations inside the map boundaries.")

    if not ev_stations:
        print("   WARNING: No EV stations found in the bounding box. "
              "The output file will be unchanged.")
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"   Saved (unchanged): {output_path}")
        return

    # ── Step 5: Snap EV stations to nearest OSM nodes ─────────────────
    print("5. Snapping EV stations to closest existing map nodes ...")
    modified_nodes = set()
    snapped_count = 0

    for station in ev_stations:
        best_id, best_dist, best_node = spatial_idx.find_nearest(
            station['lat'], station['lon'])

        if best_id and best_id not in modified_nodes and best_node is not None:
            # Inject standard OpenStreetMap EV charging tags
            ET.SubElement(best_node, 'tag', k='amenity', v='charging_station')

            if station['name']:
                ET.SubElement(best_node, 'tag', k='name', v=station['name'])
            if station['operator']:
                ET.SubElement(best_node, 'tag', k='brand', v=station['operator'])
            if station['type']:
                ET.SubElement(best_node, 'tag', k='capacity', v=station['type'])

            modified_nodes.add(best_id)
            snapped_count += 1

    print(f"   -> Snapped {snapped_count} EV stations to map nodes.")

    # ── Step 6: Save output ───────────────────────────────────────────
    print("6. Saving the updated map ...")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"   Success! Updated map saved as: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate EV charging station CSV data into an OSM file")
    parser.add_argument("--osm", default="map.osm",
                        help="Path to input .osm or .osm.gz file")
    parser.add_argument("--csv", default="ev-charging-stations-india.csv",
                        help="Path to EV stations CSV")
    parser.add_argument("--output", default="map_with_ev.osm",
                        help="Path for output .osm file")
    args = parser.parse_args()

    if not os.path.exists(args.osm):
        print(f"ERROR: OSM file not found: {args.osm}")
        return
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        return

    integrate_ev_stations(args.osm, args.csv, args.output)


if __name__ == "__main__":
    main()