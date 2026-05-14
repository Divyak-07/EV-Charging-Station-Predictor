# -*- coding: utf-8 -*-
"""
data_fetcher.py — Global EV Station Data Fetcher
=================================================
Fetches real EV charging station locations from the OpenChargeMap API
across 15+ countries, then extracts 23 spatial features via Overpass API
for ML training.

Two modes:
  --fetch    : Download station coordinates from OpenChargeMap → global_ev_stations.csv
  --extract  : Extract Overpass features for each station → global_training_data.csv

Usage:
    python data_fetcher.py --fetch
    python data_fetcher.py --extract
    python data_fetcher.py --fetch --extract   # both in sequence

Supports --resume to continue interrupted downloads/extractions.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import requests
except ImportError:
    sys.exit("ERROR: pandas, numpy, and requests are required.\n"
             "  pip install pandas numpy requests")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

# OpenChargeMap settings
OCM_API_URL = "https://api.openchargemap.io/v3/poi"
OCM_STATIONS_FILE = DATA_DIR / "global_ev_stations.csv"
OCM_FETCH_DELAY = 1.5  # seconds between OCM API calls

# Countries to fetch (ISO codes) — diverse global coverage
TARGET_COUNTRIES = [
    ("IN", "India"),
    ("US", "United States"),
    ("GB", "United Kingdom"),
    ("DE", "Germany"),
    ("FR", "France"),
    ("NL", "Netherlands"),
    ("NO", "Norway"),
    ("SE", "Sweden"),
    ("DK", "Denmark"),
    ("CA", "Canada"),
    ("AU", "Australia"),
    ("JP", "Japan"),
    ("KR", "South Korea"),
    ("CN", "China"),
    ("BR", "Brazil"),
    ("IT", "Italy"),
    ("ES", "Spain"),
    ("PT", "Portugal"),
    ("AT", "Austria"),
    ("CH", "Switzerland"),
]
MAX_PER_COUNTRY = 1000  # OpenChargeMap allows up to 5000

# Overpass / training settings
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
QUERY_RADIUS_M = 500
NEGATIVE_OFFSET_DEG = 0.015
NEGATIVES_PER_POSITIVE = 2
OVERPASS_DELAY = 4.0  # longer delay to respect fair use
TRAINING_DATA_FILE = DATA_DIR / "global_training_data.csv"
MAX_TRAINING_SAMPLES = 500  # realistic limit for Overpass fair use

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


def load_api_key():
    """Load OpenChargeMap API key from .env file or environment variable."""
    # Check environment variable first
    key = os.environ.get("OCM_API_KEY")
    if key:
        return key

    # Check .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OCM_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()

    sys.exit("ERROR: No API key found.\n"
             "  Set OCM_API_KEY in .env file or as environment variable.\n"
             "  Get a free key at: https://openchargemap.org/site/profile/applications")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: FETCH STATIONS FROM OPENCHARGEMAP
# ─────────────────────────────────────────────────────────────────────────────

def fetch_country(api_key, country_code, country_name, max_results=MAX_PER_COUNTRY):
    """Fetch EV stations for a single country from OpenChargeMap API."""
    params = {
        "key": api_key,
        "countrycode": country_code,
        "maxresults": max_results,
        "compact": True,
        "verbose": False,
        "output": "json",
    }
    headers = {
        "User-Agent": "EVStationPredictor/2.0 (academic-research)",
    }

    for attempt in range(3):
        try:
            resp = requests.get(OCM_API_URL, params=params, headers=headers, timeout=60)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"      Rate limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            stations = []
            for poi in data:
                addr = poi.get("AddressInfo", {})
                lat = addr.get("Latitude")
                lon = addr.get("Longitude")
                if lat is None or lon is None:
                    continue

                # Extract useful metadata
                city = addr.get("Town", addr.get("StateOrProvince", ""))
                operator = ""
                if poi.get("OperatorInfo"):
                    operator = poi["OperatorInfo"].get("Title", "")

                # Count connection points and max power
                num_points = 0
                max_power = 0
                for conn in poi.get("Connections", []):
                    num_points += 1
                    pw = conn.get("PowerKW") or 0
                    if pw > max_power:
                        max_power = pw

                stations.append({
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "country_code": country_code,
                    "country": country_name,
                    "city": str(city)[:100] if city else "",
                    "operator": str(operator)[:100] if operator else "",
                    "num_points": num_points,
                    "power_kw": round(max_power, 1),
                })

            return stations

        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"      ERROR fetching {country_name}: {e}")
                return []

    return []


def fetch_all_stations(api_key, resume=False):
    """Fetch stations from all target countries."""
    print("=" * 64)
    print("  FETCHING GLOBAL EV STATIONS FROM OPENCHARGEMAP")
    print("=" * 64)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support: load existing data if present
    existing_df = None
    fetched_countries = set()
    if resume and OCM_STATIONS_FILE.exists():
        existing_df = pd.read_csv(OCM_STATIONS_FILE)
        fetched_countries = set(existing_df["country_code"].unique())
        print(f"   Resuming: found {len(existing_df)} stations from "
              f"{len(fetched_countries)} countries already fetched")

    all_stations = []
    total = len(TARGET_COUNTRIES)

    for idx, (code, name) in enumerate(TARGET_COUNTRIES, 1):
        if code in fetched_countries:
            print(f"   [{idx}/{total}] {name} ({code}) — SKIPPED (already fetched)")
            continue

        print(f"   [{idx}/{total}] Fetching {name} ({code}) ...", end="", flush=True)
        stations = fetch_country(api_key, code, name)
        print(f" {len(stations)} stations")
        all_stations.extend(stations)
        time.sleep(OCM_FETCH_DELAY)

    if not all_stations and existing_df is None:
        print("   ERROR: No stations fetched!")
        return None

    # Build DataFrame
    new_df = pd.DataFrame(all_stations) if all_stations else pd.DataFrame()

    # Merge with existing if resuming
    if existing_df is not None and len(new_df) > 0:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    elif existing_df is not None:
        combined = existing_df
    else:
        combined = new_df

    # Deduplicate by coordinates (round to 4 decimal places)
    combined["lat_round"] = combined["latitude"].round(4)
    combined["lon_round"] = combined["longitude"].round(4)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["lat_round", "lon_round"])
    combined = combined.drop(columns=["lat_round", "lon_round"])
    after = len(combined)
    if before != after:
        print(f"   Deduplicated: {before} -> {after} (removed {before - after} duplicates)")

    # Save
    combined.to_csv(OCM_STATIONS_FILE, index=False)
    print(f"\n   [OK] Saved {len(combined)} stations to: {OCM_STATIONS_FILE}")

    # Summary
    print(f"\n   Country breakdown:")
    for code, name in TARGET_COUNTRIES:
        count = len(combined[combined["country_code"] == code])
        if count > 0:
            print(f"      {name:25s} {count:>5d} stations")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: EXTRACT OVERPASS FEATURES
# ─────────────────────────────────────────────────────────────────────────────

_mirror_idx = 0  # rotate across mirrors
_working_mirrors = None  # filtered at runtime

def _init_mirrors():
    """Test all mirrors and keep only the working ones."""
    global _working_mirrors
    if _working_mirrors is not None:
        return
    test_q = '[out:json][timeout:10];node(around:200,28.6139,77.2090)["amenity"];out tags;'
    _working_mirrors = []
    for url in OVERPASS_MIRRORS:
        name = url.split("/")[2]
        try:
            r = requests.post(url, data={"data": test_q}, timeout=15)
            if r.status_code == 200:
                _working_mirrors.append(url)
                print(f"      [OK] {name}")
            else:
                print(f"      [BLOCKED] {name} (HTTP {r.status_code})")
        except Exception:
            print(f"      [DOWN] {name}")
    if not _working_mirrors:
        print("      WARNING: No Overpass mirrors responding!")
        _working_mirrors = [OVERPASS_MIRRORS[0]]  # fallback

def overpass_query(lat, lon, radius_m=QUERY_RADIUS_M, retries=3):
    """Query Overpass API for all ways and nodes around (lat, lon).
    Rotates across working Overpass mirrors to avoid IP throttling."""
    global _mirror_idx
    _init_mirrors()
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
        url = _working_mirrors[_mirror_idx % len(_working_mirrors)]
        _mirror_idx += 1
        try:
            resp = requests.post(url, data={"data": query}, timeout=45)
            if resp.status_code in (429, 403, 406):
                wait = 20 * (attempt + 1)
                time.sleep(wait)
                continue
            if resp.status_code == 504:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(8 * (attempt + 1))
            else:
                return None  # None = API failure; [] = genuinely empty area
    return None


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

        if amen == "charging_station" or tags.get("ev:charging") or \
           "charging" in tags.get("name", "").lower():
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
        min(node_count / 100.0, 1.0),
        has_ev,
        min(poi_count / 15.0, 1.0),
    ]
    return vec


def generate_negative_point(lat, lon):
    """Generate a negative sample point offset from (lat, lon).
    Simple random offset — no expensive distance checks."""
    angle = random.uniform(0, 2 * math.pi)
    offset = random.uniform(NEGATIVE_OFFSET_DEG * 0.8, NEGATIVE_OFFSET_DEG * 1.3)
    nlat = lat + offset * math.sin(angle)
    nlon = lon + offset * math.cos(angle)
    return nlat, nlon


def sample_stations_global(df, max_n=MAX_TRAINING_SAMPLES):
    """Sample stations proportionally from each country for geographic diversity."""
    df = df.copy()
    country_counts = df["country_code"].value_counts()
    countries = country_counts.index.tolist()

    per_country = max(max_n // len(countries), 10)
    sampled = []

    for country in countries:
        country_df = df[df["country_code"] == country]
        n = min(per_country, len(country_df))
        sampled.append(country_df.sample(n=n, random_state=42))

    result = pd.concat(sampled).drop_duplicates(subset=["latitude", "longitude"])

    # If under budget, add more
    if len(result) < max_n:
        remaining = df[~df.index.isin(result.index)]
        extra = min(max_n - len(result), len(remaining))
        if extra > 0:
            result = pd.concat([result, remaining.sample(n=extra, random_state=42)])

    return result.head(max_n).reset_index(drop=True)


def extract_training_data(resume=False):
    """Extract Overpass features for sampled global stations."""
    print("\n" + "=" * 64)
    print("  EXTRACTING OVERPASS FEATURES FOR TRAINING DATA")
    print("=" * 64)

    if not OCM_STATIONS_FILE.exists():
        sys.exit(f"ERROR: No station data found at {OCM_STATIONS_FILE}\n"
                 f"  Run: python data_fetcher.py --fetch")

    stations_df = pd.read_csv(OCM_STATIONS_FILE)
    print(f"   Loaded {len(stations_df)} stations from {stations_df['country_code'].nunique()} countries")

    # Sample for training
    sampled = sample_stations_global(stations_df)
    print(f"   Sampled {len(sampled)} stations for feature extraction")

    # Resume support: load existing extractions
    existing_count = 0
    features = []
    labels = []

    if resume and TRAINING_DATA_FILE.exists():
        existing_df = pd.read_csv(TRAINING_DATA_FILE)
        existing_count = len(existing_df)
        features = existing_df[FEATURE_NAMES].values.tolist()
        labels = existing_df["label"].tolist()
        print(f"   Resuming: found {existing_count} existing samples")

        # Skip already-processed stations
        skip_n = existing_count // (1 + NEGATIVES_PER_POSITIVE)
        sampled = sampled.iloc[skip_n:]
        print(f"   Skipping first {skip_n} stations (already extracted)")

    # Warmup test — verify Overpass is responding
    print(f"\n   Testing Overpass API connectivity...")
    test = overpass_query(28.6139, 77.2090)  # New Delhi
    if test is None:
        print("   ERROR: Overpass API is not responding. Try again later.")
        return None, None
    print(f"   Overpass OK ({len(test)} elements from test query)")

    total = len(sampled)
    batch_save_every = 10  # Save more frequently
    consecutive_failures = 0
    max_consecutive_failures = 15  # Stop if API is clearly down

    for idx, (_, row) in enumerate(sampled.iterrows(), 1):
        lat, lon = row["latitude"], row["longitude"]
        country = row.get("country", "?")
        city = row.get("city", "?")

        print(f"   [{idx}/{total}] ({lat:.4f}, {lon:.4f}) "
              f"- {city}, {country} ...", end="", flush=True)

        # Positive sample
        elements = overpass_query(lat, lon)

        # None = API failure (different from [] = empty area)
        if elements is None:
            consecutive_failures += 1
            print(f" API ERROR ({consecutive_failures}/{max_consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n   STOPPING: {max_consecutive_failures} consecutive API failures.")
                print(f"   Run again with --resume to continue later.")
                break
            time.sleep(15)  # longer wait after failure
            continue

        # Empty area is VALID data (the model needs to learn about sparse areas too)
        consecutive_failures = 0  # reset on success
        fv = extract_features_from_overpass(elements)
        features.append(fv)
        labels.append(1)
        print(f" [OK] {len(elements)} elements")

        # Negative samples
        for _ in range(NEGATIVES_PER_POSITIVE):
            nlat, nlon = generate_negative_point(lat, lon)
            neg_elements = overpass_query(nlat, nlon)
            if neg_elements is None:
                neg_elements = []  # treat API failure as empty for negatives
            neg_fv = extract_features_from_overpass(neg_elements)
            features.append(neg_fv)
            labels.append(0)

        # Rate limiting
        time.sleep(OVERPASS_DELAY)

        # Periodic save for resume support
        if idx % batch_save_every == 0:
            _save_training_data(features, labels)
            print(f"   [SAVE] Progress saved: {len(features)} samples "
                  f"({sum(labels)} pos, {len(labels) - sum(labels)} neg)")

    # Final save
    if features:
        _save_training_data(features, labels)

    X = np.array(features, dtype=np.float32) if features else np.array([])
    y = np.array(labels, dtype=np.int32) if labels else np.array([])
    print(f"\n   [OK] Extraction complete!")
    print(f"   Total samples: {len(X)} ({y.sum()} positive, {(y==0).sum()} negative)")
    print(f"   Saved to: {TRAINING_DATA_FILE}")

    return X, y


def _save_training_data(features, labels):
    """Save current training data to disk."""
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_DATA_FILE, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch global EV station data and extract training features")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch station coordinates from OpenChargeMap API")
    parser.add_argument("--extract", action="store_true",
                        help="Extract Overpass features for training data")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted fetch/extraction")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: fetch 1 country, extract 5 stations")
    args = parser.parse_args()

    if not args.fetch and not args.extract and not args.test:
        parser.print_help()
        print("\nExample usage:")
        print("  python data_fetcher.py --fetch              # Download stations")
        print("  python data_fetcher.py --extract             # Extract features")
        print("  python data_fetcher.py --fetch --extract     # Both")
        print("  python data_fetcher.py --test                # Quick test")
        return

    api_key = load_api_key()
    print(f"   API key loaded: {api_key[:8]}...{api_key[-4:]}")

    if args.test:
        # Quick test mode: fetch 1 country, extract 5 stations
        print("\n[TEST] TEST MODE: Fetching India only, extracting 5 stations\n")
        stations = fetch_country(api_key, "IN", "India", max_results=20)
        print(f"   Fetched {len(stations)} stations from India")

        if stations:
            test_df = pd.DataFrame(stations[:5])
            print(f"\n   Sample stations:")
            for _, row in test_df.iterrows():
                print(f"     ({row['latitude']:.4f}, {row['longitude']:.4f}) "
                      f"— {row['city']}")

            # Extract features for first station
            lat, lon = stations[0]["latitude"], stations[0]["longitude"]
            print(f"\n   Extracting features for ({lat:.4f}, {lon:.4f}) ...")
            elements = overpass_query(lat, lon)
            fv = extract_features_from_overpass(elements)
            print(f"   Elements found: {len(elements)}")
            print(f"   Feature vector: {[round(v, 3) for v in fv]}")
            print("\n   [OK] Test passed! API key and pipeline are working.")
        return

    if args.fetch:
        fetch_all_stations(api_key, resume=args.resume)

    if args.extract:
        extract_training_data(resume=args.resume)


if __name__ == "__main__":
    main()
