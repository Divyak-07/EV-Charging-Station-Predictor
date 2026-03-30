"""
EV Charging Station Site Analysis — Multispectral Campus Mapper
================================================================
Parses any .osm file and produces:
  1. 2D Multispectral Map  — roads (width-classified), buildings, greenery,
                             water, barren land, amenities
  2. EV Suitability Score Map — heatmap overlay
  3. 3D Pseudo-3D Terrain Map — extruded buildings + layered terrain
  4. Legend + Summary Stats panel
  5. JSON report of top EV candidate zones

Usage:
    python ev_campus_analyzer.py --osm map.osm --output ./output

Key design principle:
    The large amenity=university boundary polygon is treated as a CAMPUS ZONE,
    not a building — it is skipped for building rendering. All features INSIDE
    it are classified individually by their own tags.
"""

import xml.etree.ElementTree as ET
import math, os, sys, argparse, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (multispectral band colours)
# ══════════════════════════════════════════════════════════════
C = {
    # ── Roads by hierarchy / width ──────────────────────────
    "road_primary":      "#E8320A",   # vivid red-orange   (> 12 m)
    "road_secondary":    "#F5961D",   # amber              (8–12 m)
    "road_tertiary":     "#F5D020",   # yellow             (6–8 m)
    "road_residential":  "#AAAAFF",   # periwinkle         (< 6 m)
    "road_service":      "#CCCCCC",   # light grey         (alleys)
    "road_footway":      "#88EE88",   # mint               (pedestrian)
    "road_path":         "#D2B48C",   # tan                (unpaved)
    "road_cycleway":     "#00CED1",   # dark turquoise
    "road_track":        "#B8860B",   # dark goldenrod

    # ── Buildings ───────────────────────────────────────────
    "bld_academic":      "#9B59B6",   # violet      (dept / lecture / lab)
    "bld_hostel":        "#3498DB",   # dodger blue (student residences)
    "bld_residential":   "#2980B9",   # steel blue  (staff / faculty housing)
    "bld_hospital":      "#E74C3C",   # red         (hospital / clinic / medical)
    "bld_worship":       "#E67E22",   # dark orange (temple / mandir)
    "bld_sports":        "#F39C12",   # golden      (gymnasium / stadium / pool)
    "bld_admin":         "#1ABC9C",   # teal        (admin / office / guest house)
    "bld_library":       "#16A085",   # dark teal   (library / museum)
    "bld_canteen":       "#E91E63",   # pink        (cafeteria / restaurant / cafe)
    "bld_generic":       "#7F8C8D",   # grey        (building=yes, unknown)
    "bld_gate":          "#BDC3C7",   # light grey  (gates)

    # ── Land use / natural ──────────────────────────────────
    "land_forest":       "#1E8449",   # dark green  (forest / wood)
    "land_grass":        "#58D68D",   # grass green (grass / meadow / lawn)
    "land_garden":       "#A9DFBF",   # pale green  (garden / flowerbed)
    "land_farmland":     "#F0E68C",   # khaki       (farmland / agriculture)
    "land_barren":       "#C8A96E",   # sandy brown (greenfield / scrub / wasteland)
    "land_recreation":   "#FAD7A0",   # peach       (sports ground / playground)
    "land_water":        "#1A9BFC",   # sky blue    (pond / lake / river)
    "land_wetland":      "#76D7EA",   # pale cyan   (wetland / marsh)
    "land_aquaculture":  "#0E6655",   # dark teal   (aquaculture / fish pond)
    "campus_boundary":   "#2C3E50",   # very dark   (campus outer polygon — outline only)

    # ── EV scoring ──────────────────────────────────────────
    "ev_high":           "#00FF88",
    "ev_medium":         "#FFD700",
    "ev_low":            "#FF4444",

    # ── Background ──────────────────────────────────────────
    "bg":                "#0D1117",
    "grid":              "#1A2030",
}

# ══════════════════════════════════════════════════════════════
#  OSM PARSER
# ══════════════════════════════════════════════════════════════
class OSMParser:
    def __init__(self, osm_path):
        print(f"[1/6] Parsing OSM file: {osm_path}")
        tree = ET.parse(osm_path)
        root = tree.getroot()

        bounds_el = root.find("bounds")
        self.bounds = ({k: float(v) for k, v in bounds_el.attrib.items()}
                       if bounds_el is not None else {})

        self.nodes = {}
        for n in root.findall("node"):
            lat, lon = n.get("lat"), n.get("lon")
            if lat and lon:
                self.nodes[n.get("id")] = (float(lat), float(lon))

        self.ways = []
        for w in root.findall("way"):
            tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
            nds  = [nd.get("ref") for nd in w.findall("nd")]
            self.ways.append({"id": w.get("id"), "tags": tags, "nodes": nds})

        print(f"   → {len(self.nodes):,} nodes | {len(self.ways):,} ways")

    def get_coords(self, node_ids):
        return [self.nodes[n] for n in node_ids if n in self.nodes]


# ══════════════════════════════════════════════════════════════
#  CLASSIFIER  — fine-grained, name-aware
# ══════════════════════════════════════════════════════════════

# Keywords in 'name' that identify specific building types
_NAME_HOSTEL  = {"hostel","hostel ","dorms","residency","accommodation",
                 "guest house","quarters","colony","apartment","apartments",
                 "flats","residential","dormitory"}
_NAME_TEMPLE  = {"mandir","temple","masjid","mosque","church","gurudwara","shrine"}
_NAME_SPORTS  = {"ground","stadium","pool","swimming","gymnasium","gymkhana",
                 "sports","court","track","pitch","arena","arena","athletic"}
_NAME_HOSPITAL= {"hospital","clinic","health","medical","dispensary","trauma",
                 "psychiatry","blood","mortuary","healthcare"}
_NAME_LIBRARY = {"library","museum","bhavan","bhavan","press","archive"}
_NAME_CANTEEN = {"canteen","cafeteria","cafe","restaurant","mess","food","juice",
                 "kitchen","dhaba"}
_NAME_ADMIN   = {"office","admin","administration","placement","proctor","vc ",
                 "vice chancellor","centre","center","international","conference",
                 "union","club","community","auditorium","hall"}
_NAME_GATE    = {"gate","entry","entrance"}
_NAME_FARM    = {"farm","nursery","garden","botanical","ayurvedic","mango","guava"}

def _name_match(name, keywords):
    n = name.lower()
    return any(k in n for k in keywords)

def _polygon_area_approx(xys):
    """Shoelace — returns signed area (we just need magnitude)."""
    if len(xys) < 3: return 0
    a = 0.0
    for i in range(len(xys)):
        x1, y1 = xys[i]
        x2, y2 = xys[(i+1) % len(xys)]
        a += x1*y2 - x2*y1
    return abs(a) / 2

def classify_way(tags, xys=None):
    """
    Returns (category_key, color, linewidth, zorder, legend_label)
    or (None, None, None, None, None) if way should be skipped.

    category_key is a string used internally for stats and 3D heights.
    """
    hw   = tags.get("highway", "")
    lu   = tags.get("landuse", "")
    nat  = tags.get("natural", "")
    bld  = tags.get("building", "")
    ww   = tags.get("waterway", "")
    lei  = tags.get("leisure", "")
    amen = tags.get("amenity", "")
    tour = tags.get("tourism", "")
    name = tags.get("name", "")
    hist = tags.get("historic", "")
    water= tags.get("water", "")
    man  = tags.get("man_made", "")
    shop = tags.get("shop", "")
    hc   = tags.get("healthcare", "")
    sport= tags.get("sport", "")

    # ── SKIP: campus boundary polygon (large amenity=university with no building tag)
    if amen == "university" and not bld:
        # Only skip if it looks like a big boundary (> some node threshold handled upstream)
        return ("campus_boundary", C["campus_boundary"], 1.5, 1, "Campus Boundary")

    # ════ ROADS ════
    if hw in ("primary", "primary_link"):
        return ("road_primary",    C["road_primary"],    3.5, 6, "Primary Road (>12 m)")
    if hw in ("secondary", "secondary_link"):
        return ("road_secondary",  C["road_secondary"],  2.5, 5, "Secondary Road (8–12 m)")
    if hw in ("tertiary", "tertiary_link"):
        return ("road_tertiary",   C["road_tertiary"],   2.0, 4, "Tertiary Road (6–8 m)")
    if hw in ("residential", "unclassified", "living_street"):
        return ("road_residential",C["road_residential"],1.3, 4, "Residential Road (<6 m)")
    if hw == "service":
        return ("road_service",    C["road_service"],    0.8, 3, "Service Road / Alley")
    if hw in ("footway", "pedestrian"):
        return ("road_footway",    C["road_footway"],    0.6, 3, "Footway / Pedestrian")
    if hw == "cycleway":
        return ("road_cycleway",   C["road_cycleway"],   0.8, 3, "Cycleway")
    if hw in ("path",):
        return ("road_path",       C["road_path"],       0.5, 3, "Path (Unpaved)")
    if hw in ("track",):
        return ("road_track",      C["road_track"],      0.6, 3, "Track / Dirt Road")
    if hw in ("rest_area", "construction"):
        return (None, None, None, None, None)  # skip minor junk

    # ════ WATER ════
    if nat == "water" or water in ("pond","basin","canal","reservoir") or ww in ("river","stream","canal","drain"):
        return ("land_water",  C["land_water"],  1.0, 3, "Water Body / Pond / River")
    if nat == "wetland":
        return ("land_wetland",C["land_wetland"],1.0, 2, "Wetland / Marsh")
    if lu == "aquaculture":
        return ("land_aquaculture", C["land_aquaculture"], 1.0, 2, "Aquaculture")

    # ════ NATURAL VEGETATION ════
    if nat in ("wood",) or lu in ("forest", "wood"):
        return ("land_forest", C["land_forest"], 1.0, 2, "Forest / Woodland")
    if nat in ("grassland", "scrub", "tree_row"):
        return ("land_grass",  C["land_grass"],  1.0, 2, "Grassland / Scrub")
    if lu in ("grass", "meadow"):
        return ("land_grass",  C["land_grass"],  1.0, 2, "Grass / Lawn")
    if lu in ("flowerbed",) or lei == "garden" or tags.get("garden:type"):
        return ("land_garden", C["land_garden"], 1.0, 2, "Garden / Flowerbed")
    if lei == "park":
        return ("land_garden", C["land_garden"], 1.0, 2, "Park / Garden")

    # ════ AGRICULTURE ════
    if lu in ("farmland", "farmyard", "orchard") or _name_match(name, _NAME_FARM):
        return ("land_farmland", C["land_farmland"], 1.0, 2, "Farmland / Nursery / Garden")

    # ════ BARREN / OPEN LAND ════
    if lu in ("greenfield", "brownfield", "construction", "industrial"):
        return ("land_barren", C["land_barren"], 1.0, 1, "Barren / Open Land")
    if nat == "scrub":
        return ("land_barren", C["land_barren"], 1.0, 1, "Scrubland")

    # ════ RECREATION / SPORTS (non-building) ════
    if lu == "recreation_ground":
        return ("land_recreation", C["land_recreation"], 1.0, 2, "Sports / Recreation Ground")
    if lei in ("pitch", "track", "golf_course", "firepit"):
        return ("land_recreation", C["land_recreation"], 1.0, 2, "Sports / Recreation Ground")
    if lei == "playground":
        return ("land_recreation", C["land_recreation"], 1.0, 2, "Playground")

    # ════ BUILDINGS — fine-grained ════
    if bld or amen or tour or hist or hc or shop or man:

        # ── GATE / BARRIER ──
        if bld == "gate" or hist == "city_gate" or _name_match(name, _NAME_GATE):
            return ("bld_gate", C["bld_gate"], 0.8, 7, "Gate / Entrance")

        # ── HOSPITAL / MEDICAL ──
        if (amen in ("hospital","clinic") or hc in ("hospital","yes") or
            bld == "hospital" or _name_match(name, _NAME_HOSPITAL)):
            return ("bld_hospital", C["bld_hospital"], 0.8, 7, "Hospital / Medical / Health Centre")

        # ── PLACE OF WORSHIP / TEMPLE ──
        if amen == "place_of_worship" or _name_match(name, _NAME_TEMPLE):
            return ("bld_worship", C["bld_worship"], 0.8, 7, "Temple / Place of Worship")

        # ── SPORTS FACILITY (building) ──
        if (lei in ("swimming_pool","sports_centre","sports_hall","fitness_centre","stadium") or
            sport or _name_match(name, _NAME_SPORTS)):
            return ("bld_sports", C["bld_sports"], 0.8, 7, "Sports Facility / Pool / Gymnasium")

        # ── LIBRARY / MUSEUM ──
        if amen == "library" or tour == "museum" or _name_match(name, _NAME_LIBRARY):
            return ("bld_library", C["bld_library"], 0.8, 7, "Library / Museum / Auditorium")

        # ── CANTEEN / FOOD ──
        if (amen in ("restaurant","cafe","fast_food","food_court") or
            shop in ("tea","yes") or _name_match(name, _NAME_CANTEEN)):
            return ("bld_canteen", C["bld_canteen"], 0.8, 7, "Canteen / Restaurant / Cafe")

        # ── HOSTEL / STUDENT ACCOMMODATION ──
        if (bld in ("apartments","dormitory") or
            tour in ("hostel","guest_house") or tags.get("guest_house") or
            _name_match(name, _NAME_HOSTEL)):
            return ("bld_hostel", C["bld_hostel"], 0.8, 7, "Hostel / Student Housing")

        # ── RESIDENTIAL (staff/faculty) ──
        if bld in ("house","residential","retail") or lu == "residential":
            return ("bld_residential", C["bld_residential"], 0.8, 6, "Residential (Staff / Faculty)")

        # ── ADMIN / OFFICE ──
        if (amen in ("community_centre","conference_centre","bank","post_office",
                     "parking","bicycle_parking","toilets","taxi","prep_school") or
            shop or man in ("water_tower","wastewater_plant","pipeline") or
            _name_match(name, _NAME_ADMIN)):
            return ("bld_admin", C["bld_admin"], 0.8, 7, "Admin / Office / Amenity")

        # ── ACADEMIC (departments, lecture theatres, labs) ──
        if (amen in ("college","school","university") or
            bld in ("college","school","university","public","barn","greenhouse") or
            lu == "education" or tags.get("office") == "educational_institution"):
            return ("bld_academic", C["bld_academic"], 0.8, 7, "Academic / Lecture / Lab")

        # ── HISTORIC ──
        if hist:
            return ("bld_admin", C["bld_admin"], 0.8, 7, "Historic Structure")

        # ── GENERIC BUILDING (building=yes, no other info) ──
        if bld == "yes" or bld:
            return ("bld_generic", C["bld_generic"], 0.8, 6, "Building (Unclassified)")

    return (None, None, None, None, None)


# ══════════════════════════════════════════════════════════════
#  EV SUITABILITY SCORER
# ══════════════════════════════════════════════════════════════
def ev_score(tags):
    score = 0.0
    hw   = tags.get("highway","")
    lu   = tags.get("landuse","")
    amen = tags.get("amenity","")
    bld  = tags.get("building","")
    name = tags.get("name","").lower()

    # Road access
    if hw in ("primary","secondary"):           score += 0.40
    elif hw in ("tertiary","unclassified"):      score += 0.28
    elif hw in ("residential","living_street"):  score += 0.20
    elif hw == "service":                        score += 0.15

    # High-traffic buildings
    if amen == "parking":                        score += 0.35
    if amen in ("university","college"):         score += 0.30
    if amen in ("hospital","library"):           score += 0.25
    if amen in ("community_centre","conference_centre"): score += 0.22
    if amen in ("restaurant","cafe","fast_food"):score += 0.12
    if amen in ("bank","post_office"):           score += 0.10

    # Land use
    if lu in ("education","residential"):        score += 0.20
    if lu == "recreation_ground":               score += 0.12
    if lu in ("grass","greenfield","farmland"):  score += 0.05

    # Building type
    if bld in ("university","college","school"): score += 0.25
    if bld in ("parking","garage"):              score += 0.30
    if bld in ("hospital",):                     score += 0.22

    # Name hints
    if any(k in name for k in ("parking","gate","entry","main","entrance")):
        score += 0.20
    if any(k in name for k in ("hospital","health","medical","trauma")):
        score += 0.15

    # Penalise water / wetland
    if tags.get("natural") == "water" or lu == "aquaculture": score -= 0.3
    if tags.get("waterway"):                     score -= 0.2

    return round(min(max(score, 0.0), 1.0), 3)


# ══════════════════════════════════════════════════════════════
#  COORDINATE NORMALISER
# ══════════════════════════════════════════════════════════════
class CoordNorm:
    def __init__(self, bounds):
        self.minlat = bounds.get("minlat", 25.258)
        self.minlon = bounds.get("minlon", 82.966)
        self.maxlat = bounds.get("maxlat", 25.283)
        self.maxlon = bounds.get("maxlon", 83.017)
        self.lat_range = self.maxlat - self.minlat or 1e-6
        self.lon_range = self.maxlon - self.minlon or 1e-6
        self.aspect = math.cos(math.radians((self.minlat + self.maxlat) / 2))

    def xy(self, lat, lon):
        x = (lon - self.minlon) / self.lon_range * self.aspect
        y = (lat - self.minlat) / self.lat_range
        return x, y

    def coords(self, pts):
        return [self.xy(lat, lon) for lat, lon in pts]

    def latlon(self, x, y):
        lat = self.minlat + y * self.lat_range
        lon = self.minlon + (x / self.aspect) * self.lon_range
        return round(lat, 6), round(lon, 6)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def centroid(xys):
    if not xys: return None, None
    return sum(x for x,_ in xys)/len(xys), sum(y for _,y in xys)/len(xys)

def is_closed(xys, tol=1e-7):
    if len(xys) < 3: return False
    return abs(xys[0][0]-xys[-1][0]) < tol and abs(xys[0][1]-xys[-1][1]) < tol

def poly_area(xys):
    a = 0.0
    for i in range(len(xys)):
        x1,y1 = xys[i]; x2,y2 = xys[(i+1)%len(xys)]
        a += x1*y2 - x2*y1
    return abs(a)/2


# ══════════════════════════════════════════════════════════════
#  MAP 1 — 2D MULTISPECTRAL
# ══════════════════════════════════════════════════════════════
def draw_2d_map(parser, norm, outpath):
    print("[2/6] Drawing 2D Multispectral Map …")
    fig, ax = plt.subplots(figsize=(20, 18), facecolor=C["bg"])
    ax.set_facecolor(C["bg"]); ax.set_aspect("equal"); ax.axis("off")

    legend_seen = {}
    stats = defaultdict(int)

    # Sort: campus boundary first, then land, then roads, then buildings
    RENDER_ORDER = {
        "campus_boundary": 0,
        "land_forest": 1, "land_grass": 1, "land_garden": 1,
        "land_farmland": 1, "land_barren": 1, "land_recreation": 1,
        "land_water": 2, "land_wetland": 2, "land_aquaculture": 2,
    }
    def sort_key(way):
        cat = classify_way(way["tags"])[0]
        if cat is None: return 99
        if cat.startswith("road"): return 5
        if cat.startswith("bld"): return 10
        return RENDER_ORDER.get(cat, 3)

    sorted_ways = sorted(parser.ways, key=sort_key)

    for way in sorted_ways:
        tags = way["tags"]
        pts  = parser.get_coords(way["nodes"])
        if len(pts) < 2: continue
        xys  = norm.coords(pts)
        cat, color, lw, zo, label = classify_way(tags, xys)
        if cat is None: continue

        xs = [x for x,_ in xys]; ys = [y for _,y in xys]

        if cat == "campus_boundary":
            # Draw only outline — do NOT fill
            if is_closed(xys):
                poly = plt.Polygon(xys, closed=True, facecolor="none",
                                   edgecolor="#4A90D9", linewidth=1.8,
                                   linestyle="--", zorder=1, alpha=0.7)
                ax.add_patch(poly)
            stats[cat] += 1

        elif cat.startswith("road"):
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=zo,
                    solid_capstyle="round", solid_joinstyle="round", alpha=0.95)
            stats["roads"] += 1

        elif cat.startswith("land"):
            if is_closed(xys) and len(xys) >= 3:
                poly = plt.Polygon(xys, closed=True, facecolor=color,
                                   edgecolor=color, linewidth=0.2,
                                   alpha=0.72, zorder=zo)
                ax.add_patch(poly)
            else:
                ax.plot(xs, ys, color=color, linewidth=lw or 1.0,
                        zorder=zo, alpha=0.8)
            stats[cat] += 1

        elif cat.startswith("bld"):
            if is_closed(xys) and len(xys) >= 3:
                poly = plt.Polygon(xys, closed=True, facecolor=color,
                                   edgecolor="#000000", linewidth=0.4,
                                   alpha=0.88, zorder=zo)
                ax.add_patch(poly)
                # Label if named and big enough
                name = tags.get("name","")
                if name and poly_area(xys) > 3e-5:
                    cx, cy = centroid(xys)
                    ax.text(cx, cy, name[:20], fontsize=4, color="white",
                            ha="center", va="center", zorder=zo+1,
                            fontweight="bold", alpha=0.85,
                            bbox=dict(boxstyle="round,pad=0.1",
                                      fc="black", ec="none", alpha=0.4))
            else:
                ax.plot(xs, ys, color=color, linewidth=0.8,
                        zorder=zo, alpha=0.9)
            stats[cat] += 1

        if label and label not in legend_seen:
            legend_seen[label] = mpatches.Patch(color=color, label=label)

    # Compass
    ax.annotate("N", xy=(0.975, 0.975), xycoords="axes fraction",
                fontsize=14, color="white", ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.3", fc="#1A2240", ec="white", lw=1.5))

    # Scale bar
    lon_km = norm.lon_range * 111 * norm.aspect
    bar_f = 0.5 / lon_km
    ax.annotate("", xy=(0.10 + bar_f, 0.018), xycoords="axes fraction",
                xytext=(0.10, 0.018),
                arrowprops=dict(arrowstyle="-", color="white", lw=2))
    ax.text(0.10 + bar_f/2, 0.024, "500 m", transform=ax.transAxes,
            color="white", ha="center", va="bottom", fontsize=9)

    ax.set_title(
        "BHU Campus — Fine-Grained Multispectral Classification Map\n"
        "EV Charging Station Site Analysis  |  Roads · Buildings · Land Use",
        color="white", fontsize=16, fontweight="bold", pad=14,
        fontfamily="monospace")

    # Legend in two columns
    handles = list(legend_seen.values())
    leg = ax.legend(handles=handles, loc="lower left", framealpha=0.88,
                    facecolor="#0A0F1E", edgecolor="#334466",
                    labelcolor="white", fontsize=7.5, ncol=2,
                    title="Land Classification", title_fontsize=9,
                    handlelength=1.4)
    leg.get_title().set_color("#AACCFF")

    plt.tight_layout(pad=0.3)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"   → Saved: {outpath}")
    return stats


# ══════════════════════════════════════════════════════════════
#  MAP 2 — EV SUITABILITY HEATMAP
# ══════════════════════════════════════════════════════════════
def draw_ev_heatmap(parser, norm, outpath):
    print("[3/6] Drawing EV Suitability Heatmap …")
    GRID = 250
    grid  = np.zeros((GRID, GRID))
    wgrid = np.zeros((GRID, GRID))

    sigma = max(4, int(GRID * 0.035))
    k_half = sigma * 2
    ks = np.arange(-k_half, k_half+1)
    KY, KX = np.meshgrid(ks, ks)
    kernel = np.exp(-(KX**2 + KY**2) / (2 * sigma**2))

    high_zones = []
    for way in parser.ways:
        tags  = way["tags"]
        pts   = parser.get_coords(way["nodes"])
        if not pts: continue
        xys   = norm.coords(pts)
        score = ev_score(tags)
        if score < 0.05: continue
        cx, cy = centroid(xys)
        if cx is None: continue
        gj = int(cx * (GRID-1)); gi = int(cy * (GRID-1))
        i0 = max(0, gi-k_half); i1 = min(GRID, gi+k_half+1)
        j0 = max(0, gj-k_half); j1 = min(GRID, gj+k_half+1)
        ki0 = i0-(gi-k_half); ki1 = ki0+(i1-i0)
        kj0 = j0-(gj-k_half); kj1 = kj0+(j1-j0)
        ki1 = min(ki1, kernel.shape[0]); kj1 = min(kj1, kernel.shape[1])
        i1 = i0+(ki1-ki0); j1 = j0+(kj1-kj0)
        if i1>i0 and j1>j0:
            grid [i0:i1, j0:j1] += score * kernel[ki0:ki1, kj0:kj1]
            wgrid[i0:i1, j0:j1] +=         kernel[ki0:ki1, kj0:kj1]
        if score >= 0.35:
            lat, lon = norm.latlon(cx, cy)
            high_zones.append((score, cx, cy, tags.get("name",""), lat, lon))

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(wgrid > 0, grid/wgrid, 0)

    fig, ax = plt.subplots(figsize=(18, 15), facecolor=C["bg"])
    ax.set_facecolor(C["bg"]); ax.set_aspect("equal"); ax.axis("off")

    cmap = LinearSegmentedColormap.from_list(
        "ev", ["#0D1117","#003355","#006699","#00BBAA","#FFD700","#FF6600","#FF0044"])
    ax.imshow(grid, extent=[0, norm.aspect, 0, 1],
              origin="lower", cmap=cmap, alpha=0.87, zorder=2,
              interpolation="bilinear")

    # Road overlay
    for way in parser.ways:
        hw = way["tags"].get("highway","")
        if hw not in ("primary","secondary","tertiary","residential"): continue
        pts = parser.get_coords(way["nodes"])
        if len(pts) < 2: continue
        xys = norm.coords(pts)
        ax.plot([x for x,_ in xys],[y for _,y in xys],
                color="white", lw=0.4, alpha=0.2, zorder=3)

    # High-score markers
    high_zones.sort(reverse=True)
    for score, cx, cy, name, lat, lon in high_zones[:40]:
        col = C["ev_high"] if score >= 0.55 else C["ev_medium"] if score >= 0.40 else C["ev_low"]
        ax.scatter(cx, cy, s=120, c=col, marker="*",
                   zorder=10, edgecolors="white", linewidths=0.6, alpha=0.95)
        if name:
            ax.text(cx, cy+0.007, name[:18], color="white", fontsize=5.5,
                    ha="center", zorder=11, alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.1",fc="black",ec="none",alpha=0.5))

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(0,1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.55)
    cbar.set_label("EV Suitability Score", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(
        f"EV Charging Station Suitability Heatmap — BHU Campus\n"
        f"★ = {len([z for z in high_zones if z[0]>=0.35])} candidate zones  "
        f"| ★ green ≥ 0.55  | ★ gold 0.40–0.55  | ★ red < 0.40",
        color="white", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout(pad=0.3)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"   → Saved: {outpath}")
    return high_zones


# ══════════════════════════════════════════════════════════════
#  MAP 3 — 3D MAP
# ══════════════════════════════════════════════════════════════

BLD_HEIGHTS = {
    "bld_academic":   0.060,
    "bld_hostel":     0.040,
    "bld_residential":0.030,
    "bld_hospital":   0.050,
    "bld_worship":    0.045,
    "bld_sports":     0.035,
    "bld_admin":      0.038,
    "bld_library":    0.042,
    "bld_canteen":    0.025,
    "bld_generic":    0.020,
    "bld_gate":       0.015,
}

ROAD_LW_3D = {
    "road_primary":    1.8,
    "road_secondary":  1.4,
    "road_tertiary":   1.1,
    "road_residential":0.8,
    "road_service":    0.5,
    "road_footway":    0.4,
    "road_path":       0.3,
    "road_cycleway":   0.5,
    "road_track":      0.4,
}

def draw_3d_map(parser, norm, outpath):
    print("[4/6] Drawing 3D Multispectral Map …")
    fig = plt.figure(figsize=(22, 15), facecolor=C["bg"])
    ax  = fig.add_subplot(111, projection="3d", facecolor=C["bg"])

    for spine in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        spine.fill = False
        spine.set_edgecolor("#1A2030")
    ax.grid(color="#1A2030", linewidth=0.4)
    ax.tick_params(colors="#555566", labelsize=7)

    LAND_Z = {
        "land_grass": 0.003, "land_garden": 0.004, "land_forest": 0.006,
        "land_farmland": 0.002, "land_barren": 0.001,
        "land_recreation": 0.003, "land_water": 0.000,
        "land_wetland": 0.001, "land_aquaculture": 0.001,
    }

    for way in parser.ways:
        tags = way["tags"]
        pts  = parser.get_coords(way["nodes"])
        if len(pts) < 2: continue
        xys  = norm.coords(pts)
        cat, color, lw, zo, _ = classify_way(tags, xys)
        if cat is None or cat == "campus_boundary": continue

        xs = [x for x,_ in xys]; ys = [y for _,y in xys]

        if cat in ROAD_LW_3D:
            ax.plot(xs, ys, [0.002]*len(xs),
                    color=color, linewidth=ROAD_LW_3D[cat], alpha=0.9)
            continue

        if cat in LAND_Z:
            if is_closed(xys) and len(xys) >= 3:
                zv = LAND_Z[cat]
                verts = [list(zip(xs, ys, [zv]*len(xs)))]
                pc = Poly3DCollection(verts, alpha=0.55,
                                      facecolor=color, edgecolor=color, linewidth=0.1)
                ax.add_collection3d(pc)
            continue

        if cat in BLD_HEIGHTS and is_closed(xys) and len(xys) >= 4:
            h = BLD_HEIGHTS[cat]
            xp = xs[:-1]; yp = ys[:-1]
            # Height override from OSM tag
            try:
                h = min(float(tags.get("height", h*1000))/1000, 0.12)
            except: pass

            # Top face
            top = [list(zip(xp, yp, [h]*len(xp)))]
            ax.add_collection3d(Poly3DCollection(top, alpha=0.85,
                                facecolor=color, edgecolor="#111111", linewidth=0.2))
            # Walls
            for i in range(len(xp)):
                j = (i+1) % len(xp)
                wall = [(xp[i],yp[i],0),(xp[j],yp[j],0),(xp[j],yp[j],h),(xp[i],yp[i],h)]
                shade = tuple(max(0, c*0.6) for c in matplotlib.colors.to_rgb(color)) + (0.65,)
                ax.add_collection3d(Poly3DCollection([wall], alpha=0.65,
                                    facecolor=shade, edgecolor="#000000", linewidth=0.1))

    ax.set_xlim(0, norm.aspect); ax.set_ylim(0, 1); ax.set_zlim(0, 0.13)
    ax.set_xlabel("Longitude →", color="#8899AA", labelpad=8, fontsize=9)
    ax.set_ylabel("Latitude →",  color="#8899AA", labelpad=8, fontsize=9)
    ax.set_zlabel("Height (relative)", color="#8899AA", labelpad=8, fontsize=9)
    ax.view_init(elev=36, azim=-52)
    ax.set_title(
        "3D Multispectral Campus Map — BHU Varanasi\n"
        "Buildings extruded by type  |  Road network  |  Land-use layers",
        color="white", fontsize=13, fontweight="bold", pad=18)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"   → Saved: {outpath}")


# ══════════════════════════════════════════════════════════════
#  MAP 4 — LEGEND & STATS
# ══════════════════════════════════════════════════════════════
def draw_legend_panel(stats, outpath):
    print("[5/6] Drawing Legend & Stats Panel …")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 10), facecolor=C["bg"])

    # ── LEFT: colour legend ──
    ax.set_facecolor(C["bg"]); ax.axis("off")
    ax.set_title("Spectral Classification Legend", color="white",
                 fontsize=13, fontweight="bold", pad=10)

    sections = [
        ("ROADS — by hierarchy / width", [
            ("Primary Road  (>12 m)",      C["road_primary"]),
            ("Secondary Road (8–12 m)",    C["road_secondary"]),
            ("Tertiary Road  (6–8 m)",     C["road_tertiary"]),
            ("Residential Road (<6 m)",    C["road_residential"]),
            ("Service Road / Alley",       C["road_service"]),
            ("Footway / Pedestrian",       C["road_footway"]),
            ("Cycleway",                   C["road_cycleway"]),
            ("Path / Track (Unpaved)",     C["road_path"]),
        ]),
        ("BUILDINGS — by function", [
            ("Academic / Dept / Lab",      C["bld_academic"]),
            ("Hostel / Student Housing",   C["bld_hostel"]),
            ("Residential (Staff/Faculty)",C["bld_residential"]),
            ("Hospital / Medical",         C["bld_hospital"]),
            ("Temple / Place of Worship",  C["bld_worship"]),
            ("Sports / Pool / Gymnasium",  C["bld_sports"]),
            ("Library / Museum / Auditm.", C["bld_library"]),
            ("Admin / Office / Amenity",   C["bld_admin"]),
            ("Canteen / Cafe / Restaurant",C["bld_canteen"]),
            ("Building (Unclassified)",    C["bld_generic"]),
        ]),
        ("LAND USE & NATURAL", [
            ("Forest / Woodland",          C["land_forest"]),
            ("Grass / Grassland / Lawn",   C["land_grass"]),
            ("Garden / Park / Flowerbed",  C["land_garden"]),
            ("Farmland / Nursery",         C["land_farmland"]),
            ("Barren / Scrub / Open Land", C["land_barren"]),
            ("Sports / Recreation Ground", C["land_recreation"]),
            ("Water Body / River / Pond",  C["land_water"]),
            ("Wetland / Marsh",            C["land_wetland"]),
            ("Aquaculture",                C["land_aquaculture"]),
        ]),
        ("EV SUITABILITY", [
            ("High Priority  (≥ 0.55)",    C["ev_high"]),
            ("Medium Priority (0.40–0.55)",C["ev_medium"]),
            ("Lower Priority  (<0.40)",    C["ev_low"]),
        ]),
    ]

    y = 0.98
    for sec_title, items in sections:
        ax.text(0.03, y, sec_title, transform=ax.transAxes,
                color="#88AAFF", fontsize=8.5, fontweight="bold", va="top")
        y -= 0.038
        for label, color in items:
            rect = mpatches.FancyBboxPatch(
                (0.03, y-0.022), 0.055, 0.020,
                transform=ax.transAxes, clip_on=False,
                boxstyle="round,pad=0.002",
                facecolor=color, edgecolor="white", linewidth=0.4)
            ax.add_patch(rect)
            ax.text(0.10, y-0.012, label, transform=ax.transAxes,
                    color="white", fontsize=7.5, va="center")
            y -= 0.030
        y -= 0.012

    # ── RIGHT: stats ──
    ax2.set_facecolor(C["bg"])
    for sp in ax2.spines.values(): sp.set_color("#334466")
    ax2.tick_params(colors="white", labelsize=8)
    ax2.xaxis.label.set_color("white")
    ax2.set_title("Feature Count by Category", color="white",
                  fontsize=13, fontweight="bold")

    STAT_COLORS = {
        "roads": C["road_primary"],
        "campus_boundary": "#4A90D9",
        "bld_academic": C["bld_academic"],
        "bld_hostel": C["bld_hostel"],
        "bld_residential": C["bld_residential"],
        "bld_hospital": C["bld_hospital"],
        "bld_worship": C["bld_worship"],
        "bld_sports": C["bld_sports"],
        "bld_library": C["bld_library"],
        "bld_admin": C["bld_admin"],
        "bld_canteen": C["bld_canteen"],
        "bld_generic": C["bld_generic"],
        "bld_gate": C["bld_gate"],
        "land_forest": C["land_forest"],
        "land_grass": C["land_grass"],
        "land_garden": C["land_garden"],
        "land_farmland": C["land_farmland"],
        "land_barren": C["land_barren"],
        "land_recreation": C["land_recreation"],
        "land_water": C["land_water"],
        "land_wetland": C["land_wetland"],
        "land_aquaculture": C["land_aquaculture"],
    }

    cats   = [k for k in stats if stats[k] > 0]
    counts = [stats[k] for k in cats]
    colors = [STAT_COLORS.get(k, "#888888") for k in cats]

    bars = ax2.barh(cats, counts, color=colors, edgecolor="#111122", linewidth=0.4)
    for bar, val in zip(bars, counts):
        ax2.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                 str(val), va="center", ha="left", color="white", fontsize=8)
    ax2.set_xlabel("Count", color="white", fontsize=10)
    ax2.set_facecolor(C["bg"])

    plt.suptitle(
        "EV Charging Site Analysis — BHU Campus Varanasi\nMultispectral OSM Feature Classification Summary",
        color="white", fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"   → Saved: {outpath}")


# ══════════════════════════════════════════════════════════════
#  JSON REPORT
# ══════════════════════════════════════════════════════════════
def write_report(parser, norm, high_zones, outpath):
    print("[6/6] Writing JSON analysis report …")
    report = {
        "campus": "Banaras Hindu University (BHU), Varanasi, UP, India",
        "bounds": parser.bounds,
        "total_ways": len(parser.ways),
        "total_nodes": len(parser.nodes),
        "data_notes": [
            "Campus boundary (amenity=university, 126 nodes) rendered as outline only — not filled.",
            "building=yes with no further tags classified as 'Building (Unclassified)'.",
            "EV score based on road access, amenity type, building function, and name keywords.",
            "Height data sparse — most buildings use default heights by category.",
            "Some features may be misclassified if OSM tags are incomplete or absent.",
        ],
        "high_ev_zones": [],
        "road_type_counts": defaultdict(int),
        "building_type_counts": defaultdict(int),
        "landuse_counts": defaultdict(int),
    }

    for way in parser.ways:
        tags = way["tags"]
        hw = tags.get("highway","")
        if hw: report["road_type_counts"][hw] += 1
        if tags.get("building"): report["building_type_counts"][tags["building"]] += 1
        if tags.get("landuse"): report["landuse_counts"][tags["landuse"]] += 1

    for score, cx, cy, name, lat, lon in sorted(high_zones, reverse=True)[:50]:
        report["high_ev_zones"].append({
            "name": name,
            "score": score,
            "lat": lat,
            "lon": lon,
            "priority": "HIGH" if score>=0.55 else "MEDIUM" if score>=0.40 else "LOW",
        })

    report["road_type_counts"]    = dict(report["road_type_counts"])
    report["building_type_counts"]= dict(report["building_type_counts"])
    report["landuse_counts"]      = dict(report["landuse_counts"])

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"   → {len(report['high_ev_zones'])} EV candidate zones saved.")
    return report


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="EV Charging Station Site Analysis from any .osm file")
    ap.add_argument("--osm",    default="map.osm",
                    help="Path to input .osm file (default: map.osm)")
    ap.add_argument("--output", default="./output",
                    help="Output directory (default: ./output)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    parser = OSMParser(args.osm)
    norm   = CoordNorm(parser.bounds)

    stats      = draw_2d_map(parser, norm,
                    os.path.join(args.output, "01_multispectral_map.png"))
    high_zones = draw_ev_heatmap(parser, norm,
                    os.path.join(args.output, "02_ev_suitability_heatmap.png"))
    draw_3d_map(parser, norm,
                    os.path.join(args.output, "03_3d_map.png"))
    draw_legend_panel(stats,
                    os.path.join(args.output, "04_legend_stats.png"))
    report = write_report(parser, norm, high_zones,
                    os.path.join(args.output, "05_ev_analysis_report.json"))

    print("\n✅  Done. Top EV zones:")
    for z in report["high_ev_zones"][:5]:
        print(f"   [{z['priority']}]  score={z['score']}  "
              f"lat={z['lat']}  lon={z['lon']}  name='{z['name']}'")

if __name__ == "__main__":
    main()