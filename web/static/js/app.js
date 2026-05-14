/* ─────────────────────────────────────────────────────────────────
   EV Charging Station Dashboard — Frontend Logic
   ───────────────────────────────────────────────────────────────── */

// ── State ─────────────────────────────────────────────────────────
let map;
let heatmapLayer = null;
let candidatesLayer = null;
let roadsLayer = null;
let drawControl = null;
let drawnItems = null;
let currentData = null;
let isDrawMode = false;

// ── Tier Colors ───────────────────────────────────────────────────
const TIER_COLORS = {
    HIGH: "#10b981",
    MEDIUM: "#f59e0b",
    LOW: "#ef4444",
};

const ROAD_COLORS = {
    primary: "#ff4444",
    secondary: "#ff8800",
    tertiary: "#ffdd00",
    residential: "#8888ff",
};

// ── Initialize ────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initMap();
    initDropzone();
    initControls();
    checkHealth();
});

function initMap() {
    // Dark tile layer
    const darkTiles = L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
            maxZoom: 19,
        }
    );

    const lightTiles = L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
            maxZoom: 19,
        }
    );

    map = L.map("map", {
        center: [20.5937, 78.9629],  // India center
        zoom: 5,
        layers: [darkTiles],
        zoomControl: false,
    });

    // Zoom control top-right
    L.control.zoom({ position: "topright" }).addTo(map);

    // Layer control
    L.control.layers(
        { "Dark": darkTiles, "Light": lightTiles },
        null,
        { position: "topright" }
    ).addTo(map);

    // Drawn items layer for bbox
    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
}

function initDropzone() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith(".osm")) {
            uploadFile(file);
        } else {
            alert("Please drop a .osm file");
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) {
            uploadFile(fileInput.files[0]);
        }
    });
}

function initControls() {
    // Draw button
    document.getElementById("btn-draw").addEventListener("click", toggleDrawMode);

    // Layer toggles
    document.getElementById("toggle-heatmap").addEventListener("change", (e) => {
        if (heatmapLayer) {
            e.target.checked ? map.addLayer(heatmapLayer) : map.removeLayer(heatmapLayer);
        }
    });
    document.getElementById("toggle-candidates").addEventListener("change", (e) => {
        if (candidatesLayer) {
            e.target.checked ? map.addLayer(candidatesLayer) : map.removeLayer(candidatesLayer);
        }
    });
    document.getElementById("toggle-roads").addEventListener("change", (e) => {
        if (roadsLayer) {
            e.target.checked ? map.addLayer(roadsLayer) : map.removeLayer(roadsLayer);
        }
    });

    // Score slider
    const slider = document.getElementById("score-filter");
    const valueLabel = document.getElementById("score-value");
    slider.addEventListener("input", () => {
        const val = slider.value / 100;
        valueLabel.textContent = val.toFixed(2);
        filterByScore(val);
    });

    // Reset
    document.getElementById("btn-reset").addEventListener("click", resetMap);
}

async function checkHealth() {
    try {
        const resp = await fetch("/api/model-info");
        const data = await resp.json();
        if (data.status === "ok") {
            document.getElementById("status-dot").classList.add("online");
            document.getElementById("status-text").textContent = "Model Ready";
            document.getElementById("model-badge").textContent =
                data.model_type + " | " + data.n_features + " features";
        }
    } catch {
        document.getElementById("status-text").textContent = "Offline";
    }
}

// ── File Upload ───────────────────────────────────────────────────
async function uploadFile(file) {
    showLoading("Uploading " + file.name + "...");

    const formData = new FormData();
    formData.append("file", file);

    try {
        showLoading("Running ML prediction pipeline...");
        const resp = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });
        const data = await resp.json();

        if (data.error) {
            alert("Error: " + data.error);
            hideLoading();
            return;
        }

        renderResults(data);
    } catch (err) {
        alert("Upload failed: " + err.message);
    } finally {
        hideLoading();
    }
}

// ── Draw Mode ─────────────────────────────────────────────────────
function toggleDrawMode() {
    const btn = document.getElementById("btn-draw");

    if (isDrawMode) {
        // Exit draw mode
        if (drawControl) {
            map.removeControl(drawControl);
            drawControl = null;
        }
        isDrawMode = false;
        btn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
            </svg>
            Draw on Map
        `;
        return;
    }

    isDrawMode = true;
    btn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
        Cancel Drawing
    `;

    drawControl = new L.Control.Draw({
        position: "topright",
        draw: {
            polygon: false,
            polyline: false,
            circle: false,
            circlemarker: false,
            marker: false,
            rectangle: {
                shapeOptions: {
                    color: "#6366f1",
                    weight: 2,
                    fillOpacity: 0.1,
                },
            },
        },
        edit: false,
    });
    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, async (e) => {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);

        const bounds = e.layer.getBounds();
        const south = bounds.getSouth();
        const west = bounds.getWest();
        const north = bounds.getNorth();
        const east = bounds.getEast();

        // Remove draw control
        if (drawControl) {
            map.removeControl(drawControl);
            drawControl = null;
        }
        isDrawMode = false;
        document.getElementById("btn-draw").innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
            </svg>
            Draw on Map
        `;

        showLoading("Downloading OSM data & predicting...");
        try {
            const resp = await fetch("/api/predict-bbox", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ south, west, north, east }),
            });
            const data = await resp.json();
            if (data.error) {
                alert("Error: " + data.error);
                return;
            }
            renderResults(data);
        } catch (err) {
            alert("Prediction failed: " + err.message);
        } finally {
            hideLoading();
        }
    });
}

// ── Render Results ────────────────────────────────────────────────
function renderResults(data) {
    currentData = data;
    clearLayers();

    const bbox = data.bbox;
    map.fitBounds([
        [bbox.south, bbox.west],
        [bbox.north, bbox.east],
    ], { padding: [30, 30] });

    // Heatmap layer (colored rectangles)
    renderHeatmap(data.heatmap);

    // Candidates layer
    renderCandidates(data.candidates);

    // Roads layer
    renderRoads(data.roads);

    // Show controls & results panels
    document.getElementById("controls-panel").style.display = "block";
    document.getElementById("results-panel").style.display = "block";

    // Render stats
    renderStats(data.stats);

    // Render candidates list
    renderCandidatesList(data.candidates);
}

function renderHeatmap(geojson) {
    heatmapLayer = L.geoJSON(geojson, {
        style: (feature) => {
            const score = feature.properties.score;
            return {
                fillColor: scoreToColor(score),
                fillOpacity: Math.min(0.7, score * 0.9),
                weight: 0,
                stroke: false,
            };
        },
    });
    if (document.getElementById("toggle-heatmap").checked) {
        heatmapLayer.addTo(map);
    }
}

function renderCandidates(geojson) {
    candidatesLayer = L.geoJSON(geojson, {
        pointToLayer: (feature, latlng) => {
            const tier = feature.properties.tier;
            const color = TIER_COLORS[tier] || TIER_COLORS.LOW;
            return L.circleMarker(latlng, {
                radius: 10,
                fillColor: color,
                color: "#ffffff",
                weight: 2,
                fillOpacity: 0.9,
            });
        },
        onEachFeature: (feature, layer) => {
            const p = feature.properties;
            const tierClass = p.tier.toLowerCase();
            layer.bindPopup(`
                <div class="popup-title">#${p.rank} Candidate Location</div>
                <div class="popup-row">
                    <span class="popup-label">Score:</span>
                    <span class="popup-value candidate-score ${tierClass}">${p.score.toFixed(3)}</span>
                </div>
                <div class="popup-row">
                    <span class="popup-label">Priority:</span>
                    <span class="popup-value">${p.tier}</span>
                </div>
                <div class="popup-row">
                    <span class="popup-label">Lat:</span>
                    <span class="popup-value">${feature.geometry.coordinates[1].toFixed(5)}</span>
                </div>
                <div class="popup-row">
                    <span class="popup-label">Lon:</span>
                    <span class="popup-value">${feature.geometry.coordinates[0].toFixed(5)}</span>
                </div>
            `);
        },
    });
    if (document.getElementById("toggle-candidates").checked) {
        candidatesLayer.addTo(map);
    }
}

function renderRoads(geojson) {
    roadsLayer = L.geoJSON(geojson, {
        pointToLayer: (feature, latlng) => {
            const hw = feature.properties.highway;
            const color = ROAD_COLORS[hw] || "#888";
            return L.circleMarker(latlng, {
                radius: 2,
                fillColor: color,
                color: color,
                weight: 1,
                fillOpacity: 0.4,
            });
        },
    });
    if (document.getElementById("toggle-roads").checked) {
        roadsLayer.addTo(map);
    }
}

function renderStats(stats) {
    const grid = document.getElementById("stats-grid");
    grid.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${stats.candidates}</div>
            <div class="stat-label">Candidates</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: var(--green)">${stats.max_score}</div>
            <div class="stat-label">Top Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${stats.grid_cells.toLocaleString()}</div>
            <div class="stat-label">Grid Cells</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${stats.nodes.toLocaleString()}</div>
            <div class="stat-label">Map Nodes</div>
        </div>
    `;
}

function renderCandidatesList(geojson) {
    const list = document.getElementById("candidates-list");
    list.innerHTML = "";

    geojson.features.forEach((f) => {
        const p = f.properties;
        const coords = f.geometry.coordinates;
        const tierClass = p.tier.toLowerCase();

        const row = document.createElement("div");
        row.className = "candidate-row";
        row.innerHTML = `
            <span class="candidate-rank">#${p.rank}</span>
            <span class="candidate-score ${tierClass}">${p.score.toFixed(3)}</span>
            <span class="candidate-coords">${coords[1].toFixed(4)}, ${coords[0].toFixed(4)}</span>
        `;
        row.addEventListener("click", () => {
            map.setView([coords[1], coords[0]], 17);
            // Find and open the popup
            if (candidatesLayer) {
                candidatesLayer.eachLayer((layer) => {
                    if (layer.feature === f) {
                        layer.openPopup();
                    }
                });
            }
        });
        list.appendChild(row);
    });
}

// ── Helpers ───────────────────────────────────────────────────────
function scoreToColor(score) {
    // Plasma-like colormap
    if (score >= 0.7) return "#f0f921";
    if (score >= 0.55) return "#fcce25";
    if (score >= 0.45) return "#f1844b";
    if (score >= 0.35) return "#cc4778";
    if (score >= 0.25) return "#9c179e";
    if (score >= 0.15) return "#6a00a8";
    if (score >= 0.08) return "#3b0f70";
    return "#0d0887";
}

function filterByScore(minScore) {
    if (!currentData) return;

    // Filter heatmap
    if (heatmapLayer) {
        map.removeLayer(heatmapLayer);
        const filtered = {
            type: "FeatureCollection",
            features: currentData.heatmap.features.filter(
                (f) => f.properties.score >= minScore
            ),
        };
        heatmapLayer = L.geoJSON(filtered, {
            style: (feature) => {
                const score = feature.properties.score;
                return {
                    fillColor: scoreToColor(score),
                    fillOpacity: Math.min(0.7, score * 0.9),
                    weight: 0,
                    stroke: false,
                };
            },
        });
        if (document.getElementById("toggle-heatmap").checked) {
            heatmapLayer.addTo(map);
        }
    }

    // Filter candidates
    if (candidatesLayer) {
        candidatesLayer.eachLayer((layer) => {
            const score = layer.feature.properties.score;
            if (score < minScore) {
                layer.setStyle({ fillOpacity: 0.1, opacity: 0.2 });
            } else {
                layer.setStyle({ fillOpacity: 0.9, opacity: 1 });
            }
        });
    }
}

function clearLayers() {
    if (heatmapLayer) { map.removeLayer(heatmapLayer); heatmapLayer = null; }
    if (candidatesLayer) { map.removeLayer(candidatesLayer); candidatesLayer = null; }
    if (roadsLayer) { map.removeLayer(roadsLayer); roadsLayer = null; }
    drawnItems.clearLayers();
}

function resetMap() {
    clearLayers();
    currentData = null;
    map.setView([20.5937, 78.9629], 5);
    document.getElementById("controls-panel").style.display = "none";
    document.getElementById("results-panel").style.display = "none";
    document.getElementById("score-filter").value = 0;
    document.getElementById("score-value").textContent = "0.00";
}

function showLoading(text) {
    document.getElementById("loading-text").textContent = text || "Analyzing map...";
    document.getElementById("loading-overlay").style.display = "flex";
}

function hideLoading() {
    document.getElementById("loading-overlay").style.display = "none";
}
