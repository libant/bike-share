"""
Phase 3b — Bus frequency layer + dual-gap map.

Identifies neighbourhoods that are BOTH bike-poor AND bus-poor:
the highest-priority candidates for new bike station investment.

Outputs:
  data/analysis_data/neighbourhood_gap_matrix.csv — bike coverage + bus frequency per neighbourhood
  other/charts/map_dual_gap.html                  — choropleth: bike × bus gap quadrant
  other/charts/chart_gap_quadrant.html            — scatter quadrant: every neighbourhood plotted
"""

import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR     = Path(__file__).parent.parent / "data" / "raw_data"
ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "analysis_data"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR      = Path(__file__).parent.parent / "other" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UTM17N = "EPSG:32617"
WGS84  = "EPSG:4326"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Compute weekday bus service frequency per stop
# ─────────────────────────────────────────────────────────────────────────────

print("Computing weekday bus service frequency per stop...")

trips_gtfs = pd.read_csv(DATA_DIR / "ttc_gtfs/trips.txt", usecols=["trip_id", "service_id", "route_id"])
routes     = pd.read_csv(DATA_DIR / "ttc_gtfs/routes.txt", usecols=["route_id", "route_type"])

# Weekday service = service_id 1; exclude subway (route_type 1) — want surface transit only
weekday_trip_ids = trips_gtfs[
    (trips_gtfs["service_id"] == 1)
].merge(routes[routes["route_type"] != 1], on="route_id")["trip_id"]

print(f"  {len(weekday_trip_ids):,} weekday surface-transit trips")

# Count departures per stop on a typical weekday
stop_times = pd.read_csv(
    DATA_DIR / "ttc_gtfs/stop_times.txt",
    usecols=["trip_id", "stop_id"]
)

weekday_stop_times = stop_times[stop_times["trip_id"].isin(weekday_trip_ids)]
trips_per_stop = (
    weekday_stop_times.groupby("stop_id")
    .size()
    .reset_index(name="daily_trips")
)

print(f"  {len(trips_per_stop):,} stops with weekday service")
print(f"  Median daily trips per stop: {trips_per_stop['daily_trips'].median():.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Spatially assign stops → neighbourhoods
# ─────────────────────────────────────────────────────────────────────────────

print("\nAssigning stops to neighbourhoods...")

stops_all = pd.read_csv(DATA_DIR / "ttc_gtfs/stops.txt",
                        usecols=["stop_id", "stop_lat", "stop_lon"])
stops_all = stops_all.merge(trips_per_stop, on="stop_id", how="inner")

gdf_stops = gpd.GeoDataFrame(
    stops_all,
    geometry=gpd.points_from_xy(stops_all["stop_lon"], stops_all["stop_lat"]),
    crs=WGS84
).to_crs(UTM17N)

gdf_nbhd = gpd.read_file(DATA_DIR / "neighbourhoods.geojson").to_crs(UTM17N)
gdf_nbhd["area_km2"] = (gdf_nbhd.geometry.area / 1e6).round(3)

stops_in_nbhd = gpd.sjoin(gdf_stops, gdf_nbhd[["AREA_NAME", "area_km2", "geometry"]],
                           how="inner", predicate="within")

bus_by_nbhd = (
    stops_in_nbhd.groupby("AREA_NAME")
    .agg(
        bus_stops        = ("stop_id", "count"),
        total_daily_trips= ("daily_trips", "sum"),
        avg_trips_per_stop=("daily_trips", "mean"),
    )
    .reset_index()
)

# Stop density: stops per km²
nbhd_areas = gdf_nbhd[["AREA_NAME", "area_km2"]].copy()
bus_by_nbhd = bus_by_nbhd.merge(nbhd_areas, on="AREA_NAME", how="left")
bus_by_nbhd["stop_density"] = (bus_by_nbhd["bus_stops"] / bus_by_nbhd["area_km2"]).round(2)

print(f"  {len(bus_by_nbhd)} neighbourhoods with bus data")
print(f"  Median avg trips/stop/day: {bus_by_nbhd['avg_trips_per_stop'].median():.0f}")
print(f"  Median stop density: {bus_by_nbhd['stop_density'].median():.1f} stops/km²")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Merge with bike coverage
# ─────────────────────────────────────────────────────────────────────────────

print("\nMerging bike coverage + bus frequency...")

bike_cov = pd.read_csv(ANALYSIS_DIR / "neighbourhood_coverage.csv",
                       usecols=["AREA_NAME", "coverage_pct", "dist_to_downtown_km"])

matrix = bike_cov.merge(bus_by_nbhd, on="AREA_NAME", how="left")

# Fill neighbourhoods with zero bus stops
matrix["bus_stops"]          = matrix["bus_stops"].fillna(0).astype(int)
matrix["avg_trips_per_stop"] = matrix["avg_trips_per_stop"].fillna(0)
matrix["stop_density"]       = matrix["stop_density"].fillna(0)

# ── Gap quadrant classification ──────────────────────────────────────────────
# Thresholds: median of each metric across all 158 neighbourhoods
cov_median = matrix["coverage_pct"].median()
freq_median = matrix["avg_trips_per_stop"].median()

def classify(row):
    bike_ok = row["coverage_pct"] >= cov_median
    bus_ok  = row["avg_trips_per_stop"] >= freq_median
    if not bike_ok and not bus_ok:
        return "Dual gap\n(bike-poor + bus-poor)"
    elif not bike_ok and bus_ok:
        return "Bike gap only\n(bus-rich)"
    elif bike_ok and not bus_ok:
        return "Bus gap only\n(bike-rich)"
    else:
        return "Well served"

matrix["gap_quadrant"] = matrix.apply(classify, axis=1)

print(f"\n  Bike coverage median: {cov_median:.1f}%")
print(f"  Bus frequency median: {freq_median:.0f} trips/stop/day")
print("\n  Gap quadrant counts:")
print(matrix["gap_quadrant"].value_counts().to_string())

dual_gap = matrix[matrix["gap_quadrant"].str.startswith("Dual")].sort_values("dist_to_downtown_km")
print(f"\n  {len(dual_gap)} DUAL-GAP neighbourhoods (highest priority for investment):")
for _, r in dual_gap.iterrows():
    print(f"    {r['AREA_NAME']:<40}  cov: {r['coverage_pct']:>5.1f}%  "
          f"bus: {r['avg_trips_per_stop']:>5.0f} trips/stop  "
          f"{r['dist_to_downtown_km']:.1f}km from downtown")

matrix.to_csv(ANALYSIS_DIR / "neighbourhood_gap_matrix.csv", index=False)
print(f"\nSaved neighbourhood_gap_matrix.csv")

# ─────────────────────────────────────────────────────────────────────────────
# CHART — Scatter quadrant: every neighbourhood
# ─────────────────────────────────────────────────────────────────────────────

QUAD_COLORS = {
    "Dual gap\n(bike-poor + bus-poor)":  "#d62728",   # red — priority
    "Bike gap only\n(bus-rich)":         "#ff7f0e",   # orange
    "Bus gap only\n(bike-rich)":         "#1f77b4",   # blue
    "Well served":                        "#2ca02c",   # green
}

fig1 = px.scatter(
    matrix,
    x="avg_trips_per_stop",
    y="coverage_pct",
    color="gap_quadrant",
    color_discrete_map=QUAD_COLORS,
    hover_name="AREA_NAME",
    hover_data={
        "coverage_pct": ":.1f",
        "avg_trips_per_stop": ":.0f",
        "dist_to_downtown_km": ":.1f",
        "gap_quadrant": False,
    },
    size="dist_to_downtown_km",
    size_max=18,
    title="Toronto Neighbourhood Transit Gap Matrix — 2023<br>"
          "<sup>Red = bike-poor AND bus-poor (highest need); Bubble size = distance from downtown</sup>",
    labels={
        "avg_trips_per_stop": "Bus Service Frequency (avg weekday trips per stop)",
        "coverage_pct":       "Bike Share Coverage (% area within 500m of station)",
        "dist_to_downtown_km":"km from downtown",
        "gap_quadrant":       "Category",
    },
)
fig1.add_vline(x=freq_median, line_dash="dot", line_color="grey")
fig1.add_hline(y=cov_median,  line_dash="dot", line_color="grey")
fig1.add_annotation(x=freq_median * 0.15, y=cov_median * 0.12,
                    text="← Dual gap zone\n(invest here first)",
                    showarrow=False, font=dict(color="#d62728", size=11))
fig1.update_layout(height=560, legend=dict(orientation="h", y=-0.18, title=""))
fig1.write_html(OUT_DIR / "chart_gap_quadrant.html")
print("Saved chart_gap_quadrant.html")

# ─────────────────────────────────────────────────────────────────────────────
# MAP — Dual-gap choropleth
# ─────────────────────────────────────────────────────────────────────────────

gdf_map = gdf_nbhd.merge(matrix, on="AREA_NAME", how="left").to_crs(WGS84)

# Encode quadrant as numeric for colour scale
quad_order = {
    "Dual gap\n(bike-poor + bus-poor)": 0,
    "Bike gap only\n(bus-rich)":        1,
    "Bus gap only\n(bike-rich)":        2,
    "Well served":                      3,
}
gdf_map["quad_num"]   = gdf_map["gap_quadrant"].map(quad_order)
gdf_map["quad_label"] = gdf_map["gap_quadrant"].str.replace("\n", " ")

fig2 = px.choropleth_map(
    gdf_map,
    geojson=gdf_map.__geo_interface__,
    locations=gdf_map.index,
    color="quad_label",
    color_discrete_map={
        "Dual gap (bike-poor + bus-poor)": "#d62728",
        "Bike gap only (bus-rich)":        "#ff7f0e",
        "Bus gap only (bike-rich)":        "#1f77b4",
        "Well served":                     "#2ca02c",
    },
    category_orders={"quad_label": [
        "Dual gap (bike-poor + bus-poor)",
        "Bike gap only (bus-rich)",
        "Bus gap only (bike-rich)",
        "Well served",
    ]},
    zoom=10,
    center={"lat": 43.718, "lon": -79.383},
    map_style="carto-positron",
    hover_name="AREA_NAME",
    hover_data={
        "coverage_pct":        ":.1f",
        "avg_trips_per_stop":  ":.0f",
        "dist_to_downtown_km": ":.1f",
        "quad_label":          False,
        "quad_num":            False,
    },
    opacity=0.75,
    title="Where Should Bike Share Expand? Dual Transit Gap Analysis<br>"
          "<sup>Red = bike-poor + bus-poor → highest ROI for new stations</sup>",
    labels={
        "quad_label":          "Category",
        "coverage_pct":        "Bike coverage %",
        "avg_trips_per_stop":  "Bus trips/stop/day",
        "dist_to_downtown_km": "km from downtown",
    },
)
fig2.update_layout(height=760, margin=dict(t=60, b=0, l=0, r=0),
                   legend=dict(title="", orientation="h", y=-0.05))
fig2.write_html(OUT_DIR / "map_dual_gap.html")
print("Saved map_dual_gap.html")

print("\nDone.")
