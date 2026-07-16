"""
Phase 3 — "The 11-Minute City meets the Coverage Gap"

Narrative:
  Bike Share works as a transit connector — but only where transit already is.
  The neighbourhoods that need last-mile connections the most have no bikes.

Analyses:
  1. Transit proximity:  how close are bike stations to subway stations?
  2. Trip connector proof: what % of trips start within walking distance of a subway?
  3. Coverage gap:        which neighbourhoods lack bike coverage?
  4. The equity frame:   coverage gap vs. distance-from-downtown proxy

Outputs:
  other/charts/map_transit_coverage.html      — master map: stations, subway, coverage buffers
  other/charts/chart_station_subway_dist.html — distribution: bike station → nearest subway distance
  other/charts/chart_trip_proximity.html      — % of trips by proximity bucket to subway
  other/charts/chart_coverage_vs_downtown.html — coverage % vs distance from downtown
  data/analysis_data/neighbourhood_coverage.csv — neighbourhood-level coverage table
"""

import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point
from pathlib import Path

DATA_DIR     = Path(__file__).parent.parent / "data" / "raw_data"
ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "analysis_data"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR      = Path(__file__).parent.parent / "other" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# UTM Zone 17N — metres-based CRS for Toronto
UTM17N = "EPSG:32617"
WGS84  = "EPSG:4326"

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data...")

stations = pd.read_csv(ANALYSIS_DIR / "stations.csv")
active   = stations[stations["has_2023_trips"] == True].copy()

subway = pd.read_csv(DATA_DIR / "subway_stations_deduped.csv")

trips = pd.read_parquet(ANALYSIS_DIR / "trips_combined.parquet",
                        columns=["trip_id", "start_station_id", "duration_min", "user_type", "hour", "is_weekend"])

nbhd = gpd.read_file(DATA_DIR / "neighbourhoods.geojson")

print(f"  {len(active)} active bike stations")
print(f"  {len(subway)} subway stations")
print(f"  {len(nbhd)} neighbourhoods")

# ─────────────────────────────────────────────────────────────────────────────
# Build GeoDataFrames and reproject to UTM for metre-based operations
# ─────────────────────────────────────────────────────────────────────────────

gdf_bike = gpd.GeoDataFrame(
    active,
    geometry=gpd.points_from_xy(active["lon"], active["lat"]),
    crs=WGS84
).to_crs(UTM17N)

gdf_subway = gpd.GeoDataFrame(
    subway,
    geometry=gpd.points_from_xy(subway["lon"], subway["lat"]),
    crs=WGS84
).to_crs(UTM17N)

gdf_nbhd = nbhd.to_crs(UTM17N)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Distance from each bike station to its nearest subway station
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing bike-station → subway distances...")

nearest = gpd.sjoin_nearest(
    gdf_bike[["station_id", "name", "total_activity", "geometry"]],
    gdf_subway[["station_name", "geometry"]],
    how="left",
    distance_col="dist_to_subway_m"
)
# sjoin_nearest may produce duplicates; keep closest
nearest = nearest.sort_values("dist_to_subway_m").drop_duplicates("station_id")

gdf_bike = gdf_bike.merge(
    nearest[["station_id", "dist_to_subway_m", "station_name"]].rename(
        columns={"station_name": "nearest_subway"}
    ),
    on="station_id", how="left"
)

print(f"  Median distance to nearest subway: {gdf_bike['dist_to_subway_m'].median():.0f} m")
print(f"  % stations within 500m of subway:  {(gdf_bike['dist_to_subway_m'] < 500).mean()*100:.1f}%")
print(f"  % stations within 1km of subway:   {(gdf_bike['dist_to_subway_m'] < 1000).mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Trip proximity: join trips to station coords, then distance to subway
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing trip-level subway proximity...")

station_subway_dist = gdf_bike[["station_id", "dist_to_subway_m"]].copy()

trip_prox = trips.merge(
    station_subway_dist.rename(columns={"station_id": "start_station_id"}),
    on="start_station_id", how="inner"
)

buckets = [0, 200, 500, 1000, 2000, float("inf")]
labels  = ["<200m", "200–500m", "500m–1km", "1–2km", ">2km"]
trip_prox["proximity_bucket"] = pd.cut(
    trip_prox["dist_to_subway_m"], bins=buckets, labels=labels
)

prox_counts = (
    trip_prox["proximity_bucket"]
    .value_counts()
    .reindex(labels)
    .reset_index()
)
prox_counts.columns = ["proximity_to_subway", "trips"]
prox_counts["pct"] = (prox_counts["trips"] / prox_counts["trips"].sum() * 100).round(1)

print("  Trip share by proximity to nearest subway:")
for _, row in prox_counts.iterrows():
    print(f"    {row['proximity_to_subway']:<12}  {row['trips']:>8,}  ({row['pct']:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Coverage gap: 500m buffer around bike stations → intersect with neighbourhoods
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing neighbourhood coverage gaps...")

bike_buffer = gdf_bike.copy()
bike_buffer["geometry"] = gdf_bike.geometry.buffer(500)
covered_union = bike_buffer.unary_union

gdf_nbhd["area_total_m2"]   = gdf_nbhd.geometry.area
gdf_nbhd["area_covered_m2"] = gdf_nbhd.geometry.intersection(covered_union).area
gdf_nbhd["coverage_pct"]    = (gdf_nbhd["area_covered_m2"] / gdf_nbhd["area_total_m2"] * 100).round(1)
gdf_nbhd["coverage_cat"]    = pd.cut(
    gdf_nbhd["coverage_pct"],
    bins=[-1, 0, 10, 40, 70, 101],
    labels=["None (0%)", "Minimal (<10%)", "Partial (10–40%)", "Good (40–70%)", "High (>70%)"]
)

# Neighbourhood centroid distance to downtown (Yonge/King intersection = city centre)
downtown = gpd.GeoSeries([Point(-79.3832, 43.6481)], crs=WGS84).to_crs(UTM17N).iloc[0]
gdf_nbhd["dist_to_downtown_km"] = (gdf_nbhd.geometry.centroid.distance(downtown) / 1000).round(2)

nbhd_out = gdf_nbhd[["AREA_NAME","coverage_pct","coverage_cat","dist_to_downtown_km"]].copy()
nbhd_out.to_csv(ANALYSIS_DIR / "neighbourhood_coverage.csv", index=False)

print("  Coverage distribution:")
print(gdf_nbhd["coverage_cat"].value_counts().sort_index().to_string())

zero_cov = gdf_nbhd[gdf_nbhd["coverage_pct"] == 0]["AREA_NAME"].tolist()
print(f"\n  {len(zero_cov)} neighbourhoods with ZERO bike coverage:")
for n in sorted(zero_cov):
    print(f"    {n}")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Histogram: bike station distance to nearest subway
# ─────────────────────────────────────────────────────────────────────────────

fig1 = px.histogram(
    gdf_bike[gdf_bike["dist_to_subway_m"] < 5000],
    x="dist_to_subway_m",
    nbins=50,
    title="Distance from Bike Station to Nearest Subway Station",
    labels={"dist_to_subway_m": "Distance (metres)", "count": "Stations"},
    color_discrete_sequence=["#2196F3"],
)
fig1.add_vline(x=500, line_dash="dash", line_color="red",
               annotation_text="500m walk threshold", annotation_position="top right")
fig1.update_layout(height=420)
fig1.write_html(OUT_DIR / "chart_station_subway_dist.html")
print("\nSaved chart_station_subway_dist.html")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Trip proximity bar chart
# ─────────────────────────────────────────────────────────────────────────────

fig2 = px.bar(
    prox_counts,
    x="proximity_to_subway", y="pct",
    text="pct",
    title="What % of Bike Share Trips Start Near a Subway Station?",
    labels={"proximity_to_subway": "Distance to Nearest Subway", "pct": "% of Trips"},
    color="proximity_to_subway",
    color_discrete_sequence=px.colors.sequential.Blues_r,
)
fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig2.update_layout(showlegend=False, height=430, yaxis_title="% of 2023 Trips")
fig2.write_html(OUT_DIR / "chart_trip_proximity.html")
print("Saved chart_trip_proximity.html")

# ─────────────────────────────────────────────────────────────────────────────
# MAP 1 — Neighbourhood coverage choropleth
# ─────────────────────────────────────────────────────────────────────────────

gdf_nbhd_wgs = gdf_nbhd.to_crs(WGS84)
gdf_nbhd_wgs["centroid_lat"] = gdf_nbhd_wgs.geometry.centroid.y
gdf_nbhd_wgs["centroid_lon"] = gdf_nbhd_wgs.geometry.centroid.x

fig3 = px.choropleth_map(
    gdf_nbhd_wgs,
    geojson=gdf_nbhd_wgs.__geo_interface__,
    locations=gdf_nbhd_wgs.index,
    color="coverage_pct",
    color_continuous_scale="RdYlGn",
    range_color=[0, 100],
    zoom=10,
    center={"lat": 43.718, "lon": -79.383},
    map_style="carto-positron",
    hover_name="AREA_NAME",
    hover_data={"coverage_pct": ":.1f", "dist_to_downtown_km": ":.1f"},
    title="Bike Share Coverage by Neighbourhood (500m station buffer)<br>"
          "<sup>Green = well covered; Red = no coverage within 500m of any bike station</sup>",
    labels={"coverage_pct": "% Area Covered", "dist_to_downtown_km": "km from downtown"},
    opacity=0.75,
)

# Overlay active bike stations
fig3.add_scattermap(
    lat=gdf_bike["lat"],
    lon=gdf_bike["lon"],
    mode="markers",
    marker=dict(size=4, color="navy", opacity=0.5),
    name="Bike stations",
    hovertext=gdf_bike["name"],
    hoverinfo="text",
)

# Overlay subway stations
fig3.add_scattermap(
    lat=subway["lat"],
    lon=subway["lon"],
    mode="markers+text",
    marker=dict(size=10, color="gold", symbol="circle"),
    text=subway["station_name"].str.replace(" Station", "", regex=False),
    textposition="top right",
    textfont=dict(size=8, color="black"),
    name="Subway stations",
    hovertext=subway["station_name"],
    hoverinfo="text",
)

fig3.update_layout(height=750, margin=dict(t=60, b=0, l=0, r=0),
                   legend=dict(orientation="h", y=-0.05))
fig3.write_html(OUT_DIR / "map_transit_coverage.html")
print("Saved map_transit_coverage.html")

# ─────────────────────────────────────────────────────────────────────────────
# MAP 2 — Scatter: coverage % vs distance from downtown (equity frame)
# ─────────────────────────────────────────────────────────────────────────────

fig4 = px.scatter(
    gdf_nbhd[gdf_nbhd["AREA_NAME"].notna()],
    x="dist_to_downtown_km",
    y="coverage_pct",
    hover_name="AREA_NAME",
    color="coverage_pct",
    color_continuous_scale="RdYlGn",
    range_color=[0, 100],
    size_max=12,
    title="Neighbourhood Coverage vs. Distance from Downtown<br>"
          "<sup>The further from downtown, the less bike coverage — regardless of need</sup>",
    labels={
        "dist_to_downtown_km": "Distance from Downtown (km)",
        "coverage_pct": "% Area with Bike Coverage (500m)",
    },
)
fig4.add_hline(y=40, line_dash="dot", line_color="grey",
               annotation_text="40% coverage threshold", annotation_position="bottom right")
fig4.update_layout(height=480)
fig4.write_html(OUT_DIR / "chart_coverage_vs_downtown.html")
print("Saved chart_coverage_vs_downtown.html")

print("\nDone. Open output/ HTML files in your browser.")
