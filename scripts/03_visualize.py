"""
Phase 2 visualizations for Bike Share Toronto 2023 station optimization.

Outputs (HTML, open in any browser):
  other/charts/map_activity.html       — bubble map: station total activity
  other/charts/map_flow.html           — bubble map: net flow imbalance
  other/charts/chart_hourly.html       — hourly demand by user type
  other/charts/chart_monthly.html      — monthly trip volume
  other/charts/chart_top_stations.html — top 20 stations bar chart
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "analysis_data"
OUT_DIR      = Path(__file__).parent.parent / "other" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

stations = pd.read_csv(ANALYSIS_DIR / "stations.csv")
trips    = pd.read_parquet(ANALYSIS_DIR / "trips_combined.parquet")

# Only stations with trips for map clarity
active = stations[stations["has_2023_trips"] == True].copy()

MAPBOX_STYLE = "carto-positron"
TORONTO_CENTER = dict(lat=43.653, lon=-79.383)

# ─────────────────────────────────────────────────────────────────────────────
# MAP 1 — Total Activity (bubble size = activity, color = utilization rate)
# ─────────────────────────────────────────────────────────────────────────────

fig1 = px.scatter_map(
    active,
    lat="lat", lon="lon",
    size="total_activity",
    color="trips_per_dock_per_day",
    color_continuous_scale="YlOrRd",
    size_max=30,
    zoom=11,
    center=TORONTO_CENTER,
    map_style=MAPBOX_STYLE,
    hover_name="name",
    hover_data={
        "total_activity": ":,",
        "capacity": True,
        "trips_per_dock_per_day": ":.2f",
        "member_share": ":.1%",
        "lat": False,
        "lon": False,
    },
    title="Bike Share Toronto 2023 — Station Activity<br><sup>Bubble size = total trips; Color = trips per dock per day</sup>",
    labels={"trips_per_dock_per_day": "Trips/dock/day"},
)
fig1.update_layout(height=700, margin=dict(t=60, b=0, l=0, r=0))
fig1.write_html(OUT_DIR / "map_activity.html")
print("Saved map_activity.html")

# ─────────────────────────────────────────────────────────────────────────────
# MAP 2 — Net Flow Imbalance
# ─────────────────────────────────────────────────────────────────────────────

active["abs_net_flow"] = active["net_flow"].abs()
active["flow_label"] = active["net_flow"].apply(
    lambda x: "Importer (bikes pile up)" if x > 0
    else ("Exporter (bikes drain)" if x < 0 else "Balanced")
)

fig2 = px.scatter_map(
    active,
    lat="lat", lon="lon",
    size="abs_net_flow",
    color="net_flow",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    size_max=28,
    zoom=11,
    center=TORONTO_CENTER,
    map_style=MAPBOX_STYLE,
    hover_name="name",
    hover_data={
        "net_flow": ":,",
        "departures": ":,",
        "arrivals": ":,",
        "capacity": True,
        "abs_net_flow": False,
        "flow_label": True,
        "lat": False,
        "lon": False,
    },
    title="Bike Share Toronto 2023 — Station Net Flow Imbalance<br>"
          "<sup>Blue = bikes accumulate (importer); Red = bikes drain (exporter); Size = magnitude</sup>",
    labels={"net_flow": "Net flow"},
)
fig2.update_layout(height=700, margin=dict(t=60, b=0, l=0, r=0))
fig2.write_html(OUT_DIR / "map_flow.html")
print("Saved map_flow.html")

# ─────────────────────────────────────────────────────────────────────────────
# MAP 3 — Member share (proxy for commuter vs tourist usage)
# ─────────────────────────────────────────────────────────────────────────────

fig3 = px.scatter_map(
    active,
    lat="lat", lon="lon",
    size="total_activity",
    color="member_share",
    color_continuous_scale="Viridis",
    size_max=25,
    zoom=11,
    center=TORONTO_CENTER,
    map_style=MAPBOX_STYLE,
    hover_name="name",
    hover_data={
        "member_share": ":.1%",
        "member_departures": ":,",
        "casual_departures": ":,",
        "total_activity": ":,",
        "lat": False,
        "lon": False,
    },
    title="Bike Share Toronto 2023 — Annual Member Share by Station<br>"
          "<sup>Darker = higher share of annual member trips; Bubble size = total trips</sup>",
    labels={"member_share": "Annual member share"},
)
fig3.update_layout(height=700, margin=dict(t=60, b=0, l=0, r=0))
fig3.write_html(OUT_DIR / "map_member_share.html")
print("Saved map_member_share.html")

# ─────────────────────────────────────────────────────────────────────────────
# CHART — Hourly demand by user type (commuter signature)
# ─────────────────────────────────────────────────────────────────────────────

hourly = (
    trips.groupby(["hour", "user_type"])
    .size()
    .reset_index(name="trips")
)

fig4 = px.line(
    hourly,
    x="hour", y="trips",
    color="user_type",
    markers=True,
    title="Hourly Trip Volume by User Type — Full Year 2023",
    labels={"hour": "Hour of Day", "trips": "Total Trips", "user_type": "User Type"},
    color_discrete_map={"Annual Member": "#2196F3", "Casual Member": "#FF9800"},
)
fig4.update_layout(xaxis=dict(tickvals=list(range(0, 24))), height=450)
fig4.write_html(OUT_DIR / "chart_hourly.html")
print("Saved chart_hourly.html")

# ─────────────────────────────────────────────────────────────────────────────
# CHART — Monthly volume
# ─────────────────────────────────────────────────────────────────────────────

monthly = (
    trips.groupby(["month", "user_type"])
    .size()
    .reset_index(name="trips")
)
month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
monthly["month_label"] = monthly["month"].map(month_labels)

fig5 = px.bar(
    monthly,
    x="month_label", y="trips",
    color="user_type",
    barmode="stack",
    category_orders={"month_label": list(month_labels.values())},
    title="Monthly Trip Volume by User Type — 2023",
    labels={"month_label": "Month", "trips": "Trips", "user_type": "User Type"},
    color_discrete_map={"Annual Member": "#2196F3", "Casual Member": "#FF9800"},
)
fig5.update_layout(height=430)
fig5.write_html(OUT_DIR / "chart_monthly.html")
print("Saved chart_monthly.html")

# ─────────────────────────────────────────────────────────────────────────────
# CHART — Top 20 stations: activity breakdown
# ─────────────────────────────────────────────────────────────────────────────

top20 = active.nlargest(20, "total_activity").copy()

fig6 = go.Figure()
fig6.add_bar(
    y=top20["name"], x=top20["casual_departures"],
    name="Casual Member", orientation="h", marker_color="#FF9800"
)
fig6.add_bar(
    y=top20["name"], x=top20["member_departures"],
    name="Annual Member", orientation="h", marker_color="#2196F3"
)
fig6.update_layout(
    barmode="stack",
    title="Top 20 Stations by Departures — User Type Breakdown",
    xaxis_title="Departures",
    yaxis=dict(autorange="reversed"),
    height=600,
    legend=dict(orientation="h", y=-0.12),
)
fig6.write_html(OUT_DIR / "chart_top_stations.html")
print("Saved chart_top_stations.html")

# ─────────────────────────────────────────────────────────────────────────────
# Quick terminal summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n── User type split (2023) ──────────────────────")
ut = trips["user_type"].value_counts()
for k, v in ut.items():
    print(f"  {k:<20} {v:>8,}  ({v/len(trips)*100:.1f}%)")

print("\n── Weekend vs weekday trips ────────────────────")
wk = trips["is_weekend"].value_counts()
print(f"  Weekday: {wk[False]:>8,}  ({wk[False]/len(trips)*100:.1f}%)")
print(f"  Weekend: {wk[True]:>8,}  ({wk[True]/len(trips)*100:.1f}%)")

print("\n── Duration distribution (minutes) ────────────")
# Clip extreme outliers (max 10h = 600min)
d = trips["duration_min"].clip(upper=600)
print(f"  Median: {d.median():.1f}  |  Mean: {d.mean():.1f}  |  P95: {d.quantile(.95):.1f}")

print("\nAll charts saved to output/")
