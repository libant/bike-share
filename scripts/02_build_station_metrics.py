"""
Phase 2 — Station metrics pipeline for Bike Share Toronto 2023.

Outputs:
  output/stations.csv          — station master with lat/lon, capacity, metrics
  output/trips_combined.parquet — full normalized trip table (fast for downstream analysis)
"""

import json
import pandas as pd
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data" / "raw_data"
OUT_DIR   = Path(__file__).parent.parent / "data" / "analysis_data"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load station master from locally cached GBFS file
# ─────────────────────────────────────────────────────────────────────────────

print("Loading station master from cached GBFS file...")
with open(DATA_DIR / "stations_gbfs.json") as f:
    gbfs = json.load(f)

stations_raw = gbfs["data"]["stations"]
stations = pd.DataFrame([{
    "station_id":   int(s["station_id"]),
    "name":         s["name"],
    "lat":          s["lat"],
    "lon":          s["lon"],
    "capacity":     s.get("capacity", None),
} for s in stations_raw])

print(f"  {len(stations)} stations loaded from GBFS")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load and normalize all 12 monthly CSVs
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading monthly CSVs...")

RENAME = {
    "Trip Id":              "trip_id",
    "Trip  Duration":       "duration_sec",   # double-space in source
    "Start Station Id":     "start_station_id",
    "Start Time":           "start_time",
    "Start Station Name":   "start_station_name",
    "End Station Id":       "end_station_id",
    "End Time":             "end_time",
    "End Station Name":     "end_station_name",
    "Bike Id":              "bike_id",
    "User Type":            "user_type",
}

dfs = []
csv_files = sorted(DATA_DIR.glob("Bike share ridership 2023-*.csv"))

for path in csv_files:
    month = path.stem.split("2023-")[-1]
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    df = df.rename(columns=RENAME)

    # Normalize user type labels (some months use different capitalisation)
    df["user_type"] = df["user_type"].str.strip().str.title()

    # Parse datetimes
    for col in ("start_time", "end_time"):
        df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False, errors="coerce")

    # Cast IDs to nullable int
    df["end_station_id"] = pd.to_numeric(df["end_station_id"], errors="coerce").astype("Int64")
    df["start_station_id"] = df["start_station_id"].astype("Int64")

    df["month"] = int(month)
    dfs.append(df)
    print(f"  2023-{month}  {len(df):>7,} rows  enc={enc}")

trips = pd.concat(dfs, ignore_index=True)
print(f"\n  Combined: {len(trips):,} trips")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Add derived time columns
# ─────────────────────────────────────────────────────────────────────────────

trips["hour"]        = trips["start_time"].dt.hour
trips["day_of_week"] = trips["start_time"].dt.day_name()
trips["is_weekend"]  = trips["start_time"].dt.dayofweek >= 5
trips["duration_min"]= trips["duration_sec"] / 60

# ─────────────────────────────────────────────────────────────────────────────
# 4. Station-level utilization metrics
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing station metrics...")

# Departures per station
departures = (
    trips.groupby("start_station_id")
    .agg(
        departures        = ("trip_id", "count"),
        member_departures = ("user_type", lambda x: (x == "Annual Member").sum()),
        casual_departures = ("user_type", lambda x: (x == "Casual Member").sum()),
        avg_duration_min  = ("duration_min", "mean"),
        peak_hour_dep     = ("hour", lambda x: x.mode()[0] if len(x) > 0 else None),
    )
    .reset_index()
    .rename(columns={"start_station_id": "station_id"})
)

# Arrivals per station
arrivals = (
    trips.dropna(subset=["end_station_id"])
    .groupby("end_station_id")
    .agg(arrivals=("trip_id", "count"))
    .reset_index()
    .rename(columns={"end_station_id": "station_id"})
)

# Merge everything onto station master
station_metrics = (
    stations
    .merge(departures, on="station_id", how="left")
    .merge(arrivals,   on="station_id", how="left")
)

station_metrics["departures"] = station_metrics["departures"].fillna(0).astype(int)
station_metrics["arrivals"]   = station_metrics["arrivals"].fillna(0).astype(int)

# Net flow: positive = more arrivals than departures (accumulator)
#           negative = more departures (generator / high-demand origin)
station_metrics["net_flow"]         = station_metrics["arrivals"] - station_metrics["departures"]
station_metrics["total_activity"]   = station_metrics["arrivals"] + station_metrics["departures"]

# Utilization rate: trips per dock per day (365 days)
station_metrics["trips_per_dock_per_day"] = (
    station_metrics["total_activity"] / (station_metrics["capacity"].clip(lower=1) * 365)
).round(3)

# Member share of departures
station_metrics["member_share"] = (
    station_metrics["member_departures"] / station_metrics["departures"].clip(lower=1)
).round(3)

# Flag unmatched stations (in GBFS but not seen in 2023 trips)
station_metrics["has_2023_trips"] = station_metrics["departures"] > 0

print(f"  {station_metrics['has_2023_trips'].sum()} / {len(station_metrics)} GBFS stations active in 2023 trips")

# Stations in trip data but not in GBFS (may have been retired)
all_trip_station_ids = pd.concat([
    trips["start_station_id"].dropna(),
    trips["end_station_id"].dropna()
]).astype(int).unique()

ghost_ids = set(all_trip_station_ids) - set(stations["station_id"])
print(f"  {len(ghost_ids)} station IDs in trips but not in current GBFS (retired stations)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save outputs
# ─────────────────────────────────────────────────────────────────────────────

station_metrics.to_csv(OUT_DIR / "stations.csv", index=False)
print(f"\nSaved → output/stations.csv  ({len(station_metrics)} rows)")

trips.to_parquet(OUT_DIR / "trips_combined.parquet", index=False)
print(f"Saved → output/trips_combined.parquet  ({len(trips):,} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Quick top/bottom station summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Top 10 stations by total activity ──────────────────────────")
top10 = station_metrics.nlargest(10, "total_activity")[
    ["name", "capacity", "total_activity", "net_flow", "member_share", "trips_per_dock_per_day"]
]
print(top10.to_string(index=False))

print("\n── Bottom 10 active stations by total activity ─────────────────")
bot10 = station_metrics[station_metrics["has_2023_trips"]].nsmallest(10, "total_activity")[
    ["name", "capacity", "total_activity", "net_flow", "member_share", "trips_per_dock_per_day"]
]
print(bot10.to_string(index=False))

print("\n── Largest net importers (bikes accumulate here) ───────────────")
print(station_metrics.nlargest(5, "net_flow")[["name", "net_flow", "total_activity"]].to_string(index=False))

print("\n── Largest net exporters (demand origin, bikes drain) ──────────")
print(station_metrics.nsmallest(5, "net_flow")[["name", "net_flow", "total_activity"]].to_string(index=False))

print("\nDone.")
