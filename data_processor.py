"""
============================================================
  DATA PROCESSOR — Bangalore Traffic Dataset
  TrafficIQ Bangalore | data_processor.py
============================================================
Loads, cleans, and engineers features from the
Bangalore traffic CSV dataset.

Source columns:
  Date, Area Name, Road/Intersection Name,
  Traffic Volume, Average Speed, Travel Time Index,
  Congestion Level, Road Capacity Utilization,
  Incident Reports, Environmental Impact,
  Public Transport Usage, Traffic Signal Compliance,
  Parking Usage, Pedestrian and Cyclist Count,
  Weather Conditions, Roadwork and Construction Activity

Engineered features (16 total) used by the ML model:
  TrafficVolume, AverageSpeed, TravelTimeIndex,
  RoadCapacityUtil, IncidentReports, EnvironmentalImpact,
  PublicTransportUsage, SignalCompliance, ParkingUsage,
  PedestrianCount, WeatherCode, RoadworkActive,
  RoadTypeCode, AreaCode, DayOfWeek, IsWeekend

Target: CongestionLevel (0–100)
============================================================
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────
DATA_PATH = "Banglore_traffic_Dataset.csv"

AREA_LIST = [
    "Koramangala", "M.G. Road", "Indiranagar", "Jayanagar",
    "Hebbal", "Whitefield", "Yeshwanthpur", "Electronic City",
]

# Road → road type mapping
ROAD_TYPE_MAP = {
    "100 Feet Road"        : "arterial_main",
    "CMH Road"             : "arterial_main",
    "Hosur Road"           : "arterial_main",
    "Tumkur Road"          : "arterial_main",
    "Ballari Road"         : "arterial_main",
    "Sarjapur Road"        : "arterial_main",
    "ITPL Main Road"       : "arterial_main",
    "Marathahalli Bridge"  : "expressway_ramp",
    "Hebbal Flyover"       : "expressway_ramp",
    "Silk Board Junction"  : "signal_junction",
    "Sony World Junction"  : "signal_junction",
    "Trinity Circle"       : "signal_junction",
    "Anil Kumble Circle"   : "signal_junction",
    "South End Circle"     : "crossroad",
    "Yeshwanthpur Circle"  : "crossroad",
    "Jayanagar 4th Block"  : "residential",
}

ROAD_TYPE_CODES = {
    "arterial_main"  : 0,
    "signal_junction": 1,
    "crossroad"      : 2,
    "expressway_ramp": 3,
    "residential"    : 4,
}

ROAD_TYPE_LABELS = {
    "arterial_main"  : "Arterial / Main Road",
    "signal_junction": "Signal Junction / Circle",
    "crossroad"      : "Crossroad / Circle",
    "expressway_ramp": "Flyover / Bridge",
    "residential"    : "Residential Street",
}

# Roads per area
AREA_ROADS = {
    "Indiranagar"    : ["100 Feet Road", "CMH Road"],
    "Koramangala"    : ["Sarjapur Road", "Sony World Junction"],
    "M.G. Road"      : ["Anil Kumble Circle", "Trinity Circle"],
    "Hebbal"         : ["Ballari Road", "Hebbal Flyover"],
    "Whitefield"     : ["ITPL Main Road", "Marathahalli Bridge"],
    "Jayanagar"      : ["Jayanagar 4th Block", "South End Circle"],
    "Yeshwanthpur"   : ["Tumkur Road", "Yeshwanthpur Circle"],
    "Electronic City": ["Hosur Road", "Silk Board Junction"],
}

WEATHER_CODES = {
    "Clear"   : 0,
    "Overcast": 1,
    "Windy"   : 2,
    "Fog"     : 3,
    "Rain"    : 4,
}

WEATHER_IMPACT = {
    "Clear"   : 0.0,
    "Overcast": 0.1,
    "Windy"   : 0.25,
    "Fog"     : 0.45,
    "Rain"    : 0.55,
}

AREA_CODES = {a: i for i, a in enumerate(sorted(AREA_LIST))}

FEATURE_COLS = [
    "TrafficVolume", "AverageSpeed", "TravelTimeIndex",
    "RoadCapacityUtil", "IncidentReports", "EnvironmentalImpact",
    "PublicTransportUsage", "SignalCompliance", "ParkingUsage",
    "PedestrianCount", "WeatherCode", "RoadworkActive",
    "RoadTypeCode", "AreaCode", "DayOfWeek", "IsWeekend",
]

TARGET_COL = "CongestionLevel"

# Policy sensitivity defaults per road type
POLICY_SENS = {
    "arterial_main"  : {"signal_opt": 0.75, "parking_enf": 0.70, "peak_restr": 0.80, "pt_boost": 0.65},
    "signal_junction": {"signal_opt": 0.90, "parking_enf": 0.85, "peak_restr": 0.85, "pt_boost": 0.70},
    "crossroad"      : {"signal_opt": 0.80, "parking_enf": 0.75, "peak_restr": 0.75, "pt_boost": 0.60},
    "expressway_ramp": {"signal_opt": 0.40, "parking_enf": 0.30, "peak_restr": 0.65, "pt_boost": 0.45},
    "residential"    : {"signal_opt": 0.55, "parking_enf": 0.90, "peak_restr": 0.40, "pt_boost": 0.50},
}

# Feature defaults for inference
FEATURE_DEFAULTS = {
    "TrafficVolume"       : 28000,
    "AverageSpeed"        : 38.0,
    "TravelTimeIndex"     : 1.3,
    "RoadCapacityUtil"    : 85.0,
    "IncidentReports"     : 0,
    "EnvironmentalImpact" : 100.0,
    "PublicTransportUsage": 50.0,
    "SignalCompliance"    : 75.0,
    "ParkingUsage"        : 65.0,
    "PedestrianCount"     : 110,
    "WeatherCode"         : 0,
    "RoadworkActive"      : 0,
    "RoadTypeCode"        : 0,
    "AreaCode"            : 0,
    "DayOfWeek"           : 2,
    "IsWeekend"           : 0,
}


# ─────────────────────────────────────────────────────────
#  LOAD & ENGINEER
# ─────────────────────────────────────────────────────────
def load_and_engineer(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load Bangalore CSV and engineer the full feature set.
    Returns a DataFrame ready for model training.
    """
    df = pd.read_csv(path)
    df = df.copy()

    # Date features
    df["Date"]      = pd.to_datetime(df["Date"])
    df["DayOfWeek"] = df["Date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Month"]     = df["Date"].dt.month

    # Rename core columns
    df = df.rename(columns={
        "Traffic Volume"                    : "TrafficVolume",
        "Average Speed"                     : "AverageSpeed",
        "Travel Time Index"                 : "TravelTimeIndex",
        "Congestion Level"                  : "CongestionLevel",
        "Road Capacity Utilization"         : "RoadCapacityUtil",
        "Incident Reports"                  : "IncidentReports",
        "Environmental Impact"              : "EnvironmentalImpact",
        "Public Transport Usage"            : "PublicTransportUsage",
        "Traffic Signal Compliance"         : "SignalCompliance",
        "Parking Usage"                     : "ParkingUsage",
        "Pedestrian and Cyclist Count"      : "PedestrianCount",
        "Weather Conditions"                : "WeatherConditions",
        "Roadwork and Construction Activity": "Roadwork",
        "Area Name"                         : "AreaName",
        "Road/Intersection Name"            : "RoadName",
    })

    # Encode categoricals
    df["WeatherCode"]   = df["WeatherConditions"].map(WEATHER_CODES).fillna(0).astype(int)
    df["RoadworkActive"]= (df["Roadwork"] == "Yes").astype(int)
    df["RoadTypeName"]  = df["RoadName"].map(ROAD_TYPE_MAP).fillna("arterial_main")
    df["RoadTypeCode"]  = df["RoadTypeName"].map(ROAD_TYPE_CODES).fillna(0).astype(int)
    df["AreaCode"]      = df["AreaName"].map(AREA_CODES).fillna(0).astype(int)

    # Roadwork penalty on congestion (for labeling purposes)
    df["RoadworkPenalty"] = df["RoadworkActive"] * 5.0
    df["CongestionLevel"] = (df["CongestionLevel"] + df["RoadworkPenalty"]).clip(0, 100)

    # Drop rows with nulls in key columns
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    print(f"[INFO] Loaded: {len(df)} rows | {len(FEATURE_COLS)} features | target: {TARGET_COL}")
    print(f"       Areas  : {df['AreaName'].nunique()} | Roads: {df['RoadName'].nunique()}")
    print(f"       Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"       Congestion: {df[TARGET_COL].min():.1f} – {df[TARGET_COL].max():.1f} "
          f"(mean {df[TARGET_COL].mean():.1f})")
    return df


# ─────────────────────────────────────────────────────────
#  AREA / ROAD STATS
# ─────────────────────────────────────────────────────────
def get_location_stats(df: pd.DataFrame, area: str, road: str) -> dict:
    """
    Return historical statistics for a specific area + road combination.
    Used to pre-fill the simulation sliders with realistic defaults.
    """
    mask = (df["AreaName"] == area) & (df["RoadName"] == road)
    sub  = df[mask]

    if sub.empty:
        return FEATURE_DEFAULTS.copy()

    stats = {}
    for col in FEATURE_COLS:
        if col in sub.columns:
            stats[col] = round(float(sub[col].mean()), 2)

    # Add rich context
    stats["AvgCongestion"]     = round(float(sub["CongestionLevel"].mean()), 1)
    stats["MaxCongestion"]     = round(float(sub["CongestionLevel"].max()), 1)
    stats["MinCongestion"]     = round(float(sub["CongestionLevel"].min()), 1)
    stats["PeakCongestion"]    = round(float(sub[sub["DayOfWeek"] < 5]["CongestionLevel"].mean()), 1)
    stats["WeekendCongestion"] = round(float(sub[sub["IsWeekend"] == 1]["CongestionLevel"].mean()), 1)
    stats["RoadTypeName"]      = ROAD_TYPE_MAP.get(road, "arterial_main")
    stats["RoadTypeLabel"]     = ROAD_TYPE_LABELS.get(stats["RoadTypeName"], "Main Road")
    stats["RecordCount"]       = len(sub)

    # Weather breakdown
    weather_cong = sub.groupby("WeatherConditions")["CongestionLevel"].mean().round(1).to_dict()
    stats["CongestionByWeather"] = weather_cong

    # Monthly trend
    monthly = sub.groupby("Month")["CongestionLevel"].mean().round(1).to_dict()
    stats["MonthlyTrend"] = monthly

    return stats


def get_area_summary(df: pd.DataFrame, area: str) -> pd.DataFrame:
    """Return per-road summary stats for an area."""
    mask = df["AreaName"] == area
    return (
        df[mask]
        .groupby("RoadName")
        .agg(
            AvgCongestion  = ("CongestionLevel",    "mean"),
            MaxCongestion  = ("CongestionLevel",    "max"),
            AvgSpeed       = ("AverageSpeed",        "mean"),
            AvgVolume      = ("TrafficVolume",        "mean"),
            CapacityUtil   = ("RoadCapacityUtil",    "mean"),
            RoadType       = ("RoadTypeName",        "first"),
        )
        .round(2)
        .reset_index()
    )


def get_historical_trend(df: pd.DataFrame, area: str, road: str) -> pd.DataFrame:
    """Return daily congestion time series for a location."""
    mask = (df["AreaName"] == area) & (df["RoadName"] == road)
    return (
        df[mask]
        .groupby("Date")
        .agg(
            CongestionLevel = ("CongestionLevel", "mean"),
            AverageSpeed    = ("AverageSpeed",    "mean"),
            TrafficVolume   = ("TrafficVolume",   "mean"),
        )
        .reset_index()
        .sort_values("Date")
    )


def get_weather_impact(df: pd.DataFrame, area: str, road: str) -> dict:
    """Return congestion breakdown by weather for a location."""
    mask = (df["AreaName"] == area) & (df["RoadName"] == road)
    sub  = df[mask]
    return sub.groupby("WeatherConditions")["CongestionLevel"].mean().round(2).to_dict()


if __name__ == "__main__":
    df = load_and_engineer(DATA_PATH)
    print("\nSample row:")
    print(df[FEATURE_COLS + [TARGET_COL]].head(2).to_string())
    print("\nArea summary — Koramangala:")
    print(get_area_summary(df, "Koramangala").to_string(index=False))
