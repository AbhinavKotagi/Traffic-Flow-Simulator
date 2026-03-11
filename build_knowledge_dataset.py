"""
============================================================
  KNOWLEDGE DATASET BUILDER
  Traffic Policy Simulator | build_knowledge_dataset.py
============================================================
Generates a rich labeled traffic knowledge dataset across:

  10 Road Types  x  4 Time Periods  x  3 Weather Conditions
  x  5 City Types  =  ~3000 rows of labeled domain knowledge

Each row captures realistic traffic behaviour for that
combination using domain-expert-defined statistical ranges.

Features (26 total):
  Traffic    : VehicleCount, Density, Stopped, WrongParked, EstSpeed
  Geometry   : Lanes, RoadWidth, HasMedian, IsDivided
  Composition: CarRatio, BikeRatio, BusRatio, TruckRatio, VehicleMix
  Context    : Pedestrians, NMVCount, Signals
  Temporal   : PeakHour, TimeOfDayCode
  Environment: WeatherImpact, WeatherCode, CityTypeCode
  Road Meta  : RoadTypeCode, LaneDisciplineCode
  Policy     : PolicySens_OneWay, PolicySens_NoParking, PolicySens_PeakRestr

Target: CongestionScore (0-100 continuous)
============================================================
"""

import numpy as np
import pandas as pd

np.random.seed(2024)

# ─────────────────────────────────────────────────────────
#  ROAD TYPE PROFILES
# ─────────────────────────────────────────────────────────
ROAD_PROFILES = {
    "highway": {
        "code": 0, "lanes_range": (4, 8), "speed_range": (60, 110),
        "vc_range": (30, 80), "stopped_frac": (0.00, 0.03), "wp_frac": (0.00, 0.01),
        "car_ratio": (0.60, 0.80), "bike_ratio": (0.00, 0.05),
        "bus_ratio": (0.05, 0.15), "truck_ratio": (0.10, 0.30),
        "pedestrians": (0, 2), "signals": (0, 1),
        "has_median": 1.0, "is_divided": 1.0,
        "policy_oneway": 0.10, "policy_nopark": 0.20, "policy_peak": 0.55,
        "lane_discipline": 0.85,
    },
    "arterial_main": {
        "code": 1, "lanes_range": (3, 6), "speed_range": (30, 60),
        "vc_range": (25, 65), "stopped_frac": (0.02, 0.10), "wp_frac": (0.01, 0.05),
        "car_ratio": (0.50, 0.70), "bike_ratio": (0.05, 0.20),
        "bus_ratio": (0.08, 0.20), "truck_ratio": (0.05, 0.15),
        "pedestrians": (2, 12), "signals": (0, 3),
        "has_median": 0.6, "is_divided": 0.5,
        "policy_oneway": 0.65, "policy_nopark": 0.70, "policy_peak": 0.75,
        "lane_discipline": 0.60,
    },
    "signal_junction": {
        "code": 2, "lanes_range": (2, 5), "speed_range": (10, 35),
        "vc_range": (20, 70), "stopped_frac": (0.10, 0.35), "wp_frac": (0.02, 0.10),
        "car_ratio": (0.45, 0.65), "bike_ratio": (0.10, 0.30),
        "bus_ratio": (0.08, 0.20), "truck_ratio": (0.05, 0.15),
        "pedestrians": (5, 25), "signals": (1, 4),
        "has_median": 0.3, "is_divided": 0.2,
        "policy_oneway": 0.55, "policy_nopark": 0.80, "policy_peak": 0.85,
        "lane_discipline": 0.40,
    },
    "crossroad": {
        "code": 3, "lanes_range": (2, 4), "speed_range": (8, 30),
        "vc_range": (15, 55), "stopped_frac": (0.08, 0.30), "wp_frac": (0.03, 0.12),
        "car_ratio": (0.40, 0.60), "bike_ratio": (0.15, 0.35),
        "bus_ratio": (0.05, 0.15), "truck_ratio": (0.03, 0.10),
        "pedestrians": (8, 30), "signals": (0, 3),
        "has_median": 0.1, "is_divided": 0.1,
        "policy_oneway": 0.70, "policy_nopark": 0.75, "policy_peak": 0.80,
        "lane_discipline": 0.35,
    },
    "residential": {
        "code": 4, "lanes_range": (1, 3), "speed_range": (10, 30),
        "vc_range": (3, 20), "stopped_frac": (0.05, 0.20), "wp_frac": (0.05, 0.20),
        "car_ratio": (0.55, 0.80), "bike_ratio": (0.10, 0.25),
        "bus_ratio": (0.00, 0.05), "truck_ratio": (0.01, 0.08),
        "pedestrians": (3, 15), "signals": (0, 2),
        "has_median": 0.0, "is_divided": 0.0,
        "policy_oneway": 0.80, "policy_nopark": 0.85, "policy_peak": 0.30,
        "lane_discipline": 0.50,
    },
    "market_street": {
        "code": 5, "lanes_range": (1, 3), "speed_range": (5, 20),
        "vc_range": (10, 50), "stopped_frac": (0.15, 0.45), "wp_frac": (0.10, 0.35),
        "car_ratio": (0.25, 0.45), "bike_ratio": (0.20, 0.40),
        "bus_ratio": (0.05, 0.15), "truck_ratio": (0.10, 0.25),
        "pedestrians": (15, 50), "signals": (0, 2),
        "has_median": 0.0, "is_divided": 0.0,
        "policy_oneway": 0.60, "policy_nopark": 0.95, "policy_peak": 0.50,
        "lane_discipline": 0.20,
    },
    "bus_corridor": {
        "code": 6, "lanes_range": (2, 4), "speed_range": (12, 35),
        "vc_range": (20, 60), "stopped_frac": (0.08, 0.25), "wp_frac": (0.05, 0.15),
        "car_ratio": (0.30, 0.50), "bike_ratio": (0.10, 0.25),
        "bus_ratio": (0.25, 0.45), "truck_ratio": (0.03, 0.10),
        "pedestrians": (8, 30), "signals": (1, 4),
        "has_median": 0.3, "is_divided": 0.3,
        "policy_oneway": 0.45, "policy_nopark": 0.90, "policy_peak": 0.70,
        "lane_discipline": 0.45,
    },
    "industrial_road": {
        "code": 7, "lanes_range": (2, 4), "speed_range": (20, 50),
        "vc_range": (10, 40), "stopped_frac": (0.03, 0.15), "wp_frac": (0.02, 0.10),
        "car_ratio": (0.30, 0.50), "bike_ratio": (0.05, 0.15),
        "bus_ratio": (0.05, 0.12), "truck_ratio": (0.30, 0.55),
        "pedestrians": (1, 8), "signals": (0, 2),
        "has_median": 0.2, "is_divided": 0.2,
        "policy_oneway": 0.40, "policy_nopark": 0.60, "policy_peak": 0.65,
        "lane_discipline": 0.55,
    },
    "school_zone": {
        "code": 8, "lanes_range": (1, 3), "speed_range": (8, 20),
        "vc_range": (5, 30), "stopped_frac": (0.15, 0.50), "wp_frac": (0.10, 0.30),
        "car_ratio": (0.50, 0.75), "bike_ratio": (0.05, 0.20),
        "bus_ratio": (0.10, 0.25), "truck_ratio": (0.01, 0.05),
        "pedestrians": (20, 60), "signals": (0, 3),
        "has_median": 0.0, "is_divided": 0.0,
        "policy_oneway": 0.75, "policy_nopark": 0.90, "policy_peak": 0.40,
        "lane_discipline": 0.30,
    },
    "expressway_ramp": {
        "code": 9, "lanes_range": (1, 3), "speed_range": (20, 60),
        "vc_range": (15, 55), "stopped_frac": (0.02, 0.12), "wp_frac": (0.00, 0.02),
        "car_ratio": (0.55, 0.75), "bike_ratio": (0.00, 0.05),
        "bus_ratio": (0.05, 0.15), "truck_ratio": (0.15, 0.35),
        "pedestrians": (0, 3), "signals": (0, 2),
        "has_median": 0.8, "is_divided": 0.9,
        "policy_oneway": 0.20, "policy_nopark": 0.30, "policy_peak": 0.60,
        "lane_discipline": 0.75,
    },
}

TIME_PROFILES = {
    "night"      : {"vc_mult": 0.15, "stop_mult": 0.60, "speed_mult": 1.30, "peak": 0.0, "code": 0},
    "off_peak"   : {"vc_mult": 0.55, "stop_mult": 0.80, "speed_mult": 1.10, "peak": 0.2, "code": 1},
    "peak"       : {"vc_mult": 1.00, "stop_mult": 1.40, "speed_mult": 0.70, "peak": 1.0, "code": 2},
    "super_peak" : {"vc_mult": 1.20, "stop_mult": 1.80, "speed_mult": 0.55, "peak": 1.0, "code": 3},
}

WEATHER_PROFILES = {
    "clear" : {"vc_mult": 1.00, "speed_mult": 1.00, "stop_mult": 1.00, "impact": 0.00, "code": 0},
    "rain"  : {"vc_mult": 0.85, "speed_mult": 0.75, "stop_mult": 1.30, "impact": 0.50, "code": 1},
    "heavy" : {"vc_mult": 0.65, "speed_mult": 0.55, "stop_mult": 1.70, "impact": 1.00, "code": 2},
}

CITY_PROFILES = {
    "metro"      : {"vc_mult": 1.30, "ped_mult": 1.50, "discipline": 0.70, "code": 0},
    "tier2_city" : {"vc_mult": 1.00, "ped_mult": 1.00, "discipline": 0.50, "code": 1},
    "small_town" : {"vc_mult": 0.65, "ped_mult": 0.70, "discipline": 0.60, "code": 2},
    "suburban"   : {"vc_mult": 0.80, "ped_mult": 0.80, "discipline": 0.65, "code": 3},
    "port_city"  : {"vc_mult": 1.10, "ped_mult": 0.90, "discipline": 0.55, "code": 4},
}


def compute_congestion(row: dict) -> float:
    vc   = row["VehicleCount"]
    st   = row["Stopped"]
    wp   = row["WrongParked"]
    spd  = max(row["EstSpeed"], 1)
    lns  = max(row["Lanes"], 1)
    ped  = row["Pedestrians"]
    sig  = row["Signals"]
    peak = row["PeakHour"]
    wx   = row["WeatherImpact"]
    bus  = row["BusRatio"]
    bike = row["BikeRatio"]
    disc = row.get("LaneDisciplineCode", 0.5)

    base       = (0.4 * vc) + (2.5 * st) + (2.0 * wp)
    spd_pen    = ((80 - min(spd, 80)) / 80) * 25
    cap_stress = (vc / lns) * 1.5
    ped_imp    = ped * (0.4 if lns <= 2 else 0.15)
    sig_delay  = sig * 3.0 * (1.5 if peak > 0.5 else 0.8)
    hv_fric    = (bus + bike * 0.5) * vc * 0.3
    disc_pen   = (1 - disc) * 10
    wx_mult    = 1.0 + (wx * 0.6)

    raw = base + spd_pen + cap_stress + ped_imp + sig_delay + hv_fric + disc_pen
    return float(np.clip(raw * wx_mult, 0, 100))


def build_knowledge_dataset(output_path: str = "road_knowledge.csv",
                             n_rows: int = 3000) -> pd.DataFrame:
    records = []
    road_types    = list(ROAD_PROFILES.keys())
    time_types    = list(TIME_PROFILES.keys())
    weather_types = list(WEATHER_PROFILES.keys())
    city_types    = list(CITY_PROFILES.keys())

    rows_per_road = n_rows // len(road_types)

    for road_name in road_types:
        rp = ROAD_PROFILES[road_name]

        for _ in range(rows_per_road):
            t_name = np.random.choice(time_types,    p=[0.15, 0.35, 0.35, 0.15])
            w_name = np.random.choice(weather_types, p=[0.65, 0.25, 0.10])
            c_name = np.random.choice(city_types)

            tp = TIME_PROFILES[t_name]
            wp = WEATHER_PROFILES[w_name]
            cp = CITY_PROFILES[c_name]

            lo, hi = rp["lanes_range"]
            lanes  = int(np.random.randint(lo, hi + 1))

            vc  = np.random.uniform(*rp["vc_range"])  * tp["vc_mult"]   * cp["vc_mult"]   * wp["vc_mult"]
            spd = np.random.uniform(*rp["speed_range"]) * tp["speed_mult"] * wp["speed_mult"]
            spd = float(np.clip(spd, 3, 120))

            sf  = np.random.uniform(*rp["stopped_frac"]) * tp["stop_mult"] * wp["stop_mult"]
            wf  = np.random.uniform(*rp["wp_frac"])       * wp["stop_mult"]
            stopped      = vc * sf
            wrong_parked = vc * wf

            car_r   = np.random.uniform(*rp["car_ratio"])
            bike_r  = np.random.uniform(*rp["bike_ratio"])
            bus_r   = np.random.uniform(*rp["bus_ratio"])
            truck_r = np.random.uniform(*rp["truck_ratio"])
            total   = car_r + bike_r + bus_r + truck_r
            car_r, bike_r, bus_r, truck_r = [x / total for x in [car_r, bike_r, bus_r, truck_r]]

            plo, phi = rp["pedestrians"]
            peds = np.random.uniform(plo, phi) * cp["ped_mult"]
            slo, shi = rp["signals"]
            sigs = int(np.random.randint(slo, shi + 1)) if shi > slo else slo

            has_median = float(np.random.random() < rp["has_median"])
            is_divided = float(np.random.random() < rp["is_divided"])
            disc = float(np.clip(rp["lane_discipline"] * cp["discipline"] + np.random.normal(0, 0.08), 0, 1))

            pol_ow = float(np.clip(rp["policy_oneway"]   + np.random.normal(0, 0.05), 0, 1))
            pol_np = float(np.clip(rp["policy_nopark"]   + np.random.normal(0, 0.05), 0, 1))
            pol_pk = float(np.clip(rp["policy_peak"]     + np.random.normal(0, 0.05), 0, 1))

            row = {
                "VehicleCount"         : round(vc, 1),
                "Density"              : round(vc / lanes, 2),
                "Stopped"              : round(stopped, 1),
                "WrongParked"          : round(wrong_parked, 1),
                "EstSpeed"             : round(spd, 1),
                "Lanes"                : lanes,
                "RoadWidth"            : round(lanes * 3.5, 1),
                "HasMedian"            : has_median,
                "IsDivided"            : is_divided,
                "CarRatio"             : round(car_r,   3),
                "BikeRatio"            : round(bike_r,  3),
                "BusRatio"             : round(bus_r,   3),
                "TruckRatio"           : round(truck_r, 3),
                "VehicleMix"           : round(bike_r + bus_r + truck_r, 3),
                "Pedestrians"          : round(peds, 1),
                "NMVCount"             : round(vc * bike_r, 1),
                "Signals"              : sigs,
                "PeakHour"             : tp["peak"],
                "TimeOfDayCode"        : tp["code"],
                "WeatherImpact"        : wp["impact"],
                "WeatherCode"          : wp["code"],
                "CityTypeCode"         : cp["code"],
                "RoadTypeCode"         : rp["code"],
                "RoadTypeName"         : road_name,
                "LaneDisciplineCode"   : round(disc, 3),
                "PolicySens_OneWay"    : round(pol_ow, 3),
                "PolicySens_NoParking" : round(pol_np, 3),
                "PolicySens_PeakRestr" : round(pol_pk, 3),
            }

            row["CongestionScore"] = round(compute_congestion(row), 2)

            # Small measurement noise
            for col in ["VehicleCount", "Stopped", "WrongParked", "EstSpeed", "Pedestrians"]:
                row[col] = round(max(0, row[col] + np.random.normal(0, row[col] * 0.03)), 1)

            records.append(row)

    df = pd.DataFrame(records)
    print(f"[INFO] Knowledge dataset : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"[INFO] CongestionScore   : {df['CongestionScore'].min():.1f} – "
          f"{df['CongestionScore'].max():.1f}  (mean {df['CongestionScore'].mean():.1f})")
    print(f"[INFO] Road type counts  :")
    for rt, cnt in df["RoadTypeName"].value_counts().items():
        print(f"         {rt:<20}: {cnt}")
    df.to_csv(output_path, index=False)
    print(f"[✓] Saved → {output_path}")
    return df


if __name__ == "__main__":
    df = build_knowledge_dataset("road_knowledge.csv", n_rows=3000)
    print("\n", df[["RoadTypeName","VehicleCount","Stopped","EstSpeed","CongestionScore"]].describe())
