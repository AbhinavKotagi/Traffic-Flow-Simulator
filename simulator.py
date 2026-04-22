"""
============================================================
  POLICY SIMULATION ENGINE — Bangalore Traffic
  TrafficIQ Bangalore | simulator.py
============================================================
Four Bangalore-specific traffic policies:

  1. Signal Optimisation   — Adaptive signal timing, better compliance
  2. Parking Enforcement   — Zero-tolerance on road-side parking
  3. Peak Hour Restriction — ODD/EVEN or permit-based entry 07–10, 17–20
  4. Public Transport Boost— BMTC frequency increase, incentives

Each policy modifies a subset of the 16 features and the
ML model predicts congestion before and after.
============================================================
"""

import numpy as np
import pandas as pd

from data_processor  import ROAD_TYPE_MAP, POLICY_SENS, FEATURE_DEFAULTS
from train_model     import predict_congestion

# ─────────────────────────────────────────────────────────
#  POLICY DEFINITIONS
# ─────────────────────────────────────────────────────────
POLICIES = {
    "signal_opt": {
        "label"      : "Signal Optimisation",
        "icon"       : "🚦",
        "color"      : "#34d399",
        "description": "Adaptive signal timing + strict compliance enforcement. Reduces junction delays and stop-and-go cycles.",
        "modifiers"  : {
            # What changes when you optimise signals
            "SignalCompliance"    : lambda v: min(v * 1.35, 100),   # +35% compliance
            "TravelTimeIndex"    : lambda v: v * 0.80,              # 20% less delay
            "AverageSpeed"       : lambda v: min(v * 1.18, 90),    # speed improves
            "TrafficVolume"      : lambda v: v * 0.95,             # slight spread
        },
    },
    "parking_enf": {
        "label"      : "Parking Enforcement",
        "icon"       : "🚫",
        "color"      : "#60a5fa",
        "description": "Zero-tolerance no-parking zones on arterials + smart parking redirection. Frees lane capacity.",
        "modifiers"  : {
            "ParkingUsage"       : lambda v: v * 0.30,             # 70% reduction
            "RoadCapacityUtil"   : lambda v: max(v * 0.82, 0),    # capacity freed
            "AverageSpeed"       : lambda v: min(v * 1.15, 90),
            "TravelTimeIndex"    : lambda v: v * 0.88,
        },
    },
    "peak_restr": {
        "label"      : "Peak Hour Restriction",
        "icon"       : "⏱",
        "color"      : "#f0c040",
        "description": "ODD/EVEN vehicle scheme or permit-based entry during 07:00–10:00 and 17:00–20:00.",
        "modifiers"  : {
            "TrafficVolume"      : lambda v: v * 0.72,             # 28% volume cut
            "RoadCapacityUtil"   : lambda v: v * 0.74,
            "AverageSpeed"       : lambda v: min(v * 1.22, 90),
            "TravelTimeIndex"    : lambda v: v * 0.78,
            "IncidentReports"    : lambda v: v * 0.80,
        },
    },
    "pt_boost": {
        "label"      : "Public Transport Boost",
        "icon"       : "🚌",
        "color"      : "#a78bfa",
        "description": "BMTC frequency doubled + dedicated bus lanes + last-mile connectivity incentives.",
        "modifiers"  : {
            "PublicTransportUsage": lambda v: min(v * 1.55, 100),  # +55% PT usage
            "TrafficVolume"       : lambda v: v * 0.80,            # modal shift
            "RoadCapacityUtil"    : lambda v: v * 0.82,
            "AverageSpeed"        : lambda v: min(v * 1.12, 90),
            "TravelTimeIndex"     : lambda v: v * 0.85,
            "EnvironmentalImpact" : lambda v: v * 0.78,            # cleaner air
        },
    },
}


# ─────────────────────────────────────────────────────────
#  POLICY SENSITIVITY SCALING
#  Not all roads benefit equally from each policy.
#  e.g. signal_opt is most impactful at junctions.
# ─────────────────────────────────────────────────────────
def _policy_scale(road_type: str, policy_key: str) -> float:
    """Return 0-1 effectiveness scale for this policy on this road type."""
    sens = POLICY_SENS.get(road_type, POLICY_SENS["arterial_main"])
    return sens.get(policy_key, 0.65)


# ─────────────────────────────────────────────────────────
#  SIMULATE SINGLE POLICY
# ─────────────────────────────────────────────────────────
def simulate_policy(features: dict, policy_key: str,
                    model, scaler, road_name: str = "") -> dict:
    """
    Apply one policy to a feature dict and measure congestion change.

    Parameters
    ----------
    features   : dict of feature values for the location
    policy_key : one of "signal_opt", "parking_enf", "peak_restr", "pt_boost"
    model, scaler : trained ML artifacts
    road_name  : used to get road-type-specific effectiveness scaling

    Returns
    -------
    dict with baseline, modified scores, reductions, and metadata
    """
    policy    = POLICIES[policy_key]
    road_type = ROAD_TYPE_MAP.get(road_name, "arterial_main")
    scale     = _policy_scale(road_type, policy_key)

    # Baseline prediction
    baseline_score = predict_congestion(features, model, scaler)
    baseline_speed = features.get("AverageSpeed", 38.0)
    baseline_tti   = features.get("TravelTimeIndex", 1.3)
    baseline_cap   = features.get("RoadCapacityUtil", 85.0)

    # Apply policy modifiers (scaled by road-type effectiveness)
    modified = dict(features)
    for feat, modifier_fn in policy["modifiers"].items():
        if feat in modified:
            original = modified[feat]
            target   = modifier_fn(original)
            # Scale: full change × effectiveness, partial otherwise
            modified[feat] = original + (target - original) * scale

    # Modified prediction
    modified_score = predict_congestion(modified, model, scaler)
    modified_speed = modified.get("AverageSpeed", baseline_speed)
    modified_tti   = modified.get("TravelTimeIndex", baseline_tti)
    modified_cap   = modified.get("RoadCapacityUtil", baseline_cap)

    reduction_pct   = round((baseline_score - modified_score) / max(baseline_score, 1) * 100, 2)
    speed_gain_pct  = round((modified_speed - baseline_speed) / max(baseline_speed, 1) * 100, 2)
    tti_improvement = round(baseline_tti - modified_tti, 3)
    cap_freed       = round(baseline_cap - modified_cap, 2)

    return {
        "policy_key"       : policy_key,
        "policy_label"     : policy["label"],
        "policy_icon"      : policy["icon"],
        "policy_color"     : policy["color"],
        "policy_description": policy["description"],
        "road_type"        : road_type,
        "effectiveness"    : round(scale * 100, 1),
        "inputs"           : dict(features),
        "modified_features": modified,
        "baseline_score"   : round(baseline_score, 2),
        "modified_score"   : round(modified_score, 2),
        "reduction_pct"    : reduction_pct,
        "baseline_speed"   : round(baseline_speed, 1),
        "modified_speed"   : round(modified_speed, 1),
        "speed_gain_pct"   : speed_gain_pct,
        "tti_improvement"  : tti_improvement,
        "cap_freed_pct"    : cap_freed,
    }


# ─────────────────────────────────────────────────────────
#  SIMULATE ALL POLICIES
# ─────────────────────────────────────────────────────────
def simulate_all_policies(features: dict, model, scaler,
                           road_name: str = "") -> dict:
    """Run all 4 policies and return results dict keyed by policy_key."""
    return {
        key: simulate_policy(features, key, model, scaler, road_name)
        for key in POLICIES
    }


# ─────────────────────────────────────────────────────────
#  TREND — sweep traffic volume
# ─────────────────────────────────────────────────────────
def simulate_trend(base_features: dict, policy_key: str,
                   model, scaler, road_name: str = "") -> pd.DataFrame:
    """
    Sweep TrafficVolume from 5000 → 70000 and compute
    baseline vs policy congestion at each point.
    """
    policy    = POLICIES[policy_key]
    road_type = ROAD_TYPE_MAP.get(road_name, "arterial_main")
    scale     = _policy_scale(road_type, policy_key)

    volumes  = np.linspace(5000, 70000, 50)
    rows     = []
    for vol in volumes:
        f = dict(base_features)
        f["TrafficVolume"]    = vol
        f["RoadCapacityUtil"] = min(vol / 700, 100)

        baseline = predict_congestion(f, model, scaler)

        mod = dict(f)
        for feat, fn in policy["modifiers"].items():
            if feat in mod:
                orig = mod[feat]
                mod[feat] = orig + (fn(orig) - orig) * scale
        after = predict_congestion(mod, model, scaler)

        rows.append({
            "TrafficVolume": int(vol),
            "Baseline"     : round(baseline, 2),
            "AfterPolicy"  : round(after, 2),
            "Gap"          : round(baseline - after, 2),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
#  COMBINED POLICIES
# ─────────────────────────────────────────────────────────
def simulate_combined(features: dict, policy_keys: list,
                      model, scaler, road_name: str = "") -> dict:
    """
    Apply multiple policies simultaneously and predict combined effect.
    """
    road_type = ROAD_TYPE_MAP.get(road_name, "arterial_main")
    baseline  = predict_congestion(features, model, scaler)
    modified  = dict(features)

    for key in policy_keys:
        policy = POLICIES[key]
        scale  = _policy_scale(road_type, key)
        for feat, fn in policy["modifiers"].items():
            if feat in modified:
                orig = modified[feat]
                modified[feat] = orig + (fn(orig) - orig) * scale

    after       = predict_congestion(modified, model, scaler)
    reduction   = round((baseline - after) / max(baseline, 1) * 100, 2)
    speed_gain  = round(
        (modified.get("AverageSpeed", features.get("AverageSpeed", 38))
         - features.get("AverageSpeed", 38))
        / max(features.get("AverageSpeed", 38), 1) * 100, 2)

    return {
        "policies_applied" : [POLICIES[k]["label"] for k in policy_keys],
        "baseline_score"   : round(baseline, 2),
        "modified_score"   : round(after, 2),
        "reduction_pct"    : reduction,
        "speed_gain_pct"   : speed_gain,
        "modified_features": modified,
    }


if __name__ == "__main__":
    from train_model import load_artifacts, FEATURE_DEFAULTS
    model, scaler = load_artifacts()
    feats = dict(FEATURE_DEFAULTS)
    feats.update({"TrafficVolume": 45000, "AverageSpeed": 28.0,
                  "RoadCapacityUtil": 97.0, "ParkingUsage": 85.0,
                  "SignalCompliance": 60.0})
    print("\nSimulation — Silk Board Junction style features:")
    for key in POLICIES:
        r = simulate_policy(feats, key, model, scaler, "Silk Board Junction")
        print(f"  {r['policy_icon']} {r['policy_label']:<28} "
              f"▼{r['reduction_pct']:.1f}%  speed +{r['speed_gain_pct']:.1f}%  "
              f"(effectiveness: {r['effectiveness']:.0f}%)")
