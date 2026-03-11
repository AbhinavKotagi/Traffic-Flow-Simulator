"""
============================================================
  PART 4 — POLICY SIMULATION ENGINE
  Traffic Policy Simulator | simulator.py
============================================================
Implements the 3 traffic policies and predicts:
  • CongestionScore reduction
  • Speed improvement estimate

Policies:
  "one_way"          — One-Way Road
  "no_parking"       — No-Parking Enforcement
  "peak_restriction" — Peak-Hour Vehicle Restriction

Usage (standalone):
  python simulator.py
============================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from train_model import FEATURE_COLS, predict_congestion

# ─────────────────────────────────────────────────────────
#  POLICY REGISTRY
# ─────────────────────────────────────────────────────────
POLICIES: Dict[str, Dict] = {
    "one_way": {
        "label"      : "One-Way Road Policy",
        "short"      : "OneWay",
        "icon"       : "↔️→",
        "description": (
            "Converts a bidirectional road to one-way traffic flow. "
            "Reduces overall vehicle volume and minimises stopping conflicts."
        ),
        "modifiers": {
            "VehicleCount": 0.85,
            "Stopped"     : 0.60,
            "WrongParked" : 1.00,
            "EstSpeed"    : 1.25,   # speed improves due to fewer conflicts
        },
        "color": "#2196F3",
        "accent": "#64B5F6",
    },
    "no_parking": {
        "label"      : "No-Parking Enforcement",
        "short"      : "NoParking",
        "icon"       : "🚫🅿",
        "description": (
            "Strictly enforces no-parking zones, clearing illegally parked "
            "vehicles and reducing lane blockages."
        ),
        "modifiers": {
            "VehicleCount": 1.00,
            "Stopped"     : 0.70,
            "WrongParked" : 0.00,   # eliminated entirely
            "EstSpeed"    : 1.15,
        },
        "color": "#FF9800",
        "accent": "#FFB74D",
    },
    "peak_restriction": {
        "label"      : "Peak-Hour Vehicle Restriction",
        "short"      : "PeakRestriction",
        "icon"       : "⏰🚗",
        "description": (
            "Restricts vehicle entry during peak hours using permit or "
            "odd/even number plate rules, reducing overall traffic volume."
        ),
        "modifiers": {
            "VehicleCount": 0.70,
            "Stopped"     : 0.80,
            "WrongParked" : 0.90,
            "EstSpeed"    : 1.20,
        },
        "color": "#9C27B0",
        "accent": "#CE93D8",
    },
}


# ─────────────────────────────────────────────────────────
#  CORE SIMULATION FUNCTION
# ─────────────────────────────────────────────────────────
def simulate_policy(
    features: dict,
    policy_key: str,
    model=None,
    scaler=None,
) -> dict:
    """
    Apply a traffic policy to the given feature set and predict outcomes.

    Workflow:
      1. Compute baseline CongestionScore (formula or ML model)
      2. Apply policy multipliers to features
      3. Compute post-policy CongestionScore
      4. Estimate speed improvement
      5. Return full result dictionary

    Parameters
    ----------
    features   : dict  Input traffic features (VehicleCount, Stopped, etc.)
    policy_key : str   Key from POLICIES dict
    model      : trained RandomForestRegressor (optional; formula used if None)
    scaler     : fitted StandardScaler (optional)

    Returns
    -------
    dict with keys: policy_key, policy_label, inputs, modified_features,
                    baseline_score, modified_score, reduction_pct,
                    baseline_speed, modified_speed, speed_improvement_pct
    """
    if policy_key not in POLICIES:
        raise ValueError(f"Unknown policy '{policy_key}'. Choose: {list(POLICIES.keys())}")

    policy  = POLICIES[policy_key]
    mods    = policy["modifiers"]

    # ── Ensure VehicleMix is present ──────────────────────
    features = _ensure_vehicle_mix(features)

    # ── Baseline ──────────────────────────────────────────
    baseline_score = _compute_score(features, model, scaler)
    baseline_speed = float(features.get("EstSpeed", 25))

    # ── Apply policy modifiers ─────────────────────────────
    modified = dict(features)
    for feat, mult in mods.items():
        if feat in modified:
            modified[feat] = max(0.0, modified[feat] * mult)

    modified = _ensure_vehicle_mix(modified)   # recompute mix after change

    # ── Post-policy prediction ─────────────────────────────
    modified_score = _compute_score(modified, model, scaler)
    modified_speed = min(float(modified.get("EstSpeed", baseline_speed)), 80.0)

    # ── Derived KPIs ──────────────────────────────────────
    reduction_pct      = _pct_change(baseline_score, modified_score)
    speed_improve_pct  = _pct_change(baseline_speed, modified_speed, higher_is_better=True)

    return {
        "policy_key"          : policy_key,
        "policy_label"        : policy["label"],
        "policy_short"        : policy["short"],
        "policy_icon"         : policy["icon"],
        "policy_color"        : policy["color"],
        "inputs"              : dict(features),
        "modified_features"   : {
            k: round(v, 2) for k, v in modified.items()
            if k in ("VehicleCount", "Stopped", "WrongParked", "EstSpeed", "VehicleMix")
        },
        "baseline_score"      : round(baseline_score,     2),
        "modified_score"      : round(modified_score,     2),
        "reduction_pct"       : round(reduction_pct,      2),
        "baseline_speed"      : round(baseline_speed,     1),
        "modified_speed"      : round(modified_speed,     1),
        "speed_improvement_pct": round(speed_improve_pct, 2),
    }


# ─────────────────────────────────────────────────────────
#  TREND SIMULATION
# ─────────────────────────────────────────────────────────
def simulate_trend(
    base_features: dict,
    policy_key: str,
    model=None,
    scaler=None,
    vc_range: Tuple[int, int] = (5, 80),
    steps: int = 35,
) -> pd.DataFrame:
    """
    Sweep VehicleCount across a range and compute congestion scores
    for baseline and after-policy — used for the Trend Graph.

    Returns DataFrame: VehicleCount | Baseline | AfterPolicy
    """
    vc_values = np.linspace(vc_range[0], vc_range[1], steps)
    rows = []
    for vc in vc_values:
        feats = dict(base_features)
        feats["VehicleCount"] = vc
        feats["Density"]      = vc / max(feats.get("Lanes", 2), 1)
        feats = _ensure_vehicle_mix(feats)

        result = simulate_policy(feats, policy_key, model, scaler)
        rows.append({
            "VehicleCount": round(vc, 1),
            "Baseline"    : result["baseline_score"],
            "AfterPolicy" : result["modified_score"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
def _ensure_vehicle_mix(features: dict) -> dict:
    """Add VehicleMix key if absent (needed by ML model)."""
    f = dict(features)
    if "VehicleMix" not in f:
        f["VehicleMix"] = f.get("BikeRatio", 0.2) + f.get("BusRatio", 0.1) + f.get("TruckRatio", 0.05)
    return f


def _compute_score(features: dict, model, scaler) -> float:
    """Use ML model if available, else formula."""
    if model is not None and scaler is not None:
        try:
            return predict_congestion(features, model, scaler)
        except Exception:
            pass
    # Fallback formula
    vc  = features.get("VehicleCount", 0)
    st  = features.get("Stopped",      0)
    return (0.5 * vc) + (2 * st)


def _pct_change(before: float, after: float, higher_is_better: bool = False) -> float:
    """Percentage change. Positive = improvement (reduction in congestion / increase in speed)."""
    if before == 0:
        return 0.0
    raw = (before - after) / before * 100
    return raw if not higher_is_better else -raw


def format_result(result: dict) -> str:
    """Human-readable one-liner output."""
    return (
        f"Policy: {result['policy_short']} | "
        f"Congestion Reduced: {result['reduction_pct']:.1f}% | "
        f"Speed Improved: {result['speed_improvement_pct']:.1f}%"
    )


# ─────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "VehicleCount": 45,
        "Stopped"     : 8,
        "WrongParked" : 4,
        "EstSpeed"    : 22,
        "Lanes"       : 4,
        "Density"     : 11.25,
        "BikeRatio"   : 0.25,
        "BusRatio"    : 0.12,
        "TruckRatio"  : 0.08,
        "Pedestrians" : 6,
    }

    print("\n" + "═"*60)
    print("         POLICY SIMULATION — STANDALONE TEST")
    print("═"*60)

    for pk in POLICIES:
        res = simulate_policy(sample, pk)
        print(f"\n  {res['policy_icon']}  {res['policy_label']}")
        print(f"     Baseline Score  : {res['baseline_score']}")
        print(f"     After Policy    : {res['modified_score']}")
        print(f"     Congestion ↓    : {res['reduction_pct']}%")
        print(f"     Speed ↑         : {res['speed_improvement_pct']}%")

    print("\n" + "═"*60 + "\n")
