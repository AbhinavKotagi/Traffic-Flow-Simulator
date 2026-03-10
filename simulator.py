"""
============================================================
 PART 3 — POLICY SIMULATION ENGINE
============================================================
Defines and applies the 3 traffic policies.
Core function: simulate_policy(data, policy_name, model, scaler)

Policies:
  1. one_way          — One-Way Road Policy
  2. no_parking       — No-Parking Enforcement
  3. peak_restriction — Peak-Hour Vehicle Restriction

Usage (standalone test):
    python simulator.py
============================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

# ── Policy definitions ────────────────────────────────────
# Each policy maps feature names to multipliers applied to
# the baseline values before ML prediction.

POLICIES: Dict[str, Dict] = {
    "one_way": {
        "label"      : "One-Way Road Policy",
        "description": "Restricts traffic to one direction, reducing vehicle volume and stopping.",
        "modifiers"  : {
            "VehicleCount": 0.85,
            "WrongParked" : 1.00,   # no direct effect
            "Stopped"     : 0.60,
        },
        "color": "#2196F3",
    },
    "no_parking": {
        "label"      : "No-Parking Enforcement",
        "description": "Clears illegally parked vehicles and reduces lane blockages.",
        "modifiers"  : {
            "VehicleCount": 1.00,
            "WrongParked" : 0.00,   # cleared entirely
            "Stopped"     : 0.70,
        },
        "color": "#FF9800",
    },
    "peak_restriction": {
        "label"      : "Peak-Hour Vehicle Restriction",
        "description": "Limits vehicle entry during peak hours using odd/even or permit rules.",
        "modifiers"  : {
            "VehicleCount": 0.70,
            "WrongParked" : 1.00,
            "Stopped"     : 0.80,
        },
        "color": "#9C27B0",
    },
}


def apply_policy_modifiers(
    vehicle_count: float,
    wrong_parked:  float,
    stopped:       float,
    policy_key:    str,
) -> Tuple[float, float, float]:
    """
    Apply policy multipliers to raw feature values.

    Parameters
    ----------
    vehicle_count, wrong_parked, stopped : baseline feature values
    policy_key : one of 'one_way', 'no_parking', 'peak_restriction'

    Returns
    -------
    Tuple of (modified_vehicle_count, modified_wrong_parked, modified_stopped)
    """
    if policy_key not in POLICIES:
        raise ValueError(f"Unknown policy '{policy_key}'. Choose from: {list(POLICIES.keys())}")

    mods = POLICIES[policy_key]["modifiers"]
    return (
        max(0, vehicle_count * mods["VehicleCount"]),
        max(0, wrong_parked  * mods["WrongParked"]),
        max(0, stopped       * mods["Stopped"]),
    )


def compute_congestion_score(vehicle_count: float, wrong_parked: float, stopped: float) -> float:
    """
    Compute CongestionScore using the defined formula.
        CongestionScore = 0.5×VehicleCount + 2×WrongParked + 2×Stopped
    """
    return (0.5 * vehicle_count) + (2 * wrong_parked) + (2 * stopped)


def simulate_policy(
    vehicle_count: float,
    wrong_parked:  float,
    stopped:       float,
    policy_key:    str,
    model=None,
    scaler=None,
) -> Dict:
    """
    Main simulation function.

    Applies policy modifiers to inputs, predicts congestion
    using ML model (or formula fallback), and returns
    a full result dictionary.

    Parameters
    ----------
    vehicle_count : baseline vehicle count
    wrong_parked  : baseline wrong-parked vehicles
    stopped       : baseline stopped vehicles
    policy_key    : policy identifier string
    model         : trained RandomForestRegressor (optional)
    scaler        : fitted StandardScaler (optional)

    Returns
    -------
    dict with keys:
        policy_key, policy_label,
        baseline_score, modified_score,
        reduction_pct, modified_features, inputs
    """
    # ── Baseline score ────────────────────────────────────
    baseline_score = compute_congestion_score(vehicle_count, wrong_parked, stopped)

    # Optionally use ML model for baseline
    if model is not None and scaler is not None:
        X_base         = np.array([[vehicle_count, wrong_parked, stopped]])
        X_base_scaled  = scaler.transform(X_base)
        baseline_score = float(model.predict(X_base_scaled)[0])

    # ── Apply policy ──────────────────────────────────────
    mod_vc, mod_wp, mod_st = apply_policy_modifiers(
        vehicle_count, wrong_parked, stopped, policy_key
    )

    # ── Modified score ────────────────────────────────────
    modified_score = compute_congestion_score(mod_vc, mod_wp, mod_st)

    if model is not None and scaler is not None:
        X_mod         = np.array([[mod_vc, mod_wp, mod_st]])
        X_mod_scaled  = scaler.transform(X_mod)
        modified_score = float(model.predict(X_mod_scaled)[0])

    # ── Reduction percentage ──────────────────────────────
    reduction_pct = ((baseline_score - modified_score) / baseline_score * 100) if baseline_score > 0 else 0

    return {
        "policy_key"      : policy_key,
        "policy_label"    : POLICIES[policy_key]["label"],
        "inputs"          : {
            "VehicleCount": vehicle_count,
            "WrongParked" : wrong_parked,
            "Stopped"     : stopped,
        },
        "modified_features": {
            "VehicleCount": round(mod_vc, 2),
            "WrongParked" : round(mod_wp, 2),
            "Stopped"     : round(mod_st, 2),
        },
        "baseline_score"  : round(baseline_score, 2),
        "modified_score"  : round(modified_score, 2),
        "reduction_pct"   : round(reduction_pct, 2),
    }


def simulate_trend(
    wrong_parked: float,
    stopped:      float,
    policy_key:   str,
    model=None,
    scaler=None,
    vc_range=(5, 60),
    steps=30,
) -> pd.DataFrame:
    """
    Simulate congestion vs vehicle count for a trend graph.

    Returns DataFrame with columns:
        VehicleCount, Baseline, AfterPolicy
    """
    vc_values = np.linspace(vc_range[0], vc_range[1], steps)
    rows = []
    for vc in vc_values:
        result = simulate_policy(vc, wrong_parked, stopped, policy_key, model, scaler)
        rows.append({
            "VehicleCount": round(vc, 1),
            "Baseline"    : result["baseline_score"],
            "AfterPolicy" : result["modified_score"],
        })
    return pd.DataFrame(rows)


def format_result_summary(result: Dict) -> str:
    """Return a formatted single-line summary of simulation result."""
    return (
        f"Policy: {result['policy_label']} | "
        f"Before: {result['baseline_score']:.1f} | "
        f"After: {result['modified_score']:.1f} | "
        f"Congestion Reduced: {result['reduction_pct']:.1f}%"
    )


# ── Standalone test ───────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("        TRAFFIC POLICY SIMULATION TEST")
    print("="*60)

    # Test all 3 policies with sample inputs
    sample = dict(vehicle_count=35, wrong_parked=4, stopped=6)

    for key in POLICIES:
        result = simulate_policy(**sample, policy_key=key)
        print(f"\n  {result['policy_label']}")
        print(f"  Baseline Score  : {result['baseline_score']}")
        print(f"  After Policy    : {result['modified_score']}")
        print(f"  Reduction       : {result['reduction_pct']}%")
        print(f"  Modified Inputs : {result['modified_features']}")

    print("\n" + "="*60 + "\n")
