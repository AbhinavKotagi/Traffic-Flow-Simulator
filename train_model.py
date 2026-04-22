"""
============================================================
  ML TRAINING PIPELINE — Bangalore Traffic Data
  TrafficIQ Bangalore | train_model.py
============================================================
Trains a stacked ensemble on the Bangalore traffic CSV.

Architecture:
  GradientBoostingRegressor  +  RandomForestRegressor
            ↓ out-of-fold stacking
        Ridge meta-learner

Features (16): TrafficVolume, AverageSpeed, TravelTimeIndex,
  RoadCapacityUtil, IncidentReports, EnvironmentalImpact,
  PublicTransportUsage, SignalCompliance, ParkingUsage,
  PedestrianCount, WeatherCode, RoadworkActive,
  RoadTypeCode, AreaCode, DayOfWeek, IsWeekend

Target: CongestionLevel (0–100)

Usage:
  python train_model.py
  python train_model.py --data Banglore_traffic_Dataset.csv
============================================================
"""

import os, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble        import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model    import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import StandardScaler

from data_processor import (
    load_and_engineer, FEATURE_COLS, TARGET_COL, FEATURE_DEFAULTS, DATA_PATH
)

MODEL_PATH    = "traffic_model.pkl"
SCALER_PATH   = "scaler.pkl"
METADATA_PATH = "model_metadata.pkl"


# ─────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Scale features and split. Returns X_tr, X_te, y_tr, y_te, scaler."""
    data = df[FEATURE_COLS + [TARGET_COL]].dropna()
    X    = data[FEATURE_COLS].values.astype(float)
    y    = data[TARGET_COL].values.astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    print(f"[INFO] Train: {len(X_tr)}  Test: {len(X_te)}  Features: {len(FEATURE_COLS)}")
    return X_tr, X_te, y_tr, y_te, scaler


# ─────────────────────────────────────────────────────────
#  STACKED ENSEMBLE
# ─────────────────────────────────────────────────────────
class StackedEnsemble:
    def __init__(self, gb, rf, meta):
        self.gb   = gb
        self.rf   = rf
        self.meta = meta

    def predict(self, X):
        p_gb   = self.gb.predict(X)
        p_rf   = self.rf.predict(X)
        return self.meta.predict(np.column_stack([p_gb, p_rf]))

    @property
    def feature_importances_(self):
        return self.gb.feature_importances_ * 0.65 + self.rf.feature_importances_ * 0.35


def train_model(X_tr, y_tr) -> StackedEnsemble:
    print("[INFO] Training GradientBoostingRegressor …")
    gb = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=5,
        min_samples_split=4, min_samples_leaf=2,
        subsample=0.85, max_features="sqrt", random_state=42,
    )
    gb.fit(X_tr, y_tr)

    print("[INFO] Training RandomForestRegressor …")
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=12,
        min_samples_split=4, min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    print("[INFO] Stacking: generating OOF predictions …")
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_gb = np.zeros(len(X_tr))
    oof_rf = np.zeros(len(X_tr))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr), 1):
        _gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.06,
                                         max_depth=4, random_state=42)
        _rf = RandomForestRegressor(n_estimators=150, max_depth=10,
                                     random_state=42, n_jobs=-1)
        _gb.fit(X_tr[tr_idx], y_tr[tr_idx])
        _rf.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_gb[val_idx] = _gb.predict(X_tr[val_idx])
        oof_rf[val_idx] = _rf.predict(X_tr[val_idx])
        print(f"         Fold {fold}/5")

    meta = Ridge(alpha=0.5)
    meta.fit(np.column_stack([oof_gb, oof_rf]), y_tr)
    print(f"[✓] Meta weights: GB={meta.coef_[0]:.3f}  RF={meta.coef_[1]:.3f}")
    return StackedEnsemble(gb, rf, meta)


# ─────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────
def evaluate_model(model, X_tr, X_te, y_tr, y_te) -> dict:
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    train_r2  = r2_score(y_tr, y_pred_tr)
    test_r2   = r2_score(y_te, y_pred_te)
    mae       = mean_absolute_error(y_te, y_pred_te)
    rmse      = float(np.sqrt(mean_squared_error(y_te, y_pred_te)))

    print("\n" + "═"*56)
    print("       BANGALORE MODEL — EVALUATION REPORT")
    print("═"*56)
    print(f"  Train R²   : {train_r2:.4f}")
    print(f"  Test  R²   : {test_r2:.4f}")
    print(f"  MAE        : {mae:.4f}")
    print(f"  RMSE       : {rmse:.4f}")
    print("═"*56)
    print("  Top Feature Importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_),
                            key=lambda x: -x[1])[:10]:
        print(f"    {feat:<28} {'█'*int(imp*55)}  {imp:.4f}")
    print()
    return {"train_r2": train_r2, "test_r2": test_r2, "MAE": mae, "RMSE": rmse}


# ─────────────────────────────────────────────────────────
#  SAVE / LOAD
# ─────────────────────────────────────────────────────────
def save_artifacts(model, scaler, metadata=None):
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    if metadata:
        joblib.dump(metadata, METADATA_PATH)
    print(f"[✓] Saved model  → {MODEL_PATH}")
    print(f"[✓] Saved scaler → {SCALER_PATH}")


def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise FileNotFoundError("Run train_model.py first.")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


# ─────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────
def predict_congestion(features: dict, model, scaler) -> float:
    """Predict CongestionLevel for a feature dict. Fills missing with defaults."""
    filled = {**FEATURE_DEFAULTS, **features}
    row    = np.array([[filled.get(f, 0) for f in FEATURE_COLS]], dtype=float)
    return float(np.clip(model.predict(scaler.transform(row))[0], 0, 100))


# ─────────────────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────────────────
def run_full_pipeline(data_csv: str = DATA_PATH) -> tuple:
    df                             = load_and_engineer(data_csv)
    X_tr, X_te, y_tr, y_te, scaler = preprocess(df)
    model                          = train_model(X_tr, y_tr)
    metrics                        = evaluate_model(model, X_tr, X_te, y_tr, y_te)
    save_artifacts(model, scaler, {
        "feature_cols": FEATURE_COLS, "target_col": TARGET_COL,
        "metrics": metrics, "n_rows": len(df),
    })
    return model, scaler, metrics, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    args   = parser.parse_args()
    model, scaler, metrics, _ = run_full_pipeline(args.data)
    print(f"\n[✓] Done  Test R²: {metrics['test_r2']:.4f}  MAE: {metrics['MAE']:.4f}\n")
