"""
============================================================
  PART 3 — DUAL-SOURCE ML TRAINING PIPELINE
  Traffic Policy Simulator | train_model.py
============================================================

Strategy (3 layers):

  LAYER 1 — KNOWLEDGE BASE
    Train on road_knowledge.csv (3000 rows, 10 road types,
    4 time periods, 3 weather conditions, 5 city types).
    Model learns GENERAL patterns: what makes a highway vs
    a market street vs a school zone congested.

  LAYER 2 — VIDEO ADAPTATION
    When video data (dataset.csv) is available, blend it in
    with oversampling so site-specific ground truth gets
    adequate influence despite smaller size.
    Video weight: 40%  |  Knowledge weight: 60%

  LAYER 3 — STACKED ENSEMBLE
    GradientBoostingRegressor (primary, captures non-linear)
    + RandomForestRegressor (secondary, captures interactions)
    → Ridge meta-learner combines via out-of-fold stacking.

Feature Set (21 features):
  Core (from video)  : VehicleCount, Density, Stopped, WrongParked,
                       EstSpeed, Lanes, RoadWidth, CarRatio, BikeRatio,
                       BusRatio, TruckRatio, VehicleMix, Pedestrians, Signals
  Extended (context) : HasMedian, IsDivided, PeakHour, WeatherImpact,
                       LaneDisciplineCode, PolicySens_OneWay,
                       PolicySens_NoParking, PolicySens_PeakRestr

Target: CongestionScore (0-100, multi-factor physics formula)

Usage:
  python train_model.py                          # auto-generates knowledge data
  python train_model.py --rebuild                # force rebuild knowledge dataset
  python train_model.py --knowledge road_knowledge.csv --video dataset.csv
============================================================
"""

import os, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble        import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model    import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import StandardScaler

# ─────────────────────────────────────────────────────────
#  FEATURE CONFIGURATION
# ─────────────────────────────────────────────────────────
# Core: always available from video extraction
CORE_FEATURES = [
    "VehicleCount", "Density", "Stopped", "WrongParked", "EstSpeed",
    "Lanes", "RoadWidth", "CarRatio", "BikeRatio", "BusRatio",
    "TruckRatio", "VehicleMix", "Pedestrians", "Signals",
]

# Extended: from knowledge dataset; filled with defaults for video rows
EXTENDED_FEATURES = [
    "HasMedian", "IsDivided", "PeakHour", "WeatherImpact",
    "LaneDisciplineCode",
    "PolicySens_OneWay", "PolicySens_NoParking", "PolicySens_PeakRestr",
]

FEATURE_COLS  = CORE_FEATURES + EXTENDED_FEATURES
TARGET_COL    = "CongestionScore"

MODEL_PATH    = "traffic_model.pkl"
SCALER_PATH   = "scaler.pkl"
METADATA_PATH = "model_metadata.pkl"

# Blend ratio: how much influence does video data get
KNOWLEDGE_WEIGHT = 0.60
VIDEO_WEIGHT     = 0.40

# ─────────────────────────────────────────────────────────
#  DEFAULT VALUES
#  Used when extended features are absent from video data
# ─────────────────────────────────────────────────────────
FEATURE_DEFAULTS = {
    "HasMedian"            : 0.30,
    "IsDivided"            : 0.30,
    "PeakHour"             : 0.50,   # unknown → neutral
    "WeatherImpact"        : 0.00,   # assume clear
    "LaneDisciplineCode"   : 0.50,   # average discipline
    "PolicySens_OneWay"    : 0.55,
    "PolicySens_NoParking" : 0.65,
    "PolicySens_PeakRestr" : 0.60,
    # Core fallbacks (for inference only)
    "RoadWidth"  : 10.5,
    "CarRatio"   : 0.55,
    "BikeRatio"  : 0.20,
    "BusRatio"   : 0.12,
    "TruckRatio" : 0.13,
    "VehicleMix" : 0.35,
    "Pedestrians": 5.0,
    "Signals"    : 1.0,
    "Density"    : 8.0,
    "WrongParked": 3.0,
}


# ─────────────────────────────────────────────────────────
#  CONGESTION FORMULA (mirrors knowledge dataset builder)
#  Used to label video rows that lack a CongestionScore
# ─────────────────────────────────────────────────────────
def _formula_congestion(row: pd.Series) -> float:
    vc   = float(row.get("VehicleCount", 0))
    st   = float(row.get("Stopped",      0))
    wp   = float(row.get("WrongParked",  0))
    spd  = max(float(row.get("EstSpeed", 30)), 1)
    lns  = max(float(row.get("Lanes",    2)), 1)
    ped  = float(row.get("Pedestrians",  0))
    sig  = float(row.get("Signals",      0))
    peak = float(row.get("PeakHour",     0.5))
    wx   = float(row.get("WeatherImpact",0))
    bus  = float(row.get("BusRatio",     0.1))
    bike = float(row.get("BikeRatio",    0.2))
    disc = float(row.get("LaneDisciplineCode", 0.5))

    base       = (0.4 * vc) + (2.5 * st) + (2.0 * wp)
    spd_pen    = ((80 - min(spd, 80)) / 80) * 25
    cap_stress = (vc / lns) * 1.5
    ped_imp    = ped * (0.4 if lns <= 2 else 0.15)
    sig_delay  = sig * 3.0 * (1.5 if peak > 0.5 else 0.8)
    hv_fric    = (bus + bike * 0.5) * vc * 0.3
    disc_pen   = (1 - disc) * 10
    wx_mult    = 1.0 + (wx * 0.6)

    return float(np.clip((base + spd_pen + cap_stress + ped_imp + sig_delay + hv_fric + disc_pen) * wx_mult, 0, 100))


# ─────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────
def load_knowledge_dataset(path: str = "road_knowledge.csv",
                            force_rebuild: bool = False) -> pd.DataFrame:
    """Load or auto-generate the road knowledge dataset."""
    if os.path.exists(path) and not force_rebuild:
        print(f"[INFO] Loading knowledge dataset: {path}")
        df = pd.read_csv(path)
        print(f"       {len(df)} rows, {len(df.columns)} cols")
        return df
    print("[INFO] Generating knowledge dataset …")
    from build_knowledge_dataset import build_knowledge_dataset
    return build_knowledge_dataset(path, n_rows=3000)


def load_video_dataset(path: str = "dataset.csv") -> "pd.DataFrame | None":
    """Load video-extracted dataset; return None if too small."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) < 10:
        print(f"[WARNING] Video dataset too small ({len(df)} rows) — skipped.")
        return None
    print(f"[INFO] Video dataset loaded: {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, source: str = "knowledge") -> pd.DataFrame:
    """
    Standardise and fill all feature columns.
    source: "knowledge" | "video"
    """
    df = df.copy()

    # Derived features
    if "VehicleMix" not in df.columns:
        df["VehicleMix"] = (
            df.get("BikeRatio",  0.2) +
            df.get("BusRatio",   0.1) +
            df.get("TruckRatio", 0.1)
        )
    if "RoadWidth" not in df.columns:
        df["RoadWidth"] = df.get("Lanes", 2) * 3.5
    if "Density" not in df.columns:
        df["Density"] = df["VehicleCount"] / df.get("Lanes", pd.Series(2, index=df.index)).clip(lower=1)

    # Fill missing extended features with defaults
    for col, default in FEATURE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # Recompute CongestionScore for video rows using the richer formula
    if source == "video" or TARGET_COL not in df.columns:
        df[TARGET_COL] = df.apply(_formula_congestion, axis=1)

    return df


# ─────────────────────────────────────────────────────────
#  DATASET BLENDING
# ─────────────────────────────────────────────────────────
def blend_datasets(knowledge_df: pd.DataFrame,
                   video_df: "pd.DataFrame | None") -> pd.DataFrame:
    """
    Merge knowledge + video datasets with weighted oversampling.

    The knowledge dataset provides broad priors across road types.
    The video dataset provides site-specific ground truth and
    is oversampled to achieve its target weight (40%).
    """
    k_eng = engineer_features(knowledge_df, source="knowledge")
    k_eng["_src"] = "knowledge"

    if video_df is None:
        print("[INFO] No video data — training on knowledge dataset only.")
        return k_eng[FEATURE_COLS + [TARGET_COL]].dropna()

    v_eng = engineer_features(video_df, source="video")
    v_eng["_src"] = "video"

    n_k = len(k_eng)
    n_v_target = int(n_k * VIDEO_WEIGHT / KNOWLEDGE_WEIGHT)
    n_v_actual = len(v_eng)

    if n_v_actual < n_v_target:
        v_eng = v_eng.sample(n=n_v_target, replace=True, random_state=42)
        print(f"[INFO] Video oversampled {n_v_actual} → {n_v_target} rows (weight={VIDEO_WEIGHT:.0%})")
    else:
        v_eng = v_eng.sample(n=n_v_target, replace=False, random_state=42)
        print(f"[INFO] Video sampled {n_v_target}/{n_v_actual} rows (weight={VIDEO_WEIGHT:.0%})")

    combined = pd.concat(
        [k_eng[FEATURE_COLS + [TARGET_COL, "_src"]],
         v_eng[FEATURE_COLS + [TARGET_COL, "_src"]]],
        ignore_index=True
    ).dropna(subset=FEATURE_COLS + [TARGET_COL])

    n_k2 = (combined["_src"] == "knowledge").sum()
    n_v2 = (combined["_src"] == "video").sum()
    print(f"[INFO] Blended: {n_k2} knowledge + {n_v2} video = {len(combined)} total rows")

    return combined.drop(columns=["_src"])


# ─────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Scale and split dataset. Returns X_tr, X_te, y_tr, y_te, scaler."""
    df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    X  = df[FEATURE_COLS].values.astype(float)
    y  = df[TARGET_COL].values.astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"[INFO] Train: {len(X_tr)}  |  Test: {len(X_te)}  |  Features: {len(FEATURE_COLS)}")
    return X_tr, X_te, y_tr, y_te, scaler


# ─────────────────────────────────────────────────────────
#  STACKED ENSEMBLE
# ─────────────────────────────────────────────────────────
class StackedEnsemble:
    """
    Layer-1: GradientBoostingRegressor + RandomForestRegressor
    Layer-2: Ridge meta-learner trained on out-of-fold predictions
    """
    def __init__(self, gb, rf, meta):
        self.gb   = gb
        self.rf   = rf
        self.meta = meta

    def predict(self, X: np.ndarray) -> np.ndarray:
        p_gb   = self.gb.predict(X)
        p_rf   = self.rf.predict(X)
        meta_X = np.column_stack([p_gb, p_rf])
        return self.meta.predict(meta_X)

    @property
    def feature_importances_(self) -> np.ndarray:
        # Weighted average: GB gets more weight on tabular data
        return self.gb.feature_importances_ * 0.65 + self.rf.feature_importances_ * 0.35


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> StackedEnsemble:
    """
    Train stacked ensemble on training data.
    Uses OOF (out-of-fold) predictions for meta-learner to prevent leakage.
    """
    # ── Full base models ──────────────────────────────────
    print("[INFO] Training GradientBoostingRegressor …")
    gb = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=5,
        min_samples_split=4, min_samples_leaf=2,
        subsample=0.85, max_features="sqrt", random_state=42,
    )
    gb.fit(X_train, y_train)

    print("[INFO] Training RandomForestRegressor …")
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=12,
        min_samples_split=4, min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ── OOF stacking ─────────────────────────────────────
    print("[INFO] Generating out-of-fold predictions for meta-learner …")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_gb = np.zeros(len(X_train))
    oof_rf = np.zeros(len(X_train))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        _gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.06,
                                         max_depth=4, random_state=42)
        _rf = RandomForestRegressor(n_estimators=150, max_depth=10,
                                     random_state=42, n_jobs=-1)
        _gb.fit(X_train[tr_idx], y_train[tr_idx])
        _rf.fit(X_train[tr_idx], y_train[tr_idx])
        oof_gb[val_idx] = _gb.predict(X_train[val_idx])
        oof_rf[val_idx] = _rf.predict(X_train[val_idx])
        print(f"         Fold {fold}/5 done")

    meta_X = np.column_stack([oof_gb, oof_rf])
    meta   = Ridge(alpha=0.5)
    meta.fit(meta_X, y_train)

    print(f"[✓] Meta-learner weights: GB={meta.coef_[0]:.3f}  RF={meta.coef_[1]:.3f}")
    return StackedEnsemble(gb, rf, meta)


# ─────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────
def evaluate_model(model, X_tr, X_te, y_tr, y_te) -> dict:
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    train_r2 = r2_score(y_tr, y_pred_tr)
    test_r2  = r2_score(y_te, y_pred_te)
    mae      = mean_absolute_error(y_te, y_pred_te)
    rmse     = float(np.sqrt(mean_squared_error(y_te, y_pred_te)))

    print("\n" + "═"*58)
    print("       DUAL-SOURCE ENSEMBLE — EVALUATION REPORT")
    print("═"*58)
    print(f"  Train R²              : {train_r2:.4f}")
    print(f"  Test  R²              : {test_r2:.4f}")
    print(f"  Mean Absolute Error   : {mae:.4f}")
    print(f"  Root Mean Sq. Error   : {rmse:.4f}")
    print("═"*58)
    print("\n  Top Feature Importances:")
    imps = model.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLS, imps), key=lambda x: -x[1])[:12]:
        bar = "█" * int(imp * 55)
        print(f"    {feat:<26} {bar}  {imp:.4f}")
    print()
    return {"train_r2": train_r2, "test_r2": test_r2, "MAE": mae, "RMSE": rmse}


# ─────────────────────────────────────────────────────────
#  SAVE / LOAD
# ─────────────────────────────────────────────────────────
def save_artifacts(model, scaler, metadata: dict = None):
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    if metadata:
        joblib.dump(metadata, METADATA_PATH)
    print(f"[✓] Model  → {MODEL_PATH}")
    print(f"[✓] Scaler → {SCALER_PATH}")


def load_model_and_scaler():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise FileNotFoundError("Run train_model.py first.")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


# ─────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────
def predict_congestion(features: dict, model, scaler) -> float:
    """
    Predict CongestionScore for a single feature dict.
    Missing features are filled with FEATURE_DEFAULTS.
    """
    filled = {**FEATURE_DEFAULTS, **features}
    row    = np.array([[filled.get(f, 0) for f in FEATURE_COLS]], dtype=float)
    scaled = scaler.transform(row)
    return float(np.clip(model.predict(scaled)[0], 0, 100))


# ─────────────────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────────────────
def run_full_pipeline(knowledge_csv: str = "road_knowledge.csv",
                      video_csv: str     = "dataset.csv",
                      force_rebuild: bool = False) -> tuple:
    """End-to-end dual-source training pipeline."""
    k_df     = load_knowledge_dataset(knowledge_csv, force_rebuild)
    v_df     = load_video_dataset(video_csv)
    combined = blend_datasets(k_df, v_df)
    X_tr, X_te, y_tr, y_te, scaler = preprocess(combined)
    model    = train_model(X_tr, y_tr)
    metrics  = evaluate_model(model, X_tr, X_te, y_tr, y_te)
    meta     = {
        "feature_cols"    : FEATURE_COLS,
        "target_col"      : TARGET_COL,
        "feature_defaults": FEATURE_DEFAULTS,
        "n_knowledge"     : len(k_df),
        "n_video"         : len(v_df) if v_df is not None else 0,
        "metrics"         : metrics,
    }
    save_artifacts(model, scaler, meta)
    return model, scaler, metrics


# ─────────────────────────────────────────────────────────
#  BACKWARD COMPATIBILITY (app.py calls these directly)
# ─────────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    from feature_extraction import generate_synthetic_dataset
    return generate_synthetic_dataset(csv_path)


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-Source Traffic Model")
    parser.add_argument("--knowledge", default="road_knowledge.csv")
    parser.add_argument("--video",     default="dataset.csv")
    parser.add_argument("--rebuild",   action="store_true",
                        help="Force rebuild knowledge dataset from scratch")
    args = parser.parse_args()

    model, scaler, metrics = run_full_pipeline(args.knowledge, args.video, args.rebuild)
    print(f"\n[✓] Done  |  Test R²: {metrics['test_r2']:.4f}  |  MAE: {metrics['MAE']:.4f}\n")
