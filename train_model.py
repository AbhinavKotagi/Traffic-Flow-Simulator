"""
============================================================
 PART 2 — ML MODEL TRAINING (RandomForestRegressor)
============================================================
Trains a Random Forest model to predict CongestionScore
from extracted traffic features.

Inputs  : VehicleCount, WrongParked, Stopped
Output  : CongestionScore (regression)

Usage:
    python train_model.py
    python train_model.py --csv my_data.csv

Saves trained model to: traffic_model.pkl
============================================================
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble          import RandomForestRegressor
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing     import StandardScaler

# ── Feature & target column names ────────────────────────
FEATURES      = ["VehicleCount", "WrongParked", "Stopped"]
TARGET        = "CongestionScore"
MODEL_PATH    = "traffic_model.pkl"
SCALER_PATH   = "scaler.pkl"


def load_or_generate_data(csv_path: str) -> pd.DataFrame:
    """Load CSV dataset; generate synthetic one if not found."""
    if os.path.exists(csv_path):
        print(f"[INFO] Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[WARNING] {csv_path} not found. Generating synthetic data …")
        from feature_extraction import generate_synthetic_dataset
        df = generate_synthetic_dataset(csv_path)
    return df


def preprocess(df: pd.DataFrame):
    """
    Clean data and split into train / test sets.
    Returns X_train, X_test, y_train, y_test, scaler
    """
    # Drop rows with any missing values in required columns
    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES].values
    y = df[TARGET].values

    # Scale features for better RF performance
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"[INFO] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor with tuned hyperparameters.
    """
    model = RandomForestRegressor(
        n_estimators=200,      # number of trees
        max_depth=10,          # limit depth to prevent overfitting
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,             # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("[✓] Model training complete.")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Print comprehensive model evaluation metrics.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2   = r2_score(y_test, y_pred_test)

    print("\n" + "="*50)
    print("         MODEL EVALUATION REPORT")
    print("="*50)
    print(f"  Train R² Score    : {r2_score(y_train, y_pred_train):.4f}")
    print(f"  Test  R² Score    : {r2:.4f}")
    print(f"  Mean Abs Error    : {mae:.4f}")
    print(f"  Root Mean Sq Err  : {rmse:.4f}")
    print("="*50)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"  5-Fold CV R²      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("="*50 + "\n")

    # Feature importances
    importances = model.feature_importances_
    print("  Feature Importances:")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {feat:<18} {bar}  ({imp:.4f})")
    print()

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_artifacts(model, scaler):
    """Save trained model and scaler to disk using joblib."""
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[✓] Model saved  → {MODEL_PATH}")
    print(f"[✓] Scaler saved → {SCALER_PATH}")


def load_model_and_scaler():
    """Load and return saved model + scaler."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ── CLI entrypoint ────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traffic Congestion ML Model")
    parser.add_argument("--csv", type=str, default="dataset.csv", help="Path to features CSV")
    args = parser.parse_args()

    # 1. Load data
    df = load_or_generate_data(args.csv)

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # 3. Train
    model = train_model(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 5. Save
    save_artifacts(model, scaler)

    print("[✓] Training pipeline complete. Ready for simulation.\n")
