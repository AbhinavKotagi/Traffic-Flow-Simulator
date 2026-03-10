# 🚦 AI-Based Traffic Policy Simulation System
### Using Video Data | YOLOv8 + OpenCV + RandomForest + Streamlit

> A decision-support tool for traffic authorities to simulate real-world policy changes using computer vision and machine learning.

---

## 📌 Project Overview

This system:
1. **Extracts** structured traffic data from videos using YOLOv8 + OpenCV
2. **Trains** a RandomForest ML model on extracted features
3. **Simulates** 3 traffic policy changes with predicted outcomes
4. **Provides** an interactive Streamlit dashboard with graphs and scenario history

---

## 📂 Project Structure

```
traffic-policy-simulator/
│
├── feature_extraction.py   # PART 1 — YOLOv8 video processing
├── train_model.py          # PART 2 — ML model training
├── simulator.py            # PART 3 — Policy simulation engine
├── app.py                  # PART 4/5 — Streamlit interactive app
├── generate_dataset.py     # Helper: generate synthetic demo dataset
├── dataset.csv             # Extracted/synthetic traffic data
├── traffic_model.pkl       # Trained RandomForest model (auto-generated)
├── scaler.pkl              # Feature scaler (auto-generated)
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly scikit-learn pandas numpy joblib opencv-python ultralytics
```

### 2. Generate Dataset (demo mode)

```bash
python feature_extraction.py --demo
```

### 3. Train the Model

```bash
python train_model.py
```

### 4. Launch the App

```bash
streamlit run app.py
```

---

## 🎥 Using Real Video Data

If you have a traffic video file:

```bash
python feature_extraction.py --video traffic_footage.mp4 --output dataset.csv
```

This will:
- Run YOLOv8 object tracking on each frame
- Detect cars, trucks, buses, motorcycles
- Estimate stopped/wrong-parked vehicles via motion analysis
- Save all features to `dataset.csv`

---

## 🧠 Features Extracted

| Feature        | Description                                  |
|----------------|----------------------------------------------|
| `VehicleCount` | Total vehicles detected per frame            |
| `WrongParked`  | Stationary vehicles near road edges          |
| `Stopped`      | Stationary vehicles blocking active lanes    |

### Target Variable

```
CongestionScore = (0.5 × VehicleCount) + (2 × WrongParked) + (2 × Stopped)
```

---

## 🔥 Traffic Policies Simulated

### 1️⃣ One-Way Road Policy
| Feature       | Modifier |
|---------------|----------|
| VehicleCount  | × 0.85   |
| Stopped       | × 0.60   |

### 2️⃣ No-Parking Enforcement
| Feature      | Modifier |
|--------------|----------|
| WrongParked  | × 0.00   |
| Stopped      | × 0.70   |

### 3️⃣ Peak-Hour Vehicle Restriction
| Feature      | Modifier |
|--------------|----------|
| VehicleCount | × 0.70   |
| Stopped      | × 0.80   |

---

## 📊 Visualizations

| Graph | Description |
|-------|-------------|
| **Before vs After Bar Chart** | Compares baseline vs post-policy congestion |
| **Trend Graph** | Congestion score vs vehicle count for current policy |
| **Feature Change Chart** | Shows how policy modifies each input feature |
| **Scenario Comparison** | Current simulation vs previous saved simulation |
| **History Bar Chart** | Reduction % across all simulations in session |

---

## 🤖 ML Model Details

- **Algorithm**: `RandomForestRegressor`
- **Inputs**: `VehicleCount`, `WrongParked`, `Stopped`
- **Output**: `CongestionScore` (regression)
- **Performance**: R² ≈ 0.98, MAE ≈ 0.73

---

## 💡 Output Format

```
Policy: No-Parking Enforcement
Baseline Score: 37.5 → After Policy: 25.9
Congestion Reduced: 30.93%
```

---

## 🧩 Module Usage (Standalone)

```python
from simulator import simulate_policy

result = simulate_policy(
    vehicle_count=35,
    wrong_parked=4,
    stopped=6,
    policy_key="no_parking",  # or "one_way", "peak_restriction"
)

print(f"Reduction: {result['reduction_pct']}%")
```

---

## 📋 Requirements

```
streamlit>=1.28.0
plotly>=5.18.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
opencv-python>=4.8.0       # for video processing
ultralytics>=8.0.0          # for YOLOv8
```

---

## 🎓 Academic Context

- **Purpose**: Final Year Project — AI & Computer Vision
- **Domain**: Intelligent Transportation Systems
- **Methodology**: CV Feature Extraction → ML Regression → Policy Simulation

---

*Built for final-year submission and viva demonstration.*
