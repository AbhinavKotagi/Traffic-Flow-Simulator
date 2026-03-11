# 🚦 TrafficIQ — AI-Based Traffic Policy Simulation System
### YOLOv8 + OpenCV + Grok AI + RandomForest + Streamlit

> A production-grade, decision-support tool for traffic authorities.  
> Upload a traffic video → extract CV features → classify road with AI → simulate policies → compare scenarios.

---

## 🏗 Architecture

```
Traffic Video (MP4)
        │
        ▼
┌───────────────────────┐
│  feature_extraction.py │   YOLOv8 + OpenCV
│  • Vehicle detection   │   • Vehicle count / density
│  • Optical flow        │   • Stopped, wrong-parked
│  • Lane detection      │   • Speed estimate (km/h)
│  • Context detection   │   • Lane count, road width
└──────────┬────────────┘   • Pedestrians, signals
           │
           ▼
┌───────────────────────┐
│  road_classifier.py   │   Grok LLM (xAI)
│  • Feature → prompt   │   • Road Type classification
│  • JSON parse         │   • Lane discipline
│  • Fallback rules     │   • Behavior summary
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  train_model.py       │   RandomForestRegressor
│  • Feature engineer   │   • CongestionScore prediction
│  • Train / evaluate   │   • R² ≈ 0.98
│  • Save .pkl          │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  simulator.py         │   Policy Engine
│  • 3 traffic policies │   • Modifier application
│  • Trend simulation   │   • Speed improvement
│  • Result formatting  │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  app.py               │   Streamlit Dashboard
│  • Video upload       │   • 4 graph types
│  • Manual sliders     │   • Session history
│  • Road profile card  │   • Multi-scenario compare
└───────────────────────┘
```

---

## 📂 Project Structure

```
traffic-policy-sim-v2/
├── feature_extraction.py   # Part 1 — CV feature extraction (YOLOv8 + OpenCV)
├── road_classifier.py      # Part 2 — LLM road classification (Grok API)
├── train_model.py          # Part 3 — ML model (RandomForestRegressor)
├── simulator.py            # Part 4 — Policy simulation engine
├── app.py                  # Part 5 — Streamlit interactive dashboard
├── dataset.csv             # Extracted / synthetic traffic data
├── traffic_model.pkl       # Trained model (auto-generated)
├── scaler.pkl              # Feature scaler (auto-generated)
├── requirements.txt        # Dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set Grok API key

```bash
export XAI_API_KEY=xai-your-key-here        # macOS / Linux
set XAI_API_KEY=xai-your-key-here           # Windows CMD
```
> Get your key at: https://console.x.ai  
> Without a key, the system uses rule-based road classification.

### 3. Launch the app

```bash
streamlit run app.py
```

The app will:
- Auto-train the model on first launch (takes ~5 seconds)
- Load synthetic demo data if no video is uploaded

---

## 🎥 Video Processing

### With a real video:
1. Upload MP4 via sidebar in the app, **OR**
2. Run from terminal:
```bash
python feature_extraction.py --video my_traffic.mp4 --output dataset.csv
```

### Demo mode (no video):
```bash
python feature_extraction.py --demo
```

Recommended video: 2–3 minutes, static camera, clear road view.

---

## 🧠 Features Extracted

| Category | Feature | Description |
|---|---|---|
| Traffic Flow | `VehicleCount` | Vehicles detected per frame |
| Traffic Flow | `Stopped` | In-lane stationary vehicles |
| Traffic Flow | `WrongParked` | Edge-zone stationary vehicles |
| Traffic Flow | `EstSpeed` | Optical flow speed (km/h) |
| Traffic Flow | `Density` | Vehicles per lane |
| Road Geometry | `Lanes` | Lane count (Hough detection) |
| Road Geometry | `RoadWidth` | Lanes × 3.5 metres |
| Composition | `CarRatio`, `BikeRatio`, `BusRatio`, `TruckRatio` | Vehicle type proportions |
| Context | `Pedestrians`, `Signals` | Per-frame counts |
| Context | `NMVCount` | Non-motorised vehicles |
| Behavioral | `FlowDirection` | Dominant flow vector |

---

## 🤖 Grok AI Road Classification

Features are sent to Grok (`grok-3-mini`) with a structured prompt. Response (JSON) includes:

```json
{
  "road_type": "Signal Junction",
  "lane_discipline": "Poor",
  "congestion_level": "High",
  "density_label": "High",
  "stopped_label": "High",
  "speed_label": "Slow",
  "vehicle_mix": "Heavy/Mixed",
  "behavior_summary": "Traffic shows high congestion..."
}
```

---

## 🔥 Traffic Policies

### 1️⃣ One-Way Road
| Feature | Modifier |
|---|---|
| VehicleCount | × 0.85 |
| Stopped | × 0.60 |
| EstSpeed | × 1.25 |

### 2️⃣ No-Parking Enforcement
| Feature | Modifier |
|---|---|
| WrongParked | × 0.00 (cleared) |
| Stopped | × 0.70 |
| EstSpeed | × 1.15 |

### 3️⃣ Peak-Hour Vehicle Restriction
| Feature | Modifier |
|---|---|
| VehicleCount | × 0.70 |
| Stopped | × 0.80 |
| EstSpeed | × 1.20 |

---

## 📊 Visualisations

| Graph | Description |
|---|---|
| Before vs After | Bar chart comparing baseline vs post-policy congestion |
| Trend Graph | Congestion vs vehicle count sweep with vertical current-VC marker |
| Feature Changes | Grouped bar: original vs modified feature values |
| Scenario Comparison | Side-by-side comparison of any two saved scenarios |
| History Chart | Reduction % across all scenarios in session |

---

## 🧪 ML Model Performance

```
Algorithm : RandomForestRegressor (300 trees)
Features  : VehicleCount, Stopped, Lanes, EstSpeed, VehicleMix, Density, WrongParked, Pedestrians
Target    : CongestionScore = 0.5 × VehicleCount + 2 × Stopped
Train R²  : ~0.995
Test  R²  : ~0.980
MAE       : ~0.73
CV R²     : ~0.98 ± 0.005
```

---

## 🎯 Output Format

```
Policy            : No-Parking Enforcement
Congestion Before : 38.50
Congestion After  : 23.40
Congestion Reduced: 39.2%
Speed Before      : 22.0 km/h
Speed After       : 25.3 km/h
Speed Improved    : 15.0%
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
openai>=1.12.0        # for Grok API (OpenAI-compatible)
opencv-python>=4.8.0  # for video processing
ultralytics>=8.0.0    # for YOLOv8
```

---

## 🎓 Academic Context

| Item | Detail |
|---|---|
| Purpose | Final Year Project — AI & Computer Vision |
| Domain | Intelligent Transportation Systems (ITS) |
| Method | CV Feature Extraction → LLM Classification → ML Regression → Policy Simulation |
| Tools | YOLOv8, OpenCV, Grok (xAI), RandomForest, Streamlit, Plotly |

---

*TrafficIQ — Decision Support for Traffic Authorities*
