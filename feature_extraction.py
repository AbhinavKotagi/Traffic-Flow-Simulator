"""
============================================================
  PART 1 — COMPUTER VISION FEATURE EXTRACTION
  Traffic Policy Simulator | feature_extraction.py
============================================================
Extracts comprehensive traffic features from video using:
  • YOLOv8       — vehicle detection & tracking
  • OpenCV        — optical flow, lane detection, frame ops
  • NumPy         — numerical computations

FEATURES EXTRACTED:
  Traffic Flow   : VehicleCount, Density, StoppedVehicles, EstSpeed
  Road Geometry  : LaneCount, RoadWidth
  Composition    : CarRatio, BikeRatio, BusRatio, TruckRatio
  Behavioral     : FlowDirection (vector)
  Context        : TrafficSignals, Pedestrians, NMVCount

Usage:
  python feature_extraction.py --video traffic.mp4
  python feature_extraction.py --demo         # synthetic data
  python feature_extraction.py --video traffic.mp4 --output my_data.csv

Requirements:
  pip install ultralytics opencv-python pandas numpy
============================================================
"""

import cv2
import pandas as pd
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

# ── YOLOv8 import guard ───────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")

# ─────────────────────────────────────────────────────────
#  COCO class IDs used in this project
# ─────────────────────────────────────────────────────────
VEHICLE_CLASSES   = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PEDESTRIAN_CLASS  = 0   # 'person'
NMV_CLASSES       = {1: "bicycle", 3: "motorcycle"}

# ── Signal-like objects: traffic light = 9 (COCO) ────────
SIGNAL_CLASS      = 9

# ── Thresholds ────────────────────────────────────────────
FRAME_SKIP        = 5      # process 1 in every N frames
STOPPED_PIXELS    = 6      # pixel movement ≤ this → "stopped"
ROAD_EDGE_FRAC    = 0.15   # top/bottom fraction = road edge zone
CONF_THRESHOLD    = 0.35   # YOLO confidence cutoff


# ─────────────────────────────────────────────────────────
#  HELPER: Is detection near road edge?
# ─────────────────────────────────────────────────────────
def _near_edge(cy: float, frame_h: int) -> bool:
    band = frame_h * ROAD_EDGE_FRAC
    return cy < band or cy > (frame_h - band)


# ─────────────────────────────────────────────────────────
#  HELPER: Optical-flow speed estimate (Farneback)
# ─────────────────────────────────────────────────────────
def _estimate_speed_optical_flow(prev_gray, curr_gray, fps: float, px_per_meter: float = 8.0) -> float:
    """
    Estimate average vehicle speed (km/h) using dense optical flow.
    px_per_meter is a calibration constant (pixels ≈ 1 meter).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    avg_mag   = np.mean(magnitude)               # px / frame
    speed_mps = (avg_mag * fps) / px_per_meter   # m/s
    return round(speed_mps * 3.6, 2)             # → km/h


# ─────────────────────────────────────────────────────────
#  HELPER: Lane count via Hough line detection
# ─────────────────────────────────────────────────────────
def _count_lanes(frame) -> int:
    """
    Estimate number of lanes using Canny + Hough lines on
    the bottom half of the frame (road region).
    """
    h, w = frame.shape[:2]
    roi   = frame[h // 2:, :]                          # bottom half
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=60, minLineLength=80, maxLineGap=30
    )

    if lines is None:
        return 2   # fallback

    # Keep only lines with steep-enough angle (lane markers are near-vertical in BEV)
    lane_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 20:           # accept diagonal & vertical
            lane_lines.append(line)

    if not lane_lines:
        return 2

    # Cluster x-midpoints to estimate distinct lane boundaries
    x_mids = sorted([(l[0][0] + l[0][2]) / 2 for l in lane_lines])
    clusters, prev = 1, x_mids[0]
    for x in x_mids[1:]:
        if x - prev > w * 0.08:   # gap > 8% width = new lane boundary
            clusters += 1
        prev = x

    return max(2, min(clusters, 6))   # clamp 2–6 lanes


# ─────────────────────────────────────────────────────────
#  HELPER: Flow direction from optical flow vectors
# ─────────────────────────────────────────────────────────
def _flow_direction(prev_gray, curr_gray) -> str:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    vx   = np.mean(flow[..., 0])
    vy   = np.mean(flow[..., 1])
    if abs(vx) < 0.5 and abs(vy) < 0.5:
        return "Static"
    return "Left→Right" if vx > 0 else "Right→Left"


# ─────────────────────────────────────────────────────────
#  MAIN: Video Feature Extraction
# ─────────────────────────────────────────────────────────
def extract_features_from_video(video_path: str, output_csv: str = "dataset.csv") -> pd.DataFrame:
    """
    Process a traffic video frame-by-frame using YOLOv8 + OpenCV.

    Returns a DataFrame with per-frame traffic features and
    saves it to `output_csv`.

    Parameters
    ----------
    video_path : str   Path to MP4 video file
    output_csv : str   Output CSV path

    Returns
    -------
    pd.DataFrame
    """
    if not YOLO_AVAILABLE:
        raise RuntimeError("Install ultralytics: pip install ultralytics")

    print(f"[INFO] Loading YOLOv8 model …")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {total} frames @ {fps:.1f} fps | {width}×{height}")

    records    = []
    prev_boxes = {}
    prev_gray  = None
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp = round(frame_idx / fps, 2)

        # ── Optical flow speed & direction ─────────────────
        speed         = 0.0
        flow_direction = "Unknown"
        if prev_gray is not None:
            speed         = _estimate_speed_optical_flow(prev_gray, curr_gray, fps / FRAME_SKIP)
            flow_direction = _flow_direction(prev_gray, curr_gray)

        # ── Lane count (run every 25 frames for speed) ─────
        lanes = _count_lanes(frame) if frame_idx % 25 == 0 else (records[-1]["Lanes"] if records else 2)

        # ── YOLO detection + tracking ──────────────────────
        results = model.track(
            frame, persist=True,
            classes=list(set(VEHICLE_CLASSES.keys()) | {PEDESTRIAN_CLASS, SIGNAL_CLASS} | set(NMV_CLASSES.keys())),
            conf=CONF_THRESHOLD, verbose=False,
        )

        # Counters
        vehicle_count = car_count = bike_count = bus_count = truck_count = 0
        stopped = wrong_parked = pedestrians = nmv_count = signals = 0
        current_boxes = {}

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                tid = int(box.id[0]) if box.id is not None else None

                # ── Pedestrians ────────────────────────────
                if cls_id == PEDESTRIAN_CLASS:
                    pedestrians += 1
                    continue

                # ── Traffic signals ────────────────────────
                if cls_id == SIGNAL_CLASS:
                    signals += 1
                    continue

                # ── Vehicles ───────────────────────────────
                if cls_id in VEHICLE_CLASSES:
                    vehicle_count += 1
                    if   cls_id == 2: car_count   += 1
                    elif cls_id == 3: bike_count  += 1
                    elif cls_id == 5: bus_count   += 1
                    elif cls_id == 7: truck_count += 1

                    # NMV check
                    if cls_id in NMV_CLASSES:
                        nmv_count += 1

                    # Motion-based stopped detection
                    is_stopped = False
                    if tid is not None and tid in prev_boxes:
                        pcx, pcy   = prev_boxes[tid]
                        movement   = np.hypot(cx - pcx, cy - pcy)
                        is_stopped = movement < STOPPED_PIXELS

                    if tid is not None:
                        current_boxes[tid] = (cx, cy)

                    if is_stopped:
                        if _near_edge(cy, height):
                            wrong_parked += 1
                        else:
                            stopped += 1

        prev_boxes = current_boxes
        prev_gray  = curr_gray

        # ── Compute derived features ───────────────────────
        density       = round(vehicle_count / max(lanes, 1), 2)
        road_width    = round(lanes * 3.5, 1)   # metres (standard lane = 3.5m)
        car_ratio     = round(car_count   / max(vehicle_count, 1), 3)
        bike_ratio    = round(bike_count  / max(vehicle_count, 1), 3)
        bus_ratio     = round(bus_count   / max(vehicle_count, 1), 3)
        truck_ratio   = round(truck_count / max(vehicle_count, 1), 3)

        # ── Congestion Score formula ───────────────────────
        congestion = round((0.5 * vehicle_count) + (2 * stopped), 2)

        records.append({
            # Timing
            "FrameIndex"   : frame_idx,
            "Timestamp"    : timestamp,
            # Traffic Flow
            "VehicleCount" : vehicle_count,
            "Density"      : density,
            "Stopped"      : stopped,
            "WrongParked"  : wrong_parked,
            "EstSpeed"     : speed,
            # Road Geometry
            "Lanes"        : lanes,
            "RoadWidth"    : road_width,
            # Composition
            "CarRatio"     : car_ratio,
            "BikeRatio"    : bike_ratio,
            "BusRatio"     : bus_ratio,
            "TruckRatio"   : truck_ratio,
            # Context
            "Pedestrians"  : pedestrians,
            "NMVCount"     : nmv_count,
            "Signals"      : min(signals, 3),
            # Behavioral
            "FlowDirection": flow_direction,
            # Target
            "CongestionScore": congestion,
        })

        if frame_idx % 50 == 0:
            print(f"  [{frame_idx}/{total}] Vehicles:{vehicle_count} Stopped:{stopped} Speed:{speed} km/h")

        frame_idx += 1

    cap.release()
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n[✓] Extracted {len(df)} frame records → {output_csv}")
    return df


# ─────────────────────────────────────────────────────────
#  SYNTHETIC DATASET (demo / fallback)
# ─────────────────────────────────────────────────────────
def generate_synthetic_dataset(output_csv: str = "dataset.csv", n: int = 600) -> pd.DataFrame:
    """
    Generate a realistic synthetic traffic dataset that mimics
    time-of-day patterns (rush hour, off-peak, night).

    Used when no real video is available.
    """
    print(f"[INFO] Generating synthetic dataset ({n} samples) …")
    np.random.seed(42)
    timestamps = np.linspace(0, 24, n)
    records    = []

    for t in timestamps:
        # Traffic intensity by time of day
        if   7  <= t <= 9:     intensity = 0.90
        elif 12 <= t <= 14:    intensity = 0.60
        elif 17 <= t <= 20:    intensity = 1.00
        elif t >= 23 or t < 5: intensity = 0.12
        else:                  intensity = 0.40

        vc   = int(np.clip(np.random.normal(40 * intensity, 6), 0, 80))
        st   = int(np.clip(np.random.normal(5  * intensity, 2), 0, 20))
        wp   = int(np.clip(np.random.normal(3  * intensity, 1), 0, 10))
        spd  = round(np.clip(np.random.normal(30 * (1 - 0.6*intensity), 5), 5, 80), 1)
        lns  = int(np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3]))
        sigs = int(np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1]))
        peds = int(np.clip(np.random.normal(4 * intensity, 2), 0, 15))
        car_r   = round(np.random.uniform(0.4, 0.7), 3)
        bike_r  = round(np.random.uniform(0.1, 0.3), 3)
        bus_r   = round(np.random.uniform(0.05, 0.2), 3)
        truck_r = round(1 - car_r - bike_r - bus_r, 3)

        congestion = round((0.5 * vc) + (2 * st), 2)

        records.append({
            "FrameIndex"     : int(t * 100),
            "Timestamp"      : round(t, 2),
            "VehicleCount"   : vc,
            "Density"        : round(vc / lns, 2),
            "Stopped"        : st,
            "WrongParked"    : wp,
            "EstSpeed"       : spd,
            "Lanes"          : lns,
            "RoadWidth"      : round(lns * 3.5, 1),
            "CarRatio"       : car_r,
            "BikeRatio"      : bike_r,
            "BusRatio"       : bus_r,
            "TruckRatio"     : truck_r,
            "Pedestrians"    : peds,
            "NMVCount"       : int(vc * bike_r),
            "Signals"        : sigs,
            "FlowDirection"  : np.random.choice(["Left→Right", "Right→Left"]),
            "CongestionScore": congestion,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Synthetic dataset saved → {output_csv}")
    return df


# ─────────────────────────────────────────────────────────
#  AGGREGATE: Summarize extracted frames into a single dict
# ─────────────────────────────────────────────────────────
def aggregate_features(df: pd.DataFrame) -> dict:
    """
    Collapse per-frame features into a single summary dict
    suitable for LLM classification and ML inference.
    """
    numeric = df.select_dtypes(include=np.number).mean()
    return {
        "VehicleCount" : round(numeric.get("VehicleCount", 0), 1),
        "Density"      : round(numeric.get("Density",      0), 2),
        "Stopped"      : round(numeric.get("Stopped",      0), 1),
        "WrongParked"  : round(numeric.get("WrongParked",  0), 1),
        "EstSpeed"     : round(numeric.get("EstSpeed",     0), 1),
        "Lanes"        : round(numeric.get("Lanes",        2)),
        "RoadWidth"    : round(numeric.get("RoadWidth",    7), 1),
        "CarRatio"     : round(numeric.get("CarRatio",     0), 3),
        "BikeRatio"    : round(numeric.get("BikeRatio",    0), 3),
        "BusRatio"     : round(numeric.get("BusRatio",     0), 3),
        "TruckRatio"   : round(numeric.get("TruckRatio",  0), 3),
        "Pedestrians"  : round(numeric.get("Pedestrians", 0), 1),
        "NMVCount"     : round(numeric.get("NMVCount",    0), 1),
        "Signals"      : round(numeric.get("Signals",     0), 1),
        "CongestionScore": round(numeric.get("CongestionScore", 0), 2),
        "FlowDirection": df["FlowDirection"].mode()[0] if "FlowDirection" in df.columns else "Unknown",
    }


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Feature Extraction")
    parser.add_argument("--video",  type=str,            help="Path to traffic video (.mp4)")
    parser.add_argument("--demo",   action="store_true", help="Generate synthetic demo dataset")
    parser.add_argument("--output", type=str, default="dataset.csv")
    args = parser.parse_args()

    if args.demo or not args.video:
        generate_synthetic_dataset(args.output)
    else:
        df = extract_features_from_video(args.video, args.output)
        summary = aggregate_features(df)
        print("\nAggregated Summary:")
        for k, v in summary.items():
            print(f"  {k:<18}: {v}")
