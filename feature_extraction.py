"""
============================================================
 PART 1 — FEATURE EXTRACTION FROM TRAFFIC VIDEO
============================================================
Uses YOLOv8 + OpenCV to detect vehicles, estimate stopped
vehicles, and estimate wrong parking from traffic footage.
Saves structured data as dataset.csv.

Usage:
    python feature_extraction.py --video <path_to_video.mp4>
    python feature_extraction.py --demo      # generate synthetic dataset

Requirements:
    pip install ultralytics opencv-python pandas numpy
============================================================
"""

import cv2
import pandas as pd
import numpy as np
import argparse
import os

# ── Try importing YOLOv8 ──────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Use --demo mode or: pip install ultralytics")


# ── YOLO vehicle class IDs (COCO dataset) ─────────────────
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── Constants ─────────────────────────────────────────────
ROAD_EDGE_RATIO   = 0.15   # top/bottom % of frame = road edge zone
STOPPED_THRESHOLD = 5      # pixel movement below this = "stopped"
FRAME_SKIP        = 5      # process every Nth frame for performance
CONF_THRESHOLD    = 0.4    # YOLO detection confidence threshold


def is_near_edge(y_center: float, frame_height: int) -> bool:
    """Return True if detection is near the top or bottom road edge (likely wrong-parked)."""
    edge_band = frame_height * ROAD_EDGE_RATIO
    return (y_center < edge_band) or (y_center > frame_height - edge_band)


def extract_features_from_video(video_path: str, output_csv: str = "dataset.csv") -> pd.DataFrame:
    """
    Process a traffic video using YOLOv8 + OpenCV.

    Extracts per sampled frame:
        VehicleCount  — total vehicles detected
        WrongParked   — stationary vehicles near road edges
        Stopped       — stationary vehicles in lane

    Saves results to CSV and returns DataFrame.
    """
    if not YOLO_AVAILABLE:
        raise RuntimeError("Install ultralytics: pip install ultralytics")

    print("[INFO] Loading YOLOv8 nano model …")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video: {total_frames} frames @ {fps:.1f} fps")

    records    = []
    prev_boxes = {}   # track_id → (cx, cy) for motion estimation
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        h, w      = frame.shape[:2]
        timestamp = frame_idx / fps

        # ── Run YOLOv8 with object tracking ───────────────
        results = model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASS_IDS.keys()),
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        vehicle_count = 0
        wrong_parked  = 0
        stopped       = 0
        current_boxes = {}

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue

                vehicle_count += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                tid = int(box.id[0]) if box.id is not None else None

                # ── Check if vehicle is stopped ────────────
                is_stopped = False
                if tid is not None and tid in prev_boxes:
                    pcx, pcy   = prev_boxes[tid]
                    movement   = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                    is_stopped = movement < STOPPED_THRESHOLD

                if tid is not None:
                    current_boxes[tid] = (cx, cy)

                # ── Classify: wrong-parked vs in-lane stop ─
                if is_stopped:
                    if is_near_edge(cy, h):
                        wrong_parked += 1
                    else:
                        stopped += 1

        prev_boxes = current_boxes

        congestion_score = (0.5 * vehicle_count) + (2 * wrong_parked) + (2 * stopped)

        records.append({
            "FrameIndex"     : frame_idx,
            "Timestamp"      : round(timestamp, 2),
            "VehicleCount"   : vehicle_count,
            "WrongParked"    : wrong_parked,
            "Stopped"        : stopped,
            "CongestionScore": round(congestion_score, 2),
        })

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames} | Vehicles: {vehicle_count} | Score: {congestion_score:.1f}")

        frame_idx += 1

    cap.release()
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n[✓] Extracted {len(df)} records → {output_csv}")
    return df


def generate_synthetic_dataset(output_csv: str = "dataset.csv", n_samples: int = 500) -> pd.DataFrame:
    """
    Generate a realistic synthetic traffic dataset for demo / testing.
    Simulates time-of-day traffic patterns (rush hour, off-peak, night).

    This is used when no real video is available.
    """
    print(f"[INFO] Generating synthetic dataset ({n_samples} samples) …")
    np.random.seed(42)

    timestamps = np.linspace(0, 24, n_samples)
    records    = []

    for t in timestamps:
        # Traffic intensity by hour
        if   7  <= t <= 9:     intensity = 0.9   # morning rush
        elif 12 <= t <= 14:    intensity = 0.6   # lunch hour
        elif 17 <= t <= 20:    intensity = 1.0   # evening rush
        elif t >= 23 or t < 5: intensity = 0.15  # late night
        else:                  intensity = 0.4   # off-peak

        vehicle_count = int(np.clip(np.random.normal(30 * intensity, 5), 0, 60))
        wrong_parked  = int(np.clip(np.random.normal(3  * intensity, 1), 0, 10))
        stopped       = int(np.clip(np.random.normal(4  * intensity, 2), 0, 15))

        congestion_score = (0.5 * vehicle_count) + (2 * wrong_parked) + (2 * stopped)

        records.append({
            "FrameIndex"     : int(t * 100),
            "Timestamp"      : round(t, 2),
            "VehicleCount"   : vehicle_count,
            "WrongParked"    : wrong_parked,
            "Stopped"        : stopped,
            "CongestionScore": round(congestion_score, 2),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Synthetic dataset saved → {output_csv}")
    print(df.describe().round(2))
    return df


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Feature Extraction")
    parser.add_argument("--video",  type=str,           help="Path to traffic video")
    parser.add_argument("--demo",   action="store_true", help="Generate synthetic demo data")
    parser.add_argument("--output", type=str, default="dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.demo or not args.video:
        generate_synthetic_dataset(args.output)
    else:
        extract_features_from_video(args.video, args.output)
