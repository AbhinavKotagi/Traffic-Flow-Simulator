"""Quick script to generate the demo dataset.csv"""
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 500
timestamps = np.linspace(0, 24, n_samples)
records = []

for t in timestamps:
    if   7  <= t <= 9:     intensity = 0.9
    elif 12 <= t <= 14:    intensity = 0.6
    elif 17 <= t <= 20:    intensity = 1.0
    elif t >= 23 or t < 5: intensity = 0.15
    else:                  intensity = 0.4

    vehicle_count = int(np.clip(np.random.normal(30 * intensity, 5), 0, 60))
    wrong_parked  = int(np.clip(np.random.normal(3  * intensity, 1), 0, 10))
    stopped       = int(np.clip(np.random.normal(4  * intensity, 2), 0, 15))
    congestion_score = (0.5 * vehicle_count) + (2 * wrong_parked) + (2 * stopped)

    records.append({
        "FrameIndex": int(t * 100),
        "Timestamp": round(t, 2),
        "VehicleCount": vehicle_count,
        "WrongParked": wrong_parked,
        "Stopped": stopped,
        "CongestionScore": round(congestion_score, 2),
    })

df = pd.DataFrame(records)
df.to_csv("dataset.csv", index=False)
print(f"Generated {len(df)} rows")
print(df.describe().round(2))
