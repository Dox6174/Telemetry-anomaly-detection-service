import numpy as np
import os

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42)

CHANNELS = ["voltage", "current", "temp1", "temp2", "gyro_rate"]
N_CHANNELS = len(CHANNELS)
WINDOW = 30  # timesteps per window

BASELINE = {
    "voltage": (28.0, 0.3),
    "current": (2.0, 0.1),
    "temp1":   (25.0, 1.0),
    "temp2":   (22.0, 1.0),
    "gyro_rate": (0.0, 0.5),
}

def make_normal_window():
    window = np.zeros((N_CHANNELS, WINDOW))
    t = np.linspace(0, 2 * np.pi, WINDOW)
    for i, ch in enumerate(CHANNELS):
        mean, std = BASELINE[ch]
        drift = 0.3 * std * np.sin(t + rng.uniform(0, 2 * np.pi))
        noise = rng.normal(0, std, WINDOW)
        window[i] = mean + drift + noise
    return window

def inject_fault(window, fault):
    window = window.copy()
    idx = CHANNELS.index(fault["channel"])
    start = rng.integers(0, WINDOW - 8)
    length = rng.integers(5, 10)
    window[idx, start:start + length] += fault["magnitude"]
    return window

FAULTS = [
    {"channel": "voltage",   "magnitude": -6.0},
    {"channel": "current",   "magnitude": 3.5},
    {"channel": "temp1",     "magnitude": 15.0},
    {"channel": "gyro_rate", "magnitude": 8.0},
]

N_TRAIN, N_TEST_NORMAL, N_TEST_ANOMALY = 2000, 300, 300

X_train = np.stack([make_normal_window() for _ in range(N_TRAIN)])
X_test_normal = np.stack([make_normal_window() for _ in range(N_TEST_NORMAL)])

X_test_anomaly, y_channel = [], []
for _ in range(N_TEST_ANOMALY):
    fault = FAULTS[rng.integers(0, len(FAULTS))]
    X_test_anomaly.append(inject_fault(make_normal_window(), fault))
    y_channel.append(CHANNELS.index(fault["channel"]))
X_test_anomaly = np.stack(X_test_anomaly)

np.save("data/X_train.npy", X_train)
np.save("data/X_test_normal.npy", X_test_normal)
np.save("data/X_test_anomaly.npy", X_test_anomaly)
np.save("data/y_test_anomaly_channel.npy", np.array(y_channel))

print(f"Train: {X_train.shape}  Test normal: {X_test_normal.shape}  Test anomaly: {X_test_anomaly.shape}")