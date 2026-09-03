import numpy as np
import torch
from model import Autoencoder, INPUT_DIM

CHANNELS = ["voltage", "current", "temp1", "temp2", "gyro_rate"]
WINDOW = 30
rng = np.random.default_rng(7)

model = Autoencoder()
model.load_state_dict(torch.load("model.pt"))
model.eval()

mean = np.load("data/norm_mean.npy")
std = np.load("data/norm_std.npy")
threshold = float(np.load("data/threshold.npy")[0])

def normalize(x):
    return (x - mean) / std

def make_normal_window():
    window = np.zeros((5, WINDOW))
    t = np.linspace(0, 2 * np.pi, WINDOW)
    baseline = {"voltage": (28.0, 0.3), "current": (2.0, 0.1), "temp1": (25.0, 1.0),
                "temp2": (22.0, 1.0), "gyro_rate": (0.0, 0.5)}
    for i, ch in enumerate(CHANNELS):
        m, s = baseline[ch]
        window[i] = m + 0.3 * s * np.sin(t + rng.uniform(0, 2 * np.pi)) + rng.normal(0, s, WINDOW)
    return window

# Sweep fault magnitude as a multiple of that channel's baseline std, from subtle to obvious
STD_MULTIPLIERS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 8, 12, 20]
BASE_STD = {"voltage": 0.3, "current": 0.1, "temp1": 1.0, "temp2": 1.0, "gyro_rate": 0.5}

for mult in STD_MULTIPLIERS:
    hits = 0
    n_trials = 200
    for _ in range(n_trials):
        ch = CHANNELS[rng.integers(0, 5)]
        idx = CHANNELS.index(ch)
        w = make_normal_window()
        start = rng.integers(0, WINDOW - 8)
        length = rng.integers(5, 10)
        w[idx, start:start+length] += mult * BASE_STD[ch] * rng.choice([-1, 1])
        flat = normalize(w.reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            recon = model(torch.tensor(flat)).numpy()
        err = ((recon - flat) ** 2).mean()
        if err > threshold:
            hits += 1
    print(f"Fault magnitude = {mult} std devs: detection rate = {hits/n_trials*100:.1f}%")