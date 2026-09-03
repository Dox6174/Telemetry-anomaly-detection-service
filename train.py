import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import Autoencoder, INPUT_DIM

X_train = np.load("data/X_train.npy").reshape(-1, INPUT_DIM).astype(np.float32)
X_test_normal = np.load("data/X_test_normal.npy").reshape(-1, INPUT_DIM).astype(np.float32)
X_test_anomaly = np.load("data/X_test_anomaly.npy").reshape(-1, INPUT_DIM).astype(np.float32)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-6
np.save("data/norm_mean.npy", mean)
np.save("data/norm_std.npy", std)

def normalize(x):
    return (x - mean) / std

loader = DataLoader(TensorDataset(torch.tensor(normalize(X_train))), batch_size=64, shuffle=True)

model = Autoencoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(60):
    total = 0.0
    for (batch,) in loader:
        opt.zero_grad()
        loss = loss_fn(model(batch), batch)
        loss.backward()
        opt.step()
        total += loss.item()
    if epoch % 10 == 0 or epoch == 59:
        print(f"epoch {epoch}: loss {total / len(loader):.5f}")

torch.save(model.state_dict(), "model.pt")
# --- split X_test_normal into a calibration half and a held-out eval half ---
n_normal = X_test_normal.shape[0]
rng2 = np.random.default_rng(0)
idx = rng2.permutation(n_normal)
calib_idx, eval_idx = idx[:n_normal // 2], idx[n_normal // 2:]

X_calib = X_test_normal[calib_idx]
X_eval_normal = X_test_normal[eval_idx]

model.eval()
with torch.no_grad():
    calib_recon = model(torch.tensor(normalize(X_calib))).numpy()
    calib_err = ((calib_recon - normalize(X_calib)) ** 2).mean(axis=1)

    threshold = float(np.percentile(calib_err, 99))
    np.save("data/threshold.npy", np.array([threshold]))

    eval_recon = model(torch.tensor(normalize(X_eval_normal))).numpy()
    eval_err = ((eval_recon - normalize(X_eval_normal)) ** 2).mean(axis=1)
    fp_rate = (eval_err > threshold).mean() * 100

    a_recon = model(torch.tensor(normalize(X_test_anomaly))).numpy()
    a_err = ((a_recon - normalize(X_test_anomaly)) ** 2).mean(axis=1)
    detection_rate = (a_err > threshold).mean() * 100

print(f"Threshold (99th pct of CALIBRATION set): {threshold:.5f}")
print(f"False positive rate on HELD-OUT normal data: {fp_rate:.1f}%")
print(f"Detection rate on full-magnitude injected faults: {detection_rate:.1f}%")