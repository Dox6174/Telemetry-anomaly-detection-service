import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("model_infer.onnx", "model_quant.onnx", weight_type=QuantType.QInt8)
print("Wrote model_quant.onnx")

def bench(path, n=500):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 150).astype(np.float32)
    for _ in range(20):
        sess.run(None, {"input": x})  # warmup
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)

for name, path in [("FP32", "model.onnx"), ("INT8", "model_quant.onnx")]:
    t = bench(path)
    print(f"{name}: mean={t.mean():.3f}ms  p50={np.percentile(t,50):.3f}ms  "
          f"p95={np.percentile(t,95):.3f}ms  p99={np.percentile(t,99):.3f}ms")

X_test_normal = np.load("data/X_test_normal.npy").reshape(-1, 150).astype(np.float32)
mean = np.load("data/norm_mean.npy")
std = np.load("data/norm_std.npy")
Xn = ((X_test_normal - mean) / std).astype(np.float32)

sess_q = ort.InferenceSession("model_quant.onnx", providers=["CPUExecutionProvider"])
recon_q = sess_q.run(None, {"input": Xn})[0]
err_q = ((recon_q - Xn) ** 2).mean(axis=1)
old_threshold = float(np.load("data/threshold.npy")[0])
new_threshold_99 = float(np.percentile(err_q, 99))
print(f"FP32 threshold: {old_threshold:.5f}   INT8 99th-pct error: {new_threshold_99:.5f}")
if abs(new_threshold_99 - old_threshold) / old_threshold > 0.15:
    print("WARNING: quantization shifted the error distribution >15% — using a separate threshold for INT8.")