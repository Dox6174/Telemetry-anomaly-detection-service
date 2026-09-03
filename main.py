import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field
import db

app = FastAPI(title="Telemetry Anomaly Detection API")
db.init_db()

CHANNELS = ["voltage", "current", "temp1", "temp2", "gyro_rate"]

_active = db.get_active_model_version()
if _active is None:
    threshold = float(np.load("static/threshold.npy")[0])
    db.register_model_version("telemetry-ae-fp32", "model.onnx", False, threshold, None)
    db.register_model_version("telemetry-ae-int8", "model_quant.onnx", True, threshold, None)
    _active = db.get_active_model_version()

MODEL_ID, ONNX_PATH, THRESHOLD = _active
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
mean = np.load("static/norm_mean.npy")
std = np.load("static/norm_std.npy")

class InferRequest(BaseModel):
    window: list[list[float]] = Field(
        ..., description="5 channels x 30 timesteps, order: voltage, current, temp1, temp2, gyro_rate"
    )

@app.post("/infer")
def infer(req: InferRequest):
    arr = np.array(req.window, dtype=np.float32)
    flat = arr.reshape(1, -1)
    norm = ((flat - mean) / std).astype(np.float32)

    t0 = time.perf_counter()
    recon = session.run(None, {"input": norm})[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    err = float(((recon - norm) ** 2).mean())
    is_anomaly = err > THRESHOLD

    suspected_channel, suspected_channel_id = None, None
    if is_anomaly:
        per_channel_err = ((recon.reshape(5, 30) - norm.reshape(5, 30)) ** 2).mean(axis=1)
        idx = int(np.argmax(per_channel_err))
        suspected_channel = CHANNELS[idx]
        suspected_channel_id = db.channel_id(suspected_channel)

    db.log_request(MODEL_ID, req.window, err, is_anomaly, suspected_channel_id, latency_ms)

    return {
        "anomaly_score": round(err, 6),
        "threshold": THRESHOLD,
        "is_anomaly": is_anomaly,
        "suspected_channel": suspected_channel,
        "latency_ms": round(latency_ms, 3),
        "model_version_id": MODEL_ID,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_version_id": MODEL_ID}

@app.get("/stats")
def stats():
    return db.get_stats()

@app.get("/channels")
def channels():
    return db.list_channels()
