# Telemetry Anomaly Detection Service

A self-hosted, containerized inference service that detects anomalies in multivariate
satellite housekeeping telemetry (power, thermal, attitude channels) using a
reconstruction-error autoencoder, served through a REST API with logged,
versioned inference history.

Built to explore the full deployment cost of putting a learned model into a
low-latency serving path — not just training accuracy, but the effect of INT8
quantization on the decision boundary, and the operational cost of running it
as a versioned, logged service rather than a notebook.

**Live demo:** `https://<your-app-name>.onrender.com`
*(Render's free tier spins down after 15 minutes idle — the first request after
that will take ~30-60s to wake up.)*

---

## Why this exists

Satellites report periodic housekeeping telemetry — bus voltage, current draw,
thermal readings, attitude/gyro rates — and operators need to flag deviations
from nominal behavior quickly, ideally without relying on hand-tuned
per-channel thresholds. This project models that as an unsupervised anomaly
detection problem: train an autoencoder to reconstruct *normal* telemetry
windows well, and treat high reconstruction error as an anomaly signal.

All data is synthetically generated (correlated multi-channel signals with
drift and noise, plus controlled fault injection), which makes ground truth
exact and removes any external data dependency — the entire pipeline runs
end-to-end with nothing to download.

---

## Architecture

```
                    ┌─────────────────────┐
   synthetic        │   gen_data.py        │
   telemetry   ───▶  │  (numpy, 5 channels, │
   windows          │   30 timesteps)       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   train.py            │
                    │  PyTorch autoencoder  │
                    │  (calib/eval split)   │
                    └──────────┬───────────┘
                               │ model.pt
                    ┌──────────▼───────────┐
                    │  export_onnx.py       │
                    │  PyTorch → ONNX       │
                    │  + parity check       │
                    └──────────┬───────────┘
                               │ model.onnx
                    ┌──────────▼───────────┐
                    │  benchmark.py         │
                    │  INT8 quantization +  │
                    │  latency benchmark +  │
                    │  threshold revalidate │
                    └──────────┬───────────┘
                               │ model_quant.onnx
                               │
        ┌──────────────────────▼──────────────────────┐
        │                 FastAPI (main.py)             │
        │  ONNXRuntime session loaded once at startup   │
        │  POST /infer  GET /health  GET /stats  /channels
        └──────────────────────┬──────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │   SQLite (db.py)      │
                    │  sensor_channels       │
                    │  model_versions        │
                    │  inference_requests    │
                    └───────────────────────┘

        packaged with Docker → deployed to Render (production)
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Model | PyTorch autoencoder (dense, 150→64→16→64→150) | Unsupervised — real anomalies are rare by definition, so training to reconstruct *normal* data is the right framing over a classifier |
| Inference runtime | ONNXRuntime (CPU) | Self-hosted, no external inference API; CPU-only mirrors power-constrained edge/onboard compute |
| Optimization | Dynamic INT8 quantization (`onnxruntime.quantization`) | Latency reduction with an explicit, measured accuracy tradeoff check (see below) |
| API | FastAPI + Pydantic | Request/response validation, auto-generated OpenAPI docs at `/docs` |
| Database | SQLite (WAL mode) | Lightweight, file-based, sufficient for logging inference history; schema designed to be swappable for managed Postgres |
| Containerization | Docker | Single image builds and runs identically locally and in production |
| Deployment | Render (free tier) | Git-based deploy from a Dockerfile, no card required |

---

## Data flow

1. `gen_data.py` generates synthetic multivariate telemetry windows (5 channels
   × 30 timesteps): normal windows with drift + noise, and fault windows with
   an injected deviation in one channel at a known magnitude.
2. `train.py` trains the autoencoder on normal data only, then splits held-out
   normal data into a **calibration set** (used to set the anomaly threshold at
   the 99th percentile of reconstruction error) and a separate **evaluation
   set** (used to measure the false positive rate honestly, out-of-sample).
3. `export_onnx.py` exports the trained model to ONNX with a fixed batch size
   of 1 (matching real serving usage) and verifies numerical parity against
   the PyTorch model before proceeding.
4. `benchmark.py` produces an INT8-quantized copy of the model, benchmarks
   FP32 vs INT8 latency (mean/p50/p95/p99) over 500 warmed-up runs, and
   **re-validates the anomaly threshold against the quantized model's error
   distribution** rather than assuming quantization leaves it unchanged.
5. `main.py` loads the ONNX model once at FastAPI startup and serves
   `POST /infer`, logging every request (score, anomaly flag, suspected
   channel, latency) to SQLite.
6. `sensitivity_sweep.py` measures detection rate as a function of fault
   magnitude (in standard deviations from baseline) — see results below.

---

## Evaluation results

**Detection sensitivity sweep** (fault magnitude vs. detection rate, 200 trials per point):

| Fault magnitude (σ from baseline) | Detection rate |
|---|---|
| 1σ | 5.0% |
| 2σ | 19.5% |
| 3σ | 63.5% |
| 5σ | 100.0% |
| 8σ | 100.0% |
| 12σ | 100.0% |
| 20σ | 100.0% |

The 50% detection crossover sits between 2σ and 3σ, with saturation to 100%
by 5σ. This is the actual sensitivity boundary of the detector, set by the
99th-percentile threshold chosen during calibration — a lower percentile
would catch smaller deviations at the cost of a higher false positive rate.

**Held-out false positive rate:** `2.00%`

**Latency (FP32 vs INT8, CPU, 500 runs):**

| Model | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| FP32 | `0.019` ms | `0.015` ms | `0.032` ms | `0.043` ms |
| INT8 | `0.023` ms | `0.020` ms | `0.025` ms | `0.052` ms |

`[Quantization gives a very modest change/overhead in this case, this is due to the fact that the model is small enough(~21k parameters) that ONNXRuntime/Python call overhead likely dominates raw compute — a larger conv-based model would be expected to show a bigger INT8 win.]`

---

## API reference

Auto-generated interactive docs are available at `/docs` on any running
instance (local or deployed).

| Endpoint | Method | Description |
|---|---|---|
| `/infer` | POST | Accepts a `{"window": [[...], [...], ...]}` payload (5 channels × 30 timesteps, order: voltage, current, temp1, temp2, gyro_rate). Returns anomaly score, threshold, anomaly flag, suspected channel, and latency. |
| `/health` | GET | Returns service status and the active model version ID. |
| `/stats` | GET | Returns aggregate latency percentiles (p50/p95/p99) across all logged requests. |
| `/channels` | GET | Returns metadata for each telemetry channel (unit, expected range). |

---

## Database schema

```sql
sensor_channels (id, channel_name, unit, expected_min, expected_max)
model_versions  (id, name, onnx_path, quantized, threshold, avg_latency_ms, created_at)
inference_requests (
    id, model_version_id → model_versions.id,
    input_hash, anomaly_score, is_anomaly,
    suspected_channel_id → sensor_channels.id,
    latency_ms, created_at
)
```

`sensor_channels` and `model_versions` are kept as separate normalized tables
from `inference_requests` rather than one flat log, because the schema needs
to support A/B comparison of model versions (FP32 vs INT8) over time and
trace an anomalous request back to a specific physical channel — two real
foreign-key relationships, not incidental structure.

---

## Running it locally

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

python gen_data.py
python train.py
python export_onnx.py
python benchmark.py

mkdir static -Force
Copy-Item data\norm_mean.npy, data\norm_std.npy, data\threshold.npy -Destination static\ -Force

pip install fastapi "uvicorn[standard]"
uvicorn main:app --reload --port 8000
```

Test it:
```bash
python demo_request.py
```

## Running it with Docker

```bash
docker compose up --build
python demo_request.py
```

## Deployment

Deployed on Render as a Docker-based web service, built directly from this
repository's `Dockerfile`. The service listens on the port Render assigns via
the `PORT` environment variable at runtime (`${PORT:-8000}` in the Dockerfile
CMD, defaulting to 8000 for local Docker Compose use).

**Known limitation:** Render's free tier has an ephemeral filesystem — the
SQLite database resets on every spin-down/redeploy. For real persistence,
the two options are Render's paid disk add-on or migrating the logging layer
to a managed Postgres instance, which would be a small, isolated change since
all SQL is behind the helper functions in `db.py`.

---

## Limitations and what I'd do with more time

- **Threshold selection is a single fixed percentile**, not a tuned
  operating point. The right next step is a full ROC/precision-recall sweep
  across thresholds to pick an operating point matched to an actual cost
  tradeoff between missed anomalies and false alarms.
- **Per-channel thresholds** would likely improve small-magnitude detection
  over a single global reconstruction-error cutoff, since channels have very
  different noise floors (e.g., gyro_rate's baseline std is ~5x current's).
- **Persistence**: SQLite is fine for a project this size but would move to
  managed Postgres for anything long-running in production.
- **Quantization benefit is model-size-dependent**: the latency win from
  INT8 here is bounded by how small this autoencoder already is; a larger
  convolutional model would be a better test of quantization's real value.
- **Data is synthetic.** The next real step would be validating this
  approach against actual satellite housekeeping telemetry or a public
  spacecraft anomaly dataset, where fault signatures are less clean than
  the injected ones used here.
