import sqlite3
import os
import hashlib
import json
import numpy as np

DB_PATH = os.environ.get("DB_PATH", "data/telemetry.db")

CHANNELS = ["voltage", "current", "temp1", "temp2", "gyro_rate"]
CHANNEL_META = {
    "voltage":   ("V", 26.0, 30.0),
    "current":   ("A", 1.5, 2.5),
    "temp1":     ("C", 20.0, 30.0),
    "temp2":     ("C", 18.0, 26.0),
    "gyro_rate": ("deg/s", -2.0, 2.0),
}

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sensor_channels (
        id INTEGER PRIMARY KEY,
        channel_name TEXT UNIQUE NOT NULL,
        unit TEXT, expected_min REAL, expected_max REAL
    );
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        onnx_path TEXT NOT NULL,
        quantized INTEGER NOT NULL,
        threshold REAL NOT NULL,
        avg_latency_ms REAL,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS inference_requests (
        id INTEGER PRIMARY KEY,
        model_version_id INTEGER REFERENCES model_versions(id),
        input_hash TEXT,
        anomaly_score REAL NOT NULL,
        is_anomaly INTEGER NOT NULL,
        suspected_channel_id INTEGER REFERENCES sensor_channels(id),
        latency_ms REAL NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    );
    """)
    for ch in CHANNELS:
        unit, lo, hi = CHANNEL_META[ch]
        conn.execute(
            "INSERT OR IGNORE INTO sensor_channels (channel_name, unit, expected_min, expected_max) VALUES (?,?,?,?)",
            (ch, unit, lo, hi),
        )
    conn.commit()
    conn.close()

def register_model_version(name, onnx_path, quantized, threshold, avg_latency_ms):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO model_versions (name, onnx_path, quantized, threshold, avg_latency_ms) VALUES (?,?,?,?,?)",
        (name, onnx_path, int(quantized), threshold, avg_latency_ms),
    )
    conn.commit()
    mid = cur.lastrowid
    conn.close()
    return mid

def get_active_model_version():
    conn = get_conn()
    row = conn.execute(
        "SELECT id, onnx_path, threshold FROM model_versions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row

def channel_id(name):
    conn = get_conn()
    row = conn.execute("SELECT id FROM sensor_channels WHERE channel_name=?", (name,)).fetchone()
    conn.close()
    return row[0] if row else None

def log_request(model_version_id, input_array, score, is_anomaly, suspected_channel_id, latency_ms):
    conn = get_conn()
    h = hashlib.sha256(json.dumps(input_array).encode()).hexdigest()[:16]
    conn.execute(
        """INSERT INTO inference_requests
           (model_version_id, input_hash, anomaly_score, is_anomaly, suspected_channel_id, latency_ms)
           VALUES (?,?,?,?,?,?)""",
        (model_version_id, h, score, int(is_anomaly), suspected_channel_id, latency_ms),
    )
    conn.commit()
    conn.close()

def get_stats():
    conn = get_conn()
    rows = conn.execute("SELECT latency_ms FROM inference_requests").fetchall()
    conn.close()
    if not rows:
        return {"count": 0}
    lat = np.array([r[0] for r in rows])
    return {
        "count": len(lat),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
    }

def list_channels():
    conn = get_conn()
    rows = conn.execute("SELECT channel_name, unit, expected_min, expected_max FROM sensor_channels").fetchall()
    conn.close()
    return [{"channel": r[0], "unit": r[1], "expected_min": r[2], "expected_max": r[3]} for r in rows]