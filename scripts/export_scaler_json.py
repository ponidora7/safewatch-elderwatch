"""
SafeWatch — Quick Scaler Serialization (No TensorFlow needed)
=============================================================
Serializes only the feature_scaler.pkl to JSON for browser use.
Run this first while TensorFlow/tf2onnx is installing.

Usage:
    python scripts/export_scaler_json.py
"""

import os
import json
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
FRONTEND_MODELS_DIR = os.path.join(PROJECT_ROOT, "frontend", "public", "models")
SCALER_JSON_PATH = os.path.join(FRONTEND_MODELS_DIR, "scaler.json")

os.makedirs(FRONTEND_MODELS_DIR, exist_ok=True)

print("SafeWatch -- Scaler Export Tool")
print("=" * 50)

if not os.path.exists(SCALER_PATH):
    print(f"[ERROR] Scaler not found at: {SCALER_PATH}")
    exit(1)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

scaler_data = {
    "mean_": scaler.mean_.tolist(),
    "scale_": scaler.scale_.tolist(),
    "n_features_in_": int(scaler.n_features_in_),
}

if hasattr(scaler, "var_") and scaler.var_ is not None:
    scaler_data["var_"] = scaler.var_.tolist()

with open(SCALER_JSON_PATH, "w") as f:
    json.dump(scaler_data, f, indent=2)

print(f"[OK] Scaler exported to: {SCALER_JSON_PATH}")
print(f"  Features: {scaler_data['n_features_in_']}")
print(f"  Mean sample: {scaler_data['mean_'][:5]}")
print(f"  Scale sample: {scaler_data['scale_'][:5]}")
print(f"\n[DONE] scaler.json saved in frontend/public/models/")
