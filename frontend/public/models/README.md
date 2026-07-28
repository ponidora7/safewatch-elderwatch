# SafeWatch — Browser Model Assets

This directory contains static model assets loaded by `onnxruntime-web` in the browser.

## Files

| File | Status | Description |
|---|---|---|
| `scaler.json` | ✅ Ready | StandardScaler parameters (35 features). Auto-generated from `feature_scaler.pkl`. |
| `fall_model.onnx` | ❌ Missing — see below | ONNX fall detection classifier. Must be converted from Keras. |

## How to generate `fall_model.onnx`

**Option 1: Google Colab (Recommended — Free, no local install needed)**
1. Open https://colab.research.google.com
2. Upload `notebooks/convert_model_colab.ipynb`
3. Upload `models/safewatch_fall_model.keras` when prompted
4. Run all cells
5. Download `fall_model.onnx`
6. Place it here: `frontend/public/models/fall_model.onnx`

**Option 2: Local Python 3.10 or 3.11**
```bash
pip install tf2onnx tensorflow onnx onnxruntime scikit-learn numpy
python scripts/convert_model_for_browser.py
```
Note: Python 3.12+ and 3.14 are NOT supported by TensorFlow.

## Without `fall_model.onnx`

The frontend will automatically fall back to full server-side inference via `/inference` endpoint.
Everything will still work — just with slightly higher latency (server cold start).
