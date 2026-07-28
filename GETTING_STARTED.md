# 🎯 Getting Started — SafeWatch ElderWatch (Production)

> [!IMPORTANT]
> Dokumen ini menjelaskan **pipeline production aktual** (YOLOv8-Pose + Knowledge Distillation + ONNX).
> Kode legacy (Streamlit, MediaPipe, train_enhanced_model.py) berada di folder `legacy/ml/` dan tidak lagi digunakan secara aktif.

---

## Arsitektur Pipeline Saat Ini

```
data/raw/human_fall/          ← Dataset YOLO (gambar + label .txt)
        │
        ▼
scripts/preprocess_raw_data.py
  • YOLOv8-Pose: ekstrak 8 joint utama → 16 koordinat
  • AdvancedFeatureEngineer: hitung 19 fitur geometris
  • Output: 35 fitur total per frame
  • Simpan: cleaned_human_fall.csv + cleaned_human_fall_enhanced.csv
        │
        ▼
scripts/train_distilled_model.py
  • Teacher Model (149 fitur temporal) → membimbing Student
  • Student Model (35 fitur, single-frame) → model production
  • Knowledge Distillation loss: 0.4×BCE + 0.6×MSE
  • Output: safewatch_fall_model_enhanced.keras + feature_scaler.pkl
        │
        ▼
scripts/convert_model_for_browser.py
  • Keras → ONNX Float32 (tf2onnx, opset 13)
  • ONNX → INT8 Quantization (dynamic)
  • Output: safewatch_model_cpu.onnx + frontend/public/models/fall_model.onnx
  • Output: frontend/public/models/scaler.json
        │
        ▼
backend/main.py (FastAPI)          frontend/src/ (Vite + React)
  • /pose-extract  ←────────────────  WebcamFeed.tsx (1500ms interval)
  • /log-incident  ←────────────────  onnxInference.ts (35 fitur → ONNX.js)
  • /health                          Supabase Realtime subscription
```

---

## 🚀 Alur Kerja Cepat

### Langkah 1: Pra-pemrosesan Data (~20–60 menit tergantung dataset)

Pastikan dataset YOLO ada di `data/raw/human_fall/` dengan struktur:
```
data/raw/human_fall/
  ├── data.yaml
  ├── train/images/ + labels/
  ├── valid/images/ + labels/
  └── test/images/  + labels/
```

Jalankan dari root project:
```bash
python scripts/preprocess_raw_data.py
```

**Output yang dihasilkan:**
- `data/processed/cleaned_human_fall.csv` — metadata + 16 koordinat keypoint
- `data/processed/cleaned_human_fall_enhanced.csv` — metadata + **35 fitur** (16 + 19 geometris)

---

### Langkah 2: Pelatihan Model (~10–30 menit)

```bash
python scripts/train_distilled_model.py
```

**Output yang dihasilkan:**
- `models/safewatch_fall_model_enhanced.keras` ✅ (Student model)
- `models/feature_scaler.pkl` ✅ (StandardScaler, 35 fitur)
- `models/safewatch_model_cpu_float32.onnx` ✅ (ONNX Float32)
- `models/safewatch_model_cpu.onnx` ✅ (ONNX INT8 quantized)
- `frontend/public/models/fall_model.onnx` ✅ (untuk browser ONNX.js)
- `frontend/public/models/scaler.json` ✅ (parameter scaler untuk browser)

---

### Langkah 3: Verifikasi Model ONNX (< 1 menit)

```bash
python scripts/verify_onnx.py
```

**Expected output:**
```
[OK] Model input dimension 35 matches expected 35 features!
[OK] Model inference test passed!
[OK] Copied valid ONNX model to backend destination
```

---

### Langkah 4: Benchmark Inference Speed (opsional)

```bash
python scripts/convert_model_for_browser.py
```

**Expected output:**
```
✓ Average inference: <10ms (target: <10ms for client-side)
```

---

### Langkah 5: Jalankan Backend API

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # isi nilai SUPABASE_URL, SUPABASE_KEY, RESEND_API_KEY
uvicorn main:app --reload --port 8000
```

Cek: `GET http://localhost:8000/health`

---

### Langkah 6: Jalankan Frontend Dashboard

```bash
cd frontend
pnpm install
cp .env.example .env.local   # isi VITE_API_URL=http://localhost:8000
pnpm dev
```

Buka: `http://localhost:3000`

---

## 📊 Spesifikasi Teknis Production

### Input Model (35 Fitur)

| Grup Fitur | Jumlah | Deskripsi |
|---|---|---|
| Koordinat keypoint (16) | 16 | X,Y 8 sendi utama dari YOLOv8-Pose (Bahu, Pinggul, Lutut, Pergelangan) |
| Sudut persendian | 7 | hip angle kiri/kanan, knee angle kiri/kanan, torso_angle, torso_tilt, spine_curvature |
| Proporsi tubuh | 8 | torso_length, avg_leg_length, hip_width, shoulder_width, body_aspect_ratio, ankle_spread, com_x, com_y |
| Deskriptor postur | 4 | is_horizontal, leg_spread_angle, shoulder_symmetry, avg_knee_angle |
| **Total** | **35** | |

### Arsitektur Student Model (Production)

```
Input(35) → Dense(128)+BN+Drop(0.3) → Dense(64)+BN+Drop(0.2) → Dense(32)+BN+Drop(0.2) → Dense(1,sigmoid)
```

### Target Performa

| Metrik | Target |
|---|---|
| Accuracy | ≥ 92% |
| Recall | ≥ 97% (safety-critical) |
| Confidence threshold | 0.65 |
| ONNX inference (CPU) | < 10ms per frame |
| Capture interval | 1.500ms (1,5 detik) |

---

## 🔧 Hardware Requirements

| Konfigurasi | RAM | Training Time |
|---|---|---|
| Minimum (CPU only) | 4 GB | ~15–30 menit |
| Direkomendasikan (GPU CUDA) | 8 GB+ | ~3–8 menit |

```bash
# Verifikasi GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

## ⚙️ Environment Variables

**Backend (`backend/.env`):**
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
RESEND_API_KEY=re_your_resend_key
EMAIL_FROM=onboarding@resend.dev
EMAIL_TO=guardian@example.com
CONFIDENCE_THRESHOLD=0.65
HF_API_URL=                    # opsional, kosongkan jika pakai local inference
HF_TOKEN=                      # opsional
```

**Frontend (`frontend/.env.local`):**
```env
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_KEY=your-supabase-anon-key
```

---

## 📁 Struktur File Penting

```
safewatch-elderwatch/
├── scripts/
│   ├── preprocess_raw_data.py       ← ETL: YOLOv8-Pose + feature engineering
│   ├── train_distilled_model.py     ← Training: Teacher-Student KD + ONNX export
│   ├── convert_model_for_browser.py ← Konversi & benchmark ONNX
│   ├── verify_onnx.py               ← Verifikasi model ONNX
│   └── verify_realtime.py           ← Tes WebSocket Supabase Realtime
├── src/
│   ├── advanced_feature_engineering.py ← Kelas AdvancedFeatureEngineer (19 fitur)
│   ├── data_cleaner.py                 ← Pembersihan dataset
│   ├── data_assessment.py              ← Asesmen kualitas dataset YOLO
│   └── imbalanced_handling.py          ← SMOTE & Focal Loss
├── models/
│   ├── safewatch_model_cpu.onnx         ← Model production (INT8)
│   └── feature_scaler.pkl               ← StandardScaler (35 fitur)
├── backend/
│   └── main.py                          ← FastAPI: /pose-extract, /inference, /health
└── frontend/
    └── src/
        ├── components/WebcamFeed.tsx    ← Monitor webcam + skeleton overlay
        ├── services/onnxInference.ts    ← ONNX.js browser inference
        └── services/api.ts              ← API client (extractPose, logIncident)
```

---

**Last Updated:** 2026-07-24 | **Pipeline Version:** 3.0 (YOLOv8-Pose + Knowledge Distillation + ONNX)
**Status:** ✅ Production Ready
