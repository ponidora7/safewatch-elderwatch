# Fall Detection Model — Technical Report (v3.0)

> [!WARNING]
> **DOKUMEN INI TELAH DIPERBARUI** pada 2026-07-24.
> Angka dan arsitektur di bawah ini mencerminkan **pipeline production v3.0** (YOLOv8-Pose + Knowledge Distillation + ONNX).
> Klaim dari versi sebelumnya (57 fitur, MediaPipe, 8.034 sampel) berasal dari pipeline legacy dan **tidak berlaku untuk kode saat ini**.

---

## Executive Summary

Dokumen ini mendokumentasikan strategi peningkatan akurasi model deteksi jatuh SafeWatch melalui tiga komponen utama: **advanced feature engineering** (35 fitur total), **Knowledge Distillation** (Teacher-Student model), dan **kuantisasi ONNX** untuk deployment berbasis CPU/browser.

**Target peningkatan: 15–20% dibanding baseline** (~87% accuracy, ~92% recall).

---

## 1. Baseline Model Analysis

### 1.1 Arsitektur Baseline (Pipeline Legacy)
- **Model Type:** Keras Functional API
- **Arsitektur:** 128 → 64 → 32 neurons (3 hidden layers)
- **Ekstraktor Keypoint:** MediaPipe Pose (16 koordinat) *(versi lama)*
- **Jumlah Fitur:** 32 raw landmark coordinates
- **Dataset (versi legacy):** ~8.034 sampel hasil oversampling dari data imbalance ekstrem

### 1.2 Keterbatasan yang Diatasi
1. **Fitur Terbatas:** Koordinat mentah tanpa konteks geometris
2. **Tidak Ada Informasi Temporal:** Setiap frame diproses secara independen
3. **Threshold Tetap:** Default 0.5 tanpa optimasi cost-sensitive
4. **Metrik Tidak Lengkap:** Tidak ada ROC-AUC / PR-AUC

---

## 2. Pipeline Production v3.0

### 2.1 Data & Sampel

| Item | Nilai Aktual |
|---|---|
| Dataset | Human Fall Dataset (format YOLO) |
| Total data mentah (Fall dataset) | 13.158 anotasi |
| Setelah cleaning | **13.151 baris** (retensi 99,95%) |
| Kelas distribusi | Sleeping 34%, Standing 25%, Falling 21%, Sitting 18%, Bending 2% |
| Label biner final | `fall` (Falling) vs `normal` (semua kelas lain) |
| Ekstraktor Keypoint | **YOLOv8-Pose** (`yolov8n-pose.pt`) |
| Joint yang diseleksi | 8 joint (indeks COCO: 5,6,11,12,13,14,15,16) |

### 2.2 Feature Engineering — 19 Fitur Geometris Baru

Diimplementasikan di `src/advanced_feature_engineering.py` — kelas `AdvancedFeatureEngineer`.

**Kelompok Sudut Persendian (7 fitur):**
- `left_hip_angle` — sudut shoulder→hip→knee kiri
- `right_hip_angle` — sudut shoulder→hip→knee kanan
- `left_knee_angle` — sudut hip→knee→ankle kiri
- `right_knee_angle` — sudut hip→knee→ankle kanan
- `torso_angle` — kemiringan batang tubuh vs. sumbu vertikal (**indikator utama jatuh:** > 60°)
- `torso_tilt` — deviasi horizontal batang tubuh
- `spine_curvature` — asimetri horizontal bahu vs pinggul

**Kelompok Proporsi & Jarak (8 fitur):**
- `torso_length`, `avg_leg_length`, `hip_width`, `shoulder_width`
- `body_aspect_ratio` — rasio tinggi/lebar tubuh (**berubah drastis saat jatuh**)
- `ankle_spread`, `com_x`, `com_y` (estimasi pusat massa)

**Kelompok Deskriptor Postur (4 fitur):**
- `is_horizontal` — biner: 1.0 jika `torso_angle` > 60°
- `leg_spread_angle`, `shoulder_symmetry`, `avg_knee_angle`

**Total Input Model: 16 koordinat + 19 geometris = 35 fitur**

### 2.3 Arsitektur Model

#### Teacher Network (untuk training saja, input: 149 fitur temporal)
```
Input(149) → Dense(256)+BN+Drop(0.3) → Dense(128)+BN+Drop(0.3)
           → Dense(64)+BN+Drop(0.2)  → Dense(32)+BN+Drop(0.2)
           → Dense(1, sigmoid)
```
*149 fitur = 35 statis + 64 velocity (diff ke-1, 4×16) + 48 acceleration (diff ke-2, 3×16) + 2 torso angle stats*

#### Student Network (model production, input: 35 fitur)
```
Input(35) → Dense(128)+BN+Drop(0.3) → Dense(64)+BN+Drop(0.2)
          → Dense(32)+BN+Drop(0.2)  → Dense(1, sigmoid)
```

#### Knowledge Distillation Loss
```
loss = α × student_BCE + (1 − α) × MSE(teacher_pred, student_pred)
     = 0.4 × BinaryCrossentropy + 0.6 × MeanSquaredError
```

### 2.4 Konfigurasi Training

| Parameter | Nilai |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Epochs (Teacher) | max 30, EarlyStopping `val_auc_pr` patience=8 |
| Epochs (Student) | max 40, EarlyStopping `val_auc_pr` patience=8 |
| Batch Size | 32 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=4) |
| Data split | Train/Valid/Test sesuai kolom `split` dataset YOLO |
| Normalisasi | StandardScaler (35 fitur) |
| Sliding window | 5 frame untuk Teacher (velocity + acceleration) |

### 2.5 ONNX Export & Kuantisasi

```
Keras (.keras) → tf2onnx (opset 13) → ONNX Float32
             → onnxruntime.quantization.quantize_dynamic (QUInt8)
             → ONNX INT8 (~270KB)
             → Copy ke frontend/public/models/fall_model.onnx
```

Scaler diekspor sebagai `scaler.json` (mean, scale, var, n_features_in_) untuk digunakan di browser.

---

## 3. Threshold & Deployment

| Parameter | Nilai |
|---|---|
| Confidence threshold | **0.65** (dikode di `backend/main.py` & `onnxInference.ts`) |
| Temporal smoothing (backend) | simple majority 3/5 frame via `deque(maxlen=5)` |
| Temporal smoothing (frontend) | weighted voting 7 frame, bobot 0.5→2.0, threshold ≥ 5.5 |
| Frame capture interval | **1.500ms** (1,5 detik) |
| ONNX inference target | **< 10ms** per frame (CPU) |

---

## 4. Target Performa

| Metrik | Baseline (legacy) | Target (production) | Peningkatan |
|--------|------------------|---------------------|-------------|
| Accuracy | ~87% | **92–94%** | +5–7% |
| Recall | ~92% | **97–99%** | +5–7% |
| Precision | ~82% | **85–90%** | +3–8% |
| F1-Score | 0.867 | **0.92–0.94** | +5–7% |
| ROC-AUC | 0.90 | **≥ 0.96** | +6% |

> [!NOTE]
> Angka target di atas adalah proyeksi berdasarkan peningkatan per fase.
> Angka evaluasi aktual dapat diperoleh dengan menjalankan `legacy/ml/evaluate_enhanced_model.py`
> pada model final setelah training selesai.

---

## 5. Validasi & Pengujian

### 5.1 Verifikasi ONNX
```bash
python scripts/verify_onnx.py
# Expected: Input shape [None, 35], inference test passed
```

### 5.2 Health Check API
```bash
curl http://localhost:8000/health
# Expected: {"yolo_pose": true, "fall_classifier": true, "scaler": true}
```

### 5.3 End-to-End Realtime Test
```bash
python scripts/verify_realtime.py
# Expected: WebSocket broadcast received < 5 detik
```

---

## 6. Sensor Fusion untuk Blindspot Kamera 2D

Kamera 2D tidak memiliki informasi kedalaman (sumbu-Z). Jika subjek jatuh lurus menghadap kamera, YOLOv8-Pose/neural network bisa gagal mendeteksi (Crop Normalization Trap).

**Solusi Logika Hybrid (`business_questions.md`):**

| Kondisi Rasio BBox (width/height) | Status |
|---|---|
| `rasio_box < 0.70` | NORMAL (berdiri/duduk tegak) |
| `0.70 ≤ rasio_box < 1.05` | WARNING: LOSING BALANCE |
| `rasio_box ≥ 1.05` | ⚠️ FALL DETECTED (override neural network) |

---

## 7. Masalah Umum & Solusi

| Masalah | Solusi |
|---|---|
| `ONNX model not found` | Jalankan `train_distilled_model.py` lalu `verify_onnx.py` |
| `feature_scaler.pkl not found` | Pastikan `train_distilled_model.py` sudah dijalankan |
| `Memory Error` saat training | Kurangi `batch_size` dari 32 → 16 |
| Accuracy lebih buruk dari baseline | Periksa apakah CSV enhanced ada di `data/processed/`; jalankan ulang `preprocess_raw_data.py` |
| Scaler shape mismatch | Pastikan scaler dilatih dengan 35 fitur (bukan 57 dari pipeline lama) |
| Frontend ONNX.js gagal load | Cek `frontend/public/models/fall_model.onnx` ada; periksa `ort.env.wasm.wasmPaths` |

---

**Last Updated:** 2026-07-24
**Version:** 3.0 (YOLOv8-Pose + Knowledge Distillation + ONNX Production)
**Status:** ✅ Implementation Complete — Model siap untuk training dan deployment
