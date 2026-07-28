# Panduan Pengujian Model SafeWatch (Production)

> [!IMPORTANT]
> Panduan ini merujuk ke pipeline **production aktual** menggunakan YOLOv8-Pose + ONNX.
> Script lama (`scripts/inference.py`, `scripts/inference_enhanced.py`, dll.) telah dipindahkan ke `legacy/ml/` dan **tidak aktif digunakan**.

---

## 1. Ringkasan

Ada dua mode pengujian utama pada pipeline production:

| Mode | Script / Tools | Keterangan |
|---|---|---|
| **Verifikasi ONNX** | `scripts/verify_onnx.py` | Tes model ONNX: input/output shape + dummy inference |
| **Tes Realtime Supabase** | `scripts/verify_realtime.py` | Tes WebSocket broadcast end-to-end |
| **Benchmark Backend** | `GET /health` via browser/curl | Cek status YOLO + ONNX + Scaler di API |

> Jalankan semua perintah dari **root project:**
> ```
> c:\Users\p\1. Kuliah\KP\safewatch-elderwatch\
> ```

---

## 2. Prasyarat

1. **Python 3.11** (sesuai `backend/.python-version` dan `Dockerfile`)
2. Dependency terpasang: `pip install -r requirements.txt`
3. Model ONNX tersedia di `models/safewatch_model_cpu.onnx` dan `frontend/public/models/fall_model.onnx`
4. Scaler tersedia di `models/feature_scaler.pkl`

Jika model belum ada, jalankan dulu pipeline training:
```bash
python scripts/preprocess_raw_data.py   # ekstrak keypoints dari data raw
python scripts/train_distilled_model.py # latih Teacher+Student, export ONNX
```

---

## 3. Verifikasi Model ONNX

Gunakan `scripts/verify_onnx.py` untuk mengecek model ONNX yang siap digunakan di backend dan browser.

```bash
python scripts/verify_onnx.py
```

### Yang dilakukan:
- Memuat model dari `frontend/public/models/fall_model.onnx`
- Menampilkan nama dan shape input/output layer
- Menjalankan dummy prediction (input 35 float acak)
- Menyalin model yang valid ke `models/safewatch_model_cpu.onnx`

### Output yang diharapkan:
```
SafeWatch ONNX Model Verification
==================================================
--- Input Nodes ---
Input 0: name='input_layer', shape=[None, 35], type=tensor(float)

--- Output Nodes ---
Output 0: name='...', shape=[None, 1], type=tensor(float)

[OK] Model input dimension 35 matches expected 35 features!
[OK] Model inference test passed!
[OK] Copied valid ONNX model to backend destination
```

---

## 4. Benchmark Kecepatan Inference ONNX

Untuk mengukur kecepatan inference CPU (target < 10ms per frame):

```bash
python scripts/convert_model_for_browser.py
```

### Output yang diharapkan:
```
✓ Average inference: X.XXms (target: <10ms for client-side)
```

---

## 5. Tes Health Backend (Live API)

Setelah backend berjalan (`uvicorn main:app`):

```bash
curl http://localhost:8000/health
```

### Output yang diharapkan:
```json
{
  "status": "ok",
  "message": "SafeWatch API is running",
  "models": {
    "yolo_pose": true,
    "fall_classifier": true,
    "scaler": true
  }
}
```

Jika salah satu bernilai `false`, cek keberadaan file model.

---

## 6. Tes Realtime WebSocket Supabase

Untuk memvalidasi koneksi broadcast Supabase Realtime (perlu `.env` dengan kredensial Supabase):

```bash
python scripts/verify_realtime.py
```

### Yang dilakukan:
1. Subscribe ke tabel `incidents` via WebSocket
2. Insert data mock incident ke database
3. Verifikasi broadcast diterima dalam < 5 detik
4. Hapus data mock (cleanup)

### Output sukses:
```
[OK] Mock incident inserted successfully with ID: XXX
[SUCCESS] Broadcast received in 0.XXXX seconds.
[SUCCESS] End-to-End Realtime WebSocket broadcast path is working perfectly!
```

---

## 7. Alur Pengujian Lengkap (Rekomendasi)

```
1. python scripts/preprocess_raw_data.py     # Pastikan data ada
2. python scripts/train_distilled_model.py   # Train & export ONNX
3. python scripts/verify_onnx.py             # Verifikasi model ONNX
4. python scripts/convert_model_for_browser.py  # Benchmark kecepatan
5. uvicorn backend/main:app --reload         # Jalankan API
6. curl localhost:8000/health                # Cek status semua model
7. python scripts/verify_realtime.py         # Tes WebSocket Supabase
```

---

## 8. Lokasi File Penting (Production)

| File | Fungsi |
|---|---|
| `scripts/preprocess_raw_data.py` | ETL: YOLOv8-Pose keypoint extraction |
| `scripts/train_distilled_model.py` | Training Teacher+Student + ONNX export |
| `scripts/convert_model_for_browser.py` | Konversi Keras → ONNX + benchmark |
| `scripts/verify_onnx.py` | Verifikasi model ONNX |
| `scripts/verify_realtime.py` | Tes WebSocket Supabase end-to-end |
| `models/safewatch_model_cpu.onnx` | Model ONNX INT8 untuk backend |
| `models/feature_scaler.pkl` | StandardScaler (35 fitur) |
| `frontend/public/models/fall_model.onnx` | Model ONNX untuk browser (ONNX.js) |
| `frontend/public/models/scaler.json` | Scaler JSON untuk browser |
| `src/advanced_feature_engineering.py` | Kelas AdvancedFeatureEngineer |
| `backend/main.py` | FastAPI: /pose-extract, /inference, /health |

> [!NOTE]
> Script lama untuk inferensi dan evaluasi (`inference.py`, `inference_enhanced.py`, `evaluate_enhanced_model.py`, `train_enhanced_model.py`) tersimpan di `legacy/ml/` sebagai referensi historis.

---

**Last Updated:** 2026-07-24 | **Python:** 3.11 | **Pipeline Version:** 3.0 (YOLOv8-Pose + KD + ONNX)
