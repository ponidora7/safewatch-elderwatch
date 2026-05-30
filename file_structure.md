# File Structure SafeWatch

Dokumen ini menjelaskan struktur file dan folder proyek SafeWatch, dengan fokus khusus pada data dan alur yang relevan untuk menyusun SRS peningkatan akurasi model.

## 1. Ringkasan Struktur Utama

- `config/` — konfigurasi path dan setup proyek.
- `dashboard/` — aplikasi dasbor Streamlit dan utilitas tampilannya.
- `data/` — semua data mentah, data pengolahan, dan contoh gambar.
- `devops/` — skrip penyiapan dan pipeline untuk lingkungan dan deployment lokal.
- `models/` — model dan artifact machine learning.
- `notebooks/` — notebook analisis data dan persiapan dashboard.
- `reports/` — dokumen hasil analisis, readiness, dan figur visual.
- `runs/` — hasil eksperimen dan pelatihan dari Ultralytics YOLO.
- `scripts/` — skrip pelatihan, inferensi, tuning, dan validasi.
- `src/` — modul aplikasi inti SafeWatch.
- `legacy/` — kode lama/eksperimental yang tidak menjadi jalur utama.

## 2. Data: Struktur dan Isi

### 2.1 `data/raw/`

Folder ini seharusnya menyimpan dataset mentah dan anotasinya.
Konfigurasi `config/paths.py` menunjukkan tiga dataset raw utama:

- `data/raw/human_fall/`
  - Data untuk deteksi jatuh manusia.
  - Biasanya berisi gambar dan anotasi YOLO untuk kelas "fall" / "person".

- `data/raw/fire_smoke_detection/`
  - Data untuk deteksi kebakaran dan asap.
  - Biasanya berisi gambar serta anotasi YOLO untuk kelas "fire" dan "smoke".

- `data/raw/person_detection/`
  - Data yang digunakan untuk mendeteksi keberadaan manusia.
  - Berguna sebagai dataset pendukung untuk model hibrida.

> Catatan: Jika folder `data/raw/` saat ini kosong, pastikan dataset mentah dipindahkan ke sini agar pipeline dapat berjalan konsisten. Kode dalam `config/paths.py` mengharapkan struktur ini.

### 2.2 `data/processed/`

Folder ini menyimpan output pipeline pemrosesan data.

Saat ini berisi:

- `cleaned_human_fall.csv`
  - CSV utama hasil pembersihan dan integrasi fitur untuk deteksi jatuh.
  - Digunakan oleh dashboard, skrip training, dan inference.

- `5_6084769851455315969.parquet`
  - File Parquet yang tersimpan sebagai versi terkompresi untuk analisis atau proses lanjutan.
  - Bisa berisi dataset fitur yang sudah diolah / ringkasan statistik.

### 2.3 `data/real_footage/`

Menyimpan contoh gambar nyata yang mungkin digunakan untuk evaluasi manual atau penyajian hasil.

- Berisi file `.jpg` untuk kejadian `FALL_...` dan `FIRE_...`.

### 2.4 `data/snapshots/`

Menyimpan tangkapan layar atau snapshot hasil deteksi.

- Berisi file gambar `FALL_...` yang bisa menjadi referensi visual.

## 3. Models

- `models/` — folder utama model.
- `models/fire_smoke.pt` — model YOLO atau model deteksi kebakaran.
- `models/safewatch_fall_model.keras` — model Keras untuk deteksi jatuh.
- `models/enhanced_fall_classifier.py` — definisi atau wrapper model klasifikasi.
- `models/yolov8n.pt` — model YOLOv8 baseline.
- `models/training_history.pkl` — history pelatihan untuk analisis metrik.

> Rekomendasi: Model besar biasanya sebaiknya dikelola dengan Git LFS atau disimpan di lokasi terpisah agar repositori tetap ringan.

## 4. Scripts

Folder `scripts/` berisi skrip operasional utama:

- `train_model.py` — pelatihan model utama.
- `inference.py` — skrip inferensi model.
- `hyperparameter_tuning.py` — eksperimen tuning parameter.
- `run_safewatch.py` — runner sistem SafeWatch.
- `verify_installation.py` — skrip pemeriksaan dependency python.
- `inspect_imports.py` — diagnostik import package.
- `check_env.py` — pemeriksaan environment ringan.
- `try_mediapipe.py` — isolasi tes `mediapipe`.

## 5. Devops / pipeline

Folder `devops/` berisi skrip dukungan lingkungan:

- `debug_and_try_tf.ps1` — debug instalasi TensorFlow di Windows.
- `run_pipeline.bat` — menjalankan pipeline atau alur kerja tertentu.
- `setup_safewatch.ps1` — setup awal environment.

## 6. Dashboard

- `dashboard/app.py` — aplikasi Streamlit utama.
- `dashboard/utils.py` — utilitas pemuatan data dan rendering metrik.

## 7. Source Code Inti

Folder `src/` berisi modul SafeWatch inti:

- `src/data_assessment.py`
- `src/data_cleaner.py`
- `src/data_loader.py`
- `src/eda_utils.py`
- `src/evaluation.py`
- `src/feature_temporal.py`
- `src/imbalanced_handling.py`
- `src/preprocessing.py`
- `src/threshold_optimizer.py`
- `src/validation.py`
- `src/validators.py`

## 8. Notebook dan Laporan

- `notebooks/` berisi analisis data, pembersihan, EDA, dan persiapan dashboard.
- `reports/` berisi dokumentasi dan hasil evaluasi seperti:
  - `business_questions.md`
  - `data_dictionary.md`
  - `data_readiness_checklist.md`
  - `data_readiness_result.csv`
  - `data_readiness_report.json`
  - `reports/figures/`

## 9. Legacy / Eksperimental

- `legacy/` berisi kode lama atau eksperimen yang tidak digunakan langsung dalam alur utama.
- Saat ini terdiri dari:
  - `extract_features.py`
  - `train_fire.py`
  - `yolov8.py`

> Saran: Jika kode ini tidak diperlukan lagi, pindahkan ke folder `archive/` atau hapus setelah diverifikasi.

## 10. Rekomendasi untuk SRS peningkatan akurasi

1. **Standardisasi data**
   - Pastikan semua dataset mentah (`data/raw/`) distandarisasi ke format YOLO atau bounding box yang sama.
   - Tentukan label kelas tunggal dan konsisten (misalnya gunakan `fall` bukan `Fall-Detected` di semua anotasi).

2. **Pisahkan data training / validation / test**
   - Buat subfolder jelas di `data/raw/` atau `data/processed/` untuk `train/`, `val/`, `test/`.

3. **Dokumentasi setiap dataset**
   - Tambahkan README kecil dalam `data/` menjelaskan isi tiap subfolder dan sumber datanya.
   - Dokumentasikan `data/processed/cleaned_human_fall.csv` — fitur apa saja, bagaimana dibuat.

4. **Pipeline ETL**
   - `data_cleaner.py` adalah titik penting untuk SRS. Jelaskan input, transformasi, output, dan metrik kualitas data.

5. **Evaluasi model**
   - Gunakan `reports/` dan `runs/` untuk menyimpan hasil evaluasi.
   - Simpan metrik akurasi, precision/recall, confusion matrix, dan dataset uji yang digunakan.

6. **Pengelolaan artifact**
   - Simpan model akhir di `models/` dan hasil pelatihan di `runs/`.
   - Pastikan `file_structure.md` ini disesuaikan jika model baru atau dataset baru ditambahkan.

---

Dokumen ini dibuat untuk membantu menyusun SRS yang jelas dan fokus pada `data/` serta alur data yang menjadi dasar peningkatan akurasi model SafeWatch.