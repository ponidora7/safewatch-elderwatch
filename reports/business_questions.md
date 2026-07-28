# ❓ Business Questions & Explanatory Analysis — SafeWatch

> [!WARNING]
> **DOKUMEN LEGACY** — Diperbarui: 2026-07-24
> Dokumen ini berisi analisis dari **pipeline versi awal** (Streamlit dashboard + MediaPipe) yang sudah digantikan oleh pipeline production v3.0 (React/Vite + YOLOv8-Pose + ONNX).
> - Angka distribusi kelas (8.007 normal / 27 fall) berasal dari **dataset versi MediaPipe lama**, bukan preprocessing YOLOv8-Pose saat ini (13.151 baris).
> - Angka akurasi 99.38% berasal dari **model baseline legacy**, bukan model Student ONNX production.
> - Referensi Streamlit, MediaPipe, dan `run_safewatch.py` sudah tidak berlaku; lihat `backend/main.py` dan `frontend/src/` untuk implementasi saat ini.

Dokumen ini menjawab 5 pertanyaan bisnis analitik yang didefinisikan untuk proyek SafeWatch. Setiap jawaban didasarkan pada data historis pipeline versi awal sebagai latar belakang alasan desain pipeline production v3.0.

---

## BQ1: Class Imbalance & The Accuracy Paradox
**Pertanyaan:** *"Apakah dataset postur manusia memiliki class imbalance yang dapat menyebabkan bias model terhadap aktivitas normal, dan bagaimana penanganannya?"*

**Jawaban & Insight:**  
Terdapat ketimpangan distribusi data yang sangat ekstrem pada dataset awal:
- Kelas **Normal** (berdiri/duduk) mendominasi dengan total **8.007 baris**.
- Kelas **Fall** (jatuh) hanya memiliki **27 baris**.
- Rasio ekstrem Normal : Fall ≈ 296 : 1.

**Implikasi & Solusi:** Ketimpangan ini awalnya memicu fenomena *Accuracy Paradox*, di mana AI berhenti belajar di Epoch 1 (Akurasi 99.69%) karena hanya menebak kelas mayoritas. Masalah ini diselesaikan menggunakan teknik **Oversampling**, menggandakan kelas minoritas hingga rasio seimbang (8.007 vs 8.007). Hasilnya, model Jaringan Saraf Tiruan (Functional API) mampu belajar secara sehat selama 19 Epoch dengan kurva validasi yang stabil dan akurasi nyata mencapai **99.38%**.

---

## BQ2: Ekstraksi Fitur Geometri vs Piksel Mentah
**Pertanyaan:** *"Apakah penggunaan ekstraksi titik sendi (pose landmarks) lebih efisien dibandingkan menyuapkan piksel gambar mentah ke dalam model klasifikasi?"*

**Jawaban & Insight:**  
Sistem menggunakan MediaPipe untuk mengekstrak 16 titik sendi krusial (bahu hingga pergelangan kaki).
- Fitur tabular (koordinat X dan Y) terbukti mereduksi beban komputasi secara drastis (hanya 16 fitur geometri per *frame*).
- Penerapan *Padding* 15% pada *Bounding Box* YOLO sebelum masuk ke MediaPipe mencegah terpotongnya titik sendi (amputasi digital) saat subjek berada di tepi jangkauan kamera.

**Kesimpulan:** Mengubah gambar mentah menjadi matriks angka spasial membuat model lebih kebal terhadap *noise* latar belakang dan variasi pencahayaan ruangan, menjadikannya sangat ideal untuk eksekusi *real-time* berkecepatan tinggi.

---

## BQ3: Data Readiness & Integritas Pipeline
**Pertanyaan:** *"Seberapa siap dan bersih data yang dihasilkan oleh pipeline ETL sebelum masuk ke ruang pelatihan arsitektur Deep Learning?"*

**Jawaban & Insight:**  
Seluruh data mentah telah melewati otomatisasi `data_cleaner.py`.
- ✅ Ekstraksi berhasil dan tersimpan rapi dalam `cleaned_human_fall.csv`.
- ✅ Data sepenuhnya berbentuk numerik (X11-X28, Y11-Y28) dalam rentang dinormalisasi (0.0 hingga 1.0).
- ✅ Pemisahan data *(Train-Test Split)* dilakukan secara *stratified* (80% Train, 20% Test) untuk menjaga proporsi distribusi.

**Readiness Score:** 100%. Data tabular dinyatakan sempurna dan siap dikonsumsi oleh lapisan *Dense Layer* pada TensorFlow.

---

## BQ4: Integrasi Deteksi Hazard (Fire & Smoke)
**Pertanyaan:** *"Bagaimana sistem menghindari tumpang tindih (overlap) atau alarm palsu (false positive) antara deteksi postur lansia dan bahaya lingkungan seperti kebakaran?"*

**Jawaban & Insight:**  
Sistem memisahkan beban kerja melalui arsitektur multi-model:
- YOLOv8 khusus objek manusia (`yolov8n.pt`) beroperasi secara independen dari model bahaya lingkungan (`fire_smoke.pt`).
- Untuk meminimalisir alarm palsu (*False Positive*) pada benda sehari-hari, sistem menetapkan *Confidence Threshold* yang ketat (> 0.45) dan filter *class name* spesifik hanya untuk `fire` dan `smoke`.

**Kesimpulan:** Pemisahan model memastikan sistem tidak kebingungan saat kondisi kritis terjadi bersamaan (misal: lansia jatuh karena menghirup asap), memungkinkan pencatatan dan pengiriman dua peringatan insiden yang berbeda secara paralel.

---

## BQ5: Keterbatasan Sensor 2D & Arsitektur Sensor Fusion
**Pertanyaan:** *"Bagaimana cara sistem mengatasi titik buta (blindspot) kamera 2D ketika korban jatuh terlentang/tengkurap lurus menghadap kamera?"*

**Jawaban & Insight:**  
Kamera 2D tidak memiliki pemahaman kedalaman (Sumbu-Z). Jatuh terlentang dapat memicu *The Crop Normalization Trap*, di mana MediaPipe membaca koordinat tubuh seperti orang berdiri (Prediksi 0.0), sehingga model Neural Network murni akan gagal.

**Solusi Logika Hibrida (*Sensor Fusion*):**
Sistem SafeWatch membypass titik buta ini dengan menggabungkan hasil prediksi AI dan **Rasio Dimensi Bounding Box YOLO (Lebar/Tinggi)**:
- **Rasio >= 1.05:** Korban terkapar mendatar (Mengabaikan nilai 0.0 dari jaringan saraf dan memaksa status `FALL DETECTED`).
- **Rasio 0.70 – 1.04:** Postur melebar abnormal / oleng (Memicu status `WARNING: LOSING BALANCE`).
- Logika spasial ini dihaluskan menggunakan *Buffer Moving Average* untuk mencegah kotak peringatan yang berkedip (*flickering*).

---

## ⚠️ Compliance Notes [Legacy]
- ✅ **Tidak ada data leakage:** Proses *Oversampling* dilakukan secara aman.
- ✅ **Real-Time Ready:** Sistem mampu mendeteksi kejadian dan mencatat log insiden di bawah 300ms per *frame* (pipeline production v3.0).
- ✅ **Dashboard Integration:** Dashboard React/Vite berlangganan Supabase Realtime untuk notifikasi instan.

*Pipeline versi ini (Streamlit) terakhir aktif: Mei 2026 | Digantikan: Juli 2026*
*Lihat `backend/main.py` dan `frontend/src/` untuk pipeline production v3.0*