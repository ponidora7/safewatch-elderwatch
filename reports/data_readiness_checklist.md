# ✅ Data Readiness Checklist — SafeWatch

Dokumen ini digunakan untuk memvalidasi kesiapan dan integritas data hasil ekstraksi *pipeline* ETL (YOLOv8 + MediaPipe) sebelum disuapkan ke dalam ruang pelatihan arsitektur *Deep Learning*.

## 🔹 Checklist Kualitas Data

### A. Integritas Dataset (DataFrame)
- [x] **DataFrame Tidak Kosong:** File ekstraksi memiliki baris data yang cukup untuk pelatihan.
- [x] **Bebas Nilai Kosong (Null/NaN):** Seluruh kolom esensial dari hasil ekstraksi terisi dengan sempurna tanpa ada nilai yang hilang (mencegah *crash* pada Keras).
- [x] **Bebas Duplikasi Data:** Baris data 100% unik, memastikan model tidak menghafal data yang sama secara berulang.

### B. Validitas Fitur Geometri (MediaPipe)
- [x] **Fitur 16 Sendi Terpenuhi:** Seluruh 16 titik *landmark* anatomi krusial (Bahu hingga Pergelangan Kaki) berhasil diekstrak (X dan Y).
- [x] **Normalisasi Koordinat:** Nilai titik koordinat X dan Y berada dalam rentang wajar spasial kamera (0.0 hingga 1.0).

### C. Validitas Spasial (YOLO Bounding Box)
- [x] **Validitas Bounding Box:** Titik pojok Bounding Box (X1, Y1, X2, Y2) valid dan berada di dalam batas resolusi *frame*.
- [x] **Padding Otomatis:** Area pemotongan (*crop*) sudah memperhitungkan toleransi ruang (15%) agar titik sendi pinggir tidak teramputasi.

### D. Keseimbangan Kelas
- [x] **Keterwakilan Kelas (Min: 1):** Seluruh target kelas (*Normal* maupun *Fall*) memiliki minimal sampel perwakilan.
- [x] **Mitigasi Imbalance:** (Terjadwal) Ketimpangan rasio kelas minoritas yang ekstrem akan ditangani menggunakan teknik *Oversampling* di tahap pelatihan `train_model.py`.

### E. Status Pipeline
- [x] File `cleaned_human_fall.csv` berhasil di-generate oleh skrip `data_cleaner.py`.
- [x] File manifest pengecekan `dats_readlines_result.csv` berhasil mencetak nilai logis `True` untuk seluruh tes.

---

## 📊 Hasil Validasi Terakhir

Tabel di bawah ini merepresentasikan status kelayakan data berdasarkan pengecekan otomatis terakhir sebelum proses pelatihan *Deep Learning* dimulai.

| Indikator Validasi | Hasil / Status | Keterangan Sistem |
|--------------------|----------------|-------------------|
| **Total Sampel Terekstrak** | `8,034` baris | Ekstraksi *frame* MediaPipe berhasil |
| **Kelengkapan Fitur (MediaPipe)** | `Passed` (True) | 16 titik sendi lengkap (X,Y) |
| **Integritas Box (YOLO)** | `Passed` (True) | Semua *box* berada di dalam *frame* |
| **Bebas Duplikasi** | `Passed` (True) | Data unik 100% |
| **Bebas Nilai Null** | `Passed` (True) | Seluruh kolom esensial terisi penuh |
| **Status Kesiapan Akhir** | **🟢 READY FOR TRAINING** | Data memenuhi standar *Input Layer Neural Network* |

*Diperbarui otomatis oleh sistem validasi Pipeline ETL SafeWatch.*