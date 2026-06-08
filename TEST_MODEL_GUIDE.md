# Panduan Pengujian Model SafeWatch

## 1. Ringkasan
Panduan ini menjelaskan langkah-langkah untuk menguji model deteksi jatuh di repositori SafeWatch.

Ada dua mode utama:
- `scripts/inference.py` — tes cepat dengan model dasar
- `scripts/inference_enhanced.py` — tes dengan model enhanced dan fitur tambahan
- `scripts/evaluate_enhanced_model.py` — evaluasi lengkap model enhanced

> Jalankan semua perintah dari root project: `SafeWatch.worktrees\agents-enhance-fall-detection-accuracy`

---

## 2. Prasyarat
1. Python terpasang (disarankan Python 3.8+)
2. Dependency terpasang sesuai `requirements.txt`
3. Data terproses tersedia di `data/processed/cleaned_human_fall.csv`
4. Folder `models/` writable untuk menyimpan model dan scaler

Jika belum memasang dependency, jalankan:

```bash
pip install -r requirements.txt
```

---

## 3. Tes cepat dengan model dasar
Gunakan `scripts/inference.py` untuk mengecek model dasar yang ada saat ini.

```bash
python scripts/inference.py
```

### Apa yang dilakukan
- Memuat model pada `models/safewatch_fall_model.keras`
- Mengambil sampel satu data non-fall dan satu data fall dari `data/processed/cleaned_human_fall.csv`
- Menampilkan prediksi probabilitas dan keputusan klasifikasi

### Hasil yang diharapkan
- Satu prediksi untuk data normal
- Satu prediksi untuk data jatuh
- Output menampilkan probabilitas `fall` dan label

---

## 4. Tes model enhanced
Gunakan `scripts/inference_enhanced.py` untuk menguji model enhanced dengan fitur lanjutan.

```bash
python scripts/inference_enhanced.py
```

### Apa yang dilakukan
- Memuat model enhanced dari `models/safewatch_fall_model_enhanced.keras`
- Memuat scaler dari `models/feature_scaler.pkl`
- Memuat CSV `data/processed/cleaned_human_fall.csv`
- Mengekstrak fitur tambahan melalui `src/advanced_feature_engineering.py`
- Menguji satu contoh normal dan satu contoh jatuh
- Menampilkan probabilitas dan status threshold

### Persiapan yang dibutuhkan
Jika belum tersedia, jalankan dulu pelatihan:

```bash
python scripts/train_enhanced_model.py
```

---

## 5. Evaluasi lengkap
Untuk melihat metrik dan laporan evaluasi model enhanced, jalankan:

```bash
python scripts/evaluate_enhanced_model.py
```

### Output utama
- Accuracy
- Loss
- F1-score
- Precision dan recall untuk threshold berbeda
- ROC-AUC dan PR-AUC
- Visualisasi evaluasi (grafik matriks kebingungan, kurva ROC/PR, distribusi probabilitas)

### Kapan gunakan
- Setelah model enhanced selesai dilatih
- Saat ingin memvalidasi performa pada data test
- Untuk memilih threshold deteksi yang tepat

---

## 6. Alur pengujian rekomendasi
1. Pastikan dependency sudah terpasang
2. Jalankan `python scripts/train_enhanced_model.py` untuk membuat model enhanced
3. Jalankan `python scripts/inference_enhanced.py` untuk tes cepat enhanced
4. Jalankan `python scripts/evaluate_enhanced_model.py` untuk evaluasi lengkap

---

## 7. Troubleshooting
- Jika muncul `Model tidak ditemukan`, pastikan file model ada di folder `models/`
- Jika data tidak ditemukan, pastikan `data/processed/cleaned_human_fall.csv` tersedia
- Jika ada error import `src.advanced_feature_engineering`, jalankan dari root project

---

## 8. Lokasi file penting
- `scripts/train_enhanced_model.py`
- `scripts/inference_enhanced.py`
- `scripts/evaluate_enhanced_model.py`
- `scripts/inference.py`
- `models/safewatch_fall_model_enhanced.keras`
- `models/safewatch_fall_model.keras`
- `models/feature_scaler.pkl`
- `data/processed/cleaned_human_fall.csv`
- `src/advanced_feature_engineering.py`
