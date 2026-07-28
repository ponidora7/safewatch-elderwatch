# 📋 SafeWatch Data Dictionary

> [!NOTE]
> **Diperbarui: 2026-07-24** | Pipeline v3.0 (YOLOv8-Pose + Knowledge Distillation + ONNX)
> Ekstraktor keypoint pada pipeline production adalah **YOLOv8-Pose**, bukan MediaPipe.
> Input model production adalah **35 fitur** (16 koordinat + 19 geometris), bukan hanya 16.

Dokumen ini mendefinisikan setiap kolom (*field*) dalam dataset matang hasil ekstraksi *pipeline* ETL SafeWatch v3.0 menggunakan **YOLOv8-Pose** + `AdvancedFeatureEngineer`.

## 🔹 Metadata & Label Dasar

| Field | Tipe | Range/Contoh | Deskripsi | Sumber |
|-------|------|--------------|-----------|--------|
| `image_path` | string | `".../data/raw/images/img001.jpg"` | Jalur absolut ke file gambar asli | Ekstraksi awal |
| `class_name` | string | `"normal"`, `"fall"` | Kategori postur manusia secara tekstual | Label gambar asli |
| `class_id` | integer | `0` (Normal), `1` (Fall) | ID numerik kelas (Target Prediksi AI) | Mapping dari `class_name` |

## 🔹 Fitur Landmark Sendi (Input Pipeline Production)

Merupakan 16 koordinat sendi (*keypoints*) yang diekstrak oleh **YOLOv8-Pose** (`yolov8n-pose.pt`) dari 8 joint anatomi utama. Semua koordinat dinormalisasi dalam rentang `0.0` - `1.0` (normalized image coordinates). Bersama 19 fitur geometris dari `AdvancedFeatureEngineer`, total input model production adalah **35 fitur**.

| Field | Tipe | Anatomi Tubuh | Sumbu | Range |
|-------|------|---------------|-------|-------|
| `X11`, `Y11` | float | Bahu Kiri (*Left Shoulder*) | Horizontal (X), Vertikal (Y) | `0.0` - `1.0` |
| `X12`, `Y12` | float | Bahu Kanan (*Right Shoulder*) | Horizontal (X), Vertikal (Y) | `0.0` - `1.0` |
| `X23`, `Y23` | float | Pinggul Kiri (*Left Hip*) | Horizontal (X), Vertikal (Y) | `0.0` - `1.0` |
| `X24`, `Y24` | float | Pinggul Kanan (*Right Hip*) | Horizontal (X), Vertikal (Y) | `0.0` - `1.0` |
| `X25`, `Y25` | float | Lutut Kiri (*Left Knee*) | Horizontal (X), Vertikal (Y) | `0.0` - `1.0` |
| `X26`, `Y26` | float | Lutut Kanan (*Right Knee*) | Horizontal (X), Vertikal (Y) | `0.0` - `1.0` |
| `X27`, `Y27` | float | Engkel Kiri (*Left Ankle*) | Horizontal (X), Vertikal (Y) | `0.0` - `1.0` |
| `X28`, `Y28` | float | Engkel Kanan (*Right Ankle*) | Horizontal (X), Vertikal (Y) | `0.0` - `1.0` |

## 🔹 Fitur Bounding Box (YOLOv8)

Fitur ini didapatkan dari `yolov8n.pt` dan digunakan untuk melakukan *Cropping* berpelindung (*Padding 15%*) serta memicu sistem Logika Hibrida saat produksi *real-time*.

| Field | Tipe | Deskripsi | Penggunaan dalam Sistem |
|-------|------|-----------|-------------------------|
| `bbox_x1`, `bbox_y1` | int (pixel) | Koordinat kiri atas (*Top-Left*) | Titik awal menggambar kotak |
| `bbox_x2`, `bbox_y2` | int (pixel) | Koordinat kanan bawah (*Bottom-Right*) | Titik akhir menggambar kotak |
| `bbox_w_px` | int (pixel) | Lebar absolut Bounding Box | Dasar perhitungan Padding 15% (Sumbu X) |
| `bbox_h_px` | int (pixel) | Tinggi absolut Bounding Box | Dasar perhitungan Padding 15% (Sumbu Y) |
| `rasio_box` | float | Rasio aspek (`bbox_w / bbox_h`) | **Sensor Fusion:** Menangani kelemahan kamera 2D |

## 🔹 Fitur Komputasi Khusus [LEGACY — Streamlit Dashboard]

> [!WARNING]
> Bagian ini merujuk pada logika dari **pipeline legacy** (Streamlit dashboard, `legacy/ml/`) yang sudah digantikan oleh dashboard React/Vite. Informasi ini disimpan sebagai referensi historis dan **tidak mencerminkan behavior kode production saat ini**.

Metrik rekayasa (*Feature Engineering*) ini digunakan pada Dasbor Analitik Streamlit (versi lama) dan logika sensor fusion berbasis rasio bounding box.

| Field / Metrik | Formula / Kondisi | Indikasi Keamanan |
|----------------|-------------------|-------------------|
| `Lebar_Pose` | `MAX(X) - MIN(X)` dari 16 Landmark | Proporsi persebaran tubuh secara horizontal |
| `Tinggi_Pose` | `MAX(Y) - MIN(Y)` dari 16 Landmark | Proporsi persebaran tubuh secara vertikal |
| `Rasio_Lebar_Tinggi` | `Lebar_Pose / Tinggi_Pose` | Indikator absolut postur jatuh di dalam data EDA |
| **Logic: NORMAL** | `rasio_box < 0.70` | Berdiri / Duduk tegak (Aman) |
| **Logic: WARNING** | `0.70 <= rasio_box < 1.05` | Kehilangan Keseimbangan (*Losing Balance*) |
| **Logic: FALL** | `rasio_box >= 1.05` | Terkapar / Jatuh Mendatar (*Fall Detected*) |

## 🔹 Penggunaan Aturan dalam Pemodelan (Production v3.0)

> [!IMPORTANT]
> **Aturan AI Production:**
> - Kolom `class_name` dan `class_id` **TIDAK BOLEH** masuk sebagai input model (mencegah *Data Leakage*).
> - Input model production adalah **35 kolom**: 16 koordinat (`X11`–`Y28`) + 19 fitur geometris (`feat_*`) setelah melalui `StandardScaler`.
> - Ketimpangan kelas (*Class Imbalance*) ditangani menggunakan **SMOTE** atau Oversampling di `src/imbalanced_handling.py`, tidak mengubah data validasi/uji.

## 🔹 Contoh Query (Pengolahan Pandas)

```python
import pandas as pd
import numpy as np

# 1. Memuat dataset matang hasil ekstraksi MediaPipe
df = pd.read_csv("data/processed/cleaned_human_fall.csv")

# 2. Memisahkan Fitur Geometri (X) dan Target Label (y)
kolom_fitur = [col for col in df.columns if col.startswith('X') or col.startswith('Y')]

X = df[kolom_fitur].values.astype(np.float32)
y = (df['class_name'] == 'fall').astype(np.float32).values # 1 untuk Fall, 0 untuk Normal

# 3. Menghitung Rasio Postur untuk Keperluan Analisis Dasbor
df['Lebar_Pose'] = df[[col for col in df.columns if col.startswith('X')]].max(axis=1) - df[[col for col in df.columns if col.startswith('X')]].min(axis=1)
df['Tinggi_Pose'] = df[[col for col in df.columns if col.startswith('Y')]].max(axis=1) - df[[col for col in df.columns if col.startswith('Y')]].min(axis=1)
df['Rasio_Lebar_Tinggi'] = df['Lebar_Pose'] / (df['Tinggi_Pose'] + 1e-6)

print(f"Total Fitur Ekstraksi: {X.shape[1]}") # Akan menghasilkan 16