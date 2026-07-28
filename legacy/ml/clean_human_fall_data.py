import sys
from pathlib import Path
import pandas as pd

# Arahkan root path ke parent folder agar modul src dan config terbaca
sys.path.insert(0, str(Path.cwd().parent))

from config.paths import DATASETS, OUTPUT_DIR
from src.data_loader import YOLOLoader
from src.data_cleaner import DataCleaner

# Konfigurasi penamaan file output secara dinamis
CLEAN_CSV = OUTPUT_DIR / "cleaned_human_fall.csv"
CLEAN_PARQUET = OUTPUT_DIR / "cleaned_human_fall.parquet"

print("[INFO] Modul Data Cleaner berhasil dimuat. Siap membedah data!")

# --- CELL ---

# ==========================================
# CELL 2: LOAD RAW DATA (YOLO FORMAT)
# ==========================================
from src.data_loader import load_all_datasets

print("=== TAHAP 1: MEMUAT DATA MENTAH ===")

# Kita gunakan fungsi mesin global agar otomatis membaca folder train dan valid
# dari seluruh dataset yang terdaftar di config/paths.py
df_raw = load_all_datasets(DATASETS)

print(f"\n[INFO] Data Mentah (Raw) berhasil dimuat: {len(df_raw):,} baris.")
df_raw.head()

# --- CELL ---

print("=== TAHAP 2: PROSES PEMBERSIHAN (DATA CLEANING) ===")
print("Mengeksekusi pipeline: Hapus duplikat, filter Out-of-Bounds BBox, & Drop NaN...")

# Menjalankan pembersihan menggunakan class OOP milikmu
cleaner = DataCleaner(df_raw, format="yolo")
df_clean = cleaner.run()

print(f"[INFO] Data Bersih (Clean): {len(df_clean):,} baris.")
print(f"[INFO] Total data yang terbuang/cacat: {len(df_raw) - len(df_clean):,} baris.")

# --- CELL ---

print("=== TAHAP 3: LAPORAN AUDIT PEMBERSIHAN ===")

# Mengonversi dictionary report dari cleaner menjadi DataFrame agar tampil cantik di Jupyter
report_df = pd.DataFrame.from_dict(
    cleaner.get_report(), 
    orient="index", 
    columns=["Jumlah / Keterangan"]
)

# Menambahkan styling agar tabel terlihat rapi di Jupyter Notebook
display(report_df.style.set_caption("Tabel Ringkasan Pembersihan Data").set_table_styles([{
    'selector': 'caption',
    'props': [('font-size', '16px'), ('font-weight', 'bold')]
}]))

# --- CELL ---

print("=== TAHAP 4: MENYIMPAN DATA BERSIH ===")

# 1. Simpan dalam format CSV (Untuk kemudahan dibaca manusia / Excel)
df_clean.to_csv(CLEAN_CSV, index=False)
print(f"[SUKSES] Tersimpan format CSV     -> {CLEAN_CSV.name}")

# 2. Simpan dalam format Parquet (Sangat ringan, kompresi tinggi, loading secepat kilat untuk AI)
# Pastikan library 'pyarrow' sudah terinstal di requirements.txt!
df_clean.to_parquet(CLEAN_PARQUET, index=False)
print(f"[SUKSES] Tersimpan format Parquet -> {CLEAN_PARQUET.name}")

print("\nData ini sekarang sudah suci dan siap divisualisasikan pada tahap EDA (03_eda_visualization.ipynb)!")