"""
src/data_cleaner.py
===================
Pipeline untuk pembersihan data (Cleaning) dan ekstraksi fitur (MediaPipe).
Memvalidasi kotak Bounding Box YOLO dan menerjemahkannya menjadi titik kordinat sendi.
"""

import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataCleaner:
    """
    Pipeline pembersihan hibrida untuk dataset SafeWatch.
    Menghapus anomali YOLO, lalu mengekstrak 16 titik tubuh menggunakan MediaPipe.
    """
    
    # 8 Landmark Utama (Bahu, Pinggul, Lutut, Kaki) sesuai revisi
    LANDMARK_PILIHAN = [11, 12, 23, 24, 25, 26, 27, 28]
    
    def __init__(self, df: pd.DataFrame, format: str = "yolo"):
        self.df = df.copy()
        self.cleaning_log = []
        self.format = format
        
        # Inisialisasi MediaPipe (Mode Foto Statis)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def run(self) -> pd.DataFrame:
        """
        Fungsi eksekusi otomatis. Memanggil semua langkah pembersihan secara berurutan.
        (Digunakan oleh notebook 02_data_cleaning.ipynb)
        """
        self.remove_missing_images()
        self.remove_duplicates()
        self.filter_invalid_bbox()
        self.normalize_to_pixels()
        
        # [EKSTRAKSI FITUR]
        # Hanya jalankan MediaPipe pada dataset 'human_fall'
        if self.df['dataset'].eq('human_fall').any():
            self.extract_mediapipe_features()
            
        return self.df

    def remove_missing_images(self) -> 'DataCleaner':
        """Hapus baris yang merujuk pada file gambar yang korup atau hilang."""
        before = len(self.df)
        self.df = self.df[self.df['img_exists'] == True].copy()
        removed = before - len(self.df)
        
        self.cleaning_log.append({"step": "remove_missing_images", "removed": removed})
        print(f"🗑️ Dihapus {removed} BBox karena file gambar tidak ditemukan.")
        return self

    def remove_duplicates(self) -> 'DataCleaner':
        """Hapus data Bounding Box yang terduplikasi secara identik."""
        before = len(self.df)
        # Identifikasi kolom yang relevan untuk pengecekan duplikat
        subset_cols = ['image_path', 'class_id', 'bbox_x_center', 'bbox_y_center']
        self.df = self.df.drop_duplicates(subset=subset_cols, keep='first').copy()
        removed = before - len(self.df)
        
        self.cleaning_log.append({"step": "remove_duplicates", "removed": removed})
        print(f"🗑️ Dihapus {removed} BBox yang duplikat secara identik.")
        return self

    def filter_invalid_bbox(self) -> 'DataCleaner':
        """Hapus nilai BBox yang cacat (misal: koordinat minus atau lebih besar dari 1)."""
        before = len(self.df)
        
        valid_x = self.df['bbox_x_center'].between(0, 1)
        valid_y = self.df['bbox_y_center'].between(0, 1)
        valid_w = self.df['bbox_width'].between(0, 1)
        valid_h = self.df['bbox_height'].between(0, 1)
        
        self.df = self.df[valid_x & valid_y & valid_w & valid_h].copy()
        removed = before - len(self.df)
        
        self.cleaning_log.append({"step": "filter_invalid_bbox", "removed": removed})
        print(f"🗑️ Dihapus {removed} BBox dengan koordinat di luar batas (0-1).")
        return self

    def normalize_to_pixels(self) -> 'DataCleaner':
        """Konversi persentase YOLO [0-1] menjadi satuan Piksel aktual."""
        self.df['bbox_x_px'] = self.df['bbox_x_center'] * self.df['img_width']
        self.df['bbox_y_px'] = self.df['bbox_y_center'] * self.df['img_height']
        self.df['bbox_w_px'] = self.df['bbox_width'] * self.df['img_width']
        self.df['bbox_h_px'] = self.df['bbox_height'] * self.df['img_height']
        
        print("📐 Koordinat BBox berhasil dikonversi ke Piksel.")
        return self

    # ==========================================
    # ─── FITUR BARU: MEDIA PIPE EXTRACTION ────
    # ==========================================
    
    def extract_mediapipe_features(self) -> 'DataCleaner':
        """
        [INTEGRASI] Melakukan Crop pada BBox, lalu mengekstrak 16 titik sendi menggunakan MediaPipe.
        """
        print("🚀 Memulai ekstraksi MediaPipe (16 Fitur X,Y) dari Bounding Box...")
        
        # Buat kolom kosong untuk menampung fitur baru
        for idx in self.LANDMARK_PILIHAN:
            self.df[f'X{idx}'] = np.nan
            self.df[f'Y{idx}'] = np.nan
            
        self.df['pose_extracted'] = False

        sukses = 0
        gagal = 0
        
        # Loop hanya pada data dataset 'human_fall'
        human_data = self.df[self.df['dataset'] == 'human_fall']
        
        for index, row in tqdm(human_data.iterrows(), total=len(human_data), desc="Pose Extraction"):
            img_path = str(row['image_path'])
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Logika CROP berdasarkan piksel
            x_center = int(row['bbox_x_px'])
            y_center = int(row['bbox_y_px'])
            box_w = int(row['bbox_w_px'])
            box_h = int(row['bbox_h_px'])
            
            x_min = max(0, x_center - (box_w // 2))
            y_min = max(0, y_center - (box_h // 2))
            x_max = min(img.shape[1], x_center + (box_w // 2))
            y_max = min(img.shape[0], y_center + (box_h // 2))
            
            crop_img = img[y_min:y_max, x_min:x_max]
            
            if crop_img.size == 0:
                continue

            # Proses AI MediaPipe
            image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            hasil = self.pose.process(image_rgb)
            
            if hasil.pose_landmarks:
                for idx in self.LANDMARK_PILIHAN:
                    landmark = hasil.pose_landmarks.landmark[idx]
                    self.df.at[index, f'X{idx}'] = landmark.x
                    self.df.at[index, f'Y{idx}'] = landmark.y
                
                self.df.at[index, 'pose_extracted'] = True
                sukses += 1
            else:
                gagal += 1
                
        # Hapus baris yang gagal di-ekstrak oleh MediaPipe (Drop NaN)
        before = len(self.df)
        self.df = self.df[self.df['pose_extracted'] == True].copy()
        
        # Buang kolom sementara
        self.df = self.df.drop(columns=['pose_extracted'])
        
        self.cleaning_log.append({
            "step": "mediapipe_extraction",
            "success_pose": sukses,
            "failed_pose_dropped": before - len(self.df)
        })
        
        print(f"✅ Ekstraksi selesai! Berhasil: {sukses} pose. Dibuang: {gagal} objek.")
        return self

    def get_report(self) -> dict:
        """Mengembalikan log ringkasan dari semua aksi pembersihan yang dilakukan."""
        return {
            "Total Awal (Raw)": self.cleaning_log[0].get('removed', 0) + len(self.df) if self.cleaning_log else len(self.df),
            "Total Akhir (Clean)": len(self.df),
            "Rincian Log": self.cleaning_log
        }


# ==========================================
# ─── FITUR LAMA: DATA SPLITTER (PREPROCESS)
# ==========================================
def split_data_for_training(df: pd.DataFrame, target_col: str = 'class_id', test_size: float = 0.2):
    """
    Menggantikan fungsi preprocess_data.py lama.
    Memisahkan data menjadi Train dan Test dengan Stratified Splitting.
    """
    # Pastikan data bebas NaN
    df_clean = df.dropna().copy()
    
    # Ambil 16 kolom fitur X dan Y
    kolom_fitur = [col for col in df_clean.columns if col.startswith('X') or col.startswith('Y')]
    
    if not kolom_fitur:
        raise ValueError("Data fitur MediaPipe (X, Y) tidak ditemukan! Ekstrak fitur terlebih dahulu.")
        
    X = df_clean[kolom_fitur].values.astype(np.float32)
    y = df_clean[target_col].values.astype(np.float32)
    
    # Stratified Split: Memastikan rasio Normal dan Jatuh seimbang di kedua set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n[INFO] Data Split (Rasio {(1-test_size)*100:.0f}:{test_size*100:.0f})")
    print(f" -> X_train: {X_train.shape}")
    print(f" -> X_test : {X_test.shape}")
    
    return X_train, X_test, y_train, y_test