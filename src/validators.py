"""
src/validators.py
=================
Sistem Validasi Kesiapan Data (Data Readiness Checks).
Memastikan bahwa data yang diekspor dari pipeline ETL bebas dari anomali
sebelum masuk ke tahap training model (Deep Learning) atau Dashboard Streamlit.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union
import pandas as pd
import numpy as np


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


class DataReadinessChecker:
    """
    Kelas penilai (*Validator*) untuk menguji kelayakan dataset akhir.

    Penggunaan:
    ----------
    checker = DataReadinessChecker(df)
    checker.run_all()
    checker.print_report()
    """

    def __init__(
        self, 
        df: pd.DataFrame, 
        min_samples_per_class: int = 50,
        group_col: Optional[str] = None  # 🔹 NEW: untuk leakage check
    ):
        self.df = df.copy()
        self.min_samples = min_samples_per_class
        self.group_col = group_col  # 🔹 NEW: kolom pengelompokan (video_id, session_id, dll)
        self.results: List[CheckResult] = []
        
        # 8 Titik Sendi yang sudah kita sepakati dengan Dosen Pembimbing
        self.landmark_pilihan = [11, 12, 23, 24, 25, 26, 27, 28]

    def run_all(self, check_leakage: bool = True) -> bool:  # 🔹 NEW: toggle leakage check
        """Mengeksekusi seluruh daftar checklist. Return True jika lulus semua."""
        self.results.clear()
        
        self._check_not_empty()
        self._check_no_nulls_in_required()
        self._check_class_balance()
        self._check_bbox_validity()
        self._check_no_duplicates()
        
        # Tambahan khusus: Cek kelengkapan MediaPipe (khusus dataset orang jatuh)
        if self.df['dataset'].eq('human_fall').any():
            self._check_mediapipe_features()
            
        # 🔹 NEW: Cek data leakage pada split (opsional)
        if check_leakage and self.group_col and self.group_col in self.df.columns:
            self._check_split_leakage()
        
        return all(r.passed for r in self.results)

    def print_report(self) -> None:
        """Mencetak laporan audit secara visual ke layar terminal/Jupyter."""
        print("\n=== LAPORAN AUDIT KELAYAKAN DATA (DATA READINESS) ===")
        
        for r in self.results:
            status = "✅ LULUS" if r.passed else "❌ GAGAL"
            print(f" [{status}] {r.name:<40} : {r.message}")
            
        overall = "SIAP DIGUNAKAN (READY)" if all(r.passed for r in self.results) else "BELUM SIAP (NOT READY)"
        print("-" * 80)
        print(f" KEPUTUSAN FINAL: {overall}")
        print("=" * 80)

    def to_dataframe(self) -> pd.DataFrame:
        """Mengonversi hasil audit ke DataFrame untuk diekspor ke CSV/JSON."""
        return pd.DataFrame([{
            "Check_Name": r.name,
            "Status_Passed": r.passed,
            "Message": r.message,
        } for r in self.results])


    # ==========================================
    # ─── DAFTAR PENGUJIAN (CHECKS) ────────────
    # ==========================================

    def _check_not_empty(self) -> None:
        passed = len(self.df) > 0
        msg = f"{len(self.df):,} baris data terdeteksi." if passed else "DataFrame kosong total!"
        self.results.append(CheckResult("1. DataFrame Tidak Kosong", passed, msg))

    def _check_no_nulls_in_required(self) -> None:
        """Memastikan metadata gambar dan label BBox tidak ada yang bolong (NaN)."""
        required = ["image_id", "class_id", "class_name", "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px"]
        
        cols = [c for c in required if c in self.df.columns]
        null_counts = self.df[cols].isnull().sum()
        has_nulls = null_counts[null_counts > 0]
        
        passed = len(has_nulls) == 0
        msg = "Semua kolom esensial terisi." if passed else f"Data kosong di kolom: {has_nulls.to_dict()}"
        self.results.append(CheckResult("2. Bebas Nilai Kosong (Null/NaN)", passed, msg))

    def _check_class_balance(self) -> None:
        """Memastikan AI tidak belajar dari kelas yang jumlahnya terlalu sedikit (Imbalanced)."""
        if "class_name" not in self.df.columns:
            self.results.append(CheckResult("3. Keseimbangan Kelas Target", False, "Kolom 'class_name' hilang!"))
            return
            
        target_classes = self.df[self.df["class_name"] != "background"]
        counts = target_classes["class_name"].value_counts()
        
        under = counts[counts < self.min_samples]
        passed = len(under) == 0
        
        msg = f"Semua target kelas ≥ {self.min_samples} sampel." if passed \
              else f"Kelas minoritas butuh lebih banyak data: {under.to_dict()}"
              
        self.results.append(CheckResult(f"3. Keseimbangan Kelas (Min: {self.min_samples})", passed, msg))

    def _check_bbox_validity(self) -> None:
        """Memastikan koordinat YOLO dalam batas persentase wajar (0.0 - 1.0)."""
        required_cols = ["bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height"]
        
        if all(col in self.df.columns for col in required_cols):
            invalid = ((self.df[required_cols] < 0) | (self.df[required_cols] > 1)).any(axis=1).sum()
        else:
            invalid = 0
            
        passed = int(invalid) == 0
        msg = "Semua Box berada di dalam frame." if passed else f"Ditemukan {invalid} Box yang menembus batas layar (Out of Bound)!"
        
        self.results.append(CheckResult("4. Validitas Bounding Box (0-1)", passed, msg))

    def _check_no_duplicates(self) -> None:
        """Mencegah Data Leakage: Cek apakah ada objek yang tercatat dua kali di posisi yang persis sama."""
        subset = ["image_id", "class_id", "bbox_x_center", "bbox_y_center"]
        cols_to_check = [c for c in subset if c in self.df.columns]
        
        dupes = self.df.duplicated(subset=cols_to_check).sum()
        passed = int(dupes) == 0
        msg = "Data unik 100%." if passed else f"Ditemukan {dupes} baris duplikat!"
        
        self.results.append(CheckResult("5. Bebas Duplikasi Data", passed, msg))

    def _check_mediapipe_features(self) -> None:
        """[FITUR BARU] Cek keberadaan 16 titik sendi MediaPipe pada dataset Human Fall."""
        human_data = self.df[self.df['dataset'] == 'human_fall']
        
        if len(human_data) == 0:
            return
            
        kolom_wajib_mp = []
        for idx in self.landmark_pilihan:
            kolom_wajib_mp.extend([f"X{idx}", f"Y{idx}"])
            
        kolom_hilang = [col for col in kolom_wajib_mp if col not in self.df.columns]
        
        if kolom_hilang:
            self.results.append(CheckResult("6. Fitur 16 Sendi MediaPipe", False, f"Kolom tidak ditemukan: {kolom_hilang[:3]}..."))
            return
            
        mp_nulls = human_data[kolom_wajib_mp].isnull().any(axis=1).sum()
        passed = int(mp_nulls) == 0
        
        msg = "16 titik sendi lengkap (X,Y)." if passed else f"{mp_nulls} foto gagal diekstrak kerangkanya."
        self.results.append(CheckResult("6. Fitur 16 Sendi MediaPipe", passed, msg))

    # 🔹🔹🔹 BARU: Check untuk Mencegah Data Leakage pada Split 🔹🔹🔹
    def _check_split_leakage(self, train_df: Optional[pd.DataFrame] = None, 
                            val_df: Optional[pd.DataFrame] = None) -> None:
        """
        Memastikan tidak ada group_id (video_id/session_id) yang overlap 
        antara training dan validation set — mencegah data leakage temporal/spasial.
        
        Parameters:
        -----------
        train_df : pd.DataFrame, optional
            DataFrame training. Jika None, akan diasumsikan self.df sudah terlabel split.
        val_df : pd.DataFrame, optional
            DataFrame validation. Jika None, akan dicari kolom 'split' di self.df.
        """
        if self.group_col not in self.df.columns:
            self.results.append(CheckResult(
                f"7. Anti-Leakage Split [{self.group_col}]", 
                False, 
                f"Kolom '{self.group_col}' tidak ditemukan untuk validasi split!"
            ))
            return
        
        # Case 1: train_df dan val_df disediakan secara eksplisit
        if train_df is not None and val_df is not None:
            train_groups = set(train_df[self.group_col].dropna().unique())
            val_groups = set(val_df[self.group_col].dropna().unique())
            
        # Case 2: self.df memiliki kolom 'split' dengan nilai 'train'/'val'/'test'
        elif 'split' in self.df.columns:
            train_groups = set(self.df[self.df['split'] == 'train'][self.group_col].dropna().unique())
            val_groups = set(self.df[self.df['split'] == 'val'][self.group_col].dropna().unique())
            
        else:
            self.results.append(CheckResult(
                f"7. Anti-Leakage Split [{self.group_col}]", 
                False, 
                "Berikan train_df/val_df atau pastikan ada kolom 'split' di DataFrame!"
            ))
            return
        
        # Hitung overlap
        overlap = train_groups.intersection(val_groups)
        passed = len(overlap) == 0
        
        if passed:
            msg = f"Tidak ada overlap {self.group_col} antara train ({len(train_groups)} group) dan val ({len(val_groups)} group)."
        else:
            msg = f"⚠️ DATA LEAKAGE: {len(overlap)} {self.group_col} muncul di train DAN val! Contoh: {list(overlap)[:3]}"
        
        self.results.append(CheckResult(f"7. Anti-Leakage Split [{self.group_col}]", passed, msg))


# =============================================================================
# 🔹 FUNGSI UTILITAS STANDALONE (bisa dipanggil tanpa class)
# =============================================================================

def verify_group_split(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    group_col: str = "video_id"
) -> tuple[bool, dict]:
    """
    Fungsi cepat untuk memverifikasi tidak ada leakage antara train dan val split.
    
    Returns:
    --------
    (is_safe: bool, report: dict)
    """
    train_groups = set(train_df[group_col].dropna().unique())
    val_groups = set(val_df[group_col].dropna().unique())
    overlap = train_groups.intersection(val_groups)
    
    report = {
        'train_groups': len(train_groups),
        'val_groups': len(val_groups),
        'overlap_count': len(overlap),
        'overlap_samples': list(overlap)[:10]  # max 10 contoh
    }
    
    is_safe = len(overlap) == 0
    return is_safe, report


def create_stratified_group_split(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    val_ratio: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membuat train/val split dengan Stratified Group Sampling:
    - Stratified: distribusi kelas target terjaga
    - Group: satu group (video_id) tidak terpecah antara train/val
    
    Reference: sklearn StratifiedGroupKFold pattern
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    # Pastikan target_col adalah numeric atau categorical yang bisa di-stratify
    if df[target_col].dtype == 'object':
        y_encoded = df[target_col].astype('category').cat.codes
    else:
        y_encoded = df[target_col]
    
    splitter = GroupShuffleSplit(
        n_splits=1, 
        test_size=val_ratio, 
        random_state=random_state
    )
    
    train_idx, val_idx = next(splitter.split(df, y_encoded, groups=df[group_col]))
    
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()