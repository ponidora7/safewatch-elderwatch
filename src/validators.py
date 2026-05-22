"""
src/validators.py
=================
Sistem Validasi Kesiapan Data (Data Readiness Checks).
Memastikan bahwa data yang diekspor dari pipeline ETL bebas dari anomali
sebelum masuk ke tahap training model (Deep Learning) atau Dashboard Streamlit.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd


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

    def __init__(self, df: pd.DataFrame, min_samples_per_class: int = 50):
        self.df = df.copy()
        self.min_samples = min_samples_per_class
        self.results: List[CheckResult] = []
        
        # 8 Titik Sendi yang sudah kita sepakati dengan Dosen Pembimbing
        self.landmark_pilihan = [11, 12, 23, 24, 25, 26, 27, 28]

    def run_all(self) -> bool:
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
            
        return all(r.passed for r in self.results)

    def print_report(self) -> None:
        """Mencetak laporan audit secara visual ke layar terminal/Jupyter."""
        print("\n=== LAPORAN AUDIT KELAYAKAN DATA (DATA READINESS) ===")
        
        for r in self.results:
            status = "✅ LULUS" if r.passed else "❌ GAGAL"
            print(f" [{status}] {r.name:<35} : {r.message}")
            
        overall = "SIAP DIGUNAKAN (READY)" if all(r.passed for r in self.results) else "BELUM SIAP (NOT READY)"
        print("-" * 75)
        print(f" KEPUTUSAN FINAL: {overall}")
        print("=" * 75)

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
        # Sesuai dengan penamaan kolom dari DataCleaner yang baru
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
            
        # Abaikan kelas background jika ada
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
            # Mencari baris yang nilainya minus atau melebihi 1
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
            return # Skip jika dataset ini kosong (seharusnya sudah tertangkap di check_not_empty)
            
        # Kumpulkan nama 16 kolom yang wajib ada (X11, Y11, X12, Y12, dst)
        kolom_wajib_mp = []
        for idx in self.landmark_pilihan:
            kolom_wajib_mp.extend([f"X{idx}", f"Y{idx}"])
            
        kolom_hilang = [col for col in kolom_wajib_mp if col not in self.df.columns]
        
        if kolom_hilang:
            self.results.append(CheckResult("6. Fitur 16 Sendi MediaPipe", False, f"Kolom tidak ditemukan: {kolom_hilang[:3]}..."))
            return
            
        # Pastikan tidak ada MediaPipe yang bernilai NaN pada data human_fall
        mp_nulls = human_data[kolom_wajib_mp].isnull().any(axis=1).sum()
        passed = int(mp_nulls) == 0
        
        msg = "16 titik sendi lengkap (X,Y)." if passed else f"{mp_nulls} foto gagal diekstrak kerangkanya."
        self.results.append(CheckResult("6. Fitur 16 Sendi MediaPipe", passed, msg))