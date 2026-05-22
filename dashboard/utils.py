"""
dashboard/utils.py
==================
Helper functions untuk Dasbor Streamlit SafeWatch.
Menangani pemuatan data secara efisien (Caching), rendering komponen UI, dan pemetaan warna cerdas.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import sys

# Memastikan root proyek terbaca agar config.paths bisa diimpor tanpa error
sys.path.insert(0, str(Path.cwd().parent))

try:
    from config.paths import OUTPUT_DIR
    CLEAN_CSV = OUTPUT_DIR / "cleaned_human_fall.csv"
except ImportError:
    # Fallback jika dijalankan di environment atau OS yang berbeda
    CLEAN_CSV = Path("../data/processed/cleaned_human_fall.csv")


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None) -> Optional[pd.DataFrame]:
    """
    Memuat data dengan fitur Caching (Menyimpan di memori RAM).
    Mencegah dasbor memuat ulang (reload) file CSV berulang kali saat tombol ditekan.
    """
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file unggahan: {e}")
            return None

    if CLEAN_CSV.exists():
        try:
            return pd.read_csv(CLEAN_CSV)
        except Exception as e:
            st.error(f"❌ Gagal membaca jalur sistem {CLEAN_CSV}: {e}")
            return None

    return None


def render_metric_cards(df: pd.DataFrame) -> None:
    """Tampilkan kartu metrik yang dinamis dan relevan dengan arsitektur YOLO + MediaPipe."""
    cols = st.columns(4)
    
    # Kalkulasi Metrik Kritis
    total_bbox = len(df)
    total_images = df['image_id'].nunique() if 'image_id' in df.columns else 0
    
    # Cek berapa banyak BBox manusia yang berhasil dipasangkan 16 titik sendi (MediaPipe)
    pose_count = df['X11'].notna().sum() if 'X11' in df.columns else 0
    pose_percentage = (pose_count / total_bbox) * 100 if total_bbox > 0 else 0
    
    metrics = [
        ("📋 Total Deteksi (BBox)", f"{total_bbox:,}", None),
        ("🖼️ Gambar Unik", f"{total_images:,}", None),
        ("🤖 Titik Pose Terekstrak", f"{pose_count:,}", f"{pose_percentage:.1f}% sukses"),
        ("📊 Kepadatan (Objek/Foto)", f"{total_bbox / max(total_images, 1):.1f}", None),
    ]
    
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta if delta else None)


def color_class_map(class_names: list[str]) -> dict[str, str]:
    """
    Pemetaan warna berbasis Semantik (Psikologis).
    Otomatis memberikan warna Hijau untuk keadaan Normal dan Merah untuk Bahaya/Jatuh.
    """
    color_map = {}
    
    # Palet warna netral untuk kelas sekunder
    palette = ["#3498DB", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]
    palette_idx = 0
    
    for name in class_names:
        name_lower = str(name).lower()
        
        # Deteksi kata kunci psikologis
        if any(keyword in name_lower for keyword in ["normal", "berdiri", "duduk", "aman"]):
            color_map[name] = "#2ecc71"  # Hijau (Aman)
        elif any(keyword in name_lower for keyword in ["jatuh", "fall", "api", "fire", "smoke", "bahaya"]):
            color_map[name] = "#e74c3c"  # Merah (Bahaya Kritis)
        else:
            # Gunakan warna cadangan dari palet jika kelas tidak dikenali
            color_map[name] = palette[palette_idx % len(palette)]
            palette_idx += 1
            
    return color_map