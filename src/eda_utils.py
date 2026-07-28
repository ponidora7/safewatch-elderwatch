"""
src/eda_utils.py
================
Utility Class untuk Exploratory Data Analysis (EDA) dan Visualisasi.
Mendukung Bounding Box (YOLO) dan Fitur Sendi Manusia (MediaPipe).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict

import config.paths as cfg

class EDAPlotter:
    """
    Mesin Utama Visualisasi SafeWatch.
    Menghasilkan figure matplotlib beresolusi tinggi yang siap diekspor ke Streamlit.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._setup_plotting()
        
    def _setup_plotting(self):
        """Menerapkan styling konsisten dari config (Dark Mode)."""
        sns.set_style('darkgrid')
        sns.set_palette(cfg.PLOT_CONFIG['palette'])
        plt.rcParams['figure.dpi'] = cfg.PLOT_CONFIG['dpi']
        plt.rcParams['figure.figsize'] = cfg.PLOT_CONFIG['figsize']
        
        # Mencegah peringatan Future Warning di Matplotlib/Seaborn terbaru
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

    def summary_stats(self):
        """Mencetak ringkasan statistik yang krusial di terminal/Jupyter."""
        print("📊 [STATISTIK DESKRIPTIF DATASET]")
        print(f" -> Total Baris   : {len(self.df):,}")
        print(f" -> Gambar Unik   : {self.df['image_id'].nunique() if 'image_id' in self.df.columns else 'Unknown':,}")
        
        # Cek ketersediaan kolom BBox sebelum menghitung
        if 'bbox_w_px' in self.df.columns and 'bbox_h_px' in self.df.columns:
            print(f" -> Rata-rata Lebar BBox  : {self.df['bbox_w_px'].mean():.1f} px")
            print(f" -> Rata-rata Tinggi BBox : {self.df['bbox_h_px'].mean():.1f} px")
            
        print("\nDistribusi Kelas:")
        print(self.df['class_name'].value_counts().to_string())

    def plot_class_distribution(self, groupby: str = 'dataset', top_n: Optional[int] = None) -> plt.Figure:
        """Plot jumlah objek berdasarkan kelas."""
        df_clean = self.df[self.df['class_name'] != 'background'].copy()
        n_unique = df_clean[groupby].nunique()
        
        fig, axes = plt.subplots(1, n_unique, figsize=(6*n_unique, 5), sharey=True)
        if n_unique == 1: axes = [axes]
        
        for ax, group in zip(axes, df_clean[groupby].unique()):
            subset = df_clean[df_clean[groupby] == group]
            order = subset['class_name'].value_counts().iloc[:top_n].index if top_n else None
            
            sns.countplot(data=subset, y='class_name', order=order, ax=ax, palette=cfg.PLOT_CONFIG['palette'])
            ax.set_title(f'Distribusi di Dataset: {group}', fontweight='bold')
            ax.set_xlabel('Jumlah Data (Objek)')
            ax.set_ylabel('Kategori' if ax == axes[0] else '')
            
            # Tambahkan angka di atas batang
            for p in ax.patches:
                ax.annotate(f'{int(p.get_width()):,}', 
                            (p.get_width(), p.get_y() + p.get_height() / 2.), 
                            ha='left', va='center', xytext=(5, 0), textcoords='offset points', color='white')

        plt.suptitle('Sebaran Data per Kelas Target', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        return fig

    def plot_bbox_size_distribution(self) -> plt.Figure:
        """Plot sebaran ukuran Bounding Box (Kecil vs Besar) menggunakan Kernel Density."""
        if 'bbox_w_px' not in self.df.columns:
            # Hitung otomatis jika belum ada
            self.df['bbox_w_px'] = self.df['bbox_width'] * self.df['img_width']
            self.df['bbox_h_px'] = self.df['bbox_height'] * self.df['img_height']
            
        self.df['bbox_area_px'] = self.df['bbox_w_px'] * self.df['bbox_h_px']
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histplot BBox Area
        sns.histplot(data=self.df, x='bbox_area_px', bins=50, kde=True, ax=axes[0], color='c')
        axes[0].set_title('Distribusi Luas Bounding Box (Piksel)', fontweight='bold')
        axes[0].set_xlabel('Luas Area (W x H)')
        
        # Boxplot Tinggi vs Dataset
        sns.boxplot(data=self.df, x='dataset', y='bbox_h_px', ax=axes[1], palette='Set2')
        axes[1].set_title('Sebaran Tinggi BBox per Dataset', fontweight='bold')
        axes[1].set_ylabel('Tinggi Objek (Piksel)')
        
        plt.tight_layout()
        return fig

    def plot_bbox_heatmap(self) -> plt.Figure:
        """Membuat Heatmap 2D untuk melihat posisi objek di dalam layar (Apakah bias ke tengah?)."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Sampel data agar heatmap tidak menjadi blok warna solid jika data > 10k
        df_plot = self.df.sample(min(5000, len(self.df)), random_state=42)
        
        sns.kdeplot(
            data=df_plot, x='bbox_x_center', y='bbox_y_center',
            cmap='magma', fill=True, thresh=0, levels=100, ax=ax
        )
        
        ax.set_title('Heatmap Lokasi Objek (Sumbu Kamera)', fontweight='bold', pad=15)
        ax.set_xlabel('Posisi Horizontal X (Normalized 0-1)')
        ax.set_ylabel('Posisi Vertikal Y (Normalized 0-1)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis() # Sesuaikan koordinat Y Kamera (0 di atas)
        
        plt.tight_layout()
        return fig

    def plot_annotations_per_image(self) -> plt.Figure:
        """Histogram untuk melihat kepadatan objek per gambar (Crowd Analysis)."""
        # Menghitung jumlah anotasi (baris) per image_id
        counts = self.df['image_id'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(counts, bins=range(1, max(counts.max()+2, 10)), discrete=True, color='coral', ax=ax)
        
        ax.set_title('Kepadatan Objek per Gambar', fontweight='bold')
        ax.set_xlabel('Jumlah Bounding Box dalam 1 Foto')
        ax.set_ylabel('Jumlah Foto (Frekuensi)')
        ax.set_xticks(range(1, max(counts.max()+1, 10)))
        
        plt.tight_layout()
        return fig