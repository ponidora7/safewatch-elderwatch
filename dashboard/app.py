import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import os
from pathlib import Path

# =====================================================================
# CONFIG & PATH SETUP
# =====================================================================
st.set_page_config(page_title="SafeWatch Analytics", page_icon="🚨", layout="wide")

# Resolusi Path agar stabil saat dipanggil dari terminal mana pun
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

CSV_PATH = DATA_DIR / "cleaned_human_fall.csv" # Menggunakan data CSV bersih
MANIFEST_PATH = DATA_DIR / "dashboard_manifest.json"
HISTORY_PATH = MODEL_DIR / "training_history.pkl"


# =====================================================================
# CACHE FUNCTIONS (Optimasi Memori RAM)
# =====================================================================
@st.cache_data
def muat_manifest():
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def muat_data_eda():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        # Hitung Rasio Pose secara real-time untuk dasbor jika MediaPipe berhasil diekstrak
        if 'X11' in df.columns:
            kolom_x = [col for col in df.columns if col.startswith('X')]
            kolom_y = [col for col in df.columns if col.startswith('Y')]
            
            df['Lebar_Pose'] = df[kolom_x].max(axis=1) - df[kolom_x].min(axis=1)
            df['Tinggi_Pose'] = df[kolom_y].max(axis=1) - df[kolom_y].min(axis=1)
            df['Rasio_Lebar_Tinggi'] = df['Lebar_Pose'] / (df['Tinggi_Pose'] + 1e-6)
        return df
    return None

def muat_data_training():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, 'rb') as f:
            return pickle.load(f)
    return None

# =====================================================================
# SIDEBAR NAVIGATION
# =====================================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=60)
st.sidebar.title("🛡️ SafeWatch")
st.sidebar.markdown("*Intelligent CV Pipeline*")

menu = st.sidebar.radio("Navigasi Dasbor:", [
    "📊 System Overview", 
    "📐 Analisis Postur (EDA)", 
    "🧠 AI Performance"
])

# Memuat semua data
manifest = muat_manifest()
df_eda = muat_data_eda()
riwayat = muat_data_training()


# =====================================================================
# HALAMAN 1: SYSTEM OVERVIEW (Status Pipeline)
# =====================================================================
if menu == "📊 System Overview":
    st.title("📊 Ringkasan Sistem SafeWatch")
    st.markdown("Dasbor pemantauan jalur pipa data (ETL) sebelum masuk ke ruang pelatihan.")
    
    if manifest:
        col1, col2, col3 = st.columns(3)
        col1.metric("Kesiapan Data (Readiness)", manifest.get("status_readiness", "UNKNOWN"))
        col2.metric("Total Sampel Tersedia", f"{manifest['metrics'].get('total_samples', 0):,}")
        col3.metric("Waktu Generate Terakhir", manifest.get("generated_at", "N/A"))
        
        st.divider()
        st.info(f"📁 **Sumber Data:** `{manifest['data_paths'].get('clean_csv_path')}`")
    else:
        st.warning("⚠️ File Manifest tidak ditemukan. Selesaikan tahap Notebook 04 terlebih dahulu.")


# =====================================================================
# HALAMAN 2: EDA (Exploratory Data Analysis)
# =====================================================================
elif menu == "📐 Analisis Postur (EDA)":
    st.title("📐 Analisis Geometri Postur (MediaPipe)")
    st.markdown("""
    Berdasarkan ekstraksi **16 Titik Sendi MediaPipe**, sistem SafeWatch mendeteksi anomali postur menggunakan matriks rasio geometri. 
    Kami membuktikan bahwa **Rasio Lebar vs Tinggi** adalah indikator mutlak untuk membedakan orang berdiri dan orang terjatuh.
    """)
    
    if df_eda is not None and 'Rasio_Lebar_Tinggi' in df_eda.columns:
        # Peta Kelas
        df_eda['Status'] = df_eda['class_id'].map({0: 'Normal (Berdiri/Duduk)', 1: 'Bahaya (Jatuh)'})
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("1. Distribusi Kelas")
            distribusi = df_eda['Status'].value_counts().reset_index()
            distribusi.columns = ['Status', 'Jumlah Sampel']
            fig_pie = px.pie(distribusi, values='Jumlah Sampel', names='Status', 
                             color='Status', color_discrete_map={'Normal (Berdiri/Duduk)':'#2ecc71', 'Bahaya (Jatuh)':'#e74c3c'},
                             hole=0.5)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.subheader("2. Analisis Rasio Horizontal")
            st.markdown("Semakin tinggi rasionya (melampaui 1.0), semakin horizontal posisi tubuh pasien (terkapar).")
            fig_box = px.box(df_eda, x='Status', y='Rasio_Lebar_Tinggi', 
                             color='Status', color_discrete_map={'Normal (Berdiri/Duduk)':'#2ecc71', 'Bahaya (Jatuh)':'#e74c3c'},
                             points="all")
            # Garis ambang batas (Threshold)
            fig_box.add_hline(y=1.0, line_dash="dash", line_color="orange", annotation_text="Batas Horizontal (Rasio 1:1)")
            st.plotly_chart(fig_box, use_container_width=True)
            
        st.divider()
        st.subheader("3. Peta Sebaran (Scatter Plot) Ukuran Objek")
        fig_scatter = px.scatter(df_eda, x="bbox_w_px", y="bbox_h_px", color="Status",
                                 color_discrete_map={'Normal (Berdiri/Duduk)':'#2ecc71', 'Bahaya (Jatuh)':'#e74c3c'},
                                 title="Lebar vs Tinggi Bounding Box (Piksel)",
                                 labels={'bbox_w_px': 'Lebar Tubuh', 'bbox_h_px': 'Tinggi Tubuh'})
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    else:
        st.warning("⚠️ Fitur MediaPipe (X, Y) tidak ditemukan dalam data. Pastikan DataCleaner telah dijalankan!")

# =====================================================================
# HALAMAN 3: Performa Model AI
# =====================================================================
elif menu == "🧠 AI Performance":
    st.title("🤖 Evaluasi Jaringan Saraf Tiruan (Deep Learning)")
    
    if riwayat is not None:
        # Metrik Puncak
        ak_akhir = riwayat.get('akurasi_akhir', 0) * 100
        val_akhir = riwayat['val_akurasi'][-1] * 100 if 'val_akurasi' in riwayat else 0
        
        col1, col2 = st.columns(2)
        col1.metric(label="Akurasi Final (Data Latih)", value=f"{ak_akhir:.2f}%")
        col2.metric(label="Akurasi Evaluasi (Data Uji)", value=f"{val_akhir:.2f}%")
        
        st.divider()
        col_chart1, col_chart2 = st.columns(2)
        
        # Grafik Accuracy
        with col_chart1:
            st.subheader("📈 Kurva Pembelajaran (Akurasi)")
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=riwayat['akurasi'], mode='lines', name='Train Acc', line=dict(color='#3498db')))
            fig_acc.add_trace(go.Scatter(y=riwayat['val_akurasi'], mode='lines', name='Val Acc', line=dict(color='#f1c40f')))
            fig_acc.update_layout(xaxis_title='Epoch', yaxis_title='Akurasi (0-1)')
            st.plotly_chart(fig_acc, use_container_width=True)

        # Grafik Loss
        with col_chart2:
            st.subheader("📉 Penurunan Kesalahan (Loss)")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=riwayat['loss'], mode='lines', name='Train Loss', line=dict(color='#e74c3c')))
            fig_loss.add_trace(go.Scatter(y=riwayat['val_loss'], mode='lines', name='Val Loss', line=dict(color='#9b59b6')))
            fig_loss.update_layout(xaxis_title='Epoch', yaxis_title='Loss Value')
            st.plotly_chart(fig_loss, use_container_width=True)
            
        st.divider()
        st.subheader("🎯 Matriks Kebingungan (Confusion Matrix)")
        st.markdown("Mengukur seberapa sering AI memberikan Alarm Palsu (False Positive).")
        cm = riwayat.get('confusion_matrix')
        if cm is not None:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Magma',
                               labels=dict(x="Tebakan AI (Prediksi)", y="Kejadian Asli (Aktual)"),
                               x=['Normal', 'Jatuh'], y=['Normal', 'Jatuh'])
            st.plotly_chart(fig_cm, use_container_width=True)
            
    else:
        st.warning("⚠️ File 'training_history.pkl' tidak ditemukan di folder `models/`. Silakan jalankan `train_model.py` terlebih dahulu.")