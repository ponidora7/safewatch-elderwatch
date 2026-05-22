import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np

def run_inference():
    MODEL_PATH = "models/safewatch_fall_model.keras"
    CSV_PATH = "data/processed/cleaned_human_fall.csv"
    
    print("⏳ 1. Memuat model AI...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model tidak ditemukan di {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("⏳ 2. Mengambil sampel data asli dari CSV...")
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Data CSV tidak ditemukan di {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    kolom_fitur = [col for col in df.columns if col.startswith('X') or col.startswith('Y')]
    
    # Comot 1 baris data orang Normal (berdiri/duduk)
    data_asli_normal = df[df['class_name'] != 'fall'].iloc[0]
    # Comot 1 baris data orang Jatuh
    data_asli_jatuh = df[df['class_name'] == 'fall'].iloc[0]
    
    # Ubah formatnya menjadi matrix (1 baris, 16 kolom) yang dimengerti TensorFlow
    input_normal = data_asli_normal[kolom_fitur].values.astype(np.float32).reshape(1, -1)
    input_jatuh = data_asli_jatuh[kolom_fitur].values.astype(np.float32).reshape(1, -1)
    
    print("\n🤖 3. Memulai Proses Inference (Tebak Postur)...")
    
    # Tebak Data 1
    pred1 = model.predict(input_normal, verbose=0)[0][0]
    hasil1 = "JATUH (BAHAYA!)" if pred1 > 0.5 else "NORMAL (AMAN)"
    print(f" -> Uji Gambar Normal : Terdeteksi sebagai [{hasil1}] (Probabilitas Jatuh: {pred1:.4f})")
    
    # Tebak Data 2
    pred2 = model.predict(input_jatuh, verbose=0)[0][0]
    hasil2 = "JATUH (BAHAYA!)" if pred2 > 0.5 else "NORMAL (AMAN)"
    print(f" -> Uji Gambar Jatuh  : Terdeteksi sebagai [{hasil2}] (Probabilitas Jatuh: {pred2:.4f})")

if __name__ == "__main__":
    run_inference()