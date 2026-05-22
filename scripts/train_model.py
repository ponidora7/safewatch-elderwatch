import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

JALUR_CSV = 'data/processed/cleaned_human_fall.csv'
JALUR_MODEL = 'models/safewatch_fall_model.keras'
JALUR_HISTORY = 'models/training_history.pkl'

def latih_model():
    print("\n[INFO] 1. Memuat data bersih hasil pipeline ETL...")
    df = pd.read_csv(JALUR_CSV)
    
    kolom_fitur = [col for col in df.columns if col.startswith('X') or col.startswith('Y')]
    
    # ---------------------------------------------------------
    # FITUR BARU: OVERSAMPLING (MENYEIMBANGKAN DATA)
    # ---------------------------------------------------------
    print("[INFO] 2. Menyeimbangkan Data (Oversampling)...")
    df_normal = df[df['class_name'] != 'fall']
    df_jatuh = df[df['class_name'] == 'fall']
    
    print(f" -> Jumlah Asli Normal : {len(df_normal)} baris")
    print(f" -> Jumlah Asli Jatuh  : {len(df_jatuh)} baris")
    
    # Gandakan data 'jatuh' agar jumlahnya sama dengan data 'normal'
    df_jatuh_digandakan = df_jatuh.sample(len(df_normal), replace=True, random_state=42)
    
    # Gabungkan kembali dan acak urutannya
    df_seimbang = pd.concat([df_normal, df_jatuh_digandakan], axis=0).sample(frac=1, random_state=42)
    
    print(f" -> Setelah Oversampling : {len(df_seimbang[df_seimbang['class_name'] != 'fall'])} Normal vs {len(df_seimbang[df_seimbang['class_name'] == 'fall'])} Jatuh")

    # Ekstrak Fitur (X) dan Label (y)
    X = df_seimbang[kolom_fitur].values.astype(np.float32)
    y = (df_seimbang['class_name'] == 'fall').astype(np.float32).values
    
    print("\n[INFO] 3. Membagi data (Train 80% / Test 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    jumlah_fitur = X_train.shape[1]

    # Kita turunkan targetnya sedikit karena data sudah riil seimbang
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=5, 
        restore_best_weights=True
    )

    print("\n[INFO] 4. Membangun arsitektur Functional API...")
    inputs = Input(shape=(jumlah_fitur,), name="Input_Layer")
    x = Dense(128, activation='relu', name="Dense_1")(inputs)
    x = Dropout(0.2, name="Dropout_1")(x)
    x = Dense(64, activation='relu', name="Dense_2")(x)
    x = Dropout(0.2, name="Dropout_2")(x)
    x = Dense(32, activation='relu', name="Dense_3")(x)
    outputs = Dense(1, activation='sigmoid', name="Output_Layer")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="SafeWatch_AI")
    
    # Learning rate standar agar AI belajar dengan tenang
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\n[INFO] 5. Memulai proses training pada data seimbang...")
    history = model.fit(
        X_train, y_train,
        epochs=30, # 30 Epoch sudah cukup untuk data seimbang
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n[INFO] 6. Mengevaluasi model dengan data uji...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f" -> Akurasi Akhir Model: {accuracy * 100:.2f}%")
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred)
    
    os.makedirs('models', exist_ok=True)
    model.save(JALUR_MODEL)
    
    riwayat_latih = {
        'akurasi': history.history['accuracy'],
        'val_akurasi': history.history.get('val_accuracy', []),
        'loss': history.history['loss'],
        'val_loss': history.history.get('val_loss', []),
        'confusion_matrix': cm,
        'akurasi_akhir': accuracy
    }
    
    with open(JALUR_HISTORY, 'wb') as f:
        pickle.dump(riwayat_latih, f)
        
    print(f"\n[SUKSES] Model yang telah diobati dari The Accuracy Paradox berhasil disimpan!")

if __name__ == "__main__":
    latih_model()