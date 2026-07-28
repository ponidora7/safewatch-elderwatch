"""
scripts/train_enhanced_model.py
===============================
Enhanced training pipeline with:
- Advanced geometric features
- Improved model architecture with BatchNorm and Dropout
- Cost-sensitive training
- Comprehensive evaluation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.advanced_feature_engineering import AdvancedFeatureEngineer


JALUR_CSV = 'data/processed/cleaned_human_fall.csv'
JALUR_MODEL_ENHANCED = 'models/safewatch_fall_model_enhanced.keras'
JALUR_HISTORY = 'models/training_history_enhanced.pkl'
JALUR_SCALER = 'models/feature_scaler.pkl'


def latih_model_enhanced():
    """Enhanced training with advanced features and architecture."""
    
    print("\n" + "="*70)
    print("🚀 ENHANCED FALL DETECTION MODEL TRAINING")
    print("="*70)
    
    print("\n[INFO] 1. Memuat data bersih hasil pipeline ETL...")
    df = pd.read_csv(JALUR_CSV)
    print(f"   Loaded {len(df)} samples")
    
    print("\n[INFO] 2. Mengekstrak fitur geometri lanjutan...")
    df_enhanced = AdvancedFeatureEngineer.enhance_dataframe(df)
    
    # Kolom fitur: original landmarks + engineered features
    kolom_fitur_original = [col for col in df.columns if col.startswith('X') or col.startswith('Y')]
    kolom_fitur_engineered = [col for col in df_enhanced.columns if col.startswith('feat_')]
    kolom_fitur_semua = kolom_fitur_original + kolom_fitur_engineered
    
    print(f"   Original features: {len(kolom_fitur_original)}")
    print(f"   Engineered features: {len(kolom_fitur_engineered)}")
    print(f"   Total features: {len(kolom_fitur_semua)}")
    
    print("\n[INFO] 3. Menyeimbangkan Data (Oversampling)...")
    df_normal = df_enhanced[df_enhanced['class_name'] != 'fall']
    df_jatuh = df_enhanced[df_enhanced['class_name'] == 'fall']
    
    print(f"   Normal samples: {len(df_normal)}")
    print(f"   Fall samples: {len(df_jatuh)}")
    
    # Oversampling
    df_jatuh_digandakan = df_jatuh.sample(len(df_normal), replace=True, random_state=42)
    df_seimbang = pd.concat([df_normal, df_jatuh_digandakan], axis=0).sample(frac=1, random_state=42)
    
    print(f"   After oversampling: {len(df_seimbang[df_seimbang['class_name'] != 'fall'])} Normal vs {len(df_seimbang[df_seimbang['class_name'] == 'fall'])} Fall")
    
    # Extract features and labels
    X = df_seimbang[kolom_fitur_semua].values.astype(np.float32)
    y = (df_seimbang['class_name'] == 'fall').astype(np.float32).values
    
    print("\n[INFO] 4. Normalisasi fitur (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    # Save scaler for inference
    os.makedirs('models', exist_ok=True)
    with open(JALUR_SCALER, 'wb') as f:
        pickle.dump(scaler, f)
    print("   ✓ Scaler saved for inference")
    
    print("\n[INFO] 5. Membagi data (Train 70% / Validation 15% / Test 15%)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    jumlah_fitur = X_train.shape[1]
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"   Feature dimension: {jumlah_fitur}")
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_f1_score_metric',
        patience=8,
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_f1_score_metric',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
        mode='max'
    )
    
    print("\n[INFO] 6. Membangun arsitektur model yang ditingkatkan...")
    
    # Enhanced architecture with BatchNormalization
    inputs = Input(shape=(jumlah_fitur,), name="Input_Layer")
    
    # Layer 1: 256 neurons
    x = Dense(256, activation='relu', name="Dense_1")(inputs)
    x = BatchNormalization(name="BatchNorm_1")(x)
    x = Dropout(0.3, name="Dropout_1")(x)
    
    # Layer 2: 128 neurons
    x = Dense(128, activation='relu', name="Dense_2")(x)
    x = BatchNormalization(name="BatchNorm_2")(x)
    x = Dropout(0.3, name="Dropout_2")(x)
    
    # Layer 3: 64 neurons
    x = Dense(64, activation='relu', name="Dense_3")(x)
    x = BatchNormalization(name="BatchNorm_3")(x)
    x = Dropout(0.2, name="Dropout_3")(x)
    
    # Layer 4: 32 neurons
    x = Dense(32, activation='relu', name="Dense_4")(x)
    x = BatchNormalization(name="BatchNorm_4")(x)
    x = Dropout(0.2, name="Dropout_4")(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid', name="Output_Layer")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="SafeWatch_Enhanced")
    
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Custom F1 metric for training
    def f1_score_metric(y_true, y_pred):
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred_binary)
        fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
        fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1
    
    # Compile with custom learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', f1_score_metric]
    )
    
    print("\n[INFO] 7. Memulai proses training dengan data seimbang dan fitur lanjutan...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1,
        class_weight={0: 1.0, 1: 1.0}  # Balanced weights
    )
    
    print("\n[INFO] 8. Evaluasi pada data test...")
    loss, accuracy, f1 = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print(f"   F1-Score: {f1:.4f}")
    
    # Generate predictions for detailed metrics
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # Detailed metrics
    cm = confusion_matrix(y_test, y_pred_binary)
    report = classification_report(y_test, y_pred_binary, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Dynamically find keys for report to handle float/int/string label variations
    fall_key = None
    for k in report.keys():
        if str(k) in ['1.0', '1', 'True']:
            fall_key = k
            break
            
    print("\n📊 Detailed Evaluation Metrics:")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    if fall_key is not None:
        print(f"   Precision (Fall): {report[fall_key]['precision']:.4f}")
        print(f"   Recall (Fall): {report[fall_key]['recall']:.4f}")
        print(f"   F1-Score (Fall): {report[fall_key]['f1-score']:.4f}")
    else:
        print("   Could not compute detailed metrics: Fall key not found in report.")
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Save model
    model.save(JALUR_MODEL_ENHANCED)
    print(f"\n✅ Model enhanced berhasil disimpan: {JALUR_MODEL_ENHANCED}")
    
    # Save training history
    riwayat_latih = {
        'akurasi': history.history['accuracy'],
        'val_akurasi': history.history.get('val_accuracy', []),
        'loss': history.history['loss'],
        'val_loss': history.history.get('val_loss', []),
        'confusion_matrix': cm,
        'akurasi_akhir': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'report': report,
        'feature_count': jumlah_fitur,
        'feature_names': kolom_fitur_semua
    }
    
    with open(JALUR_HISTORY, 'wb') as f:
        pickle.dump(riwayat_latih, f)
    
    print(f"✅ Training history saved: {JALUR_HISTORY}")
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = latih_model_enhanced()
