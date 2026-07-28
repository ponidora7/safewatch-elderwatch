"""
scripts/inference_enhanced.py
==============================
Enhanced inference script using advanced features and optimized threshold.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from src.advanced_feature_engineering import AdvancedFeatureEngineer


MODEL_PATH = "models/safewatch_fall_model_enhanced.keras"
SCALER_PATH = "models/feature_scaler.pkl"
CSV_PATH = "data/processed/cleaned_human_fall.csv"
HISTORY_PATH = "models/training_history_enhanced.pkl"


def run_enhanced_inference():
    """Run inference with enhanced features and optimized threshold."""
    
    print("\n" + "="*70)
    print("🔍 ENHANCED FALL DETECTION INFERENCE")
    print("="*70)
    
    print("\n[1] Memuat model enhanced...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model tidak ditemukan di {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'f1_score_metric': lambda y_true, y_pred: y_pred})
    print("   ✓ Model loaded")
    
    print("\n[2] Memuat scaler untuk normalisasi...")
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Error: Scaler tidak ditemukan di {SCALER_PATH}")
        return
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("   ✓ Scaler loaded")
    
    print("\n[3] Memuat training history untuk informasi optimal threshold...")
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)
        print(f"   Model achieved:")
        print(f"     • Accuracy: {history.get('akurasi_akhir', 0)*100:.2f}%")
        print(f"     • F1-Score: {history.get('f1_score', 0):.4f}")
        print(f"     • ROC-AUC: {history.get('roc_auc', 0):.4f}")
    
    print("\n[4] Mengambil sampel data untuk testing...")
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Data CSV tidak ditemukan di {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    
    # Enhance features
    df_enhanced = AdvancedFeatureEngineer.enhance_dataframe(df)
    
    # Get feature columns
    feature_cols = [col for col in df_enhanced.columns if col.startswith('X') or col.startswith('Y') or col.startswith('feat_')]
    
    # Sample data
    data_normal = df_enhanced[df_enhanced['class_name'] != 'fall'].iloc[0]
    data_fall = df_enhanced[df_enhanced['class_name'] == 'fall'].iloc[0]
    
    print("\n[5] Melakukan inferensi...")
    
    # Prepare inputs
    X_normal = data_normal[feature_cols].values.astype(np.float32).reshape(1, -1)
    X_fall = data_fall[feature_cols].values.astype(np.float32).reshape(1, -1)
    
    # Scale
    X_normal_scaled = scaler.transform(X_normal)
    X_fall_scaled = scaler.transform(X_fall)
    
    # Predict with default threshold (0.5)
    prob_normal = model.predict(X_normal_scaled, verbose=0)[0][0]
    prob_fall = model.predict(X_fall_scaled, verbose=0)[0][0]
    
    # Determine predictions
    threshold = 0.5
    pred_normal = "🟢 NORMAL (AMAN)" if prob_normal <= threshold else "🔴 FALL (BAHAYA!)"
    pred_fall = "🟢 NORMAL (AMAN)" if prob_fall <= threshold else "🔴 FALL (BAHAYA!)"
    
    print(f"\n📊 Test Results with Enhanced Features:")
    print(f"\n   Sample 1 - Normal Posture:")
    print(f"     Prediction: {pred_normal}")
    print(f"     Probability of Fall: {prob_normal:.4f}")
    print(f"     Confidence: {max(1-prob_normal, prob_normal)*100:.2f}%")
    
    print(f"\n   Sample 2 - Fall Posture:")
    print(f"     Prediction: {pred_fall}")
    print(f"     Probability of Fall: {prob_fall:.4f}")
    print(f"     Confidence: {max(1-prob_fall, prob_fall)*100:.2f}%")
    
    print(f"\n   Threshold: {threshold}")
    print("\n" + "="*70)


if __name__ == "__main__":
    run_enhanced_inference()
