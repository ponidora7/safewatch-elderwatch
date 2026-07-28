import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Insert root to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_enhanced_model import latih_model_enhanced
from src.advanced_feature_engineering import AdvancedFeatureEngineer

def main():
    print("\n" + "="*70)
    print("🧠 RUNNING END-TO-END TRAINING AND ONNX CPU OPTIMIZATION")
    print("="*70)
    
    # Step 1: Train the enhanced model
    print("\n=== STEP 1: TRAINING ENHANCED KERAS MODEL ===")
    model, scaler = latih_model_enhanced()
    
    # Step 2: Determine feature dimension dynamically from dataset
    print("\n=== STEP 2: DETERMINING FEATURE SIGNATURE ===")
    df = pd.read_csv('data/processed/cleaned_human_fall.csv')
    df_enhanced = AdvancedFeatureEngineer.enhance_dataframe(df)
    kolom_fitur_original = [col for col in df.columns if col.startswith('X') or col.startswith('Y')]
    kolom_fitur_engineered = [col for col in df_enhanced.columns if col.startswith('feat_')]
    kolom_fitur_semua = kolom_fitur_original + kolom_fitur_engineered
    feature_dim = len(kolom_fitur_semua)
    print(f"Detected feature dimension: {feature_dim}")
    
    # Step 3: Convert Keras model to ONNX Float32
    print("\n=== STEP 3: CONVERTING KERAS TO ONNX FLOAT32 ===")
    onnx_path_float32 = "models/safewatch_model_cpu_float32.onnx"
    spec = (tf.TensorSpec((None, feature_dim), tf.float32, name="Input_Layer"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    os.makedirs('models', exist_ok=True)
    with open(onnx_path_float32, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"✓ ONNX float32 model saved to: {onnx_path_float32}")
    
    # Step 4: Apply Dynamic INT8 Quantization
    print("\n=== STEP 4: APPLYING DYNAMIC INT8 QUANTIZATION ===")
    onnx_path_quant = "models/safewatch_model_cpu.onnx"
    
    quantize_dynamic(
        model_input=onnx_path_float32,
        model_output=onnx_path_quant,
        weight_type=QuantType.QUInt8
    )
    print(f"✓ Quantized INT8 model saved to: {onnx_path_quant}")
    
    # Step 5: Benchmark ONNX Quantized Model Inference Speed
    print("\n=== STEP 5: BENCHMARKING CPU INFERENCE SPEED ===")
    sess = ort.InferenceSession(onnx_path_quant)
    input_name = sess.get_inputs()[0].name
    
    # Benchmark over 100 runs
    dummy_input = np.random.randn(1, feature_dim).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = sess.run(None, {input_name: dummy_input})
        
    start_time = time.time()
    for _ in range(100):
        _ = sess.run(None, {input_name: dummy_input})
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) / 100.0) * 1000.0
    print(f"  → Average Inference Speed: {avg_time_ms:.4f} ms per frame")
    
    if avg_time_ms < 500.0:
        print("  → ✅ Inference speed matches PRD requirement (< 500ms)")
    else:
        print("  → ❌ Warning: Inference speed is slower than 500ms")
        
    print("\n" + "="*70)
    print("🎉 PROCESS COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == '__main__':
    main()
