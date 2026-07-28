import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Insert root to system path for importing src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.advanced_feature_engineering import AdvancedFeatureEngineer

def convert_keras_to_onnx(keras_model_path, onnx_output_path):
    print(f"\n[INFO] Loading Keras model from {keras_model_path}...")
    # Load model without compiling to avoid custom metric loading issues
    model = tf.keras.models.load_model(keras_model_path, compile=False)
    
    # Get feature dimension dynamically from model input shape
    feature_dim = model.input_shape[1]
    print(f"✓ Model loaded successfully. Input feature dimension: {feature_dim}")
    
    print(f"[INFO] Converting Keras model to ONNX Float32...")
    spec = (tf.TensorSpec((None, feature_dim), tf.float32, name="Input_Layer"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    float_onnx_path = onnx_output_path.replace(".onnx", "_float32.onnx")
    with open(float_onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"✓ Float32 ONNX model saved to: {float_onnx_path}")
    return float_onnx_path, feature_dim

def quantize_onnx_model(float_onnx_path, quantized_onnx_path):
    print(f"\n[INFO] Applying dynamic INT8 quantization...")
    quantize_dynamic(
        model_input=float_onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QUInt8
    )
    print(f"✓ Quantized INT8 ONNX model saved to: {quantized_onnx_path}")

def get_sample_input(feature_dim):
    print("\n[INFO] Constructing benchmark input...")
    csv_path = 'data/processed/cleaned_human_fall.csv'
    scaler_path = 'models/feature_scaler.pkl'
    
    # Try loading a real sample from the processed CSV
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Find the feature columns (starting with X/Y or feat_)
            feature_cols = [col for col in df.columns if col.startswith('X') or col.startswith('Y')]
            
            # Run feature engineering enhancement
            df_enhanced = AdvancedFeatureEngineer.enhance_dataframe(df.head(1))
            enhanced_cols = [col for col in df_enhanced.columns if col.startswith('feat_')]
            all_cols = feature_cols + enhanced_cols
            
            # Select first row's feature values
            if len(all_cols) == feature_dim:
                raw_features = df_enhanced[all_cols].values.astype(np.float32)
                
                # Scale if scaler is available
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    scaled_features = scaler.transform(raw_features).astype(np.float32)
                    print("✓ Loaded sample features from cleaned dataset and applied scaling.")
                    return scaled_features
                else:
                    print("⚠️ feature_scaler.pkl not found. Returning unscaled dataset features.")
                    return raw_features
        except Exception as e:
            print(f"⚠️ Failed to load sample from CSV: {e}")
            
    # Fallback to random features matching model signature
    print("⚠️ Dataset sample extraction failed. Generating synthetic feature vector.")
    return np.random.randn(1, feature_dim).astype(np.float32)

def benchmark_onnx_model(onnx_path, input_features):
    print(f"\n[INFO] Benchmarking ONNX model inference speed on CPU...")
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    
    # Warmup
    print("Running warmup...")
    for _ in range(10):
        _ = sess.run(None, {input_name: input_features})
        
    # Benchmark over 100 runs
    print("Running 100 benchmark iterations...")
    start_time = time.time()
    for _ in range(100):
        _ = sess.run(None, {input_name: input_features})
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / 100.0) * 1000.0
    print(f"✓ Benchmark finished.")
    print(f"  → Average Inference Latency: {avg_latency_ms:.4f} ms per frame")
    
    if avg_latency_ms < 500.0:
        print("  → ✅ Inference speed matches PRD requirement (< 500ms)")
    else:
        raise RuntimeError(f"❌ Error: Inference latency is {avg_latency_ms:.1f}ms (exceeds 500ms limit!)")
    
    return avg_latency_ms

def main():
    keras_model_path = "models/safewatch_fall_model_enhanced.keras"
    onnx_output_path = "models/safewatch_model_cpu.onnx"
    
    # 1. Convert Keras model to ONNX Float32
    float_onnx_path, feature_dim = convert_keras_to_onnx(keras_model_path, onnx_output_path)
    
    # 2. Quantize the ONNX model to INT8
    quantize_onnx_model(float_onnx_path, onnx_output_path)
    
    # 3. Get sample input features for benchmarking
    sample_features = get_sample_input(feature_dim)
    
    # 4. Run CPU benchmark
    benchmark_onnx_model(onnx_output_path, sample_features)
    
    print("\n" + "="*70)
    print("🎉 MODEL CONVERSION AND OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == '__main__':
    main()
