"""
SafeWatch — Model & Scaler Conversion Script
============================================
Converts the Keras fall detection model to ONNX format
and serializes the sklearn scaler to JSON for use in the browser (ONNX.js).

Usage:
    python scripts/convert_model_for_browser.py

Output:
    - models/safewatch_model_cpu.onnx         (ONNX model for backend)
    - frontend/public/models/fall_model.onnx  (ONNX model for browser)
    - frontend/public/models/scaler.json      (Scaler params for browser)
"""

import os
import sys
import json
import pickle
import numpy as np

# Resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "safewatch_fall_model.keras")
ONNX_BACKEND_PATH = os.path.join(PROJECT_ROOT, "models", "safewatch_model_cpu.onnx")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")

FRONTEND_MODELS_DIR = os.path.join(PROJECT_ROOT, "frontend", "public", "models")
ONNX_FRONTEND_PATH = os.path.join(FRONTEND_MODELS_DIR, "fall_model.onnx")
SCALER_JSON_PATH = os.path.join(FRONTEND_MODELS_DIR, "scaler.json")

os.makedirs(FRONTEND_MODELS_DIR, exist_ok=True)


def convert_keras_to_onnx():
    """Convert .keras model to ONNX format."""
    print("=" * 60)
    print("Step 1: Converting Keras model → ONNX")
    print("=" * 60)

    # Check if ONNX already exists
    if os.path.exists(ONNX_BACKEND_PATH):
        print(f"✓ ONNX model already exists: {ONNX_BACKEND_PATH}")
        print("  Skipping conversion. Delete the file to re-convert.")
    else:
        try:
            import tf2onnx
            import tensorflow as tf

            # Setup legacy Keras module redirections in sys.modules
            try:
                import sys as sys_lib
                import tf_keras as tfk
                import tf_keras.src.engine.functional as tfk_functional
                import tf_keras.src.engine.sequential as tfk_sequential
                
                sys_lib.modules['keras.src.engine.functional'] = tfk_functional
                sys_lib.modules['keras.src.engine.sequential'] = tfk_sequential
                sys_lib.modules['keras.src.engine'] = tfk.src.engine
                sys_lib.modules['keras.src'] = tfk.src
                sys_lib.modules['keras'] = tfk
                print("✓ Successfully redirected legacy Keras import paths to tf_keras.")
            except Exception as redirect_err:
                print(f"Warning: Legacy import redirection failed: {redirect_err}")

            print(f"Loading Keras model from: {KERAS_MODEL_PATH}")
            try:
                import tf_keras as tfk
                print("Using tf_keras to load legacy Keras 2 model (compile=False)...")
                model = tfk.models.load_model(KERAS_MODEL_PATH, compile=False)
            except Exception as e:
                print(f"tf_keras loading failed: {e}. Trying standard tf.keras (compile=False)...")
                model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
            
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")

            # Get input signature
            input_signature = [
                tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input")
            ]

            # Convert to ONNX
            onnx_model, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
                opset=13
            )

            # Save backend ONNX
            import onnx
            onnx.save(onnx_model, ONNX_BACKEND_PATH)
            print(f"✓ ONNX model saved to: {ONNX_BACKEND_PATH}")

        except ImportError as e:
            print(f"✗ Missing dependency: {e}")
            print("  Install with: pip install tf2onnx tensorflow onnx")
            return False
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
            return False

    # Copy ONNX to frontend/public/models/
    import shutil
    shutil.copy2(ONNX_BACKEND_PATH, ONNX_FRONTEND_PATH)
    print(f"✓ ONNX model copied to frontend: {ONNX_FRONTEND_PATH}")
    return True


def serialize_scaler():
    """Serialize sklearn StandardScaler to JSON for browser use."""
    print("\n" + "=" * 60)
    print("Step 2: Serializing sklearn scaler → JSON")
    print("=" * 60)

    if not os.path.exists(SCALER_PATH):
        print(f"✗ Scaler not found at: {SCALER_PATH}")
        return False

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        scaler_data = {
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "var_": scaler.var_.tolist() if hasattr(scaler, "var_") else None,
            "n_features_in_": int(scaler.n_features_in_),
            "n_samples_seen_": int(scaler.n_samples_seen_) if hasattr(scaler, "n_samples_seen_") else None
        }

        with open(SCALER_JSON_PATH, "w") as f:
            json.dump(scaler_data, f, indent=2)

        print(f"✓ Scaler serialized: {SCALER_JSON_PATH}")
        print(f"  Features: {scaler_data['n_features_in_']}")
        print(f"  Mean range: [{min(scaler_data['mean_']):.4f}, {max(scaler_data['mean_']):.4f}]")
        return True

    except Exception as e:
        print(f"✗ Scaler serialization failed: {e}")
        return False


def benchmark_onnx():
    """Quick benchmark to confirm ONNX model works and measure speed."""
    print("\n" + "=" * 60)
    print("Step 3: Benchmarking ONNX model")
    print("=" * 60)

    try:
        import onnxruntime as ort
        import time

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(ONNX_BACKEND_PATH, sess_options=opts)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        print(f"  Input name: {input_name}")
        print(f"  Input shape: {input_shape}")

        # Warm up + benchmark
        dummy_input = np.random.randn(1, 35).astype(np.float32)
        times = []
        for i in range(20):
            t0 = time.time()
            out = sess.run(None, {input_name: dummy_input})
            times.append((time.time() - t0) * 1000)

        avg_ms = np.mean(times[5:])  # skip warmup
        print(f"✓ Average inference: {avg_ms:.2f}ms (target: <10ms for client-side)")
        print(f"  Sample output: {out[0][0]}")

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")


if __name__ == "__main__":
    print("\n🔧 SafeWatch Model Conversion Tool\n")

    ok1 = convert_keras_to_onnx()
    ok2 = serialize_scaler()

    if ok1:
        benchmark_onnx()

    print("\n" + "=" * 60)
    if ok1 and ok2:
        print("✅ All done! Next steps:")
        print(f"   1. ONNX model at: {ONNX_FRONTEND_PATH}")
        print(f"   2. Scaler JSON at: {SCALER_JSON_PATH}")
        print("   3. Run: cd frontend && pnpm install onnxruntime-web")
        print("   4. Start frontend: pnpm dev")
    else:
        print("⚠️  Some steps failed. Check output above.")
    print("=" * 60)
