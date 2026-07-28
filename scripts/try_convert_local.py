"""
SafeWatch — Try to convert Keras model without full TensorFlow install.
Uses tf2onnx's standalone mode if available.

Usage: python scripts/try_convert_local.py
"""
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

KERAS_PATH = os.path.join(PROJECT_ROOT, "models", "safewatch_fall_model.keras")
ONNX_OUT   = os.path.join(PROJECT_ROOT, "models", "safewatch_model_cpu.onnx")
FRONTEND_OUT = os.path.join(PROJECT_ROOT, "frontend", "public", "models", "fall_model.onnx")

print("SafeWatch local ONNX conversion attempt")
print("=" * 50)
print(f"Keras model: {KERAS_PATH}")
print(f"ONNX output: {ONNX_OUT}")

# --- Attempt 1: via tf2onnx CLI (subprocess, avoids Python TF import) ------
import subprocess, shutil

def attempt_tf2onnx_cli():
    print("\n[Attempt 1] tf2onnx CLI via subprocess...")
    result = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert",
         "--keras", KERAS_PATH,
         "--output", ONNX_OUT,
         "--opset", "13"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("[OK] tf2onnx CLI succeeded!")
        return True
    else:
        print(f"[FAIL] tf2onnx CLI error:\n{result.stderr[-500:]}")
        return False

# --- Attempt 2: keras standalone (keras 3.x can load without TF sometimes) -
def attempt_keras_standalone():
    print("\n[Attempt 2] keras standalone...")
    try:
        os.environ["KERAS_BACKEND"] = "numpy"  # try numpy backend
        import keras
        model = keras.models.load_model(KERAS_PATH)
        print(f"  Loaded model: input={model.input_shape}, output={model.output_shape}")
        # Try export
        model.export(ONNX_OUT.replace(".onnx", "_savedmodel"), format="tf_saved_model")
        print("  Exported SavedModel — now converting with tf2onnx...")
        result = subprocess.run(
            [sys.executable, "-m", "tf2onnx.convert",
             "--saved-model", ONNX_OUT.replace(".onnx", "_savedmodel"),
             "--output", ONNX_OUT, "--opset", "13"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[OK] SavedModel → ONNX succeeded!")
            return True
        else:
            print(f"[FAIL] {result.stderr[-300:]}")
    except Exception as e:
        print(f"[FAIL] keras standalone: {e}")
    return False

# --- Verify ONNX ----------------------------------------------------------
def verify_onnx():
    print(f"\n[Verify] Testing ONNX model at {ONNX_OUT}...")
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(ONNX_OUT)
        name = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape
        print(f"  Input: '{name}' shape={shape}")
        dummy = np.random.randn(1, 35).astype("float32")
        out = sess.run(None, {name: dummy})
        print(f"  Test output: {out[0][0]:.4f} [PASS]")
        # Copy to frontend
        os.makedirs(os.path.dirname(FRONTEND_OUT), exist_ok=True)
        shutil.copy2(ONNX_OUT, FRONTEND_OUT)
        print(f"  Copied to frontend: {FRONTEND_OUT}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

# --- Check existing ONNX (maybe already converted) -------------------------
if os.path.exists(ONNX_OUT):
    print(f"\n[INFO] ONNX already exists: {ONNX_OUT}")
    if verify_onnx():
        print("\n[DONE] Existing ONNX model is valid and copied to frontend!")
        sys.exit(0)

ok = attempt_tf2onnx_cli() or attempt_keras_standalone()

if ok:
    verify_onnx()
    print("\n[SUCCESS] Conversion complete!")
else:
    print("\n[FAILED] Local conversion not possible with Python 3.14 + no TensorFlow.")
    print("  -> Use the Google Colab notebook instead:")
    print("  -> notebooks/convert_model_colab.ipynb")
    print("  -> Upload safewatch_fall_model.keras, run all cells, download fall_model.onnx")
    print("  -> Place it at: frontend/public/models/fall_model.onnx")
