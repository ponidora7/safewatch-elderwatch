"""
scripts/verify_onnx.py
======================
Verifies the ONNX model at frontend/public/models/fall_model.onnx.
Checks input/output layers and prints metadata.
Copies the model to models/safewatch_model_cpu.onnx if valid.
"""

import os
import shutil
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ONNX_SOURCE = os.path.join(PROJECT_ROOT, "frontend", "public", "models", "fall_model.onnx")
ONNX_BACKEND_DEST = os.path.join(PROJECT_ROOT, "models", "safewatch_model_cpu.onnx")

print("SafeWatch ONNX Model Verification")
print("=" * 50)
print(f"Source file: {ONNX_SOURCE}")

if not os.path.exists(ONNX_SOURCE):
    print(f"❌ Error: ONNX file not found at {ONNX_SOURCE}")
    sys.exit(1)

try:
    import onnxruntime as ort
    import numpy as np
except ImportError:
    print("❌ Error: onnxruntime or numpy is not installed.")
    print("Please install them using: pip install onnxruntime numpy")
    sys.exit(1)

try:
    # Load session
    sess = ort.InferenceSession(ONNX_SOURCE)
    
    # Get inputs
    inputs = sess.get_inputs()
    print("\n--- Input Nodes ---")
    for idx, inp in enumerate(inputs):
        print(f"Input {idx}: name='{inp.name}', shape={inp.shape}, type={inp.type}")
    
    # Get outputs
    outputs = sess.get_outputs()
    print("\n--- Output Nodes ---")
    for idx, out in enumerate(outputs):
        print(f"Output {idx}: name='{out.name}', shape={out.shape}, type={out.type}")
        
    # Verify input dimension
    input_shape = inputs[0].shape
    expected_dim = 35
    
    if len(input_shape) == 2 and input_shape[1] == expected_dim:
        print(f"\n[OK] Success: Model input dimension {input_shape[1]} matches expected 35 features!")
    else:
        print(f"\n[WARN] Warning: Model input shape {input_shape} does not match expected (batch, {expected_dim}).")
        
    # Run dummy prediction
    print("\nRunning dummy prediction...")
    dummy_in = np.random.randn(1, input_shape[1]).astype(np.float32)
    res = sess.run(None, {inputs[0].name: dummy_in})
    print(f"Prediction output: {res[0]}")
    print("[OK] Model inference test passed!")
    
    # Copy to backend
    os.makedirs(os.path.dirname(ONNX_BACKEND_DEST), exist_ok=True)
    shutil.copy2(ONNX_SOURCE, ONNX_BACKEND_DEST)
    print(f"[OK] Copied valid ONNX model to backend destination: {ONNX_BACKEND_DEST}")
    
except Exception as e:
    print(f"[ERROR] Error verifying model: {e}")
    sys.exit(1)
