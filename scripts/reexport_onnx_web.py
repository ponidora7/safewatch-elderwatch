"""
Re-export ONNX model with opset 12 + static batch dim for onnxruntime-web compatibility.
Run from project root: .venv\Scripts\python.exe scripts\reexport_onnx_web.py
"""
import onnx
import json
import pickle
from onnx import version_converter

# Paths
src = 'models/safewatch_model_cpu_float32.onnx'
dst = 'frontend/public/models/fall_model.onnx'
scaler_src = 'models/feature_scaler.pkl'
scaler_dst = 'frontend/public/models/scaler.json'

print(f'Loading {src}...')
model = onnx.load(src)
print(f'Original opset: {model.opset_import[0].version}, IR: {model.ir_version}')

# Convert to opset 12 which is best supported by onnxruntime-web
print('Converting to opset 12...')
try:
    model_12 = version_converter.convert_version(model, 12)
    print(f'After conversion opset: {model_12.opset_import[0].version}')
except Exception as e:
    print(f'Conversion failed ({e}), using original')
    model_12 = model

# Fix dynamic batch dim -> static 1 (required for onnxruntime-web WASM)
for inp in model_12.graph.input:
    for dim in inp.type.tensor_type.shape.dim:
        if dim.dim_param:  # dynamic dim like 'unk__6'
            dim.dim_param = ''
            dim.dim_value = 1

# Fix dynamic output dims too
for out in model_12.graph.output:
    for dim in out.type.tensor_type.shape.dim:
        if dim.dim_param:
            dim.dim_param = ''
            dim.dim_value = 1

onnx.checker.check_model(model_12)
onnx.save(model_12, dst)
print(f'Saved model to {dst}')

# Export scaler as JSON for browser
print(f'Loading scaler from {scaler_src}...')
with open(scaler_src, 'rb') as f:
    scaler = pickle.load(f)

scaler_data = {
    'mean_': scaler.mean_.tolist(),
    'scale_': scaler.scale_.tolist(),
    'n_features_in_': int(scaler.n_features_in_)
}
with open(scaler_dst, 'w') as f:
    json.dump(scaler_data, f)
print(f'Scaler saved to {scaler_dst} ({scaler.n_features_in_} features)')
print('All done! Refresh the browser to use the new model.')
