#!/usr/bin/env python
# scripts/check_env.py
import importlib
checks = {
    'opencv-python': 'cv2',
    'mediapipe': 'mediapipe',
    'ultralytics': 'ultralytics',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'streamlit': 'streamlit',
    'matplotlib': 'matplotlib',
    'keras': 'keras',
}
for name, mod in checks.items():
    try:
        m = importlib.import_module(mod)
        v = getattr(m, '__version__', 'Unknown')
        print(f"OK  {name:20s} -> import '{mod}' v{v}")
    except Exception as e:
        print(f"MISS {name:20s} -> import '{mod}' failed: {e}")
