#!/usr/bin/env python
# scripts/verify_installation.py
"""
Script untuk memverifikasi semua dependencies terinstal dengan benar
"""
import sys
import subprocess

# Map package display name -> (import_name, expected_version)
REQUIRED_PACKAGES = {
    'tensorflow': ('tensorflow', '2.15.0'),
    'keras': ('keras', '2.15.0'),
    'opencv-python': ('cv2', '4.11.0.86'),
    'mediapipe': ('mediapipe', '0.10.5'),
    'ultralytics': ('ultralytics', None),  # Version flexible
    'pandas': ('pandas', '2.1.4'),
    'numpy': ('numpy', '1.26.4'),
    'scikit-learn': ('sklearn', '1.3.2'),
    'streamlit': ('streamlit', None),
    'matplotlib': ('matplotlib', '3.8.4'),
}

# Prefer to check lightweight packages first; check TensorFlow last to avoid
# noisy oneDNN startup logs hiding earlier results.
CHECK_ORDER = [k for k in REQUIRED_PACKAGES.keys() if k != 'tensorflow'] + ['tensorflow']

def check_package(display_name, import_name, expected_version=None):
    # Import the package in a fresh subprocess to avoid heavy C-extension
    # initialization (e.g., TensorFlow) flooding stdout/stderr of this script.
    import json
    cmd = [sys.executable, "-c", (
        "import importlib, json;\n"
        f"try:\n  m = importlib.import_module('{import_name}');\n"
        "  v = getattr(m, '__version__', getattr(getattr(m, 'version', None), '__version__', 'Unknown'));\n"
        "  print(json.dumps({'ok': True, 'version': str(v)}))\n"
        "except Exception as e:\n  import traceback, sys;\n  tb = traceback.format_exc();\n  print(json.dumps({'ok': False, 'error': str(e), 'trace': tb}))\n"
    )]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out = proc.stdout.strip()
        if not out:
            # sometimes package writes to stderr only
            out = proc.stderr.strip()
        try:
            info = json.loads(out)
        except Exception:
            print(f"❌ {display_name:20s} IMPORT FAILED (no JSON output)\n{proc.stdout}\n{proc.stderr}")
            return False
        if not info.get('ok'):
            print(f"❌ {display_name:20s} NOT INSTALLED or import error: {info.get('error')}")
            return False
        version = info.get('version', 'Unknown')
        ok = (expected_version is None) or (version.startswith(expected_version) or expected_version in version)
        status = "✅" if ok else "⚠️"
        print(f"{status} {display_name:20s} v{version}")
        return ok
    except subprocess.TimeoutExpired:
        print(f"❌ {display_name:20s} IMPORT TIMEOUT")
        return False

def main():
    print("🔍 SafeWatch Dependency Verification")
    print("=" * 50)
    
    results = []
    for pkg in CHECK_ORDER:
        import_name, ver = REQUIRED_PACKAGES[pkg]
        results.append(check_package(pkg, import_name, ver))
    
    print("=" * 50)
    if all(results):
        print("🎉 Semua dependencies terinstal dengan benar!")
        return 0
    else:
        print("⚠️  Beberapa package belum terinstal. Jalankan: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())