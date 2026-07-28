"""
scripts/setup_vosk_model.py
============================
Downloads and extracts the Vosk Indonesian language model for offline
speech recognition. Only needs to be run once.

Model: vosk-model-small-id-0.22 (~40MB)
Target: models/vosk-id/
"""

import os
import sys
import io
import zipfile
import urllib.request

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "vosk-id")

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-id-0.22.zip"
MODEL_ZIP_NAME = "vosk-model-small-id-0.22"


def download_with_progress(url: str, dest_path: str):
    """Download a file with a simple progress indicator."""
    print(f"   Downloading from: {url}")
    
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get('Content-Length', 0))
    downloaded = 0
    block_size = 8192
    
    with open(dest_path, 'wb') as f:
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded * 100 // total_size
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progress: {mb_down:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)
    
    print()  # newline after progress


def main():
    print("=" * 50)
    print("SafeWatch — Vosk Indonesian Model Setup")
    print("=" * 50)
    
    # Check if model already exists
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"\n✅ Model sudah ada di: {MODEL_DIR}")
        print("   Jika ingin mendownload ulang, hapus folder tersebut terlebih dahulu.")
        return
    
    print(f"\n📦 Mengunduh model Vosk Bahasa Indonesia...")
    print(f"   Target folder: {MODEL_DIR}")
    
    # Download zip
    zip_path = os.path.join(PROJECT_ROOT, "models", f"{MODEL_ZIP_NAME}.zip")
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    
    try:
        download_with_progress(MODEL_URL, zip_path)
    except Exception as e:
        print(f"\n❌ Gagal mengunduh model: {e}")
        print("   Pastikan koneksi internet aktif.")
        sys.exit(1)
    
    # Extract zip
    print(f"\n📂 Mengekstrak model...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(os.path.join(PROJECT_ROOT, "models"))
        
        # Rename extracted folder to vosk-id
        extracted_dir = os.path.join(PROJECT_ROOT, "models", MODEL_ZIP_NAME)
        if os.path.isdir(extracted_dir):
            os.rename(extracted_dir, MODEL_DIR)
        
        print(f"   ✓ Model berhasil diekstrak ke: {MODEL_DIR}")
    except Exception as e:
        print(f"\n❌ Gagal mengekstrak: {e}")
        sys.exit(1)
    finally:
        # Cleanup zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"   ✓ File zip sementara dihapus.")
    
    # Verify
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"\n✅ Setup selesai! Model Vosk Indonesia siap digunakan.")
    else:
        print(f"\n❌ Setup gagal — folder model kosong.")
        sys.exit(1)


if __name__ == "__main__":
    main()
