from pathlib import Path
import os

# Root project & directory paths
PROJECT_ROOT = Path(__file__).parent.parent
BASE_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"          # Direktori baru khusus untuk menyimpan otak AI (.keras)
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Pastikan folder output otomatis terbuat jika belum ada
for d in [OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ✅ DICTIONARY DATASET (Telah Disesuaikan dengan Folder Nyata)
DATASETS = {
    "fall_detection": {
        "path": BASE_DATA_DIR / "human_fall", # Sinkron dengan data/raw/human_fall/
        "annotation_type": "yolo"             # Menggunakan format Label Bounding Box .txt
    },
    "smoke_fire": {
        "path": BASE_DATA_DIR / "fire_smoke_detection", # Sinkron dengan data/raw/fire_smoke/
        "annotation_type": "yolo"             # Diseragamkan menggunakan format YOLO
    },
    "person_detection": {
        "path": BASE_DATA_DIR / "person_detection",
        "annotation_type": "yolo"
    }
}

PLOT_CONFIG = {
       'style': 'dark_background', # <-- GANTI INI
       'palette': 'viridis', # (Biarkan saja nilai palette-nya)
       'dpi': 120,           # (Biarkan saja nilai dpi-nya)
       'figsize': (10, 6)    # (Biarkan saja nilai figsize-nya)
   }