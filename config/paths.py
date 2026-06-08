# config/paths.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw Dataset Directories
RAW_FALL_DIR       = os.path.join(BASE_DIR, "data", "raw", "human_fall")
RAW_FIRE_SMOKE_DIR = os.path.join(BASE_DIR, "data", "raw", "fire_smoke_detection")
RAW_PERSON_DIR     = os.path.join(BASE_DIR, "data", "raw", "person_detection")

# Data YAML
FALL_YAML       = os.path.join(RAW_FALL_DIR,       "data.yaml")
FIRE_SMOKE_YAML = os.path.join(RAW_FIRE_SMOKE_DIR, "data.yaml")
PERSON_YAML     = os.path.join(RAW_PERSON_DIR,     "data.yaml")

# Processed Output
PROCESSED_DIR       = os.path.join(BASE_DIR, "data", "processed")
ASSESSED_FALL_CSV   = os.path.join(PROCESSED_DIR, "assessed_fall.csv")
ASSESSED_FIRE_CSV   = os.path.join(PROCESSED_DIR, "assessed_fire_smoke.csv")
ASSESSED_PERSON_CSV = os.path.join(PROCESSED_DIR, "assessed_person.csv")
CLEANED_FALL_CSV    = os.path.join(PROCESSED_DIR, "cleaned_fall.csv")
CLEANED_FIRE_CSV    = os.path.join(PROCESSED_DIR, "cleaned_fire_smoke.csv")
CLEANED_PERSON_CSV  = os.path.join(PROCESSED_DIR, "cleaned_person.csv")
COMBINED_CSV        = os.path.join(PROCESSED_DIR, "combined_summary.csv")

# Model Paths
MODEL_DIR         = os.path.join(BASE_DIR, "models")
BASE_YOLO_MODEL   = os.path.join(MODEL_DIR, "yolov8n.pt")
FALL_YOLO_MODEL   = os.path.join(MODEL_DIR, "fall_model.pt")
FIRE_YOLO_MODEL   = os.path.join(MODEL_DIR, "fire_smoke_model.pt")
PERSON_YOLO_MODEL = os.path.join(MODEL_DIR, "person_model.pt")
FALL_TF_MODEL     = os.path.join(MODEL_DIR, "safewatch_fall_model.keras")
LABEL_ENCODER     = os.path.join(MODEL_DIR, "label_encoder.pkl")
TRAINING_HISTORY  = os.path.join(MODEL_DIR, "training_history.pkl")

# Reports
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# YOLO Training Output
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")

# Auto-create semua folder yang diperlukan
for _dir in [PROCESSED_DIR, MODEL_DIR, REPORTS_DIR, FIGURES_DIR, RUNS_DIR]:
    os.makedirs(_dir, exist_ok=True)