# SafeWatch Project - AI Assistant Guidelines

Welcome to the SafeWatch project repository. When operating within this workspace, please adhere to the following architecture, conventions, and workflows.

## 1. Project Context & Documentation
- **Primary Goal:** A dual-purpose computer vision system for Fall Detection (Keras/TensorFlow) and Fire/Smoke Detection (YOLOv8).
- **Start Here:** Always read `DOCUMENTATION_INDEX.md` and `IMPLEMENTATION_SUMMARY.md` before starting a new complex task to understand current architectural decisions and recent changes.
- **Project Structure:** Reference `file_structure.md` or `workspace_structure.txt` for the layout.

## 2. Technical Stack
- **Languages:** Python (Primary for ML pipelines/APIs), PowerShell/Batch (DevOps/Setup).
- **Core Libraries:**
  - TensorFlow / Keras (Fall Detection Model)
  - Ultralytics YOLO (Fire & Smoke Detection Model)
  - OpenCV, MediaPipe (Video processing and Pose extraction)
  - Scikit-learn, Pandas, NumPy (Data handling)

## 3. Engineering & AI Rules
- **No Direct Edits to Raw Data:** Do not attempt to modify anything in `data/raw/` or `data/real_footage/`.
- **Model Files:** Pre-trained or trained models are saved in `models/`. Core models are tracked in git (`!models/...` in `.gitignore`), do not arbitrarily delete them.
- **Idempotent Scripts:** Scripts in `scripts/` (e.g., `assess_all.py`, `train_fall.py`) should be designed to be run multiple times safely.
- **Python Conventions:**
  - Follow PEP8 style guidelines.
  - Use type hints wherever possible for new functions.
  - Rely on `config/paths.py` for all file path resolving to maintain cross-platform compatibility. Do not hardcode paths.
- **Notebooks:** `.ipynb` files in `notebooks/` are for exploration (EDA) and experimental training. Production pipeline code should live in `src/` and `scripts/`.

## 4. Workflows
- **Data Pipeline:** Handled in `src/` (`data_cleaner.py`, `feature_temporal.py`, etc.).
- **Testing:** Before suggesting changes to the model inference logic, refer to `TEST_MODEL_GUIDE.md`.

## 5. Environment & Git
- Do not commit `.env` files or large untracked model files unless specifically instructed.
- When generating reports or large output logs, save them to the `reports/` folder, which is appropriately managed by `.gitignore`.
