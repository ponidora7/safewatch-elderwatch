import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from config.paths import FALL_YAML, FALL_YOLO_MODEL, RUNS_DIR

model = YOLO("models/yolov8n.pt")

results = model.train(
    data=FALL_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    name="fall_v1",
    project=RUNS_DIR,
    # Augmentasi (dataset tidak punya augmentasi bawaan)
    flipud=0.3,
    fliplr=0.5,
    hsv_v=0.4,
    degrees=10,
    mosaic=1.0,
)

best_pt = os.path.join(str(results.save_dir), "weights", "best.pt")
shutil.copy(best_pt, FALL_YOLO_MODEL)
print(f"\n✅ Fall model tersimpan: {FALL_YOLO_MODEL}")