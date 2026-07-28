import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from config.paths import FIRE_SMOKE_YAML, FIRE_YOLO_MODEL, RUNS_DIR

model = YOLO("models/yolov8n.pt")

results = model.train(
    data=FIRE_SMOKE_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    name="fire_smoke_v1",
    project=RUNS_DIR,
    hsv_s=0.5,   # variasi saturasi warna penting untuk api/asap
    hsv_v=0.5,   # variasi kecerahan
    fliplr=0.5,
    mosaic=1.0,
)

best_pt = os.path.join(str(results.save_dir), "weights", "best.pt")
shutil.copy(best_pt, FIRE_YOLO_MODEL)
print(f"\n✅ Fire/Smoke model tersimpan: {FIRE_YOLO_MODEL}")