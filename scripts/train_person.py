import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from config.paths import PERSON_YAML, PERSON_YOLO_MODEL, RUNS_DIR

model = YOLO("models/yolov8n.pt")

results = model.train(
    data=PERSON_YAML,
    epochs=30,
    imgsz=640,
    batch=16,
    name="person_v1",
    project=RUNS_DIR,
    fliplr=0.5,
    degrees=5,
)

best_pt = os.path.join(str(results.save_dir), "weights", "best.pt")
shutil.copy(best_pt, PERSON_YOLO_MODEL)
print(f"\n✅ Person model tersimpan: {PERSON_YOLO_MODEL}")