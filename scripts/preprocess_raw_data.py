"""
scripts/preprocess_raw_data.py
==============================
Preprocesses the raw dataset using YOLOv8-pose to extract keypoints (compatible with Python 3.14).
Then applies advanced Biological Normalization to construct a 35-feature scale-invariant dataset.

Outputs:
    - data/processed/cleaned_human_fall_enhanced.csv (metadata + 35 normalized features)
"""

import os
import sys
import yaml
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Insert project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.advanced_feature_engineering import AdvancedFeatureEngineer

# Directories
RAW_FALL_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "human_fall")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

YAML_PATH = os.path.join(RAW_FALL_DIR, "data.yaml")
YOLO_POSE_MODEL_PATH = "yolov8n-pose.pt"

def load_classes():
    if not os.path.exists(YAML_PATH):
        print(f"[ERROR] data.yaml not found at {YAML_PATH}")
        sys.exit(1)
    with open(YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("names", [])

def main():
    print("SafeWatch Data Preprocessing (Scale-Invariant Biological Normalization)")
    print("=" * 70)
    
    classes = load_classes()
    
    try:
        from ultralytics import YOLO
        print("[INFO] Loading YOLOv8-Pose model...")
        pose_model = YOLO(YOLO_POSE_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv8-pose: {e}")
        sys.exit(1)

    splits = ["train", "valid", "test"]
    all_records = []
    
    # 5: L Shoulder, 6: R Shoulder, 11: L Hip, 12: R Hip, 13: L Knee, 14: R Knee, 15: L Ankle, 16: R Ankle
    YOLO_POSE_INDICES = [5, 6, 11, 12, 13, 14, 15, 16]
    
    for split in splits:
        img_dir = os.path.join(RAW_FALL_DIR, split, "images")
        lbl_dir = os.path.join(RAW_FALL_DIR, split, "labels")
        
        if not os.path.exists(lbl_dir):
            continue
            
        label_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        print(f"\nProcessing split '{split}': found {len(label_files)} annotation files.")
        
        for lbl_path in tqdm(label_files, desc=f"Split {split}"):
            base_name = os.path.splitext(os.path.basename(lbl_path))[0]
            
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                p = os.path.join(img_dir, base_name + ext)
                if os.path.exists(p):
                    img_path = p
                    break
                    
            if not img_path:
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            with open(lbl_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # Run YOLOv8-pose on the FULL UN-CROPPED image
                # This ensures we get true pixel coordinates that map perfectly to the inference pipeline!
                results = pose_model(img, verbose=False)
                
                if results and len(results) > 0 and results[0].keypoints is not None:
                    # Use 'xy' (raw pixels) instead of 'xyn'
                    xy_all = results[0].keypoints.xy 
                    if xy_all is not None and len(xy_all) > 0:
                        # We might have multiple people detected. 
                        # We should pick the person whose bounding box most closely matches the annotation.
                        # For simplicity, if there's only 1 person, use it. If multiple, find closest center.
                        boxes = results[0].boxes.xywhn.cpu().numpy()
                        best_idx = 0
                        min_dist = float('inf')
                        
                        for i, box in enumerate(boxes):
                            dist = (box[0] - cx)**2 + (box[1] - cy)**2
                            if dist < min_dist:
                                min_dist = dist
                                best_idx = i
                                
                        if min_dist > 0.05: # if the closest person is too far from annotation, skip
                            continue
                            
                        kp = xy_all[best_idx].cpu().numpy()  # shape (17, 2) - pure pixels
                        
                        if kp.shape[0] >= 17:
                            selected_kp = kp[YOLO_POSE_INDICES]  # shape (8, 2)
                            landmarks = selected_kp.flatten()  # shape (16,)
                            
                            if np.sum(np.abs(landmarks)) < 1.0: # pure pixels, sum should be >> 1
                                continue
                                
                            # Extract 35 biologically normalized features (19 geometric + 16 coords)
                            features_35 = AdvancedFeatureEngineer.extract_geometric_features(landmarks)
                            
                            record = {
                                "dataset": "human_fall",
                                "split": split,
                                "image_id": base_name,
                                "class_name": "fall" if class_name == "Falling" else "normal",
                            }
                            
                            # Add all 35 features exactly as they are (all are scale/translation invariant)
                            for k, v in features_35.items():
                                record[k] = v
                                
                            all_records.append(record)
                            
    if not all_records:
        print("[ERROR] No landmarks extracted from dataset!")
        return
        
    df_enhanced = pd.DataFrame(all_records)
    print(f"\n[INFO] Total processed samples: {len(df_enhanced)}")
    print(df_enhanced["class_name"].value_counts())
    
    enhanced_csv_path = os.path.join(PROCESSED_DIR, "cleaned_human_fall_enhanced.csv")
    df_enhanced.to_csv(enhanced_csv_path, index=False)
    print(f"[OK] Saved enhanced 35-feature dataset to: {enhanced_csv_path}")

if __name__ == "__main__":
    main()
