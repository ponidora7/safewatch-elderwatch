import os
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Global objects to avoid re-initializing MediaPipe in every execution step
mp_pose = None
pose = None

def init_process():
    global mp_pose, pose
    # Disable tensorflow logs inside child processes to avoid console spam
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_single_pose(args):
    idx, row_dict = args
    img_path = row_dict['image_path']
    if not os.path.exists(img_path):
        return idx, None
        
    img = cv2.imread(img_path)
    if img is None:
        return idx, None
        
    h, w = img.shape[:2]
    
    # Bbox crop coordinates
    x_center = int(row_dict['bbox_x_px'])
    y_center = int(row_dict['bbox_y_px'])
    box_w = int(row_dict['bbox_w_px'])
    box_h = int(row_dict['bbox_h_px'])
    
    x_min = max(0, x_center - (box_w // 2))
    y_min = max(0, y_center - (box_h // 2))
    x_max = min(w, x_center + (box_w // 2))
    y_max = min(h, y_center + (box_h // 2))
    
    crop_img = img[y_min:y_max, x_min:x_max]
    if crop_img.size == 0:
        return idx, None
        
    try:
        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        
        if result.pose_landmarks:
            landmark_indices = [11, 12, 23, 24, 25, 26, 27, 28]
            landmarks = {}
            for l_idx in landmark_indices:
                lm = result.pose_landmarks.landmark[l_idx]
                landmarks[f'X{l_idx}'] = lm.x
                landmarks[f'Y{l_idx}'] = lm.y
            return idx, landmarks
    except Exception:
        pass
        
    return idx, None

def main():
    csv_path = 'data/processed/cleaned_fall_new.csv'
    output_path = 'data/processed/cleaned_human_fall.csv'
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    # 1. Update paths to point to current workspace
    df['image_path'] = df['image_path'].apply(lambda x: str(x).replace(
        'D:\\2. Semester 6\\Konversi DBS\\capstone\\safewatch2\\',
        'D:\\safewatch-elderwatch\\'
    ) if pd.notna(x) else x)
    df['label_path'] = df['label_path'].apply(lambda x: str(x).replace(
        'D:\\2. Semester 6\\Konversi DBS\\capstone\\safewatch2\\',
        'D:\\safewatch-elderwatch\\'
    ) if pd.notna(x) else x)
    
    # 2. Map class names: 'Falling' -> 'fall'
    df['class_name'] = df['class_name'].apply(lambda x: 'fall' if x == 'Falling' else x)
    
    # Calculate pixel coords if not present
    df['bbox_x_px'] = df['bbox_x_center'] * df['img_width']
    df['bbox_y_px'] = df['bbox_y_center'] * df['img_height']
    
    # Pre-add coordinate columns
    landmark_indices = [11, 12, 23, 24, 25, 26, 27, 28]
    for idx in landmark_indices:
        df[f'X{idx}'] = np.nan
        df[f'Y{idx}'] = np.nan
        
    df['pose_extracted'] = False
    
    # Prepare arguments for multiprocessing
    tasks = [(idx, row.to_dict()) for idx, row in df.iterrows()]
    
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Starting parallel landmark extraction with {num_workers} workers...")
    
    success = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_process) as executor:
        # Wrap execution mapping with tqdm to show real-time progress
        for idx, landmarks in tqdm(executor.map(extract_single_pose, tasks), total=len(tasks), desc="Pose Extraction"):
            if landmarks is not None:
                for k, v in landmarks.items():
                    df.at[idx, k] = v
                df.at[idx, 'pose_extracted'] = True
                success += 1
            else:
                failed += 1
                
    print(f"\nExtraction completed. Success: {success}, Failed: {failed}")
    
    # Drop rows that failed pose extraction
    df_extracted = df[df['pose_extracted'] == True].copy()
    df_extracted.drop(columns=['pose_extracted'], inplace=True)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_extracted.to_csv(output_path, index=False)
    print(f"Saved extracted data to {output_path}. Total rows: {len(df_extracted)}")

if __name__ == '__main__':
    main()
