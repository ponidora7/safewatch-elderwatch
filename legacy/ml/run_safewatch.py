import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import mediapipe as mp
from collections import deque

# ====================================
# CONFIGURATION
# ====================================
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FALL_THRESHOLD = 0.65      
WARNING_THRESHOLD = 0.35   
ALERT_COOLDOWN = 10        

LANDMARK_PILIHAN = [11, 12, 23, 24, 25, 26, 27, 28]
pred_buffer = deque(maxlen=5) 

# Membuat folder "real_footage" secara otomatis jika belum ada
FOOTAGE_DIR = "real_footage"
os.makedirs(FOOTAGE_DIR, exist_ok=True)

# ====================================
# LOAD MODELS
# ====================================
print("⏳ Memuat seluruh arsitektur kecerdasan buatan...")
human_model = YOLO("models/yolov8n.pt")
fire_model = YOLO("models/fire_smoke.pt") 
fall_model = tf.keras.models.load_model("models/safewatch_fall_model.keras")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 

print("✅ Seluruh model AI berhasil dimuat ke memori!")

def save_snapshot(frame, label):
    """Menyimpan frame saat ini sebagai gambar JPG di folder lokal"""
    filename = os.path.join(FOOTAGE_DIR, f"{label}_{int(time.time())}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Snapshot tersimpan lokal: {filename}")
    return filename

def log_incident_alert(message, photo_path=None):
    """Mencatat alert insiden ke konsol (pengganti Telegram)"""
    photo_info = f" [Photo: {photo_path}]" if photo_path else ""
    print(f"📡 INCIDENT ALERT LOGGED: {message}{photo_info}")

# ====================================
# MAIN REAL-TIME PIPELINE
# ====================================
def run_safewatch_system(source=0):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    print("🚀 Sistem Pengawasan SafeWatch Berjalan...")
    
    last_fall_alert = 0
    last_fire_alert = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. MEDIAPIPE POSE 
        pose_result = pose.process(rgb_frame)
        nn_prediction = 0.0
        skeleton_success = False

        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            keypoints = []
            for idx in LANDMARK_PILIHAN:
                lm = pose_result.pose_landmarks.landmark[idx]
                keypoints.extend([lm.x, lm.y]) 

            if len(keypoints) == 16:
                input_data = np.array(keypoints, dtype=np.float32).reshape(1, -1)
                nn_prediction = fall_model.predict(input_data, verbose=0)[0][0]
                skeleton_success = True

        # 2. YOLO HUMAN DETECTION & FUSION LOGIC
        human_results = human_model(frame, verbose=False)
        
        for result in human_results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        box_w = x2 - x1
                        box_h = y2 - y1
                        rasio_box = box_w / (box_h + 1e-6)
                        
                        if rasio_box >= 1.05:
                            final_score = max(nn_prediction, 0.92)
                        elif 0.70 <= rasio_box < 1.05:
                            final_score = max(nn_prediction, 0.45)
                        else:
                            final_score = nn_prediction if skeleton_success else 0.0

                        pred_buffer.append(final_score)
                        smooth_score = sum(pred_buffer) / len(pred_buffer)

                        if smooth_score >= FALL_THRESHOLD:
                            label, color = "FALL DETECTED!", (0, 0, 255) 
                        elif WARNING_THRESHOLD <= smooth_score < FALL_THRESHOLD:
                            label, color = "WARNING: LOSING BALANCE", (0, 165, 255) 
                        else:
                            label, color = "NORMAL", (0, 255, 0) 

                        # PERHATIAN: Lukis kotak DULU sebelum mengambil snapshot!
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} ({smooth_score:.2f})", (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # SIMPAN SNAPSHOT DAN ALERT UNTUK DARURAT JATUH
                        current_time = time.time()
                        if smooth_score >= FALL_THRESHOLD and (current_time - last_fall_alert > ALERT_COOLDOWN):
                            img_path = save_snapshot(frame, "FALL")
                            pesan_darurat = f"🚨 DARURAT: Korban terjatuh/terkapar! (Confidence: {smooth_score*100:.1f}%)"
                            log_incident_alert(pesan_darurat, photo_path=img_path)
                            last_fall_alert = current_time
        
        # 3. YOLO FIRE & SMOKE DETECTION
        fire_results = fire_model(frame, verbose=False)
        for result in fire_results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    fconf = float(box.conf[0])
                    fire_label = fire_model.names[cls_id].lower()
                    
                    if fconf > 0.45 and (fire_label == 'fire' or fire_label == 'smoke'): 
                        fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                        
                        # Lukis kotak DULU sebelum mengambil snapshot!
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                        cv2.putText(frame, f"HAZARD: {fire_label.upper()} ({fconf:.2f})", (fx1, max(20, fy1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # SIMPAN SNAPSHOT DAN ALERT UNTUK BAHAYA API
                        current_time = time.time()
                        if current_time - last_fire_alert > ALERT_COOLDOWN:
                            img_path = save_snapshot(frame, "FIRE")
                            pesan_api = f"🔥 BAHAYA: Terdeteksi indikasi {fire_label.upper()} di area pemantauan! (Confidence: {fconf*100:.1f}%)"
                            log_incident_alert(pesan_api, photo_path=img_path)
                            last_fire_alert = current_time

        cv2.imshow("SafeWatch Integrated Security System", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Sistem Dimatikan dengan Aman.")

if __name__ == "__main__":
    run_safewatch_system(0)