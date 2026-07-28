import os
import sys
import io

# Fix standard output encoding for Windows terminal (for emojis)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Suppress Tensorflow warning spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("⏳ Memeriksa dependensi sistem...")
try:
    import cv2
    import numpy as np
    import tensorflow as tf
    import mediapipe as mp
    from ultralytics import YOLO
    from collections import deque
    import time
    import pickle
    from src.advanced_feature_engineering import AdvancedFeatureEngineer
except ImportError as e:
    print(f"\n❌ Error: Dependensi '{e.name}' belum terinstall.")
    print("Silakan jalankan perintah berikut untuk menginstall terlebih dahulu:")
    print("pip install opencv-python tensorflow mediapipe ultralytics numpy requests")
    sys.exit(1)

# Import Audio Distress Detector (optional — graceful fallback if unavailable)
audio_detector = None
try:
    from src.audio_detector import AudioDistressDetector
    audio_detector = AudioDistressDetector()
    print("   ✓ Modul Audio Distress Detector tersedia.")
except ImportError as e:
    print(f"   ⚠️ Audio Detector tidak tersedia (missing: {e.name}). Deteksi suara dinonaktifkan.")

# ====================================
# CONFIGURATION
# ====================================
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FALL_THRESHOLD = 0.75      
WARNING_THRESHOLD = 0.45   
ALERT_COOLDOWN = 10        

LANDMARK_PILIHAN = [11, 12, 23, 24, 25, 26, 27, 28]
pred_buffer = deque(maxlen=5)

# ====================================
# PATH RESOLVING
# ====================================
MODEL_ENHANCED_PATH = "models/safewatch_fall_model_enhanced.keras"
MODEL_BASE_PATH = "models/safewatch_fall_model.keras"
SCALER_PATH = "models/feature_scaler.pkl"
YOLO_HUMAN_PATH = "models/yolov8n.pt"
YOLO_FIRE_PATH = "models/fire_smoke.pt"

# ====================================
# LOAD MODELS
# ====================================
print("\n⏳ Memuat model kecerdasan buatan ke memori...")

# 1. Load Fall Model (Coba model Enhanced dahulu, jika gagal pakai model Base)
fall_model_path = MODEL_ENHANCED_PATH if os.path.exists(MODEL_ENHANCED_PATH) else MODEL_BASE_PATH
if not os.path.exists(fall_model_path):
    print(f"❌ Error: Model klasifikasi jatuh tidak ditemukan di 'models/'")
    sys.exit(1)

print(f"   ✓ Memuat Fall Model dari: {fall_model_path}")
try:
    # Memuat model Keras
    fall_model = tf.keras.models.load_model(fall_model_path, compile=False)
except Exception as e:
    print(f"❌ Gagal memuat model Keras: {e}")
    sys.exit(1)

# Load Scaler
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"   ✓ Memuat Scaler dari: {SCALER_PATH}")
    except Exception as e:
        print(f"⚠️ Peringatan: Gagal memuat scaler dari {SCALER_PATH}: {e}")

# 2. Load YOLO Human Model
if not os.path.exists(YOLO_HUMAN_PATH):
    print(f"⚠️ Peringatan: {YOLO_HUMAN_PATH} tidak ditemukan, mencoba mendownload...")
human_model = YOLO(YOLO_HUMAN_PATH)

# 3. Load YOLO Fire & Smoke Model
fire_model = None
if os.path.exists(YOLO_FIRE_PATH):
    print(f"   ✓ Memuat Fire/Smoke Model dari: {YOLO_FIRE_PATH}")
    fire_model = YOLO(YOLO_FIRE_PATH)
else:
    print(f"   ⚠️ Peringatan: {YOLO_FIRE_PATH} tidak ditemukan. Deteksi api/asap akan dinonaktifkan.")

# 4. Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

print("✅ Seluruh model AI berhasil dimuat!")

def log_incident_alert(message):
    """Mencatat alert insiden ke konsol (pengganti Telegram)"""
    print(f"📡 INCIDENT ALERT LOGGED: {message}")

# ====================================
# LIVE CAMERA PIPELINE
# ====================================
def run_live_monitor(camera_index=0):
    # Akses kamera laptop
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ Gagal mengakses kamera dengan index {camera_index}. Coba ganti index kamera (misal: 1 atau 2).")
        return
        
    print(f"\n🎥 Kamera berhasil dibuka. Tekan 'ESC' pada jendela kamera untuk keluar.")
    print("SafeWatch sedang memantau...")
    
    last_fall_alert = 0
    last_fire_alert = 0
    audio_alert_display = None    # Current audio alert to render on screen
    audio_alert_display_time = 0  # When the audio alert was first displayed
    AUDIO_ALERT_DISPLAY_DURATION = 4.0  # Seconds to show audio alert overlay

    # Start audio detector thread
    if audio_detector is not None:
        audio_detector.start()
        # Give the thread a moment to initialize
        time.sleep(0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Gagal membaca frame dari webcam.")
            break

        # Resize dan persiapan gambar
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. MediaPipe Pose Landmark Extraction
        pose_result = pose.process(rgb_frame)
        nn_prediction = 0.0
        skeleton_success = False

        if pose_result.pose_landmarks:
            # Gambar skeleton di frame
            mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Ekstrak 8 landmark pilihan (X, Y)
            keypoints = []
            for idx in LANDMARK_PILIHAN:
                lm = pose_result.pose_landmarks.landmark[idx]
                keypoints.extend([lm.x, lm.y]) 

            if len(keypoints) == 16:
                input_data = np.array(keypoints, dtype=np.float32)
                
                # Deteksi dimensi input model secara dinamis
                model_input_dim = fall_model.input_shape[1]
                
                if model_input_dim > 16:
                    # Ekstrak fitur geometri tambahan
                    geom_features = AdvancedFeatureEngineer.extract_geometric_features(input_data)
                    engineered_vector = np.array(list(geom_features.values()), dtype=np.float32)
                    
                    if model_input_dim == len(engineered_vector):
                        input_data = engineered_vector
                    else:
                        input_data = np.concatenate([input_data, engineered_vector])
                
                # Reshape untuk input model (1, num_features)
                input_data = input_data.reshape(1, -1)
                
                # Lakukan normalisasi jika scaler terload
                if scaler is not None:
                    try:
                        input_data = scaler.transform(input_data)
                    except Exception as e:
                        pass
                
                # Jalankan prediksi
                nn_prediction = float(fall_model.predict(input_data, verbose=0)[0][0])
                skeleton_success = True

        # 2. YOLO Human Detection & Fusion Logic
        human_results = human_model(frame, verbose=False)
        
        for result in human_results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # Kelas 0 = Person di YOLO
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Hitung aspect ratio dari bounding box untuk fusion logic
                        box_w = x2 - x1
                        box_h = y2 - y1
                        rasio_box = box_w / (box_h + 1e-6)
                        
                        # Fusion logic: perpaduan antara pose skeleton dan geometri bounding box
                        if skeleton_success:
                            # Jika pose MediaPipe terdeteksi, gunakan hasil Keras Neural Network (95.81% akurat)
                            final_score = nn_prediction
                        else:
                            # Jika pose MediaPipe gagal dideteksi, gunakan rasio kotak sebagai fallback kasar
                            if rasio_box >= 1.25:  # Diperketat dari 1.05 menjadi 1.25 agar tidak mudah false alarm
                                final_score = 0.85
                            elif 0.85 <= rasio_box < 1.25:
                                final_score = 0.45
                            else:
                                final_score = 0.0

                        pred_buffer.append(final_score)
                        smooth_score = sum(pred_buffer) / len(pred_buffer)

                        # Tentukan label status berdasarkan skor threshold
                        if smooth_score >= FALL_THRESHOLD:
                            label, color = "FALL DETECTED! (BAHAYA)", (0, 0, 255) 
                        elif WARNING_THRESHOLD <= smooth_score < FALL_THRESHOLD:
                            label, color = "WARNING: LOSING BALANCE", (0, 165, 255) 
                        else:
                            label, color = "NORMAL (AMAN)", (0, 255, 0) 

                        # Gambar box dan label di layar
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} ({smooth_score:.2f})", (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # Kirim alert jika terjadi jatuh (dengan cooldown)
                        current_time = time.time()
                        if smooth_score >= FALL_THRESHOLD and (current_time - last_fall_alert > ALERT_COOLDOWN):
                            pesan_darurat = f"🚨 DARURAT: SafeWatch mendeteksi orang jatuh! (Confidence: {smooth_score*100:.1f}%)"
                            log_incident_alert(pesan_darurat)
                            last_fall_alert = current_time
        
        # 3. YOLO Fire & Smoke Detection (Jika model terload)
        if fire_model is not None:
            fire_results = fire_model(frame, verbose=False)
            for result in fire_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        fconf = float(box.conf[0])
                        fire_label = fire_model.names[cls_id].lower()
                        
                        if fconf > 0.65 and (fire_label == 'fire' or fire_label == 'smoke'): 
                            fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                            
                            # Gambar box api/asap
                            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 69, 255), 2)
                            cv2.putText(frame, f"HAZARD: {fire_label.upper()} ({fconf:.2f})", (fx1, max(20, fy1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 69, 255), 2)
                            
                            # Kirim alert bahaya api
                            current_time = time.time()
                            if current_time - last_fire_alert > ALERT_COOLDOWN:
                                pesan_api = f"🔥 BAHAYA: Terdeteksi indikasi {fire_label.upper()}! (Confidence: {fconf*100:.1f}%)"
                                log_incident_alert(pesan_api)
                                last_fire_alert = current_time

        # 4. Audio Distress Detection (from background thread)
        if audio_detector is not None:
            alert = audio_detector.get_alert()
            if alert is not None:
                audio_alert_display = alert
                audio_alert_display_time = time.time()
                # Log to console
                if alert['type'] == 'keyword':
                    pesan_suara = f"🔊 DARURAT SUARA: {alert['detail']}"
                else:
                    pesan_suara = f"🔊 TERIAKAN: {alert['detail']}"
                log_incident_alert(pesan_suara)

        # Render audio alert overlay on frame
        if audio_alert_display is not None:
            elapsed = time.time() - audio_alert_display_time
            if elapsed < AUDIO_ALERT_DISPLAY_DURATION:
                # Semi-transparent red banner at the top
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 70), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Alert text
                if audio_alert_display['type'] == 'keyword':
                    alert_text = f"DISTRESS VOICE: {audio_alert_display['detail']}"
                else:
                    alert_text = f"SCREAM DETECTED: {audio_alert_display['detail']}"
                cv2.putText(frame, alert_text, (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "EMERGENCY AUDIO ALERT", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            else:
                audio_alert_display = None

        # Mic status indicator (bottom-left corner)
        if audio_detector is not None:
            if audio_detector.is_ready:
                mic_color = (0, 255, 0)  # Green = active
                mic_text = "MIC: Active"
            elif audio_detector.error_message:
                mic_color = (0, 0, 255)  # Red = error
                mic_text = "MIC: Error"
            else:
                mic_color = (0, 165, 255)  # Orange = loading
                mic_text = "MIC: Loading..."
            cv2.circle(frame, (20, FRAME_HEIGHT - 20), 6, mic_color, -1)
            cv2.putText(frame, mic_text, (32, FRAME_HEIGHT - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, mic_color, 1)

        # Tampilkan Window Preview
        cv2.imshow("SafeWatch Live Monitor (Webcam)", frame)
        
        # Keluar jika tombol ESC ditekan
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Cleanup
    if audio_detector is not None:
        audio_detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Kamera dan mikrofon ditutup. Sistem dinonaktifkan dengan aman.")

if __name__ == "__main__":
    run_live_monitor(0)
