import os
# Matikan log info dan warning TensorFlow agar terminal bersih
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import numpy as np
from ultralytics import YOLO

# ============================================
# LOAD YOLOv8 MODELS
# ============================================
print("🚀 Loading YOLOv8 models...")

# Model 1: Menggunakan model default untuk deteksi manusia
human_model = YOLO("yolov8n.pt")

# Model 2: DIKEMBALIKAN untuk menggunakan file kustom fire_smoke.pt Anda
# Pastikan file ini sudah dipindahkan ke folder 'models' di direktori proyek Anda
fire_model = YOLO("models/fire_smoke.pt")

print("✅ All YOLOv8 models loaded successfully!")

# ============================================
# CAMERA SETUP
# ============================================
camera_index = 0

# Menggunakan CAP_DSHOW khusus Windows agar kamera terbuka lebih cepat dan stabil
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

# Set resolusi kamera ke HD
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ============================================
# CHECK CAMERA
# ============================================
if not cap.isOpened():
    print("❌ ERROR: Camera tidak dapat dibuka!")
    exit()

print("✅ Webcam berhasil dijalankan!")
print("📹 SafeWatch Multi-Object Detection Running...")
print("⌨ Tekan tombol 'ESC' untuk keluar.")

# Perhitungan FPS
prev_time = 0

# ============================================
# MAIN LOOP
# ============================================
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ ERROR: Gagal membaca frame kamera")
        break

    # ========================================
    # 1. DETEKSI MANUSIA (HUMAN DETECTION)
    # ========================================
    human_results = human_model(frame, conf=0.5, verbose=False)

    for result in human_results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = human_model.names[class_id]

            # Filter hanya untuk kelas 'person'
            if class_name == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Visualisasi Manusia: Warna HIJAU
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label teks & background
                label = f"{class_name} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + 150, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Titik Tengah Manusia
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

    # ========================================
    # 2. DETEKSI REAL API & ASAP (FIRE DETECTION)
    # ========================================
    fire_results = fire_model(frame, conf=0.4, verbose=False)

    for result in fire_results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            fire_label = fire_model.names[class_id]
            
            # KODE DIKEMBALIKAN: Membaca kelas asli hasil training Roboflow Anda
            if fire_label in ["fire", "smoke"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Visualisasi Api/Asap: Warna MERAH
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Menampilkan label asli (FIRE / SMOKE) sesuai hasil deteksi model kustom
                label = f"{fire_label.upper()} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + 150, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                print(f"🚨 {fire_label.upper()} DETECTED! | Confidence: {confidence:.2f}")

    # ========================================
    # STATISTIK & OVERLAY TEXT
    # ========================================
    # Hitung FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Tampilkan informasi di layar
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, "SafeWatch AI Hybrid System", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Tampilkan jendela gambar
    cv2.imshow("SafeWatch - Real Time Multi-Detection", frame)

    # Keluar program jika menekan tombol ESC (Key Code: 27)
    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 Program dihentikan oleh pengguna.")
        break

# ============================================
# RELEASE RESOURCE
# ============================================
cap.release()
cv2.destroyAllWindows()
print("👋 SafeWatch program finished.")