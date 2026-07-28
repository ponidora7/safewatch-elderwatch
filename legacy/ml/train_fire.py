import os
# Tambahkan baris ini untuk manajemen memori Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

def main():
    # 1. Inisialisasi model dasar (YOLOv8 Nano)
    # Kita gunakan yolov8n.pt sebagai arsitektur dasar (pre-trained weights)
    print("🚀 Memuat arsitektur dasar YOLOv8n...")
    model = YOLO("yolov8n.pt")

    # Path menuju file data.yaml milik dataset fire_smoke_detection
    yaml_path = "data/raw/fire_smoke_detection/data.yaml"

    # 2. Mulai Proses Pelatihan (Training)
    print("🏋️‍♂️ Memulai proses training untuk mendeteksi Api & Asap...")
    
    model.train(
        data=yaml_path,       
        epochs=50,            
        imgsz=640,            
        batch=4,              # <<< TURUNKAN ke 4 atau 2 (Defaultnya 16 terlalu berat untuk RAM)
        device="cpu",         
        workers=0,            # <<< GANTI ke 0 untuk mematikan multiprocessing yang rawan crash di Windows
        name="safewatch_fire" 
    )
    
    print("✅ Training selesai!")
    print("Model terbaik Anda disimpan di: runs/detect/safewatch_fire/weights/best.pt")

if __name__ == "__main__":
    main()