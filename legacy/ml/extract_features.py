import os
import cv2
import csv
import glob
import mediapipe as mp

# Inisialisasi MediaPipe Pose (Mode Foto Statis)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# SARAN 2: Hanya gunakan 8 landmark utama (Bahu - Kaki)
LANDMARK_PILIHAN = [11, 12, 23, 24, 25, 26, 27, 28] 

# ==========================================
# ⚠️ PENGATURAN KELAS (MAPPING KELAS YOLO)
# Kita meleburkan 4 kelas YOLO menjadi 2 kelas AI Biner
MAPPING_KELAS = {
    0: 0, # Kelas YOLO 0 (Normal) -> Jadi Label 0 di AI kita
    1: 0, # Kelas YOLO 1 (Normal) -> Jadi Label 0 di AI kita
    2: 0, # Kelas YOLO 2 (Normal) -> Jadi Label 0 di AI kita
    3: 1  # Kelas YOLO 3 (Jatuh) -> Jadi Label 1 di AI kita (BAHAYA!)
}
# ==========================================

def proses_gambar_yolo(image_path, label_path, output_csv):
    # Baca gambar
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    H, W = img.shape[:2]
    nama_file_foto = os.path.basename(image_path)
    
    # Baca file teks YOLO
    if not os.path.exists(label_path):
        return 0
        
    sukses_ekstrak = 0
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    with open(output_csv, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        for line in lines:
            data = line.strip().split()
            if len(data) < 5: continue
                
            class_id = int(data[0])
            
            # Lewati jika ID kelas tidak ada di mapping kita
            if class_id not in MAPPING_KELAS:
                continue
                
            label_final_ai = MAPPING_KELAS[class_id]
            
            # Ekstrak kordinat YOLO (x_center, y_center, width, height)
            x_c, y_c, w, h = map(float, data[1:5])
            
            # SARAN 1: Bounding Box & Expand
            # Konversi format YOLO ke pixel gambar
            box_w, box_h = int(w * W), int(h * H)
            x_center, y_center = int(x_c * W), int(y_c * H)
            
            x_min = x_center - (box_w // 2)
            y_min = y_center - (box_h // 2)
            x_max = x_center + (box_w // 2)
            y_max = y_center + (box_h // 2)
            
            # EXPAND (Perlebar kotak 15% agar tangan/kaki tidak terpotong saat jatuh)
            pad_x = int(box_w * 0.15)
            pad_y = int(box_h * 0.15)
            
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(W, x_max + pad_x)
            y_max = min(H, y_max + pad_y)
            
            # CROP GAMBAR (Potong hanya area manusia)
            crop_img = img[y_min:y_max, x_min:x_max]
            if crop_img.size == 0: continue
                
            # Proses Crop Image menggunakan MediaPipe
            image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            hasil = pose.process(image_rgb)
            
            if hasil.pose_landmarks:
                baris_data = [f"{nama_file_foto}_obj{sukses_ekstrak}"]
                
                # Ekstrak 16 koordinat murni X dan Y dari kotak yang di-crop
                for idx in LANDMARK_PILIHAN:
                    landmark = hasil.pose_landmarks.landmark[idx]
                    baris_data.extend([landmark.x, landmark.y]) # Lupakan Z
                    
                baris_data.append(label_final_ai)
                writer.writerow(baris_data)
                sukses_ekstrak += 1
                
    return sukses_ekstrak

if __name__ == "__main__":
    folder_dataset = "dataset"
    nama_file_csv = os.path.join(folder_dataset, "dataset_fall.csv")
    os.makedirs(folder_dataset, exist_ok=True)
    
    # Hapus CSV lama jika ada
    if os.path.exists(nama_file_csv):
        os.remove(nama_file_csv)
        
    # Buat Header (16 Fitur X,Y)
    with open(nama_file_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['Image_ID']
        for idx in LANDMARK_PILIHAN:
            header.extend([f'X{idx}', f'Y{idx}'])
        header.append('Label')
        writer.writerow(header)

    print("=== MEMULAI EKSTRAKSI DATA DENGAN BOUNDING BOX ===")
    
    # Folder data YOLO kamu
    jalur_base = os.path.join("data", "raw", "human_fall")
    folder_splits = ["train", "valid"]
    
    total_sukses = 0
    
    for split in folder_splits:
        folder_images = os.path.join(jalur_base, split, "images")
        folder_labels = os.path.join(jalur_base, split, "labels")
        
        if not os.path.exists(folder_images):
            print(f"[SKIP] Folder {folder_images} tidak ditemukan.")
            continue
            
        daftar_gambar = glob.glob(os.path.join(folder_images, "*.*"))
        print(f"Mengekstrak {len(daftar_gambar)} foto dari folder '{split}'...")
        
        for img_path in daftar_gambar:
            # Cari nama file .txt yang cocok dengan gambar
            nama_file = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(folder_labels, f"{nama_file}.txt")
            
            hasil = proses_gambar_yolo(img_path, label_path, nama_file_csv)
            total_sukses += hasil
            
    print("\n=== SELURUH PROSES EKSTRAKSI SELESAI ===")
    print(f"Berhasil mengekstrak {total_sukses} objek manusia ke dalam CSV!")
    print(f"File tersimpan di: {nama_file_csv}")