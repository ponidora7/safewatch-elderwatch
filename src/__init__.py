"""
Module untuk loading dan parsing annotations dataset Computer Vision.
Fokus Utama: Mengekstrak metadata Bounding Box dari format YOLOv8 (TXT).
"""
import os
import cv2
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

def load_yolo_annotations(
    dataset_path: Path, 
    split: str = "train",
    class_mapping: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Parse YOLOv8 .txt annotations ke dalam DataFrame terstruktur.
    Mengekstrak kordinat Bounding Box, dimensi gambar, dan keberadaan file.
    """
    # 1. Parsing File Konfigurasi Kelas (data.yaml)
    yaml_path = dataset_path / "data.yaml"
    if class_mapping is None and yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
        # Jika names berupa list, ubah jadi dictionary {0: 'name', 1: 'name'}
        if isinstance(data_yaml.get('names'), list):
            class_mapping = {i: name for i, name in enumerate(data_yaml['names'])}
        else:
            class_mapping = data_yaml.get('names', {})
    
    img_dir = dataset_path / split / "images"
    label_dir = dataset_path / split / "labels"
    
    if not label_dir.exists():
        print(f"  [SKIPPED] Folder label tidak ditemukan: {label_dir}")
        return pd.DataFrame()
    
    records = []
    label_files = list(label_dir.glob("*.txt"))
    
    for label_file in label_files:
        img_name_stem = label_file.stem
        img_path = None
        img_w, img_h = 0, 0
        img_exists = False
        
        # 2. Sinkronisasi File Teks YOLO dengan File Gambar
        # Menyapu ekstensi yang umum digunakan
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            potential_img = img_dir / f"{img_name_stem}{ext}"
            if potential_img.exists():
                img_path = potential_img
                img_exists = True
                
                # Baca dimensi gambar dengan aman tanpa menyimpan seluruh pixel ke memori
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_h, img_w = img.shape[:2]
                    del img # Bebaskan memori segera setelah baca dimensi
                break
        
        # 3. Baca Isi Bounding Box YOLO
        with open(label_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        # Jika file teks kosong (tidak ada objek di gambar)
        if not lines:
            records.append({
                "dataset": dataset_path.name,
                "split": split,
                "image_path": str(img_path) if img_path else None,
                "label_path": str(label_file),
                "img_exists": img_exists,
                "img_width": img_w,
                "img_height": img_h,
                "class_id": None,
                "class_name": "background", # Objek tidak terdeteksi
                "bbox_x_center": None,
                "bbox_y_center": None,
                "bbox_width": None,
                "bbox_height": None,
                "annotation_count": 0
            })
            continue
        
        # 4. Parsing Koordinat
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue # Hindari error jika baris cacat
            
            try:
                cid = int(parts[0])
                nama_kelas = class_mapping.get(cid, f"class_{cid}") if class_mapping else f"class_{cid}"
                
                records.append({
                    "dataset": dataset_path.name,
                    "split": split,
                    "image_path": str(img_path) if img_path else None,
                    "label_path": str(label_file),
                    "img_exists": img_exists,
                    "img_width": img_w,
                    "img_height": img_h,
                    "class_id": cid,
                    "class_name": nama_kelas,
                    "bbox_x_center": float(parts[1]),
                    "bbox_y_center": float(parts[2]),
                    "bbox_width": float(parts[3]),
                    "bbox_height": float(parts[4]),
                    "annotation_count": len(lines)
                })
            except (ValueError, IndexError) as e:
                print(f"⚠️ Error parsing BBox pada {label_file.name}: {e}")
                continue
    
    return pd.DataFrame(records)


def load_all_datasets(datasets_config: Dict) -> pd.DataFrame:
    """
    Menyapu bersih dan menggabungkan seluruh data dari berbagai folder dataset.
    Berdasarkan dictionary DATASETS di config/paths.py.
    """
    all_dfs = []
    
    for name, config in datasets_config.items():
        print(f"\n📥 Memuat dataset: {name.upper()}...")
        path = Path(config['path'])
        
        # Pastikan kita hanya memproses dataset format YOLO sesuai revisi terakhir
        if config.get('annotation_type') != 'yolo':
            print(f"  [SKIPPED] Dataset {name} bukan format YOLO. Mengabaikan...")
            continue
        
        # Mengecek data train, valid, dan (jika ada) test
        for split in ['train', 'valid', 'test']:
            df_split = load_yolo_annotations(path, split)
            
            if not df_split.empty:
                all_dfs.append(df_split)
                print(f"  ✓ {split.capitalize():<6}: {len(df_split):,} Bounding Box diparsing.")
    
    if not all_dfs:
        print("\n❌ GAGAL: Tidak ada satupun data yang berhasil di-load. Cek struktur folder data/raw/ !")
        return pd.DataFrame()
    
    # Gabungkan semua data menjadi satu DataFrame raksasa
    combined = pd.concat(all_dfs, ignore_index=True)
    
    total_gambar_unik = combined['image_path'].nunique()
    print(f"\n✅ TOTAL: {len(combined):,} kotak anotasi dari {total_gambar_unik:,} gambar unik.")
    return combined