"""
src/data_loader.py
==================
Parser untuk format anotasi YOLO (.txt).
Menghasilkan pandas DataFrame standar untuk dieksekusi oleh pipeline selanjutnya.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Optional, Dict

import cv2
import pandas as pd
from tqdm import tqdm

# ==========================================
# ─── YOLO Loader Class ────────────────────
# ==========================================

class YOLOLoader:
    """
    Mesin pembaca anotasi berformat YOLOv8.
    Mengekstrak informasi ID Kelas, Bounding Box (cx, cy, w, h), 
    dan secara aman memverifikasi keberadaan serta dimensi gambar aslinya.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.class_mapping = self._load_class_mapping()

    def _load_class_mapping(self) -> Dict[int, str]:
        """Secara otomatis mendeteksi nama kelas dari file data.yaml milik YOLO."""
        yaml_path = self.dataset_path / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    data_yaml = yaml.safe_load(f)
                
                # Menangani format 'names' yang berupa List maupun Dictionary
                names = data_yaml.get('names')
                if isinstance(names, list):
                    return {i: name for i, name in enumerate(names)}
                elif isinstance(names, dict):
                    return names
            except Exception as e:
                print(f"⚠️ Peringatan: Gagal membaca data.yaml di {self.dataset_path.name} -> {e}")
                
        return {} # Jika tidak ada yaml, kembalikan kamus kosong

    def _find_image(self, images_dir: Path, stem: str) -> tuple[Optional[Path], int, int]:
        """Cari file gambar dan kembalikan jalur serta dimensinya (Lebar, Tinggi) tanpa membebani RAM."""
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            img_path = images_dir / f"{stem}{ext}"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    del img  # Sangat krusial untuk mencegah Memory Leak saat memproses 10.000+ gambar
                    return img_path, w, h
        return None, 0, 0

    def load_split(self, split: str) -> pd.DataFrame:
        """Memuat data spesifik untuk folder 'train', 'valid', atau 'test'."""
        images_dir = self.dataset_path / split / "images"
        labels_dir = self.dataset_path / split / "labels"

        if not labels_dir.exists():
            return pd.DataFrame()

        records: list[dict] = []
        label_files = sorted(labels_dir.glob("*.txt"))

        for lbl_path in tqdm(label_files, desc=f"Memuat {self.dataset_path.name}/{split}"):
            img_path, img_w, img_h = self._find_image(images_dir, lbl_path.stem)
            img_exists = img_path is not None

            with open(lbl_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Jika file TXT kosong (Background / Tidak ada deteksi)
            if not lines:
                records.append(self._create_empty_record(split, lbl_path, img_path, img_w, img_h, img_exists))
                continue

            for line_no, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) < 5:
                    continue  # Abaikan baris yang cacat strukturnya

                try:
                    class_id = int(parts[0])
                    records.append({
                        "dataset": self.dataset_path.name,
                        "split": split,
                        "image_id": lbl_path.stem,
                        "image_path": str(img_path) if img_path else None,
                        "label_path": str(lbl_path),
                        "img_exists": img_exists,
                        "img_width": img_w,
                        "img_height": img_h,
                        "line_no": line_no,
                        "class_id": class_id,
                        "class_name": self.class_mapping.get(class_id, f"class_{class_id}"),
                        "bbox_x_center": float(parts[1]),
                        "bbox_y_center": float(parts[2]),
                        "bbox_width": float(parts[3]),
                        "bbox_height": float(parts[4]),
                        "format": "yolo",
                    })
                except (ValueError, IndexError):
                    continue
                    
        return pd.DataFrame(records)

    def _create_empty_record(self, split: str, lbl_path: Path, img_path: Optional[Path], img_w: int, img_h: int, img_exists: bool) -> dict:
        """Helper untuk membuat record Background Image (Tidak ada objek)."""
        return {
            "dataset": self.dataset_path.name,
            "split": split,
            "image_id": lbl_path.stem,
            "image_path": str(img_path) if img_path else None,
            "label_path": str(lbl_path),
            "img_exists": img_exists,
            "img_width": img_w,
            "img_height": img_h,
            "line_no": 0,
            "class_id": None,
            "class_name": "background",
            "bbox_x_center": None,
            "bbox_y_center": None,
            "bbox_width": None,
            "bbox_height": None,
            "format": "yolo",
        }


# ==========================================
# ─── Global Dataset Loader Function ───────
# ==========================================

def load_all_datasets(datasets_config: Dict) -> pd.DataFrame:
    """
    Fungsi utama untuk menyapu bersih dan menggabungkan seluruh dataset 
    yang terdaftar di file config/paths.py.
    """
    all_dfs = []
    
    for name, config in datasets_config.items():
        print(f"\n📥 Inisialisasi Loader untuk: {name.upper()}")
        base_path = Path(config['path'])
        ann_type = config.get('annotation_type')
        
        if ann_type != 'yolo':
            print(f"  [SKIPPED] {name} bukan format YOLO. Mengabaikan...")
            continue
            
        loader = YOLOLoader(dataset_path=base_path)
        
        for split in ['train', 'valid', 'test']:
            df_split = loader.load_split(split)
            if not df_split.empty:
                all_dfs.append(df_split)
                print(f"  ✓ Berhasil memuat {len(df_split):,} baris dari folder {split}.")
                
    if not all_dfs:
        print("\n❌ GAGAL: Tidak ada satupun dataset yang berhasil diproses.")
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    total_unik = combined_df['image_id'].nunique()
    print(f"\n✅ PROSES SELESAI: Menghasilkan {len(combined_df):,} kotak anotasi (BBox) dari {total_unik:,} gambar unik.")
    
    return combined_df