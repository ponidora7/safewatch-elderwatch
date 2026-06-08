# src/data_assessment.py
"""
Modul generik untuk menilai kualitas dataset YOLO.
Bisa dipakai untuk dataset fall, fire/smoke, maupun person.
"""

import os
import glob
import yaml
import cv2
import pandas as pd
from typing import Optional


def assess_dataset(
    dataset_dir: str,
    dataset_name: str,
    splits: Optional[list] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Membaca semua anotasi YOLO dari folder dataset dan mengembalikan DataFrame.

    Parameters
    ----------
    dataset_dir  : str  — path ke folder dataset (harus punya data.yaml)
    dataset_name : str  — nama pengenal dataset ('fall', 'fire_smoke', 'person')
    splits       : list — daftar split yang dibaca, default ['train','valid','test']
    verbose      : bool — cetak ringkasan ke terminal

    Returns
    -------
    pd.DataFrame dengan kolom:
        dataset, split, image_id, image_path, label_path,
        img_exists, img_width, img_height,
        class_id, class_name,
        bbox_x_center, bbox_y_center, bbox_width, bbox_height,
        has_annotation
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    # Baca daftar kelas dari data.yaml
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"data.yaml tidak ditemukan di: {dataset_dir}\n"
            "Pastikan dataset sudah diekstrak dengan benar."
        )
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    class_names: list = cfg.get("names", [])

    if verbose:
        print(f"\n{'='*50}")
        print(f"  ASSESSING: {dataset_name.upper()}")
        print(f"{'='*50}")
        print(f"  Folder  : {dataset_dir}")
        print(f"  Kelas   : {class_names}")

    records = []

    for split in splits:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")

        if not os.path.exists(lbl_dir):
            if verbose:
                print(f"  [SKIP] Split '{split}' tidak ditemukan.")
            continue

        label_files = glob.glob(os.path.join(lbl_dir, "*.txt"))

        for lbl_path in label_files:
            base = os.path.splitext(os.path.basename(lbl_path))[0]

            # Cari file gambar (jpg atau png)
            img_path = os.path.join(img_dir, base + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, base + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, base + ".jpeg")

            img_exists = os.path.exists(img_path)
            img_width = img_height = None

            if img_exists:
                img = cv2.imread(img_path)
                if img is not None:
                    img_height, img_width = img.shape[:2]

            # Baca isi label
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            # Gambar tanpa anotasi — tetap dicatat agar terdeteksi saat assessing
            if not lines:
                records.append({
                    "dataset": dataset_name,
                    "split": split,
                    "image_id": base,
                    "image_path": img_path,
                    "label_path": lbl_path,
                    "img_exists": img_exists,
                    "img_width": img_width,
                    "img_height": img_height,
                    "class_id": None,
                    "class_name": None,
                    "bbox_x_center": None,
                    "bbox_y_center": None,
                    "bbox_width": None,
                    "bbox_height": None,
                    "has_annotation": False,
                })
                continue

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue  # skip baris malformat

                cid = int(parts[0])
                cname = class_names[cid] if cid < len(class_names) else f"unknown_{cid}"

                records.append({
                    "dataset": dataset_name,
                    "split": split,
                    "image_id": base,
                    "image_path": img_path,
                    "label_path": lbl_path,
                    "img_exists": img_exists,
                    "img_width": img_width,
                    "img_height": img_height,
                    "class_id": cid,
                    "class_name": cname,
                    "bbox_x_center": float(parts[1]),
                    "bbox_y_center": float(parts[2]),
                    "bbox_width":    float(parts[3]),
                    "bbox_height":   float(parts[4]),
                    "has_annotation": True,
                })

    df = pd.DataFrame(records)

    if verbose and len(df) > 0:
        bbox_cols = ["bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height"]
        df_ann = df[df["has_annotation"]]
        oob_mask = (
            (df_ann["bbox_x_center"] < 0) | (df_ann["bbox_x_center"] > 1) |
            (df_ann["bbox_y_center"] < 0) | (df_ann["bbox_y_center"] > 1) |
            (df_ann["bbox_width"]    < 0) | (df_ann["bbox_width"]    > 1) |
            (df_ann["bbox_height"]   < 0) | (df_ann["bbox_height"]   > 1)
        )

        print(f"\n  [HASIL ASSESSING]")
        print(f"  Total anotasi     : {len(df):,}")
        print(f"  Gambar unik       : {df['image_id'].nunique():,}")
        print(f"  Gambar hilang     : {(~df['img_exists']).sum():,}")
        print(f"  Tanpa anotasi     : {(~df['has_annotation']).sum():,}")
        print(f"  Bbox NaN          : {df[bbox_cols].isna().sum().sum():,}")
        print(f"  Bbox out-of-range : {oob_mask.sum():,}")
        print(f"  Duplikat baris    : {df.duplicated().sum():,}")
        print(f"\n  Distribusi split:")
        for split, cnt in df["split"].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"    {split:8s}: {cnt:6,} ({pct:.1f}%)")
        print(f"\n  Distribusi kelas:")
        for cls, cnt in df["class_name"].value_counts().items():
            print(f"    {str(cls):20s}: {cnt:6,}")

    return df


def assess_all_datasets(
    raw_fall_dir: str,
    raw_fire_smoke_dir: str,
    raw_person_dir: str,
    processed_dir: str,
    verbose: bool = True,
) -> dict:
    """
    Menjalankan assess_dataset untuk ketiga dataset sekaligus
    dan menyimpan hasilnya ke folder processed.

    Returns dict: {"fall": df, "fire_smoke": df, "person": df, "combined": df}
    """
    os.makedirs(processed_dir, exist_ok=True)

    results = {}

    configs = [
        (raw_fall_dir,       "fall",       os.path.join(processed_dir, "assessed_fall.csv")),
        (raw_fire_smoke_dir, "fire_smoke", os.path.join(processed_dir, "assessed_fire_smoke.csv")),
        (raw_person_dir,     "person",     os.path.join(processed_dir, "assessed_person.csv")),
    ]

    dfs = []
    for dir_path, name, out_csv in configs:
        if not os.path.exists(dir_path):
            print(f"\n[WARNING] Folder tidak ditemukan: {dir_path}")
            print(f"  Pastikan dataset '{name}' sudah diekstrak ke folder tersebut.")
            continue

        df = assess_dataset(dir_path, name, verbose=verbose)
        df.to_csv(out_csv, index=False)
        if verbose:
            print(f"\n  💾 Tersimpan: {out_csv}")
        results[name] = df
        dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined_path = os.path.join(processed_dir, "combined_summary.csv")
        combined.to_csv(combined_path, index=False)
        results["combined"] = combined

        if verbose:
            print(f"\n{'='*50}")
            print(f"  RINGKASAN 3 DATASET")
            print(f"{'='*50}")
            summary = combined.groupby(["dataset", "split"]).size().unstack(fill_value=0)
            print(summary.to_string())
            print(f"\n  Total keseluruhan: {len(combined):,} anotasi")
            print(f"  💾 Combined tersimpan: {combined_path}")

    return results