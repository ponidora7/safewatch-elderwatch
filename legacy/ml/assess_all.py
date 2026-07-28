#!/usr/bin/env python
# scripts/assess_all.py
"""
Jalankan asesmen kualitas untuk ketiga dataset SafeWatch sekaligus.

Cara pakai:
    python scripts/assess_all.py

Output:
    data/processed/assessed_fall.csv
    data/processed/assessed_fire_smoke.csv
    data/processed/assessed_person.csv
    data/processed/combined_summary.csv
"""

import sys
import os

# Tambahkan root project ke sys.path agar import config dan src bisa berjalan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.paths import (
    RAW_FALL_DIR,
    RAW_FIRE_SMOKE_DIR,
    RAW_PERSON_DIR,
    PROCESSED_DIR,
)
from src.data_assessment import assess_all_datasets


def check_folders():
    """Pastikan folder dataset sudah ada sebelum mulai."""
    missing = []
    for name, path in [
        ("human_fall",          RAW_FALL_DIR),
        ("fire_smoke_detection", RAW_FIRE_SMOKE_DIR),
        ("person_detection",    RAW_PERSON_DIR),
    ]:
        if not os.path.exists(path):
            missing.append((name, path))

    if missing:
        print("\n⚠️  FOLDER DATASET TIDAK DITEMUKAN:")
        for name, path in missing:
            print(f"   {name:25s} → {path}")
        print("\nPastikan kamu sudah:")
        print("  1. Download ZIP dataset dari Roboflow (format YOLOv8)")
        print("  2. Ekstrak ke folder yang sesuai:")
        print(f"     falling-udxfz-3.zip    → {RAW_FALL_DIR}")
        print(f"     fire-smoke-lwjte-1.zip → {RAW_FIRE_SMOKE_DIR}")
        print(f"     person-vcnmd-2.zip     → {RAW_PERSON_DIR}")
        print("\nLanjutkan hanya dengan dataset yang tersedia? (y/n): ", end="")
        ans = input().strip().lower()
        if ans != "y":
            print("Dibatalkan.")
            sys.exit(0)


def main():
    print("\n" + "=" * 60)
    print("  SafeWatch — Data Assessment Pipeline")
    print("  Dataset: Fall | Fire/Smoke | Person")
    print("=" * 60)

    check_folders()

    results = assess_all_datasets(
        raw_fall_dir       = RAW_FALL_DIR,
        raw_fire_smoke_dir = RAW_FIRE_SMOKE_DIR,
        raw_person_dir     = RAW_PERSON_DIR,
        processed_dir      = PROCESSED_DIR,
        verbose            = True,
    )

    print("\n" + "=" * 60)
    print("  ✅ Assessment selesai!")
    print(f"  Dataset berhasil diproses: {len([k for k in results if k != 'combined'])}")
    print(f"  Output tersimpan di      : {PROCESSED_DIR}/")
    print("\n  Langkah berikutnya:")
    print("    python scripts/clean_all.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()