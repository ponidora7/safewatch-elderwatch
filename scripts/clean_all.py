#!/usr/bin/env python
# scripts/clean_all.py
"""
Jalankan cleaning + feature engineering untuk ketiga dataset SafeWatch.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.paths import PROCESSED_DIR, REPORTS_DIR
from src.data_cleaner import clean_all_datasets, get_class_distribution_report


def check_assessed_files():
    """Pastikan file hasil assess sudah ada."""
    from config.paths import ASSESSED_FALL_CSV, ASSESSED_FIRE_CSV, ASSESSED_PERSON_CSV
    
    required = [
        ASSESSED_FALL_CSV,
        ASSESSED_FIRE_CSV,
        ASSESSED_PERSON_CSV,
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n⚠️  File assessed belum ada:")
        for f in missing:
            print(f"   {f}")
        print("\nJalankan terlebih dahulu:")
        print("   python scripts/assess_all.py")
        sys.exit(1)


def save_readiness_report(processed_dir: str, reports_dir: str):
    """Simpan laporan distribusi kelas ke reports/."""
    os.makedirs(reports_dir, exist_ok=True)
    report = get_class_distribution_report(processed_dir)

    if report.empty:
        return

    out_path = os.path.join(reports_dir, "data_readiness_result.csv")
    report.to_csv(out_path, index=False)

    print(f"\n  📋 Data Readiness Report:")
    print(f"  {'Dataset':12s} {'Kelas':20s} {'Jumlah':>8s} {'%':>7s}")
    print(f"  {'-'*52}")
    for _, row in report.iterrows():
        print(f"  {row['dataset']:12s} {str(row['class_name']):20s} "
              f"{int(row['count']):8,} {row['pct']:7.1f}%")
    print(f"\n  💾 Tersimpan: {out_path}")


def main():
    print("\n" + "=" * 60)
    print("  SafeWatch — Data Cleaning Pipeline")
    print("  Dataset: Fall | Fire/Smoke | Person")
    print("=" * 60)

    check_assessed_files()

    results = clean_all_datasets(
        processed_dir=PROCESSED_DIR,
        verbose=True,
    )

    save_readiness_report(PROCESSED_DIR, REPORTS_DIR)

    print("\n" + "=" * 60)
    print("  ✅ Cleaning selesai!")
    print(f"  Dataset berhasil dibersihkan: {len(results)}")
    print(f"  Output tersimpan di         : {PROCESSED_DIR}/")
    print("\n  Langkah berikutnya:")
    print("    1. Buka notebooks/04_eda_combined.ipynb untuk EDA")
    print("    2. python scripts/train_fall.py")
    print("    3. python scripts/train_fire_smoke.py")
    print("    4. python scripts/train_person.py")
    print("    5. python scripts/train_tf_classifier.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()