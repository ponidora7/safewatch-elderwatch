 """
scripts/run_data_pipeline.py
============================
End-to-end data wrangling pipeline for SafeWatch ElderWatch.
Orchestrates:
1. Loading the extracted geometric features (from preprocess_raw_data.py)
2. Data Cleaning (removing NaNs, duplicates)
3. Train-Test Split (maintaining original splits or performing custom splits)
4. Data Balancing (SMOTE) on Training Data ONLY
5. Feature Scaling (StandardScaler) and exporting the scaler
6. Exporting finalized datasets for ML training
"""

import os
import sys
import io
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fix standard output encoding for Windows terminal (for emojis like checkmarks)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Insert project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.imbalanced_handling import create_balanced_dataset

def main():
    print("=" * 60)
    print("SafeWatch: End-to-End Data Wrangling Pipeline")
    print("=" * 60)
    
    # 1. Configuration & Paths
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    INPUT_CSV = os.path.join(PROCESSED_DIR, "cleaned_human_fall_enhanced.csv")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: File dataset awal tidak ditemukan: {INPUT_CSV}")
        print("Silakan jalankan 'python scripts/preprocess_raw_data.py' terlebih dahulu.")
        sys.exit(1)
        
    print(f"1. Membaca dataset: {os.path.basename(INPUT_CSV)}")
    df = pd.read_csv(INPUT_CSV)
    print(f"   -> Total baris awal: {len(df)}")
    
    # 2. Data Cleaning
    print("\n2. Melakukan Data Cleaning (Menghapus NaN & Duplikat)...")
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"   -> Sisa baris setelah cleaning: {len(df)}")
    
    # Define features and metadata columns
    metadata_cols = ['dataset', 'split', 'image_id', 'class_name']
    
    # Drop rows where class_name is not known (if any)
    if 'class_name' in df.columns:
        valid_classes = ['normal', 'fall']
        df = df[df['class_name'].isin(valid_classes)]
        print(f"   -> Sisa baris (hanya normal & fall): {len(df)}")
    else:
        print("❌ Error: Kolom 'class_name' tidak ditemukan.")
        sys.exit(1)

    # 3. Train-Test Split (Memanfaatkan kolom 'split' dari YOLO jika ada, atau buat baru)
    print("\n3. Memisahkan Train & Test Set...")
    # We will use the original YOLO 'split' column (train/valid/test) to avoid data leakage
    # Combine 'valid' and 'test' into a single evaluation set called 'test'
    df_train = df[df['split'] == 'train'].copy()
    df_test = df[df['split'].isin(['valid', 'test'])].copy()
    
    if len(df_train) == 0 or len(df_test) == 0:
        print("   ⚠️ Kolom 'split' tidak valid. Melakukan random split 80:20...")
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class_name'])
        
    X_train_raw = df_train.drop(columns=metadata_cols, errors='ignore')
    y_train_raw = df_train['class_name']
    
    X_test_raw = df_test.drop(columns=metadata_cols, errors='ignore')
    y_test_raw = df_test['class_name']
    
    print(f"   -> Train set: {len(X_train_raw)} sampel")
    print(f"   -> Test set: {len(X_test_raw)} sampel")
    print("   -> Distribusi Train (Sebelum SMOTE):")
    print(y_train_raw.value_counts().to_string(header=False))
    
    # 4. Imbalanced Handling (SMOTE) on Training Data Only
    print("\n4. Menangani Data Imbalance dengan SMOTE (Hanya pada Train set)...")
    
    # Encode target to numeric (fall=1, normal=0) for SMOTE
    y_train_num = y_train_raw.map({'fall': 1, 'normal': 0})
    y_test_num = y_test_raw.map({'fall': 1, 'normal': 0})
    
    X_train_bal, y_train_bal = create_balanced_dataset(X_train_raw, y_train_num, strategy='smart', random_state=42)
    
    print("   -> Distribusi Train (Sesudah SMOTE):")
    print(pd.Series(y_train_bal).map({1: 'fall', 0: 'normal'}).value_counts().to_string(header=False))
    
    # 5. Feature Scaling
    print("\n5. Feature Scaling (StandardScaler)...")
    scaler = StandardScaler()
    
    # Fit & transform on balanced training data
    X_train_scaled = scaler.fit_transform(X_train_bal)
    
    # Transform test data using the fitted scaler
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Save the scaler for inference
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Scaler berhasil disimpan ke: {scaler_path}")
    
    # 6. Export Datasets
    print("\n6. Mengekspor Final Dataset...")
    
    # Convert back to DataFrame for easy CSV export
    feature_names = X_train_raw.columns
    
    df_train_final = pd.DataFrame(X_train_scaled, columns=feature_names)
    df_train_final['class_id'] = y_train_bal
    
    df_test_final = pd.DataFrame(X_test_scaled, columns=feature_names)
    df_test_final['class_id'] = y_test_num.values
    
    train_out = os.path.join(PROCESSED_DIR, "final_X_train.csv")
    test_out = os.path.join(PROCESSED_DIR, "final_X_test.csv")
    
    df_train_final.to_csv(train_out, index=False)
    df_test_final.to_csv(test_out, index=False)
    
    print(f"   ✓ Data latih diekspor ke: {train_out}")
    print(f"   ✓ Data uji diekspor ke: {test_out}")
    print("\n✅ Proses Data Wrangling Selesai!")

if __name__ == "__main__":
    main()
