import os
import pandas as pd

def clean_base(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Membersihkan satu file CSV dari baris kosong (NaN) dan duplikat.
    """
    if not os.path.exists(file_path):
        if verbose:
            print(f"⚠️ File tidak ditemukan: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    before_count = len(df)

    # Proses Pembersihan Data
    df = df.dropna()  # Hapus baris kosong
    df = df.drop_duplicates()  # Hapus baris duplikat
    after_count = len(df)

    if verbose:
        print(f"Processing: {os.path.basename(file_path)}")
        print(f"  Before: {before_count}")
        print(f"  After : {after_count}")
        if 'class_name' in df.columns:
            print(df['class_name'].value_counts())
        elif 'label' in df.columns:
            print(df['label'].value_counts())
        print("-" * 30)

    return df

def clean_all_datasets(processed_dir: str, verbose: bool = True) -> dict:
    """
    Membersihkan ketiga dataset SafeWatch (Fall, Fire, Person).
    Menyimpan hasil ke folder processed_dir.
    """
    base_dir = os.path.dirname(processed_dir) if processed_dir.endswith('/') else processed_dir

    # Jalur file input berdasarkan output riil dari skrip assess_all.py
    dataset_tasks = {
        'fall': (os.path.join(base_dir, "assessed_fall.csv"), "cleaned_fall.csv"),
        'fire': (os.path.join(base_dir, "assessed_fire_smoke.csv"), "cleaned_fire_smoke.csv"),
        'person': (os.path.join(base_dir, "assessed_person.csv"), "cleaned_person.csv")
    }

    cleaned_results = {}

    for name, (input_path, output_name) in dataset_tasks.items():
        df_cleaned = clean_base(input_path, verbose=verbose)
        
        if not df_cleaned.empty:
            out_path = os.path.join(processed_dir, output_name)
            df_cleaned.to_csv(out_path, index=False)
            cleaned_results[name] = out_path

    return cleaned_results

def get_class_distribution_report(processed_dir: str) -> pd.DataFrame:
    """
    Membaca semua file cleaned_* di dalam processed_dir 
    dan merangkum distribusi kelasnya untuk Data Readiness Report.
    """
    dataset_files = {
        'Fall': "cleaned_fall.csv",
        'Fire/Smoke': "cleaned_fire_smoke.csv",
        'Person': "cleaned_person.csv"
    }

    report_data = []

    for dataset_name, file_name in dataset_files.items():
        file_path = os.path.join(processed_dir, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Cari kolom kelas yang tersedia
            class_col = 'class_name' if 'class_name' in df.columns else ('label' if 'label' in df.columns else None)
            
            if class_col:
                counts = df[class_col].value_counts()
                total = counts.sum()
                
                for cls, count in counts.items():
                    pct = (count / total) * 100
                    report_data.append({
                        'dataset': dataset_name,
                        'class_name': cls,
                        'count': count,
                        'pct': pct
                    })

    return pd.DataFrame(report_data)

if __name__ == "__main__":
    print("Skrip src/data_cleaner.py siap digunakan sebagai modul.")
