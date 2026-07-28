import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Membangun visualisasi untuk Laporan KP...")
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # 1. Visualisasi Distribusi SMOTE
    # Berdasarkan log sebelumnya: Sebelum (Normal: 6636, Fall: 1291), Sesudah (Normal: 6636, Fall: 6636)
    
    data = {
        'Kondisi': ['Sebelum SMOTE', 'Sebelum SMOTE', 'Sesudah SMOTE', 'Sesudah SMOTE'],
        'Kelas': ['Normal', 'Jatuh (Fall)', 'Normal', 'Jatuh (Fall)'],
        'Jumlah Sampel': [6636, 1291, 6636, 6636]
    }
    df_smote = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(x='Kondisi', y='Jumlah Sampel', hue='Kelas', data=df_smote, palette=['#3498db', '#e74c3c'])
    plt.title('Distribusi Kelas Pada Data Latih: Sebelum vs Sesudah SMOTE', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Jumlah Sampel', fontsize=12)
    plt.xlabel('', fontsize=12)
    
    # Add labels on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontweight='bold')
                    
    plt.tight_layout()
    smote_path = os.path.join(REPORTS_DIR, "smote_distribution.png")
    plt.savefig(smote_path, dpi=300)
    plt.close()
    print(f"✓ Berhasil menyimpan visualisasi SMOTE ke {smote_path}")

if __name__ == "__main__":
    main()
