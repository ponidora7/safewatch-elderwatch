# src/data_assessment.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def assess_data_quality(df, target_col='fall_label'):
    """
    Comprehensive data quality assessment dengan visualisasi
    """
    report = {}
    
    # 1. Class Distribution Analysis
    class_dist = df[target_col].value_counts()
    report['class_distribution'] = class_dist
    report['imbalance_ratio'] = class_dist.max() / class_dist.min()
    
    # Visualisasi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    axes[0].pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%')
    axes[0].set_title('Distribusi Kelas')
    
    # Bar chart dengan imbalance indicator
    sns.barplot(x=class_dist.index, y=class_dist.values, ax=axes[1])
    axes[1].axhline(y=class_dist.mean(), color='r', linestyle='--', 
                   label=f'Mean: {class_dist.mean():.0f}')
    axes[1].set_title('Imbalance Detection')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('outputs/01_class_distribution.png', dpi=300, bbox_inches='tight')
    
    # 2. Missing Value Analysis
    missing = df.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    # 3. Feature Statistics per Class
    numeric_cols = df.select_dtypes(include='number').columns.drop(target_col, errors='ignore')
    if len(numeric_cols) > 0:
        report['stats_by_class'] = df.groupby(target_col)[numeric_cols].describe()
    
    return report, fig

# Usage
df = pd.read_csv('data/processed/fall_features.csv')
report, fig = assess_data_quality(df)
print(f"⚠️ Imbalance Ratio: {report['imbalance_ratio']:.2f}x")