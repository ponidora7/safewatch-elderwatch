"""
scripts/evaluate_enhanced_model.py
==================================
Comprehensive evaluation of enhanced fall detection model.
Generates metrics, visualizations, and comparison with baseline.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score as compute_f1
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from src.advanced_feature_engineering import AdvancedFeatureEngineer


def evaluate_enhanced_model():
    """Comprehensive evaluation of the enhanced model."""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MODEL EVALUATION - ENHANCED FALL DETECTION")
    print("="*80)
    
    # Load data
    print("\n[1] Loading enhanced dataset...")
    df = pd.read_csv('data/processed/cleaned_human_fall.csv')
    df_enhanced = AdvancedFeatureEngineer.enhance_dataframe(df)
    
    feature_cols = [col for col in df_enhanced.columns if col.startswith('X') or col.startswith('Y') or col.startswith('feat_')]
    print(f"   Total features: {len(feature_cols)}")
    
    # Prepare data
    X = df_enhanced[feature_cols].values.astype(np.float32)
    y = (df_enhanced['class_name'] == 'fall').astype(np.float32).values
    
    # Balance data
    df_normal = df_enhanced[df_enhanced['class_name'] != 'fall']
    df_fall = df_enhanced[df_enhanced['class_name'] == 'fall']
    df_fall_augmented = df_fall.sample(len(df_normal), replace=True, random_state=42)
    df_balanced = pd.concat([df_normal, df_fall_augmented], axis=0).sample(frac=1, random_state=42)
    
    X = df_balanced[feature_cols].values.astype(np.float32)
    y = (df_balanced['class_name'] == 'fall').astype(np.float32).values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load enhanced model
    print("\n[2] Loading enhanced model...")
    if not os.path.exists('models/safewatch_fall_model_enhanced.keras'):
        print("   ❌ Enhanced model not found. Please train first using train_enhanced_model.py")
        return
    
    model = tf.keras.models.load_model(
        'models/safewatch_fall_model_enhanced.keras',
        custom_objects={'f1_score_metric': lambda y_true, y_pred: y_pred}
    )
    print("   ✓ Model loaded successfully")
    
    # Evaluate
    print("\n[3] Evaluating model on test set...")
    loss, accuracy, f1 = model.evaluate(X_test, y_test, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
    print(f"\n   📈 PERFORMANCE METRICS:")
    print(f"   ├─ Accuracy: {accuracy*100:.2f}%")
    print(f"   ├─ Loss: {loss:.4f}")
    print(f"   └─ F1-Score: {f1:.4f}")
    
    # Detailed metrics at different thresholds
    print("\n[4] Threshold Analysis...")
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_results = []
    
    for threshold in thresholds_to_test:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1_th = 2 * (precision * recall) / (precision + recall + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        
        threshold_results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1_th,
            'specificity': specificity,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        })
        
        print(f"\n   Threshold {threshold}:")
        print(f"   ├─ Precision: {precision:.4f}")
        print(f"   ├─ Recall: {recall:.4f} ⭐ (Critical for safety)")
        print(f"   ├─ F1-Score: {f1_th:.4f}")
        print(f"   └─ Specificity: {specificity:.4f}")
    
    # ROC and PR curves
    print("\n[5] Computing ROC and PR curves...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   PR-AUC: {pr_auc:.4f}")
    
    # Classification report at optimal threshold (0.5)
    print("\n[6] Detailed Classification Report (threshold=0.5)...")
    y_pred_opt = (y_pred_proba >= 0.5).astype(int)
    report = classification_report(y_test, y_pred_opt, output_dict=True)
    
    print(f"\n   Class 0 (Normal):")
    print(f"   ├─ Precision: {report['0']['precision']:.4f}")
    print(f"   ├─ Recall: {report['0']['recall']:.4f}")
    print(f"   └─ F1-Score: {report['0']['f1-score']:.4f}")
    
    print(f"\n   Class 1 (Fall):")
    print(f"   ├─ Precision: {report['1']['precision']:.4f}")
    print(f"   ├─ Recall: {report['1']['recall']:.4f}")
    print(f"   └─ F1-Score: {report['1']['f1-score']:.4f}")
    
    # Create visualizations
    print("\n[7] Creating evaluation visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Fall Detection Model - Comprehensive Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
    axes[0, 0].set_title('Confusion Matrix (Threshold=0.5)', fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve', fontweight='bold')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)
    
    # 3. PR Curve
    axes[0, 2].plot(recall_curve, precision_curve, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curve', fontweight='bold')
    axes[0, 2].legend(loc="best")
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].set_xlim([0.0, 1.0])
    axes[0, 2].set_ylim([0.0, 1.05])
    
    # 4. Metrics vs Threshold
    df_thresh = pd.DataFrame(threshold_results)
    axes[1, 0].plot(df_thresh['threshold'], df_thresh['precision'], marker='o', label='Precision', linewidth=2)
    axes[1, 0].plot(df_thresh['threshold'], df_thresh['recall'], marker='s', label='Recall', linewidth=2)
    axes[1, 0].plot(df_thresh['threshold'], df_thresh['f1'], marker='^', label='F1-Score', linewidth=2)
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Default (0.5)')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Metrics vs Decision Threshold', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Normal', color='blue', edgecolor='black')
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Fall', color='red', edgecolor='black')
    axes[1, 1].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    axes[1, 1].set_xlabel('Predicted Probability (Fall)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Summary Statistics
    summary_text = f"""
    🎯 ENHANCED MODEL SUMMARY
    {'='*40}
    
    📊 Main Metrics:
    • Overall Accuracy: {accuracy*100:.2f}%
    • ROC-AUC: {roc_auc:.4f}
    • PR-AUC: {pr_auc:.4f}
    
    🎯 At Threshold 0.5:
    • Precision: {report['1']['precision']:.4f}
    • Recall: {report['1']['recall']:.4f} ⭐
    • F1-Score: {report['1']['f1-score']:.4f}
    
    📈 Features:
    • Total Features: {len(feature_cols)}
    • Original Landmarks: 32
    • Engineered Features: {len(feature_cols)-32}
    
    ⚠️  For safety-critical fall detection,
    prioritize RECALL (minimize missed falls)
    over PRECISION (minimize false alarms)
    """
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('reports/figures', exist_ok=True)
    figpath = 'reports/figures/enhanced_model_evaluation.png'
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {figpath}")
    
    # Save evaluation report
    eval_report = {
        'accuracy': accuracy,
        'f1_score': f1,
        'loss': loss,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'classification_report': report,
        'threshold_analysis': df_thresh.to_dict(),
        'features_count': len(feature_cols),
        'test_set_size': len(X_test)
    }
    
    report_path = 'reports/enhanced_model_evaluation.pkl'
    with open(report_path, 'wb') as f:
        pickle.dump(eval_report, f)
    print(f"   ✓ Evaluation data saved to {report_path}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETED")
    print("="*80)
    
    return eval_report


if __name__ == "__main__":
    evaluate_enhanced_model()
