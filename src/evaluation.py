# src/evaluation.py
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, auc
)

def comprehensive_evaluation(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluasi komprehensif dengan visualisasi untuk submission Dicoding
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # 3. ROC & PR Curves
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    
    # 4. Threshold Analysis
    thresholds = np.arange(0.3, 0.8, 0.05)
    threshold_metrics = []
    for t in thresholds:
        pred_t = (y_pred_proba >= t).astype(int)
        threshold_metrics.append({
            'threshold': t,
            'precision': precision_score(y_true, pred_t),
            'recall': recall_score(y_true, pred_t),
            'f1': f1_score(y_true, pred_t)
        })
    
    # Visualisasi
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    
    # ROC Curve
    axes[0,1].plot(fpr, tpr, label=f'AUC-ROC = {roc_auc:.3f}')
    axes[0,1].plot([0,1], [0,1], 'k--')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend()
    
    # PR Curve (Critical untuk imbalanced data)
    axes[0,2].plot(recall, precision, label=f'AUC-PR = {pr_auc:.3f}')
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].set_title('Precision-Recall Curve')
    axes[0,2].legend()
    
    # Threshold vs Metrics
    df_thresh = pd.DataFrame(threshold_metrics)
    df_thresh.plot(x='threshold', y=['precision', 'recall', 'f1'], ax=axes[1,0])
    axes[1,0].set_title('Metrics vs Threshold')
    axes[1,0].axvline(x=threshold, color='r', linestyle='--', label=f'Current: {threshold}')
    
    # Prediction Distribution
    axes[1,1].hist(y_pred_proba[y_true==0], alpha=0.5, label='Normal', bins=30)
    axes[1,1].hist(y_pred_proba[y_true==1], alpha=0.5, label='Fall', bins=30)
    axes[1,1].axvline(x=threshold, color='r', linestyle='--')
    axes[1,1].set_xlabel('Predicted Probability')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Prediction Distribution')
    axes[1,1].legend()
    
    # Metric Summary Table
    axes[1,2].axis('off')
    summary_text = f"""
    📊 EVALUATION SUMMARY
    {'='*30}
    AUC-ROC: {roc_auc:.4f}
    AUC-PR:  {pr_auc:.4f} ⭐
    
    @ Threshold {threshold}:
    • Precision: {report['1']['precision']:.4f}
    • Recall:    {report['1']['recall']:.4f}
    • F1-Score:  {report['1']['f1-score']:.4f}
    
    ⚠️ Untuk fall detection, prioritaskan Recall 
    (jangan sampai miss fall event!)
    """
    axes[1,2].text(0.1, 0.9, summary_text, fontsize=9, 
                  verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('outputs/04_evaluation_dashboard.png', dpi=300, bbox_inches='tight')
    
    return {
        'confusion_matrix': cm,
        'report': report,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'threshold_analysis': df_thresh
    }