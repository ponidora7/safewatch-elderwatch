# src/validation.py
from sklearn.model_selection import StratifiedGroupKFold

def robust_cross_validation(X, y, groups, model_fn, n_splits=5):
    """
    Cross-validation yang mencegah data leakage dari video yang sama
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(X, y, groups=groups)
    ):
        print(f"\n🔄 Fold {fold+1}/{n_splits}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Build and train model
        model = model_fn()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc_pr',
                    patience=10,
                    restore_best_weights=True,
                    mode='max'
                )
            ],
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_val).flatten()
        metrics = comprehensive_evaluation(y_val, y_pred_proba)
        
        cv_results.append({
            'fold': fold,
            'best_epoch': np.argmax(history.history['val_auc_pr']),
            'val_auc_pr': metrics['pr_auc'],
            'val_recall': metrics['report']['1']['recall']
        })
        
        print(f"✅ Fold {fold+1} - AUC-PR: {metrics['pr_auc']:.4f}, Recall: {metrics['report']['1']['recall']:.4f}")
    
    # Aggregate results
    results_df = pd.DataFrame(cv_results)
    print(f"\n📈 CV Summary (mean ± std):")
    print(f"• AUC-PR: {results_df['val_auc_pr'].mean():.4f} ± {results_df['val_auc_pr'].std():.4f}")
    print(f"• Recall: {results_df['val_recall'].mean():.4f} ± {results_df['val_recall'].std():.4f}")
    
    return results_df