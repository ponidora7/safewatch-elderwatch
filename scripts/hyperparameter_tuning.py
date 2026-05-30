# scripts/hyperparameter_tuning.py
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial, X, y, cv_splits=3):
    """
    Optuna objective function untuk hyperparameter optimization
    """
    # Hyperparameters to tune
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'l2_reg': trial.suggest_float('l2_reg', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'use_focal_loss': trial.suggest_categorical('use_focal_loss', [True, False])
    }
    
    # Build and compile model
    model = build_enhanced_fall_model(
        dropout_rate=params['dropout_rate'],
        l2_reg=params['l2_reg']
    )
    compile_model(model, 
                 learning_rate=params['learning_rate'],
                 use_focal_loss=params['use_focal_loss'])
    
    # Early stopping callback
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc_pr',  # Use AUC-PR untuk imbalanced data
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Simple cross-validation (untuk demo - di production gunakan proper CV)
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=params['batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    
    # Return best validation AUC-PR (lebih informatif untuk imbalanced data)
    return max(history.history['val_auc_pr'])

# Run optimization
def run_hyperparameter_search(X, y, n_trials=30):
    study = optuna.create_study(
        direction='maximize',
        study_name='fall_detection_optimization'
    )
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print(f"✅ Best AUC-PR: {study.best_value:.4f}")
    print(f"📋 Best params: {study.best_params}")
    
    return study.best_params
