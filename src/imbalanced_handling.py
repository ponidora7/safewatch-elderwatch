# src/imbalanced_handling.py
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import tensorflow as tf
import pandas as pd

def create_balanced_dataset(X, y, strategy='smart', random_state=42):
    """
    Apply advanced balancing strategy berdasarkan karakteristik data
    """
    imbalance_ratio = pd.Series(y).value_counts().max() / pd.Series(y).value_counts().min()
    
    if strategy == 'smart':
        # Auto-select berdasarkan severity imbalance
        if imbalance_ratio < 5:
            strategy = 'none'
        elif imbalance_ratio < 10:
            strategy = 'smote'
        else:
            strategy = 'combined'
    
    if strategy == 'none':
        return X, y
    
    elif strategy == 'smote':
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        return smote.fit_resample(X, y)
    
    elif strategy == 'combined':
        # SMOTE + Tomek Links untuk cleaner boundaries
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('tomek', RandomUnderSampler(sampling_strategy='majority', random_state=random_state))
        ])
        return pipeline.fit_resample(X, y)
    
    elif strategy == 'focal_loss':
        # Return class weights untuk Focal Loss
        class_counts = pd.Series(y).value_counts().sort_index()
        total = len(y)
        weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
        return X, y, weights

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss implementation untuk TensorFlow/Keras
    Reference: Lin et al. (2017) - Focal Loss for Dense Object Detection
    """
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + \
                 (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    return loss_fn