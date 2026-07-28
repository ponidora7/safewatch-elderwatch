"""
scripts/train_distilled_model.py
================================
Trains a Temporal Fall Classifier (Teacher model) using sequential pose features (149 features).
Then distills its predictions into a single-frame Fall Classifier (Student model) using 35 features.
Finally, exports the student model to ONNX and applies dynamic INT8 quantization.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score

# Insert project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.advanced_feature_engineering import AdvancedFeatureEngineer

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Paths
ENHANCED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_human_fall_enhanced.csv")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "safewatch_fall_model_enhanced.keras")
ONNX_FLOAT_PATH = os.path.join(PROJECT_ROOT, "models", "safewatch_model_cpu_float32.onnx")
ONNX_QUANT_PATH = os.path.join(PROJECT_ROOT, "models", "safewatch_model_cpu.onnx")
ONNX_FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend", "public", "models", "fall_model.onnx")

def extract_sliding_windows(df, window_size=5):
    """
    Constructs sequential windows of size `window_size` to calculate velocity/acceleration.
    Returns:
        X_teacher: shape (N, 149) -> 35 static/geom + 64 velocity + 48 acceleration + 2 torso angle stats
        X_student: shape (N, 35) -> 35 static/geom of current frame
        y: shape (N,) -> binary label (fall/normal)
        splits: shape (N,) -> string split (train/valid/test)
    """
    # Sort dataset logically to simulate sequences
    df = df.sort_values(by=["split", "class_name", "image_id"]).reset_index(drop=True)
    
    # Identify landmark columns (now biologically normalized)
    expected_order = []
    for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
        expected_order.extend([f"norm_X{idx}", f"norm_Y{idx}"])
        
    # All 35 feature columns are everything except the metadata
    all_35_cols = [col for col in df.columns if col not in ["dataset", "split", "image_id", "class_name"]]
    

    X_teacher_list = []
    X_student_list = []
    y_list = []
    splits_list = []
    
    # We will build windows per split + class_name grouping to prevent leakage/mixing
    grouped = df.groupby(["split", "class_name"])
    
    for (split_val, class_val), group in grouped:
        if len(group) < window_size:
            continue
            
        landmarks_seq = group[expected_order].values.astype(np.float32)
        all_35_seq = group[all_35_cols].values.astype(np.float32)
        labels_seq = (group["class_name"] == "fall").astype(np.float32).values
        
        for i in range(window_size - 1, len(group)):
            # 1. Static features (current frame): 35 features
            static = all_35_seq[i]
            
            # 2. Get window of coordinates for velocity & acceleration: shape (5, 16)
            coord_window = landmarks_seq[i - (window_size - 1): i + 1]
            
            # Velocity: np.diff -> shape (4, 16) -> 64 features
            velocity = np.diff(coord_window, axis=0).flatten()
            
            # Acceleration: np.diff(n=2) -> shape (3, 16) -> 48 features
            acceleration = np.diff(coord_window, n=2, axis=0).flatten()
            
            # Torso angles from geom features in window: 2 features (mean and std)
            torso_angles = group.iloc[i - (window_size - 1): i + 1]["torso_angle"].values
            angle_stats = np.array([np.mean(torso_angles), np.std(torso_angles)], dtype=np.float32)
            
            # Concatenate for Teacher input (35 + 64 + 48 + 2 = 149 features)
            combined_teacher = np.concatenate([static, velocity, acceleration, angle_stats])
            
            X_teacher_list.append(combined_teacher)
            X_student_list.append(static)
            y_list.append(labels_seq[i])
            splits_list.append(split_val)
            
    return (
        np.array(X_teacher_list),
        np.array(X_student_list),
        np.array(y_list),
        np.array(splits_list)
    )

class DistillationStudent(tf.keras.Model):
    """
    Subclass model that handles training with KD loss:
    loss = alpha * student_loss + (1 - alpha) * distillation_loss
    """
    def __init__(self, student_network, alpha=0.4):
        super().__init__()
        self.student_network = student_network
        self.alpha = alpha
        
    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.distill_loss_fn = tf.keras.losses.MeanSquaredError()
        
    def train_step(self, data):
        x, y_combined = data
        y_true = y_combined[:, 0:1]
        y_teacher = y_combined[:, 1:2]
        
        with tf.GradientTape() as tape:
            y_pred = self.student_network(x, training=True)
            student_loss = self.loss_fn(y_true, y_pred)
            distillation_loss = self.distill_loss_fn(y_teacher, y_pred)
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            
        gradients = tape.gradient(loss, self.student_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_network.trainable_variables))
        
        self.compiled_metrics.update_state(y_true, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        results["student_loss"] = student_loss
        results["distill_loss"] = distillation_loss
        return results
        
    def test_step(self, data):
        x, y_combined = data
        y_true = y_combined[:, 0:1]
        y_teacher = y_combined[:, 1:2]
        
        y_pred = self.student_network(x, training=False)
        student_loss = self.loss_fn(y_true, y_pred)
        distillation_loss = self.distill_loss_fn(y_teacher, y_pred)
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        self.compiled_metrics.update_state(y_true, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        results["student_loss"] = student_loss
        results["distill_loss"] = distillation_loss
        return results
        
    def call(self, inputs):
        return self.student_network(inputs)

def build_teacher_network():
    inputs = Input(shape=(148,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs, name="Teacher_Temporal")

def build_student_network():
    inputs = Input(shape=(34,))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs, name="Student_Frame")

def main():
    print("SafeWatch Distillation Model Training (Scenario A)")
    print("=" * 70)
    
    if not os.path.exists(ENHANCED_CSV_PATH):
        print(f"[ERROR] Processed data not found at {ENHANCED_CSV_PATH}. Please run preprocess_raw_data.py first.")
        sys.exit(1)
        
    print("[INFO] Loading enhanced CSV and extracting sequential windows...")
    df = pd.read_csv(ENHANCED_CSV_PATH)
    X_teacher, X_student, y, splits = extract_sliding_windows(df, window_size=5)
    
    print(f"Windows extracted: {len(X_teacher)}")
    print(f"Teacher feature shape: {X_teacher.shape}")
    print(f"Student feature shape: {X_student.shape}")
    
    # Splits
    train_mask = (splits == "train")
    val_mask = (splits == "valid")
    test_mask = (splits == "test")
    
    X_teacher_train, X_student_train, y_train = X_teacher[train_mask], X_student[train_mask], y[train_mask]
    X_teacher_val, X_student_val, y_val = X_teacher[val_mask], X_student[val_mask], y[val_mask]
    X_teacher_test, X_student_test, y_test = X_teacher[test_mask], X_student[test_mask], y[test_mask]
    
    print(f"Split sizes -> Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Scale student features (35 features)
    print("\n[INFO] Normalizing student features...")
    scaler = StandardScaler()
    X_student_train_scaled = scaler.fit_transform(X_student_train).astype(np.float32)
    X_student_val_scaled = scaler.transform(X_student_val).astype(np.float32)
    X_student_test_scaled = scaler.transform(X_student_test).astype(np.float32)
    
    # Save scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Saved scaler to {SCALER_PATH}")
    
    # Scale teacher features (149 features)
    teacher_scaler = StandardScaler()
    X_teacher_train_scaled = teacher_scaler.fit_transform(X_teacher_train).astype(np.float32)
    X_teacher_val_scaled = teacher_scaler.transform(X_teacher_val).astype(np.float32)
    
    # Train Teacher Model (Temporal Classifier)
    print("\n--- STEP 1: TRAINING TEACHER MODEL (TEMPORAL) ---")
    teacher = build_teacher_network()
    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc_pr", curve="PR")]
    )
    
    callbacks = [
        EarlyStopping(monitor="val_auc_pr", patience=8, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
    ]
    
    teacher.fit(
        X_teacher_train_scaled, y_train,
        validation_data=(X_teacher_val_scaled, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print("[OK] Teacher Model training finished.")
    
    # Generate teacher soft targets
    teacher_train_preds = teacher.predict(X_teacher_train_scaled, verbose=0)
    teacher_val_preds = teacher.predict(X_teacher_val_scaled, verbose=0)
    
    # Distillation of Student Model
    print("\n--- STEP 2: DISTILLING TEACHER KNOWLEDGE TO STUDENT (FRAME) ---")
    student_net = build_student_network()
    distilled_student = DistillationStudent(student_net, alpha=0.4)
    distilled_student.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc_pr", curve="PR")]
    )
    
    # Combine targets
    y_combined_train = np.hstack([y_train.reshape(-1, 1), teacher_train_preds])
    y_combined_val = np.hstack([y_val.reshape(-1, 1), teacher_val_preds])
    
    student_callbacks = [
        EarlyStopping(monitor="val_auc_pr", patience=8, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
    ]
    
    distilled_student.fit(
        X_student_train_scaled, y_combined_train,
        validation_data=(X_student_val_scaled, y_combined_val),
        epochs=40,
        batch_size=32,
        callbacks=student_callbacks,
        verbose=1
    )
    
    print("[OK] Student Model distillation finished.")
    
    # Save student keras model
    student_net.save(KERAS_MODEL_PATH)
    print(f"[OK] Distilled Student Keras model saved to {KERAS_MODEL_PATH}")
    
    # Evaluate distilled student on test set
    preds_prob = student_net.predict(X_student_test_scaled, verbose=0).flatten()
    preds_binary = (preds_prob >= 0.5).astype(np.float32)
    
    print("\n--- EVALUATION METRICS ON TEST SPLIT ---")
    print(f"Accuracy: {np.mean(preds_binary == y_test):.4f}")
    print(f"Recall: {recall_score(y_test, preds_binary):.4f}")
    print(f"Precision: {precision_score(y_test, preds_binary):.4f}")
    print(f"F1-Score: {f1_score(y_test, preds_binary):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, preds_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds_binary))
    
    # ONNX Conversion
    print("\n--- STEP 3: CONVERTING DISTILLED MODEL TO ONNX ---")
    try:
        import tf2onnx
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        spec = (tf.TensorSpec((None, 34), tf.float32, name="input_layer"),)
        model_proto, _ = tf2onnx.convert.from_keras(student_net, input_signature=spec, opset=13)
        
        with open(ONNX_FLOAT_PATH, "wb") as f:
            f.write(model_proto.SerializeToString())
        print(f"[OK] Saved Float32 ONNX model to: {ONNX_FLOAT_PATH}")
        
        # INT8 Quantization
        print("\n--- STEP 4: APPLYING DYNAMIC INT8 QUANTIZATION ---")
        quantize_dynamic(
            model_input=ONNX_FLOAT_PATH,
            model_output=ONNX_QUANT_PATH,
            weight_type=QuantType.QUInt8
        )
        print(f"[OK] Saved Quantized INT8 ONNX model to: {ONNX_QUANT_PATH}")
        
        # Copy to frontend
        import shutil
        os.makedirs(os.path.dirname(ONNX_FRONTEND_PATH), exist_ok=True)
        shutil.copy2(ONNX_QUANT_PATH, ONNX_FRONTEND_PATH)
        print(f"[OK] Copied Quantized ONNX model to frontend: {ONNX_FRONTEND_PATH}")
        
        # Copy scaler to frontend as JSON
        scaler_data = {
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "var_": scaler.var_.tolist() if hasattr(scaler, "var_") else None,
            "n_features_in_": int(scaler.n_features_in_)
        }
        scaler_json_path = os.path.join(os.path.dirname(ONNX_FRONTEND_PATH), "scaler.json")
        import json
        with open(scaler_json_path, "w") as f:
            json.dump(scaler_data, f, indent=2)
        print(f"[OK] Copied Scaler JSON to frontend: {scaler_json_path}")
        
    except ImportError as e:
        print(f"[WARN] Distilled model converted to Keras, but ONNX dependencies missing: {e}")
    except Exception as e:
        print(f"[ERROR] Failed ONNX conversion: {e}")

if __name__ == "__main__":
    main()
