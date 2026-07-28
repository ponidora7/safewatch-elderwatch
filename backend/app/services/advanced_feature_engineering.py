"""
src/advanced_feature_engineering.py
====================================
Advanced geometric feature engineering for fall detection.
Computes joint angles, distances, aspect ratios, and dynamic features
from 16 MediaPipe landmarks to dramatically improve model accuracy.
"""

import numpy as np
import pandas as pd


class AdvancedFeatureEngineer:
    """
    Extract advanced geometric and dynamic features from MediaPipe landmarks.
    Original landmark columns (8 joints):
      - Left/Right Shoulder (11, 12)
      - Left/Right Hip (23, 24)
      - Left/Right Knee (25, 26)
      - Left/Right Ankle (27, 28)
    Mapped to 2D array coordinates (indices 0 to 7):
      0: left_shoulder,  1: right_shoulder
      2: left_hip,       3: right_hip
      4: left_knee,      5: right_knee
      6: left_ankle,     7: right_ankle
    """
    
    @staticmethod
    def compute_distance(p1, p2):
        """Euclidean distance between two points."""
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    @staticmethod
    def compute_angle(p1, center, p2):
        """
        Compute angle at center point between p1 and p2.
        Returns angle in degrees (0-180).
        """
        v1 = p1 - center
        v2 = p2 - center
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)
    
    @staticmethod
    def extract_geometric_features(landmarks_array):
        """
        Extract geometric features from a single frame's landmarks.
        
        Args:
            landmarks_array: Array of shape (16,) - 8 landmarks * (x, y)
        
        Returns:
            Dictionary of computed geometric features (25 features)
        """
        # Reshape to 8 landmarks x 2 coordinates
        landmarks = landmarks_array.reshape(8, 2)
        
        features = {}
        
        # 1. KEY JOINT ANGLES (7 features)
        # Left Hip Angle (Shoulder -> Hip -> Knee)
        left_hip = landmarks[2]
        left_shoulder = landmarks[0]
        left_knee = landmarks[4]
        features['left_hip_angle'] = AdvancedFeatureEngineer.compute_angle(
            left_shoulder, left_hip, left_knee
        )
        
        # Right Hip Angle
        right_hip = landmarks[3]
        right_shoulder = landmarks[1]
        right_knee = landmarks[5]
        features['right_hip_angle'] = AdvancedFeatureEngineer.compute_angle(
            right_shoulder, right_hip, right_knee
        )
        
        # Left Knee Angle (Hip -> Knee -> Ankle)
        left_knee_angle = AdvancedFeatureEngineer.compute_angle(
            left_hip, left_knee, landmarks[6]
        )
        features['left_knee_angle'] = left_knee_angle
        
        # Right Knee Angle
        right_knee_angle = AdvancedFeatureEngineer.compute_angle(
            right_hip, right_knee, landmarks[7]
        )
        features['right_knee_angle'] = right_knee_angle
        
        # Torso angle relative to vertical
        torso_top = (left_shoulder + right_shoulder) / 2
        torso_bottom = (left_hip + right_hip) / 2
        torso_vector = torso_bottom - torso_top
        # Angle from vertical (negative y-axis)
        vertical = np.array([0, -1])
        cos_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        features['torso_angle'] = np.degrees(np.arccos(cos_angle))
        
        # Body tilt (horizontal deviation of torso from center)
        features['torso_tilt'] = abs(torso_top[0] - torso_bottom[0]) / (np.linalg.norm(torso_vector) + 1e-8)
        
        # Spine curvature (deviation from straight)
        features['spine_curvature'] = abs((left_shoulder[0] + right_shoulder[0]) / 2 - 
                                          (left_hip[0] + right_hip[0]) / 2)
        
        # 2. BODY PROPORTIONS & DISTANCES (8 features)
        # Height (estimated from shoulder to ankle)
        torso_length = AdvancedFeatureEngineer.compute_distance(torso_top, torso_bottom)
        features['torso_length'] = torso_length
        
        # Leg length (hip to ankle average)
        left_leg_len = AdvancedFeatureEngineer.compute_distance(left_hip, landmarks[6])
        right_leg_len = AdvancedFeatureEngineer.compute_distance(right_hip, landmarks[7])
        features['avg_leg_length'] = (left_leg_len + right_leg_len) / 2
        
        # Hip width
        hip_width = AdvancedFeatureEngineer.compute_distance(left_hip, right_hip)
        features['hip_width'] = hip_width
        
        # Shoulder width
        shoulder_width = AdvancedFeatureEngineer.compute_distance(left_shoulder, right_shoulder)
        features['shoulder_width'] = shoulder_width
        
        # Body aspect ratio (height/width) - critical for fall detection
        body_height = features['torso_length'] + features['avg_leg_length']
        body_width = hip_width
        features['body_aspect_ratio'] = body_height / (body_width + 1e-8)
        
        # Ankle spread (horizontal distance)
        ankle_spread = abs(landmarks[6][0] - landmarks[7][0])
        features['ankle_spread'] = ankle_spread
        
        # Center of mass (estimated)
        com_y = (torso_top[1] + torso_bottom[1]) / 2  # vertical center
        com_x = (landmarks[6][0] + landmarks[7][0]) / 2  # horizontal center (avg ankles)
        features['com_x'] = com_x
        features['com_y'] = com_y
        
        # 3. POSTURE DESCRIPTORS (4 features)
        # Is person horizontal? (torso angle > 60 deg suggests falling/lying)
        features['is_horizontal'] = 1.0 if features['torso_angle'] > 60 else 0.0
        
        # Leg spread angle (how far apart legs are)
        leg_vector_left = landmarks[6] - left_knee
        leg_vector_right = landmarks[7] - right_knee
        cos_leg_angle = np.dot(leg_vector_left, leg_vector_right) / (
            np.linalg.norm(leg_vector_left) * np.linalg.norm(leg_vector_right) + 1e-8
        )
        cos_leg_angle = np.clip(cos_leg_angle, -1.0, 1.0)
        features['leg_spread_angle'] = np.degrees(np.arccos(cos_leg_angle))
        
        # Arm symmetry (estimate using shoulders)
        features['shoulder_symmetry'] = abs(left_shoulder[1] - right_shoulder[1])
        
        # Knee bend state (average of both knees)
        avg_knee_angle = (features['left_knee_angle'] + features['right_knee_angle']) / 2
        features['avg_knee_angle'] = avg_knee_angle
        
        return features
    
    @staticmethod
    def enhance_dataframe(df):
        """
        Add engineered features to a DataFrame containing landmark coordinates.
        
        Args:
            df: DataFrame with columns X11, Y11, ...
        
        Returns:
            DataFrame with additional engineered feature columns
        """
        df_enhanced = df.copy()
        
        # Extract landmark columns (X11, Y11, etc.)
        landmark_cols = [col for col in df.columns if col.startswith('X') or col.startswith('Y')]
        
        if len(landmark_cols) < 16:
            print(f"Warning: Expected 16 landmark columns, found {len(landmark_cols)}")
            return df_enhanced
        
        # Ensure correct column ordering
        # X11, Y11, X12, Y12, X23, Y23, X24, Y24, X25, Y25, X26, Y26, X27, Y27, X28, Y28
        expected_order = []
        for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
            expected_order.extend([f'X{idx}', f'Y{idx}'])
            
        # Create engineered features for each row
        feature_names = []
        engineered_features = []
        
        for idx, row in df.iterrows():
            landmarks_array = row[expected_order].values.astype(np.float32)
            features = AdvancedFeatureEngineer.extract_geometric_features(landmarks_array)
            
            if idx == 0:
                feature_names = list(features.keys())
            
            engineered_features.append(list(features.values()))
        
        # Add to dataframe
        for i, fname in enumerate(feature_names):
            df_enhanced[f'feat_{fname}'] = [row[i] for row in engineered_features]
        
        return df_enhanced


def apply_advanced_features(csv_path, output_path=None):
    """
    Convenience function to apply advanced feature engineering to a CSV.
    
    Args:
        csv_path: Path to input CSV with landmark coordinates
        output_path: Path to save enhanced CSV (default: same as input with _enhanced suffix)
    
    Returns:
        Enhanced DataFrame
    """
    df = pd.read_csv(csv_path)
    df_enhanced = AdvancedFeatureEngineer.enhance_dataframe(df)
    
    if output_path is None:
        base = csv_path.rsplit('.', 1)[0]
        output_path = f"{base}_enhanced.csv"
    
    df_enhanced.to_csv(output_path, index=False)
    print(f"✅ Enhanced dataset saved to: {output_path}")
    print(f"   Original features: {len(df.columns)}")
    print(f"   Enhanced features: {len(df_enhanced.columns)}")
    
    return df_enhanced
