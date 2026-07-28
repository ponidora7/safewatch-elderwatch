"""
src/advanced_feature_engineering.py
====================================
Advanced geometric feature engineering for fall detection.
Computes joint angles, distances, aspect ratios, and dynamic features
from 16 MediaPipe/YOLO landmarks.

NEW: Uses Biological Normalization (Scale & Translation Invariant)
All input must be raw pixel coordinates (xy).
"""

import numpy as np
import pandas as pd


class AdvancedFeatureEngineer:
    """
    Extract advanced geometric and dynamic features from landmarks.
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
        
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        cos_angle = np.dot(v1, v2) / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)
    
    @staticmethod
    def extract_geometric_features(landmarks_array):
        """
        Extract geometric features from a single frame's raw pixel landmarks.
        Input MUST be raw pixel (xy) coordinates to preserve aspect ratio.
        
        Args:
            landmarks_array: Array of shape (16,) - 8 landmarks * (x, y)
        
        Returns:
            Dictionary of computed geometric features (19 features)
            + 16 biologically normalized base coordinates.
        """
        # Reshape to 8 landmarks x 2 coordinates
        landmarks = landmarks_array.reshape(8, 2)
        
        features = {}
        
        # 1. KEY JOINT ANGLES (7 features) - Computed on pure pixels for mathematical accuracy!
        left_hip = landmarks[2]
        left_shoulder = landmarks[0]
        left_knee = landmarks[4]
        features['left_hip_angle'] = AdvancedFeatureEngineer.compute_angle(left_shoulder, left_hip, left_knee)
        
        right_hip = landmarks[3]
        right_shoulder = landmarks[1]
        right_knee = landmarks[5]
        features['right_hip_angle'] = AdvancedFeatureEngineer.compute_angle(right_shoulder, right_hip, right_knee)
        
        left_knee_angle = AdvancedFeatureEngineer.compute_angle(left_hip, left_knee, landmarks[6])
        features['left_knee_angle'] = left_knee_angle
        
        right_knee_angle = AdvancedFeatureEngineer.compute_angle(right_hip, right_knee, landmarks[7])
        features['right_knee_angle'] = right_knee_angle
        
        # Torso angle relative to vertical
        torso_top = (left_shoulder + right_shoulder) / 2
        torso_bottom = (left_hip + right_hip) / 2
        torso_vector = torso_bottom - torso_top
        
        vertical = np.array([0, 1]) # y-axis goes down in images
        torso_mag = np.linalg.norm(torso_vector)
        if torso_mag == 0:
            features['torso_angle'] = 0.0
            features['torso_tilt'] = 0.0
        else:
            cos_angle = np.dot(torso_vector, vertical) / torso_mag
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            features['torso_angle'] = np.degrees(np.arccos(cos_angle))
            features['torso_tilt'] = abs(torso_top[0] - torso_bottom[0]) / torso_mag
            
        features['spine_curvature'] = abs(torso_top[0] - torso_bottom[0])
        
        # 2. BODY PROPORTIONS (Scale Invariant)
        torso_length = torso_mag + 1e-6
        
        left_leg_len = AdvancedFeatureEngineer.compute_distance(left_hip, landmarks[6])
        right_leg_len = AdvancedFeatureEngineer.compute_distance(right_hip, landmarks[7])
        avg_leg_len = (left_leg_len + right_leg_len) / 2
        features['torso_to_leg_ratio'] = torso_length / (avg_leg_len + 1e-6)
        
        hip_width = AdvancedFeatureEngineer.compute_distance(left_hip, right_hip)
        features['hip_width_ratio'] = hip_width / torso_length
        
        shoulder_width = AdvancedFeatureEngineer.compute_distance(left_shoulder, right_shoulder)
        features['shoulder_width_ratio'] = shoulder_width / torso_length
        
        body_height = torso_length + avg_leg_len
        body_width = max(hip_width, shoulder_width, 1.0)
        features['body_aspect_ratio'] = body_height / body_width
        
        ankle_spread = abs(landmarks[6][0] - landmarks[7][0])
        features['ankle_spread_ratio'] = ankle_spread / torso_length
        
        # 3. POSTURE DESCRIPTORS
        features['is_horizontal'] = 1.0 if features['torso_angle'] > 60 else 0.0
        
        leg_vector_left = landmarks[6] - left_knee
        leg_vector_right = landmarks[7] - right_knee
        mag_leg_left = np.linalg.norm(leg_vector_left)
        mag_leg_right = np.linalg.norm(leg_vector_right)
        
        if mag_leg_left == 0 or mag_leg_right == 0:
            features['leg_spread_angle'] = 0.0
        else:
            cos_leg_angle = np.dot(leg_vector_left, leg_vector_right) / (mag_leg_left * mag_leg_right)
            cos_leg_angle = np.clip(cos_leg_angle, -1.0, 1.0)
            features['leg_spread_angle'] = np.degrees(np.arccos(cos_leg_angle))
            
        features['shoulder_symmetry'] = abs(left_shoulder[1] - right_shoulder[1]) / torso_length
        features['avg_knee_angle'] = (features['left_knee_angle'] + features['right_knee_angle']) / 2
        
        # Center of mass (Biologically Normalized)
        com_y = (torso_top[1] + torso_bottom[1]) / 2
        com_x = (landmarks[6][0] + landmarks[7][0]) / 2
        features['com_x_norm'] = (com_x - torso_bottom[0]) / torso_length
        features['com_y_norm'] = (com_y - torso_bottom[1]) / torso_length
        
        # 4. BIOLOGICAL NORMALIZATION FOR BASE COORDINATES (16 features)
        # We normalize all 16 raw coordinates to be centered at the hips, scaled by torso length
        hip_center = torso_bottom
        scale = torso_length
        
        expected_order = [11, 12, 23, 24, 25, 26, 27, 28]
        for i, idx in enumerate(expected_order):
            pt = landmarks[i]
            features[f'norm_X{idx}'] = (pt[0] - hip_center[0]) / scale
            features[f'norm_Y{idx}'] = (pt[1] - hip_center[1]) / scale

        return features
