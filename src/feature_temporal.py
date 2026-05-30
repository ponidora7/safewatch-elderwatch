# src/feature_temporal.py
import numpy as np
from collections import deque

class TemporalFeatureExtractor:
    """
    Extract temporal features menggunakan sliding window
    Reference: PoseAware FallNet [[12]] - temporal window analysis
    """
    def __init__(self, window_size=5, step_size=1):
        self.window_size = window_size
        self.step_size = step_size
        
    def extract_temporal_features(self, landmark_sequence):
        """
        landmark_sequence: array [T, 16, 2] - T frames, 16 landmarks, (x,y)
        Returns: enhanced features dengan temporal context
        """
        features = []
        
        for t in range(len(landmark_sequence) - self.window_size + 1):
            window = landmark_sequence[t:t + self.window_size]
            
            # 1. Static features (current frame)
            static = window[-1].flatten()  # 32 features
            
            # 2. Velocity features (first derivative)
            velocity = np.diff(window, axis=0).flatten()  # 30 features
            
            # 3. Acceleration features (second derivative)
            if len(window) >= 3:
                acceleration = np.diff(window, n=2, axis=0).flatten()  # 28 features
            else:
                acceleration = np.zeros(28)
            
            # 4. Angle features: torso angle relative to vertical
            torso_angles = self._compute_torso_angles(window)
            
            # Concatenate all
            combined = np.concatenate([static, velocity, acceleration, torso_angles])
            features.append(combined)
        
        return np.array(features)
    
    def _compute_torso_angles(self, window):
        """
        Hitung angle antara torso dan vertical axis
        Critical untuk deteksi fall state [[18]]
        """
        angles = []
        for frame in window:
            # Landmark indices: shoulders (5,6), hips (11,12)
            left_shoulder = frame[5]
            right_shoulder = frame[6]
            left_hip = frame[11]
            right_hip = frame[12]
            
            # Torso center
            torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
            torso_vector = (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2
            
            # Angle with vertical axis
            vertical = np.array([0, -1])
            angle = np.arccos(np.dot(torso_vector, vertical) / 
                            (np.linalg.norm(torso_vector) * np.linalg.norm(vertical)))
            angles.append(np.degrees(angle))
        
        return np.array([np.mean(angles), np.std(angles)])  # 2 features