"""
src/data_augmentor.py
=====================
Advanced data augmentation for fall detection.
Augments pose landmark coordinates to increase dataset diversity.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


class PoseAugmentor:
    """
    Advanced augmentation techniques for pose landmark data.
    Applies geometric transformations without changing semantic meaning of fall/normal.
    """
    
    @staticmethod
    def rotate_pose(landmarks: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate all landmarks by given angle (degrees) around center.
        
        Args:
            landmarks: Array of shape (16, 2) - 16 landmarks with (x, y)
            angle_deg: Rotation angle in degrees
        
        Returns:
            Rotated landmarks array
        """
        # Compute center of mass as rotation pivot
        center = np.mean(landmarks, axis=0)
        
        # Convert angle to radians
        angle_rad = np.radians(angle_deg)
        
        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply rotation
        rotated = np.zeros_like(landmarks)
        for i, point in enumerate(landmarks):
            shifted = point - center
            rotated[i] = (rot_matrix @ shifted) + center
        
        return rotated
    
    @staticmethod
    def scale_pose(landmarks: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scale pose around its center.
        
        Args:
            landmarks: Array of shape (16, 2)
            scale_factor: Scaling factor (e.g., 1.1 = 10% larger)
        
        Returns:
            Scaled landmarks array
        """
        center = np.mean(landmarks, axis=0)
        scaled = (landmarks - center) * scale_factor + center
        
        # Clamp to valid range [0, 1]
        return np.clip(scaled, 0.0, 1.0)
    
    @staticmethod
    def translate_pose(landmarks: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        Translate pose by (dx, dy).
        
        Args:
            landmarks: Array of shape (16, 2)
            dx: Horizontal translation
            dy: Vertical translation
        
        Returns:
            Translated landmarks array
        """
        translated = landmarks.copy()
        translated[:, 0] += dx  # X translation
        translated[:, 1] += dy  # Y translation
        
        # Clamp to valid range
        return np.clip(translated, 0.0, 1.0)
    
    @staticmethod
    def add_gaussian_noise(landmarks: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to landmarks (simulates detection noise).
        
        Args:
            landmarks: Array of shape (16, 2)
            noise_std: Standard deviation of Gaussian noise
        
        Returns:
            Noisy landmarks array
        """
        noise = np.random.normal(0, noise_std, landmarks.shape)
        noisy = landmarks + noise
        return np.clip(noisy, 0.0, 1.0)
    
    @staticmethod
    def horizontal_flip(landmarks: np.ndarray) -> np.ndarray:
        """
        Horizontally flip pose (mirror around x=0.5).
        
        Args:
            landmarks: Array of shape (16, 2)
        
        Returns:
            Flipped landmarks array
        """
        flipped = landmarks.copy()
        flipped[:, 0] = 1.0 - flipped[:, 0]  # Mirror X coordinates
        
        # Swap left-right joints
        # Landmark indices for left-right pairs (MediaPipe model)
        swap_pairs = [
            (11, 12),  # Shoulders
            (23, 24),  # Hips
            (25, 26),  # Knees
            (27, 28),  # Ankles
        ]
        
        for left_idx, right_idx in swap_pairs:
            flipped[[left_idx, right_idx]] = flipped[[right_idx, left_idx]]
        
        return flipped
    
    @staticmethod
    def occlusion_mask(landmarks: np.ndarray, occlusion_ratio: float = 0.1) -> np.ndarray:
        """
        Simulate partial occlusion by zeroing out some landmarks.
        
        Args:
            landmarks: Array of shape (16, 2)
            occlusion_ratio: Fraction of landmarks to occlude (0.0-0.3)
        
        Returns:
            Partially occluded landmarks array
        """
        occluded = landmarks.copy()
        num_landmarks = len(landmarks)
        num_to_occlude = max(1, int(num_landmarks * occlusion_ratio))
        
        # Randomly select landmarks to occlude
        occlude_indices = np.random.choice(num_landmarks, num_to_occlude, replace=False)
        occluded[occlude_indices] = 0.0  # Zero out occluded points
        
        return occluded
    
    @staticmethod
    def augment_row(row: pd.Series, landmark_cols: List[str], 
                   augmentation_type: str = 'random') -> pd.Series:
        """
        Apply augmentation to a single row of landmark data.
        
        Args:
            row: DataFrame row with landmark coordinates
            landmark_cols: List of landmark column names
            augmentation_type: Type of augmentation ('rotate', 'scale', 'translate', 'noise', 'flip', 'occlusion', 'random')
        
        Returns:
            Augmented row (as Series)
        """
        # Extract landmarks
        landmarks_array = row[landmark_cols].values.reshape(16, 2).astype(np.float32)
        
        # Apply augmentation
        if augmentation_type == 'rotate':
            angle = np.random.uniform(-15, 15)  # ±15 degrees
            augmented = PoseAugmentor.rotate_pose(landmarks_array, angle)
        
        elif augmentation_type == 'scale':
            scale = np.random.uniform(0.85, 1.15)  # ±15% scaling
            augmented = PoseAugmentor.scale_pose(landmarks_array, scale)
        
        elif augmentation_type == 'translate':
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            augmented = PoseAugmentor.translate_pose(landmarks_array, dx, dy)
        
        elif augmentation_type == 'noise':
            augmented = PoseAugmentor.add_gaussian_noise(landmarks_array, noise_std=0.01)
        
        elif augmentation_type == 'flip':
            augmented = PoseAugmentor.horizontal_flip(landmarks_array)
        
        elif augmentation_type == 'occlusion':
            augmented = PoseAugmentor.occlusion_mask(landmarks_array, occlusion_ratio=0.15)
        
        elif augmentation_type == 'random':
            # Randomly select augmentation
            aug_types = ['rotate', 'scale', 'translate', 'noise', 'flip', 'occlusion']
            selected = np.random.choice(aug_types)
            augmented = PoseAugmentor.augment_row(row, landmark_cols, selected)
        
        else:
            augmented = landmarks_array
        
        # Update row with augmented values
        augmented_row = row.copy()
        augmented_flat = augmented.flatten()
        augmented_row[landmark_cols] = augmented_flat
        
        return augmented_row
    
    @staticmethod
    def augment_dataframe(df: pd.DataFrame, multiplier: int = 2,
                         augmentation_types: List[str] = None) -> pd.DataFrame:
        """
        Augment entire DataFrame with multiple augmented copies.
        
        Args:
            df: DataFrame with landmark columns
            multiplier: How many times to augment dataset (2 = double size)
            augmentation_types: List of augmentation types to apply
        
        Returns:
            DataFrame with original + augmented rows
        """
        if augmentation_types is None:
            augmentation_types = ['rotate', 'scale', 'translate', 'noise', 'flip']
        
        # Extract landmark columns
        landmark_cols = [col for col in df.columns if col.startswith('X') or col.startswith('Y')]
        
        augmented_dfs = [df]  # Start with original
        
        for mult_idx in range(multiplier - 1):
            aug_type = augmentation_types[mult_idx % len(augmentation_types)]
            print(f"   Augmenting with {aug_type}...")
            
            augmented_subset = df.copy()
            for idx in range(len(augmented_subset)):
                augmented_subset.iloc[idx] = PoseAugmentor.augment_row(
                    augmented_subset.iloc[idx],
                    landmark_cols,
                    aug_type
                )
            
            augmented_dfs.append(augmented_subset)
        
        # Combine all
        result = pd.concat(augmented_dfs, ignore_index=True)
        return result


def augment_csv(input_csv: str, output_csv: str = None, multiplier: int = 2):
    """
    Augment dataset CSV file.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path to output CSV (default: input_csv with _augmented suffix)
        multiplier: Dataset size multiplier (2 = double size)
    
    Returns:
        Augmented DataFrame
    """
    print(f"\n📊 Augmenting dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   Original size: {len(df)} rows")
    
    df_augmented = PoseAugmentor.augment_dataframe(df, multiplier=multiplier)
    print(f"   Augmented size: {len(df_augmented)} rows (+{len(df_augmented)-len(df)} new samples)")
    
    if output_csv is None:
        base = input_csv.rsplit('.', 1)[0]
        output_csv = f"{base}_augmented.csv"
    
    df_augmented.to_csv(output_csv, index=False)
    print(f"   ✅ Saved to: {output_csv}")
    
    return df_augmented


if __name__ == "__main__":
    # Example usage
    try:
        df_augmented = augment_csv('data/processed/cleaned_human_fall.csv', multiplier=2)
        print(f"\nAugmentation complete!")
        print(f"Class distribution after augmentation:")
        print(df_augmented['class_name'].value_counts())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure cleaned_human_fall.csv exists in data/processed/")
