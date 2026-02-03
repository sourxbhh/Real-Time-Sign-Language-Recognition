"""
Preprocessing Utilities
=======================

Functions for preprocessing hand landmarks and preparing data for the model.
"""

import numpy as np
from typing import Optional, Tuple


def normalize_landmarks(
    landmarks: np.ndarray,
    center_on_wrist: bool = True,
    scale_by_palm: bool = True
) -> np.ndarray:
    """
    Normalize hand landmarks for model input.

    Args:
        landmarks: Raw landmarks array of shape (21, 3) or (N, 21, 3)
        center_on_wrist: Center landmarks on wrist (landmark 0)
        scale_by_palm: Scale by palm size for size invariance

    Returns:
        Normalized landmarks with same shape as input

    Why Normalize?
    --------------
    Raw landmarks are in image-relative coordinates (0-1). Normalizing:
    1. Centers the hand at origin → position invariant
    2. Scales by palm size → size invariant
    3. Allows model to focus on hand shape, not absolute position
    """
    landmarks = np.array(landmarks, dtype=np.float32)
    original_shape = landmarks.shape

    # Handle batch dimension
    if landmarks.ndim == 2:
        landmarks = landmarks[np.newaxis, ...]  # Add batch dimension

    normalized = landmarks.copy()

    for i in range(len(normalized)):
        # Center on wrist
        if center_on_wrist:
            wrist = normalized[i, 0:1, :]  # Keep dimension for broadcasting
            normalized[i] = normalized[i] - wrist

        # Scale by palm size
        if scale_by_palm:
            # Palm size = distance from wrist to middle finger MCP
            wrist = landmarks[i, 0]
            middle_mcp = landmarks[i, 9]
            palm_size = np.linalg.norm(middle_mcp - wrist)

            if palm_size > 1e-6:  # Avoid division by zero
                normalized[i] = normalized[i] / palm_size

    # Remove batch dimension if input didn't have one
    if len(original_shape) == 2:
        normalized = normalized[0]

    return normalized


def augment_landmarks(
    landmarks: np.ndarray,
    rotation_range: float = 15.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    translation_range: float = 0.1,
    noise_std: float = 0.02,
    flip_horizontal: bool = False
) -> np.ndarray:
    """
    Apply data augmentation to landmarks.

    Args:
        landmarks: Landmarks array of shape (21, 3) or (N, 21, 3)
        rotation_range: Maximum rotation in degrees (±)
        scale_range: Min and max scale factors
        translation_range: Maximum translation as fraction
        noise_std: Standard deviation of Gaussian noise
        flip_horizontal: Whether to randomly flip horizontally

    Returns:
        Augmented landmarks

    Augmentation Strategies:
    ------------------------
    1. **Rotation**: Simulates different hand orientations
    2. **Scale**: Simulates different distances from camera
    3. **Translation**: Simulates different hand positions
    4. **Noise**: Simulates detection inaccuracies
    5. **Flip**: Creates mirror images (careful with handedness!)
    """
    landmarks = np.array(landmarks, dtype=np.float32)
    original_shape = landmarks.shape

    # Handle batch dimension
    if landmarks.ndim == 2:
        landmarks = landmarks[np.newaxis, ...]

    augmented = landmarks.copy()

    for i in range(len(augmented)):
        # Random rotation around z-axis
        if rotation_range > 0 and np.random.random() < 0.5:
            angle = np.random.uniform(-rotation_range, rotation_range)
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            augmented[i] = augmented[i] @ rotation_matrix.T

        # Random scale
        if scale_range[0] != 1.0 or scale_range[1] != 1.0:
            if np.random.random() < 0.5:
                scale = np.random.uniform(scale_range[0], scale_range[1])
                augmented[i] = augmented[i] * scale

        # Random translation
        if translation_range > 0 and np.random.random() < 0.5:
            translation = np.random.uniform(
                -translation_range, translation_range, size=(1, 3)
            )
            augmented[i] = augmented[i] + translation

        # Gaussian noise
        if noise_std > 0 and np.random.random() < 0.3:
            noise = np.random.normal(0, noise_std, augmented[i].shape)
            augmented[i] = augmented[i] + noise

        # Horizontal flip (careful - changes handedness!)
        if flip_horizontal and np.random.random() < 0.5:
            augmented[i, :, 0] = -augmented[i, :, 0]

    # Remove batch dimension if input didn't have one
    if len(original_shape) == 2:
        augmented = augmented[0]

    return augmented.astype(np.float32)


def landmarks_to_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert landmarks to finger joint angles.

    Args:
        landmarks: Landmarks array of shape (21, 3)

    Returns:
        Array of joint angles (shape depends on representation)

    This provides an alternative representation that may be more
    invariant to hand position and size.
    """
    angles = []

    # Finger joint chains: MCP → PIP → TIP
    finger_chains = [
        [1, 2, 3, 4],    # Thumb
        [5, 6, 7, 8],    # Index
        [9, 10, 11, 12], # Middle
        [13, 14, 15, 16],# Ring
        [17, 18, 19, 20] # Pinky
    ]

    for chain in finger_chains:
        for i in range(len(chain) - 2):
            # Three consecutive joints
            p1 = landmarks[chain[i]]
            p2 = landmarks[chain[i + 1]]
            p3 = landmarks[chain[i + 2]]

            # Vectors
            v1 = p1 - p2
            v2 = p3 - p2

            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)

    return np.array(angles, dtype=np.float32)


def landmarks_to_distances(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert landmarks to pairwise distances.

    Args:
        landmarks: Landmarks array of shape (21, 3)

    Returns:
        Array of selected distances

    Distance-based features are inherently position and rotation invariant.
    """
    distances = []

    # Key landmark pairs for sign language
    pairs = [
        # Fingertip to wrist
        (4, 0), (8, 0), (12, 0), (16, 0), (20, 0),
        # Fingertip to thumb tip
        (8, 4), (12, 4), (16, 4), (20, 4),
        # Adjacent fingertips
        (4, 8), (8, 12), (12, 16), (16, 20),
        # Fingertip to palm center (approximated by MCP average)
        # ... add more as needed
    ]

    for i, j in pairs:
        dist = np.linalg.norm(landmarks[i] - landmarks[j])
        distances.append(dist)

    return np.array(distances, dtype=np.float32)


def create_heatmap(
    landmarks: np.ndarray,
    image_size: Tuple[int, int] = (64, 64),
    sigma: float = 2.0
) -> np.ndarray:
    """
    Create Gaussian heatmap from landmarks.

    Args:
        landmarks: Normalized landmarks (21, 3)
        image_size: Output heatmap size (H, W)
        sigma: Gaussian sigma

    Returns:
        Heatmap array of shape (21, H, W)

    This is useful for CNN-based approaches that expect spatial input.
    """
    H, W = image_size
    num_landmarks = landmarks.shape[0]
    heatmaps = np.zeros((num_landmarks, H, W), dtype=np.float32)

    # Create coordinate grids
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    for i in range(num_landmarks):
        # Convert normalized coords to pixel coords
        # Landmarks are in range [-1, 1] after normalization
        lx = (landmarks[i, 0] + 1) / 2 * W
        ly = (landmarks[i, 1] + 1) / 2 * H

        # Gaussian heatmap
        heatmaps[i] = np.exp(-((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * sigma ** 2))

    return heatmaps
