"""
Dataset Module for Sign Language Recognition
=============================================

This module handles data loading, preprocessing, and augmentation for
training sign language recognition models.

Dataset Options:
----------------

1. **ASL Alphabet Dataset** (Recommended for start)
   - 87,000+ images of ASL alphabet
   - 29 classes (A-Z + space, del, nothing)
   - Good for testing pipeline
   - Source: Kaggle ASL Alphabet

2. **WLASL (Word-Level ASL)**
   - 2,000 words
   - Video dataset
   - More challenging
   - Source: WLASL GitHub

3. **RWTH-PHOENIX-Weather 2014T**
   - German Sign Language
   - Continuous signing
   - Research benchmark

4. **MS-ASL**
   - 25,000+ videos
   - 1,000 signs
   - Large scale

For this project, we start with ASL Alphabet for simplicity,
then optionally extend to WLASL for word-level recognition.

Data Flow:
----------
Raw Images/Videos → Frame Extraction → Landmark Detection →
Normalization → Augmentation → Tensor → Model

Why Landmarks Instead of Raw Images?
------------------------------------
1. Much smaller data (63 floats vs 224x224x3 = 150528 values)
2. Position/scale invariant after normalization
3. Lighting/background invariant
4. Faster training and inference
5. Model focuses on hand shape, not irrelevant features
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, Tuple, List, Dict, Callable, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ASLDataset(Dataset):
    """
    Dataset for ASL sign language recognition.

    Supports two modes:
    1. **Landmark mode**: Pre-extracted landmarks (recommended)
    2. **Image mode**: Raw images with on-the-fly landmark extraction

    For training, use landmark mode - it's much faster.

    Data Format (Landmark Mode):
    ----------------------------
    Expects a directory structure:
    data/processed/
    ├── landmarks.npy      # All landmarks (N, 21, 3)
    ├── labels.npy         # All labels (N,)
    ├── metadata.json      # Class names, statistics
    └── split_indices.json # Train/val/test splits

    Or per-class structure:
    data/processed/
    ├── A/
    │   ├── 0001.npy       # Landmarks for sample 1
    │   ├── 0002.npy
    │   └── ...
    ├── B/
    │   └── ...
    └── ...

    Usage:
    ------
    >>> dataset = ASLDataset(data_dir='data/processed')
    >>> landmarks, label = dataset[0]
    >>> print(f"Landmarks shape: {landmarks.shape}, Label: {label}")

    With augmentation:
    >>> dataset = ASLDataset(
    ...     data_dir='data/processed',
    ...     augmentation=True,
    ...     sequence_length=30  # For temporal models
    ... )
    """

    # Class mapping (ASL Alphabet + special tokens)
    CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space', 'del', 'nothing']
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 1,  # 1 for static, >1 for temporal
        augmentation: bool = True,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to processed data directory
            split: 'train', 'val', or 'test'
            sequence_length: Number of frames per sample (1 for static)
            augmentation: Whether to apply data augmentation
            transform: Optional additional transform function
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.augmentation = augmentation and split == 'train'
        self.transform = transform

        # Load data
        self.landmarks: np.ndarray = None
        self.labels: np.ndarray = None
        self.sequences: List[List[int]] = []  # For temporal mode

        self._load_data()

        logger.info(f"ASLDataset initialized: {len(self)} samples, "
                   f"split={split}, seq_len={sequence_length}, aug={self.augmentation}")

    def _load_data(self) -> None:
        """Load data from disk."""
        # Check for consolidated files first
        landmarks_file = self.data_dir / 'landmarks.npy'
        labels_file = self.data_dir / 'labels.npy'

        if landmarks_file.exists() and labels_file.exists():
            self._load_consolidated()
        else:
            self._load_per_class()

        # Load or create split indices
        self._apply_split()

    def _load_consolidated(self) -> None:
        """Load from consolidated numpy files."""
        self.landmarks = np.load(self.data_dir / 'landmarks.npy')
        self.labels = np.load(self.data_dir / 'labels.npy')

        logger.info(f"Loaded {len(self.labels)} samples from consolidated files")

    def _load_per_class(self) -> None:
        """Load from per-class directory structure."""
        all_landmarks = []
        all_labels = []

        for class_name in self.CLASSES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            class_idx = self.CLASS_TO_IDX[class_name]

            for npy_file in sorted(class_dir.glob('*.npy')):
                try:
                    landmarks = np.load(npy_file)
                    all_landmarks.append(landmarks)
                    all_labels.append(class_idx)
                except Exception as e:
                    logger.warning(f"Failed to load {npy_file}: {e}")

        if all_landmarks:
            self.landmarks = np.array(all_landmarks)
            self.labels = np.array(all_labels)
            logger.info(f"Loaded {len(self.labels)} samples from per-class directories")
        else:
            # Create dummy data for testing
            logger.warning("No data found, creating dummy dataset for testing")
            self._create_dummy_data()

    def _create_dummy_data(self, num_samples: int = 1000) -> None:
        """Create dummy data for testing pipeline."""
        self.landmarks = np.random.randn(num_samples, 21, 3).astype(np.float32)
        self.labels = np.random.randint(0, len(self.CLASSES), num_samples)
        logger.warning(f"Created {num_samples} dummy samples")

    def _apply_split(self) -> None:
        """Apply train/val/test split."""
        split_file = self.data_dir / 'split_indices.json'

        if split_file.exists():
            with open(split_file) as f:
                splits = json.load(f)
            indices = splits.get(self.split, list(range(len(self.labels))))
        else:
            # Create random split
            n = len(self.labels)
            indices = np.random.permutation(n)

            train_end = int(0.8 * n)
            val_end = int(0.9 * n)

            if self.split == 'train':
                indices = indices[:train_end]
            elif self.split == 'val':
                indices = indices[train_end:val_end]
            else:  # test
                indices = indices[val_end:]

        # Apply split
        self.landmarks = self.landmarks[indices]
        self.labels = self.labels[indices]

        # For temporal mode, group into sequences
        if self.sequence_length > 1:
            self._create_sequences()

    def _create_sequences(self) -> None:
        """Group samples into sequences for temporal models."""
        # For demonstration, create sequences by sliding window
        # In real data, sequences would come from video frames
        n = len(self.labels)
        self.sequences = []

        for i in range(n - self.sequence_length + 1):
            # Check if all frames in sequence have same label
            sequence_labels = self.labels[i:i + self.sequence_length]
            if len(set(sequence_labels)) == 1:  # Same class
                self.sequences.append(list(range(i, i + self.sequence_length)))

        logger.info(f"Created {len(self.sequences)} sequences of length {self.sequence_length}")

    def _augment_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to landmarks.

        Augmentation strategies for landmarks:
        1. Rotation: Rotate hand slightly (±15°)
        2. Scale: Vary hand size (0.9-1.1x)
        3. Translation: Small position shifts
        4. Noise: Add Gaussian noise
        5. Flip: Horizontal flip (careful - changes handedness!)

        Note: These are applied to normalized landmarks, so they
        simulate natural variation in signing.
        """
        aug_landmarks = landmarks.copy()

        # Random rotation around z-axis (in image plane)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            aug_landmarks = aug_landmarks @ rotation.T

        # Random scale
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            aug_landmarks = aug_landmarks * scale

        # Random translation
        if np.random.random() < 0.5:
            translation = np.random.uniform(-0.1, 0.1, size=(1, 3))
            aug_landmarks = aug_landmarks + translation

        # Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, aug_landmarks.shape)
            aug_landmarks = aug_landmarks + noise

        return aug_landmarks.astype(np.float32)

    def __len__(self) -> int:
        if self.sequence_length > 1 and self.sequences:
            return len(self.sequences)
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Returns:
            Tuple of (landmarks_tensor, label)
            - For static: landmarks shape (21, 3)
            - For temporal: landmarks shape (seq_len, 21, 3)
        """
        if self.sequence_length > 1 and self.sequences:
            # Temporal mode
            seq_indices = self.sequences[idx]
            landmarks = self.landmarks[seq_indices]  # (seq_len, 21, 3)
            label = self.labels[seq_indices[0]]

            if self.augmentation:
                # Augment each frame consistently
                aug_landmarks = []
                for frame in landmarks:
                    aug_landmarks.append(self._augment_landmarks(frame))
                landmarks = np.stack(aug_landmarks)
        else:
            # Static mode
            landmarks = self.landmarks[idx]
            label = self.labels[idx]

            if self.augmentation:
                landmarks = self._augment_landmarks(landmarks)

        # Apply custom transform
        if self.transform:
            landmarks = self.transform(landmarks)

        return torch.FloatTensor(landmarks), int(label)

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.

        Returns:
            Tensor of weights for each class

        This is useful for weighted loss functions or samplers when
        some classes have fewer samples than others.
        """
        class_counts = np.bincount(self.labels, minlength=len(self.CLASSES))
        class_counts = np.maximum(class_counts, 1)  # Avoid division by zero

        # Inverse frequency weighting
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(self.CLASSES)

        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Get per-sample weights for weighted random sampling.

        Returns:
            Tensor of weight for each sample
        """
        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.labels]
        return sample_weights


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    sequence_length: int = 1,
    num_workers: int = 4,
    balanced_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_dir: Path to processed data
        batch_size: Batch size for training
        sequence_length: Number of frames per sample
        num_workers: Number of data loading workers
        balanced_sampling: Use weighted sampling for class balance

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ASLDataset(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        augmentation=True
    )

    val_dataset = ASLDataset(
        data_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        augmentation=False
    )

    test_dataset = ASLDataset(
        data_dir=data_dir,
        split='test',
        sequence_length=sequence_length,
        augmentation=False
    )

    # Create samplers
    train_sampler = None
    if balanced_sampling and len(train_dataset) > 0:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created data loaders: train={len(train_dataset)}, "
               f"val={len(val_dataset)}, test={len(test_dataset)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# Data Preprocessing Utilities
# =============================================================================

def extract_landmarks_from_images(
    image_dir: str,
    output_dir: str,
    max_samples_per_class: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract landmarks from image dataset using MediaPipe.

    This preprocesses a raw image dataset (like ASL Alphabet from Kaggle)
    into landmark format for efficient training.

    Args:
        image_dir: Path to raw images (organized by class folders)
        output_dir: Path to save extracted landmarks
        max_samples_per_class: Limit samples per class (for testing)

    Returns:
        Dictionary with extraction statistics
    """
    import cv2
    from ..src.detection import LandmarkDetector

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = LandmarkDetector(max_hands=1)
    stats = {'total': 0, 'success': 0, 'failed': 0, 'per_class': {}}

    for class_name in ASLDataset.CLASSES:
        class_input = image_dir / class_name
        class_output = output_dir / class_name
        class_output.mkdir(exist_ok=True)

        if not class_input.exists():
            logger.warning(f"Class directory not found: {class_input}")
            continue

        class_stats = {'success': 0, 'failed': 0}
        images = list(class_input.glob('*.jpg')) + list(class_input.glob('*.png'))

        if max_samples_per_class:
            images = images[:max_samples_per_class]

        for i, img_path in enumerate(images):
            stats['total'] += 1

            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    raise ValueError("Failed to load image")

                # Detect landmarks
                hands, _ = detector.detect(image)

                if not hands:
                    raise ValueError("No hand detected")

                # Get normalized landmarks
                landmarks = hands[0].normalize()

                # Save
                output_path = class_output / f"{i:05d}.npy"
                np.save(output_path, landmarks)

                stats['success'] += 1
                class_stats['success'] += 1

            except Exception as e:
                stats['failed'] += 1
                class_stats['failed'] += 1
                if stats['failed'] <= 10:  # Only log first 10 failures
                    logger.warning(f"Failed to process {img_path}: {e}")

        stats['per_class'][class_name] = class_stats
        logger.info(f"Processed {class_name}: {class_stats['success']}/{len(images)} success")

    detector.close()

    # Save metadata
    metadata = {
        'classes': ASLDataset.CLASSES,
        'num_classes': len(ASLDataset.CLASSES),
        'stats': stats
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Extraction complete: {stats['success']}/{stats['total']} success")
    return stats


# =============================================================================
# Testing
# =============================================================================

def test_dataset():
    """Test dataset loading."""
    print("Testing ASLDataset...")

    # Test with dummy data
    dataset = ASLDataset(
        data_dir='data/processed',
        split='train',
        sequence_length=1,
        augmentation=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.CLASSES}")

    # Get a sample
    landmarks, label = dataset[0]
    print(f"Sample landmarks shape: {landmarks.shape}")
    print(f"Sample label: {label} ({dataset.IDX_TO_CLASS[label]})")

    # Test data loader
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir='data/processed',
        batch_size=4,
        sequence_length=1
    )

    # Get a batch
    batch_landmarks, batch_labels = next(iter(train_loader))
    print(f"\nBatch landmarks shape: {batch_landmarks.shape}")
    print(f"Batch labels: {batch_labels}")

    # Test temporal mode
    print("\n--- Testing Temporal Mode ---")
    temporal_dataset = ASLDataset(
        data_dir='data/processed',
        split='train',
        sequence_length=10,
        augmentation=True
    )

    print(f"Temporal dataset size: {len(temporal_dataset)}")
    if len(temporal_dataset) > 0:
        landmarks, label = temporal_dataset[0]
        print(f"Temporal sample shape: {landmarks.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataset()
