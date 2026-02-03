#!/usr/bin/env python3
"""
Dataset Preprocessing Script
============================

This script preprocesses the ASL Alphabet dataset by:
1. Loading images from the raw dataset
2. Detecting hand landmarks using MediaPipe
3. Normalizing landmarks
4. Saving as numpy arrays for efficient training

Usage:
------
python scripts/preprocess_dataset.py --input data/raw/asl_alphabet_train --output data/processed

Or with options:
python scripts/preprocess_dataset.py \
    --input data/raw/asl_alphabet_train \
    --output data/processed \
    --max-per-class 1000 \
    --visualize
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ASL Alphabet classes
CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space', 'del', 'nothing']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def setup_mediapipe():
    """Initialize MediaPipe Hands."""
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,  # For images (not video)
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    return hands


def extract_landmarks(image: np.ndarray, hands) -> Optional[np.ndarray]:
    """
    Extract hand landmarks from an image.

    Args:
        image: BGR image from OpenCV
        hands: MediaPipe Hands instance

    Returns:
        Normalized landmarks array (21, 3) or None if no hand detected
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    results = hands.process(rgb_image)

    if not results.multi_hand_landmarks:
        return None

    # Get first hand landmarks
    hand_landmarks = results.multi_hand_landmarks[0]

    # Convert to numpy array
    landmarks = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ], dtype=np.float32)

    # Normalize landmarks
    landmarks = normalize_landmarks(landmarks)

    return landmarks


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks for model input.

    Steps:
    1. Center on wrist (landmark 0)
    2. Scale by palm size
    """
    # Center on wrist
    wrist = landmarks[0:1]  # Keep dimension
    centered = landmarks - wrist

    # Scale by palm size (wrist to middle finger MCP)
    palm_size = np.linalg.norm(landmarks[9] - landmarks[0])
    if palm_size > 1e-6:
        normalized = centered / palm_size
    else:
        normalized = centered

    return normalized.astype(np.float32)


def process_class_directory(
    class_dir: Path,
    class_name: str,
    hands,
    max_samples: Optional[int] = None,
    visualize: bool = False
) -> Tuple[List[np.ndarray], Dict]:
    """
    Process all images in a class directory.

    Returns:
        Tuple of (landmarks_list, stats_dict)
    """
    landmarks_list = []
    stats = {'total': 0, 'success': 0, 'failed': 0}

    # Get all image files
    image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

    if max_samples:
        image_files = image_files[:max_samples]

    for img_path in tqdm(image_files, desc=f"Processing {class_name}", leave=False):
        stats['total'] += 1

        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # Extract landmarks
            landmarks = extract_landmarks(image, hands)

            if landmarks is None:
                raise ValueError("No hand detected")

            landmarks_list.append(landmarks)
            stats['success'] += 1

            # Visualize if requested
            if visualize and stats['success'] <= 3:
                visualize_landmarks(image, landmarks, class_name, img_path.stem)

        except Exception as e:
            stats['failed'] += 1
            if stats['failed'] <= 5:  # Only log first few failures
                logger.warning(f"Failed to process {img_path.name}: {e}")

    return landmarks_list, stats


def visualize_landmarks(image: np.ndarray, landmarks: np.ndarray, class_name: str, filename: str):
    """Visualize landmarks on image for debugging."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original image with landmarks
    ax1 = axes[0]
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Denormalize landmarks for visualization
    h, w = image.shape[:2]
    # Note: landmarks are normalized, need to approximate original positions
    ax1.set_title(f"Class: {class_name}")

    # Normalized landmarks plot
    ax2 = axes[1]
    ax2.scatter(landmarks[:, 0], -landmarks[:, 1], c='blue', s=50)

    # Connect landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]

    for start, end in connections:
        ax2.plot(
            [landmarks[start, 0], landmarks[end, 0]],
            [-landmarks[start, 1], -landmarks[end, 1]],
            'g-', linewidth=1
        )

    ax2.set_title(f"Normalized Landmarks: {class_name}")
    ax2.set_aspect('equal')
    ax2.grid(True)

    plt.tight_layout()

    # Save visualization
    viz_dir = Path('data/visualizations')
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_dir / f"{class_name}_{filename}.png")
    plt.close()

    logger.info(f"Saved visualization: {class_name}_{filename}.png")


def create_train_val_test_split(
    num_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Dict[str, List[int]]:
    """Create random train/val/test split indices."""
    indices = np.random.permutation(num_samples)

    train_end = int(train_ratio * num_samples)
    val_end = int((train_ratio + val_ratio) * num_samples)

    return {
        'train': indices[:train_end].tolist(),
        'val': indices[train_end:val_end].tolist(),
        'test': indices[val_end:].tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess ASL Alphabet Dataset')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw/asl_alphabet_train',
        help='Path to raw dataset directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed',
        help='Path to output directory'
    )
    parser.add_argument(
        '--max-per-class',
        type=int,
        default=None,
        help='Maximum samples per class (for testing)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save visualization of first few samples'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.info("\nPlease download the ASL Alphabet dataset from Kaggle:")
        logger.info("https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        logger.info(f"\nThen extract it to: {input_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ASL Alphabet Dataset Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max samples per class: {args.max_per_class or 'all'}")
    logger.info("=" * 60)

    # Initialize MediaPipe
    logger.info("\nInitializing MediaPipe Hands...")
    hands = setup_mediapipe()

    # Process each class
    all_landmarks = []
    all_labels = []
    total_stats = {'total': 0, 'success': 0, 'failed': 0, 'per_class': {}}

    logger.info("\nProcessing classes...")

    for class_name in CLASSES:
        class_dir = input_dir / class_name

        if not class_dir.exists():
            # Try alternate names
            alt_names = [class_name.lower(), class_name.upper()]
            found = False
            for alt in alt_names:
                alt_dir = input_dir / alt
                if alt_dir.exists():
                    class_dir = alt_dir
                    found = True
                    break

            if not found:
                logger.warning(f"Class directory not found: {class_name}")
                continue

        # Process class
        landmarks_list, stats = process_class_directory(
            class_dir,
            class_name,
            hands,
            max_samples=args.max_per_class,
            visualize=args.visualize
        )

        # Add to arrays
        class_idx = CLASS_TO_IDX[class_name]
        for landmarks in landmarks_list:
            all_landmarks.append(landmarks)
            all_labels.append(class_idx)

        # Update stats
        total_stats['total'] += stats['total']
        total_stats['success'] += stats['success']
        total_stats['failed'] += stats['failed']
        total_stats['per_class'][class_name] = stats

        logger.info(f"  {class_name}: {stats['success']}/{stats['total']} samples")

    # Close MediaPipe
    hands.close()

    if not all_landmarks:
        logger.error("No samples were successfully processed!")
        return

    # Convert to numpy arrays
    logger.info("\nConverting to numpy arrays...")
    landmarks_array = np.array(all_landmarks, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    logger.info(f"  Landmarks shape: {landmarks_array.shape}")
    logger.info(f"  Labels shape: {labels_array.shape}")

    # Create train/val/test split
    logger.info("\nCreating train/val/test split...")
    split_indices = create_train_val_test_split(len(labels_array))

    logger.info(f"  Train: {len(split_indices['train'])} samples")
    logger.info(f"  Val: {len(split_indices['val'])} samples")
    logger.info(f"  Test: {len(split_indices['test'])} samples")

    # Save data
    logger.info("\nSaving preprocessed data...")

    np.save(output_dir / 'landmarks.npy', landmarks_array)
    np.save(output_dir / 'labels.npy', labels_array)

    with open(output_dir / 'split_indices.json', 'w') as f:
        json.dump(split_indices, f, indent=2)

    # Save metadata
    metadata = {
        'classes': CLASSES,
        'num_classes': len(CLASSES),
        'num_samples': len(labels_array),
        'landmarks_shape': list(landmarks_array.shape),
        'class_to_idx': CLASS_TO_IDX,
        'stats': total_stats
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples processed: {total_stats['success']}/{total_stats['total']}")
    logger.info(f"Success rate: {100 * total_stats['success'] / total_stats['total']:.1f}%")
    logger.info(f"\nOutput files saved to: {output_dir}")
    logger.info("  - landmarks.npy")
    logger.info("  - labels.npy")
    logger.info("  - split_indices.json")
    logger.info("  - metadata.json")
    logger.info("\nYou can now train the model with:")
    logger.info(f"  python main.py --mode train --data-dir {output_dir}")


if __name__ == "__main__":
    main()
