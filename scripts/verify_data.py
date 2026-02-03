#!/usr/bin/env python3
"""
Data Verification Script
========================

Verifies that the preprocessed data is correct and ready for training.

Usage:
------
python scripts/verify_data.py --data-dir data/processed
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Verify preprocessed data')
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data/processed',
        help='Path to processed data directory'
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)

    # Check if directory exists
    if not data_dir.exists():
        print(f"\n❌ ERROR: Data directory not found: {data_dir}")
        print("\nPlease run preprocessing first:")
        print("  python scripts/preprocess_dataset.py")
        return False

    # Check required files
    required_files = ['landmarks.npy', 'labels.npy', 'metadata.json', 'split_indices.json']
    missing_files = []

    print("\n📁 Checking required files...")
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {filename} ({size:.2f} MB)")
        else:
            print(f"  ✗ {filename} - MISSING")
            missing_files.append(filename)

    if missing_files:
        print(f"\n❌ ERROR: Missing files: {missing_files}")
        return False

    # Load and verify data
    print("\n📊 Loading data...")
    landmarks = np.load(data_dir / 'landmarks.npy')
    labels = np.load(data_dir / 'labels.npy')

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    with open(data_dir / 'split_indices.json') as f:
        splits = json.load(f)

    # Print statistics
    print("\n📈 Data Statistics:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Verify shapes
    print("\n🔍 Verifying data integrity...")

    if landmarks.shape[0] != labels.shape[0]:
        print(f"  ✗ Mismatch: {landmarks.shape[0]} landmarks vs {labels.shape[0]} labels")
        return False
    print(f"  ✓ Landmarks and labels match: {len(labels)} samples")

    if landmarks.shape[1:] != (21, 3):
        print(f"  ✗ Unexpected landmark shape: {landmarks.shape[1:]}")
        return False
    print(f"  ✓ Landmark shape correct: (21, 3)")

    # Check for NaN or Inf
    if np.isnan(landmarks).any():
        print("  ✗ Found NaN values in landmarks!")
        return False
    if np.isinf(landmarks).any():
        print("  ✗ Found Inf values in landmarks!")
        return False
    print("  ✓ No NaN or Inf values")

    # Check label range
    num_classes = metadata['num_classes']
    if labels.min() < 0 or labels.max() >= num_classes:
        print(f"  ✗ Labels out of range: [{labels.min()}, {labels.max()}]")
        return False
    print(f"  ✓ Labels in valid range: [0, {num_classes-1}]")

    # Check splits
    total_in_splits = len(splits['train']) + len(splits['val']) + len(splits['test'])
    if total_in_splits != len(labels):
        print(f"  ✗ Split indices don't match: {total_in_splits} vs {len(labels)}")
        return False
    print(f"  ✓ Split indices valid")

    # Class distribution
    print("\n📊 Class Distribution:")
    classes = metadata['classes']
    class_counts = np.bincount(labels, minlength=num_classes)

    for i, (cls, count) in enumerate(zip(classes, class_counts)):
        bar = '█' * (count // 100) if count > 0 else ''
        print(f"  {cls:>8}: {count:5d} {bar}")

    # Check for empty classes
    empty_classes = [classes[i] for i in range(num_classes) if class_counts[i] == 0]
    if empty_classes:
        print(f"\n⚠️ WARNING: Empty classes: {empty_classes}")

    # Split statistics
    print("\n📊 Split Statistics:")
    print(f"  Train: {len(splits['train'])} samples ({100*len(splits['train'])/len(labels):.1f}%)")
    print(f"  Val:   {len(splits['val'])} samples ({100*len(splits['val'])/len(labels):.1f}%)")
    print(f"  Test:  {len(splits['test'])} samples ({100*len(splits['test'])/len(labels):.1f}%)")

    # Landmark statistics
    print("\n📊 Landmark Statistics:")
    print(f"  Mean: {landmarks.mean():.4f}")
    print(f"  Std:  {landmarks.std():.4f}")
    print(f"  Min:  {landmarks.min():.4f}")
    print(f"  Max:  {landmarks.max():.4f}")

    # Sample a few landmarks to visualize
    print("\n🔍 Sample Landmarks (first sample):")
    sample = landmarks[0]
    print(f"  Wrist (0):       [{sample[0, 0]:.4f}, {sample[0, 1]:.4f}, {sample[0, 2]:.4f}]")
    print(f"  Index tip (8):   [{sample[8, 0]:.4f}, {sample[8, 1]:.4f}, {sample[8, 2]:.4f}]")
    print(f"  Pinky tip (20):  [{sample[20, 0]:.4f}, {sample[20, 1]:.4f}, {sample[20, 2]:.4f}]")

    print("\n" + "=" * 60)
    print("✅ DATA VERIFICATION PASSED")
    print("=" * 60)
    print("\nYour data is ready for training! Run:")
    print(f"  python main.py --mode train --data-dir {data_dir}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
