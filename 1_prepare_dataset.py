"""
Dataset Preparation Script for Recyclable Items Classification
Downloads and organizes a dataset for training and testing
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

# Dataset configuration
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Recyclable item categories
CLASSES = ["plastic", "paper", "glass", "metal", "cardboard"]

# Image configuration
IMG_SIZE = 224
IMG_CHANNELS = 3

def create_directory_structure():
    """Create directory structure for the dataset"""
    for directory in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        for class_name in CLASSES:
            os.makedirs(os.path.join(directory, class_name), exist_ok=True)
    print("Directory structure created successfully!")

def download_sample_dataset():
    """
    For demonstration, we'll create synthetic data or use a small dataset.
    In production, you would download from a real dataset source.
    
    Note: This script creates placeholder structure. For actual training,
    you'll need to populate these directories with real images.
    """
    print("\n" + "="*60)
    print("Dataset Preparation for Recyclable Items Classification")
    print("="*60)
    
    # Create directory structure
    create_directory_structure()
    
    # Create a sample dataset info file
    dataset_info = {
        "classes": CLASSES,
        "image_size": IMG_SIZE,
        "channels": IMG_CHANNELS,
        "train_dir": TRAIN_DIR,
        "test_dir": TEST_DIR,
        "val_dir": VAL_DIR,
        "total_classes": len(CLASSES)
    }
    
    print(f"\nDataset structure created with {len(CLASSES)} classes:")
    for i, class_name in enumerate(CLASSES, 1):
        print(f"  {i}. {class_name}")
    
    print(f"\nImage configuration:")
    print(f"  Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Channels: {IMG_CHANNELS}")
    
    print("\n" + "-"*60)
    print("IMPORTANT: Dataset setup required")
    print("-"*60)
    print("Option 1: Create synthetic test dataset (for testing pipeline)")
    print("  Run: python create_test_dataset.py")
    print("\nOption 2: Use real dataset")
    print("  You can:")
    print("  1. Download the TrashNet dataset from GitHub")
    print("  2. Use TensorFlow datasets API")
    print("  3. Use your own collection of recyclable item images")
    print("\nPlace your images in the following structure:")
    print(f"  {TRAIN_DIR}/")
    print("    plastic/")
    print("    paper/")
    print("    glass/")
    print("    metal/")
    print("    cardboard/")
    print(f"\n  {TEST_DIR}/ (same structure)")
    print(f"  {VAL_DIR}/ (same structure)")
    print("\nRecommended: At least 50-100 images per class in training set")
    
    # Save dataset info
    import json
    with open(os.path.join(DATASET_DIR, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset info saved to {DATASET_DIR}/dataset_info.json")
    print("\n" + "="*60)
    
    return dataset_info

def create_synthetic_dataset(num_samples_per_class=10):
    """
    Create a small synthetic dataset for testing purposes.
    This generates simple colored rectangles as placeholder images.
    """
    print("\nCreating synthetic dataset for testing...")
    
    for split_dir in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        for class_name in CLASSES:
            class_dir = os.path.join(split_dir, class_name)
            
            # Generate synthetic images
            num_samples = num_samples_per_class if split_dir == TRAIN_DIR else num_samples_per_class // 3
            
            for i in range(num_samples):
                # Create a simple colored image
                img_array = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = os.path.join(class_dir, f"{class_name}_{i}.jpg")
                img.save(img_path)
    
    print(f"Synthetic dataset created with {num_samples_per_class} samples per class in training set")

if __name__ == "__main__":
    # Create directory structure
    dataset_info = download_sample_dataset()
    
    # Create synthetic dataset for testing (optional)
    # Uncomment the line below to create synthetic test images
    # create_synthetic_dataset(num_samples_per_class=20)
    
    print("\n✓ Dataset preparation complete!")

