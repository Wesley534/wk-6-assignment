"""
Create a small synthetic test dataset for demonstration purposes
This generates simple placeholder images so you can test the pipeline
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Configuration
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
VAL_DIR = os.path.join(DATASET_DIR, "val")

CLASSES = ["plastic", "paper", "glass", "metal", "cardboard"]
IMG_SIZE = 224

# Color mapping for each class (for synthetic images)
CLASS_COLORS = {
    "plastic": (255, 20, 147),    # Deep pink
    "paper": (255, 255, 255),     # White
    "glass": (173, 216, 230),     # Light blue
    "metal": (192, 192, 192),     # Silver
    "cardboard": (139, 69, 19),   # Brown
}

def create_synthetic_image(class_name, index, img_size=224):
    """
    Create a synthetic image with class-specific characteristics
    """
    # Create base image with class color
    base_color = CLASS_COLORS.get(class_name, (128, 128, 128))
    
    # Create image with slight variation
    img_array = np.random.randint(
        max(0, base_color[0] - 30),
        min(255, base_color[0] + 30),
        (img_size, img_size, 3),
        dtype=np.uint8
    )
    
    img = Image.fromarray(img_array)
    
    # Add some pattern to make it more realistic
    draw = ImageDraw.Draw(img)
    
    # Add text label (helps with visualization)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Add class name in corner
    text = class_name[0].upper()  # First letter
    draw.text((10, 10), text, fill=(0, 0, 0), font=font)
    
    # Add some random shapes/patterns
    for _ in range(5):
        x1 = np.random.randint(0, img_size)
        y1 = np.random.randint(0, img_size)
        x2 = np.random.randint(0, img_size)
        y2 = np.random.randint(0, img_size)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
    
    return img

def create_test_dataset(samples_per_class_train=50, samples_per_class_val=10, samples_per_class_test=10):
    """
    Create a synthetic test dataset
    
    Args:
        samples_per_class_train: Number of training samples per class
        samples_per_class_val: Number of validation samples per class
        samples_per_class_test: Number of test samples per class
    """
    print("\n" + "="*60)
    print("Creating Synthetic Test Dataset")
    print("="*60)
    print("\nThis will create placeholder images for testing the pipeline.")
    print("For real training, replace these with actual images of recyclable items.")
    print()
    
    total_samples = (samples_per_class_train + samples_per_class_val + samples_per_class_test) * len(CLASSES)
    print(f"Creating {total_samples} synthetic images...")
    print(f"  - Training: {samples_per_class_train} per class")
    print(f"  - Validation: {samples_per_class_val} per class")
    print(f"  - Test: {samples_per_class_test} per class")
    print()
    
    splits = [
        ("train", TRAIN_DIR, samples_per_class_train),
        ("val", VAL_DIR, samples_per_class_val),
        ("test", TEST_DIR, samples_per_class_test)
    ]
    
    for split_name, split_dir, samples_per_class in splits:
        print(f"Creating {split_name} set...", end=" ")
        
        for class_name in CLASSES:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(samples_per_class):
                img = create_synthetic_image(class_name, i, IMG_SIZE)
                img_path = os.path.join(class_dir, f"{class_name}_{i:04d}.jpg")
                img.save(img_path, quality=85)
        
        print(f"✓ ({samples_per_class * len(CLASSES)} images)")
    
    print("\n" + "="*60)
    print("✓ Synthetic dataset created successfully!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"  Training: {samples_per_class_train * len(CLASSES)} images")
    print(f"  Validation: {samples_per_class_val * len(CLASSES)} images")
    print(f"  Test: {samples_per_class_test * len(CLASSES)} images")
    print(f"  Total: {total_samples} images")
    print("\nYou can now run: python 2_train_model.py")
    print("\nNote: These are synthetic placeholder images.")
    print("For real-world accuracy, replace with actual photos of recyclable items.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create synthetic test dataset')
    parser.add_argument('--train', type=int, default=50,
                       help='Training samples per class (default: 50)')
    parser.add_argument('--val', type=int, default=10,
                       help='Validation samples per class (default: 10)')
    parser.add_argument('--test', type=int, default=10,
                       help='Test samples per class (default: 10)')
    
    args = parser.parse_args()
    
    create_test_dataset(
        samples_per_class_train=args.train,
        samples_per_class_val=args.val,
        samples_per_class_test=args.test
    )

