"""
Train a Lightweight Image Classification Model for Recyclable Items
Uses Transfer Learning with MobileNet for efficient edge deployment
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_CLASSES = 5  # plastic, paper, glass, metal, cardboard

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_DIR = "models"
CLASSES = ["plastic", "paper", "glass", "metal", "cardboard"]
os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset_info():
    """Load dataset information"""
    info_path = os.path.join(DATASET_DIR, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            return json.load(f)
    return {"classes": ["plastic", "paper", "glass", "metal", "cardboard"]}

def create_data_generators():
    """Create data generators for training and validation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Check if dataset exists
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory {TRAIN_DIR} not found!")
        print("Please run 1_prepare_dataset.py first and populate with images.")
        return None, None
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

def build_model(num_classes):
    """
    Build a lightweight model using MobileNetV2 transfer learning
    MobileNet is optimized for mobile and edge devices
    """
    print("\nBuilding MobileNetV2-based model for edge deployment...")
    
    # Load pre-trained MobileNetV2 (without top layer)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0  # Width multiplier (1.0 = full width, smaller = lighter model)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add dropout for regularization
    x = layers.Dropout(0.2)(x)
    
    # Dense layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    print(f"Model created with {model.count_params():,} total parameters")
    
    return model, base_model

def train_model(model, train_gen, val_gen, epochs=EPOCHS):
    """Train the model"""
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model

def fine_tune_model(model, base_model, train_gen, val_gen, epochs=10):
    """Fine-tune the model by unfreezing some base layers"""
    
    print("\n" + "="*60)
    print("Starting Fine-tuning...")
    print("="*60)
    
    # Unfreeze top layers of base model
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    # Continue training
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        verbose=1
    )
    
    return history, model

def plot_training_history(history):
    """Plot training history"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to {MODEL_DIR}/training_history.png")
    plt.close()

def save_model_summary(model):
    """Save model summary to file"""
    summary_path = os.path.join(MODEL_DIR, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {summary_path}")

def main():
    print("\n" + "="*60)
    print("Recyclable Items Image Classification - Model Training")
    print("="*60)
    
    # Load dataset info
    dataset_info = load_dataset_info()
    num_classes = len(dataset_info.get("classes", NUM_CLASSES))
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    if train_gen is None:
        print("\nPlease prepare your dataset first!")
        return
    
    print(f"\nClasses found: {list(train_gen.class_indices.keys())}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Check if dataset is empty
    if train_gen.samples == 0 or val_gen.samples == 0:
        print("\n" + "="*60)
        print("ERROR: Empty Dataset Detected!")
        print("="*60)
        print("\nThe dataset directories exist but contain no images.")
        print("\nTo fix this:")
        print("1. Add your images to the following directories:")
        print(f"   - {TRAIN_DIR}/<class_name>/")
        print(f"   - {VAL_DIR}/<class_name>/")
        print(f"   - {TEST_DIR}/<class_name>/")
        print("\n2. Class names should be: " + ", ".join(CLASSES))
        print("\n3. Each class directory should contain image files (.jpg, .png, etc.)")
        print("\n4. Recommended: At least 50-100 images per class in training set")
        print("\nExample structure:")
        print("   dataset/train/plastic/image1.jpg")
        print("   dataset/train/plastic/image2.jpg")
        print("   dataset/train/paper/image1.jpg")
        print("   ...")
        print("\nQuick fix - Create a synthetic test dataset:")
        print("  python create_test_dataset.py")
        print("\nAlternatively, you can:")
        print("  - Use a dataset download script")
        print("  - Use TensorFlow datasets API")
        print("  - Download the TrashNet dataset from GitHub")
        print("\n" + "="*60)
        return
    
    # Build model
    model, base_model = build_model(num_classes)
    
    # Save model summary
    save_model_summary(model)
    
    # Train model
    history, model = train_model(model, train_gen, val_gen, epochs=EPOCHS)
    
    # Optional: Fine-tuning
    print("\nDo you want to fine-tune the model? (This will take additional time)")
    print("For now, skipping fine-tuning. Uncomment the line below to enable.")
    # history_ft, model = fine_tune_model(model, base_model, train_gen, val_gen, epochs=10)
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, 'final_model.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # Save training metrics
    metrics = {
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'epochs_trained': len(history.history['accuracy']),
        'model_params': model.count_params()
    }
    
    with open(os.path.join(MODEL_DIR, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {metrics['best_val_accuracy']:.4f}")
    print(f"Final Validation Accuracy: {metrics['final_val_accuracy']:.4f}")
    print(f"Model Parameters: {metrics['model_params']:,}")
    print("\nNext step: Run 3_convert_to_tflite.py to convert model to TensorFlow Lite")

if __name__ == "__main__":
    main()

