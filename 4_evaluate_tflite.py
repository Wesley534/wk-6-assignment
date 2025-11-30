"""
Evaluate TensorFlow Lite Model Performance
Tests accuracy, inference time, and compares with Keras model
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

# Paths
DATASET_DIR = "dataset"
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_DIR = "models"
TFLITE_DIR = "tflite_models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model paths
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
TFLITE_MODEL_PATH = os.path.join(TFLITE_DIR, "recyclable_items.tflite")
TFLITE_FLOAT16_PATH = os.path.join(TFLITE_DIR, "recyclable_items_float16.tflite")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32

def load_test_data():
    """Load test dataset"""
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory {TEST_DIR} not found!")
        return None, None
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator, test_generator.class_indices

def predict_with_keras(model_path, test_generator):
    """Make predictions using Keras model"""
    print(f"\nEvaluating Keras model: {model_path}")
    
    model = keras.models.load_model(model_path)
    
    start_time = time.time()
    predictions = model.predict(test_generator, verbose=1)
    inference_time = time.time() - start_time
    
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    accuracy = np.mean(predicted_classes == true_classes)
    
    return {
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'accuracy': accuracy,
        'inference_time': inference_time,
        'avg_time_per_image': inference_time / len(true_classes)
    }

def predict_with_tflite(tflite_path, test_generator):
    """Make predictions using TFLite model"""
    print(f"\nEvaluating TFLite model: {tflite_path}")
    
    if not os.path.exists(tflite_path):
        print(f"  Model not found: {tflite_path}")
        return None
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    inference_times = []
    
    # Process all test images
    test_generator.reset()
    num_batches = len(test_generator)
    
    for batch_idx in range(num_batches):
        batch_images, batch_labels = test_generator.next()
        
        batch_predictions = []
        for img in batch_images:
            # Preprocess if needed
            if input_details[0]['dtype'] == np.uint8:
                input_data = (img * 255).astype(np.uint8)
            else:
                input_data = img.astype(np.float32)
            
            # Add batch dimension
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            inference_time = time.time() - start_time
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            batch_predictions.append(output_data[0])
            inference_times.append(inference_time)
        
        predictions.extend(batch_predictions)
    
    predictions = np.array(predictions)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes[:len(predicted_classes)]
    
    accuracy = np.mean(predicted_classes == true_classes)
    total_time = sum(inference_times)
    avg_time = np.mean(inference_times)
    
    return {
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'accuracy': accuracy,
        'inference_time': total_time,
        'avg_time_per_image': avg_time,
        'model_size_mb': os.path.getsize(tflite_path) / (1024 * 1024)
    }

def generate_classification_report(y_true, y_pred, class_names, title="Classification Report"):
    """Generate and save classification report"""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Print report
    print(f"\n{title}:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return report

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Confusion matrix saved to {save_path}")
    plt.close()

def compare_models(results_dict):
    """Compare performance of different models"""
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    comparison_data = []
    
    for model_name, results in results_dict.items():
        if results is not None:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Avg Inference Time (ms)': f"{results['avg_time_per_image'] * 1000:.2f}",
                'Total Inference Time (s)': f"{results['inference_time']:.2f}",
                'Model Size (MB)': f"{results.get('model_size_mb', 'N/A')}"
            })
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    
    return df

def main():
    print("\n" + "="*60)
    print("TensorFlow Lite Model Evaluation")
    print("="*60)
    
    # Load test data
    test_generator, class_indices = load_test_data()
    
    if test_generator is None:
        print("\nPlease prepare your test dataset first!")
        return
    
    class_names = list(class_indices.keys())
    num_samples = test_generator.samples
    
    print(f"\nTest dataset loaded:")
    print(f"  Classes: {class_names}")
    print(f"  Total samples: {num_samples}")
    
    results = {}
    
    # Evaluate Keras model
    try:
        keras_path = KERAS_MODEL_PATH if os.path.exists(KERAS_MODEL_PATH) else FALLBACK_MODEL_PATH
        if os.path.exists(keras_path):
            keras_results = predict_with_keras(keras_path, test_generator)
            results['Keras Model'] = keras_results
            
            # Generate reports for Keras model
            report = generate_classification_report(
                keras_results['true_classes'],
                keras_results['predicted_classes'],
                class_names,
                "Keras Model Classification Report"
            )
            
            plot_confusion_matrix(
                keras_results['true_classes'],
                keras_results['predicted_classes'],
                class_names,
                "Keras Model - Confusion Matrix",
                os.path.join(RESULTS_DIR, "confusion_matrix_keras.png")
            )
        else:
            print(f"\nKeras model not found. Skipping Keras evaluation.")
    except Exception as e:
        print(f"\nError evaluating Keras model: {e}")
    
    # Evaluate TFLite Float32 model
    tflite_results = predict_with_tflite(TFLITE_MODEL_PATH, test_generator)
    if tflite_results:
        results['TFLite (Float32)'] = tflite_results
        
        report = generate_classification_report(
            tflite_results['true_classes'],
            tflite_results['predicted_classes'],
            class_names,
            "TFLite Float32 Classification Report"
        )
        
        plot_confusion_matrix(
            tflite_results['true_classes'],
            tflite_results['predicted_classes'],
            class_names,
            "TFLite Float32 - Confusion Matrix",
            os.path.join(RESULTS_DIR, "confusion_matrix_tflite_float32.png")
        )
    
    # Evaluate TFLite Float16 model
    tflite_f16_results = predict_with_tflite(TFLITE_FLOAT16_PATH, test_generator)
    if tflite_f16_results:
        results['TFLite (Float16)'] = tflite_f16_results
        
        report = generate_classification_report(
            tflite_f16_results['true_classes'],
            tflite_f16_results['predicted_classes'],
            class_names,
            "TFLite Float16 Classification Report"
        )
        
        plot_confusion_matrix(
            tflite_f16_results['true_classes'],
            tflite_f16_results['predicted_classes'],
            class_names,
            "TFLite Float16 - Confusion Matrix",
            os.path.join(RESULTS_DIR, "confusion_matrix_tflite_float16.png")
        )
    
    # Compare models
    comparison_df = compare_models(results)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    
    # Save detailed results
    evaluation_results = {}
    for model_name, model_results in results.items():
        if model_results:
            evaluation_results[model_name] = {
                'accuracy': float(model_results['accuracy']),
                'avg_inference_time_ms': float(model_results['avg_time_per_image'] * 1000),
                'total_inference_time_s': float(model_results['inference_time']),
                'model_size_mb': float(model_results.get('model_size_mb', 0))
            }
    
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ Evaluation Complete!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("  - model_comparison.csv")
    print("  - evaluation_results.json")
    print("  - confusion_matrix_*.png")
    
    print("\nSummary:")
    for model_name, model_results in results.items():
        if model_results:
            print(f"  {model_name}:")
            print(f"    Accuracy: {model_results['accuracy']:.4f}")
            print(f"    Avg inference: {model_results['avg_time_per_image']*1000:.2f} ms/image")

if __name__ == "__main__":
    main()

