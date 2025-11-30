"""
Convert Trained Model to TensorFlow Lite Format
Optimized for edge device deployment (Raspberry Pi, mobile devices)
"""

import os
import tensorflow as tf
import numpy as np
import json

MODEL_DIR = "models"
TFLITE_DIR = "tflite_models"
os.makedirs(TFLITE_DIR, exist_ok=True)

# Model paths
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")

# TFLite output paths
TFLITE_MODEL_PATH = os.path.join(TFLITE_DIR, "recyclable_items.tflite")
TFLITE_QUANTIZED_PATH = os.path.join(TFLITE_DIR, "recyclable_items_quantized.tflite")
TFLITE_FLOAT16_PATH = os.path.join(TFLITE_DIR, "recyclable_items_float16.tflite")

def load_keras_model():
    """Load the trained Keras model"""
    model_path = KERAS_MODEL_PATH if os.path.exists(KERAS_MODEL_PATH) else FALLBACK_MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first using 2_train_model.py"
        )
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    return model

def convert_to_tflite(model, output_path, optimization=None):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model: Trained Keras model
        output_path: Path to save TFLite model
        optimization: None, 'quantization', or 'float16'
    """
    print(f"\nConverting to TensorFlow Lite...")
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    if optimization == 'quantization':
        print("  Applying INT8 quantization (post-training)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for quantization
        def representative_dataset_gen():
            # Generate representative samples
            for _ in range(100):
                yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
    elif optimization == 'float16':
        print("  Applying Float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
    print(f"✓ TFLite model saved to {output_path}")
    print(f"  Model size: {file_size:.2f} MB")
    
    return file_size

def get_model_info(model_path):
    """Get information about the TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    info = {
        "input_shape": str(input_details[0]['shape']),
        "input_type": str(input_details[0]['dtype']),
        "output_shape": str(output_details[0]['shape']),
        "output_type": str(output_details[0]['dtype']),
        "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2)
    }
    
    return info

def test_tflite_model(tflite_path):
    """Test the TFLite model with a dummy input"""
    print(f"\nTesting TFLite model: {tflite_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input type: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output type: {output_details[0]['dtype']}")
    
    # Prepare test input
    if input_details[0]['dtype'] == np.uint8:
        input_data = np.random.randint(0, 255, input_details[0]['shape'], dtype=np.uint8)
    else:
        input_data = np.random.random(input_details[0]['shape']).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  Inference successful!")
    print(f"  Output shape: {output_data.shape}")
    print(f"  Output sum (should be ~1.0 for probabilities): {np.sum(output_data):.4f}")
    
    return True

def compare_model_sizes():
    """Compare sizes of different model formats"""
    print("\n" + "="*60)
    print("Model Size Comparison")
    print("="*60)
    
    sizes = {}
    
    # Keras model size
    if os.path.exists(KERAS_MODEL_PATH):
        sizes['Keras Model'] = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        sizes['Keras Model'] = os.path.getsize(FALLBACK_MODEL_PATH) / (1024 * 1024)
    
    # TFLite model sizes
    if os.path.exists(TFLITE_MODEL_PATH):
        sizes['TFLite (Float32)'] = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    
    if os.path.exists(TFLITE_FLOAT16_PATH):
        sizes['TFLite (Float16)'] = os.path.getsize(TFLITE_FLOAT16_PATH) / (1024 * 1024)
    
    if os.path.exists(TFLITE_QUANTIZED_PATH):
        sizes['TFLite (INT8 Quantized)'] = os.path.getsize(TFLITE_QUANTIZED_PATH) / (1024 * 1024)
    
    print("\nModel Sizes (MB):")
    for model_type, size in sizes.items():
        print(f"  {model_type:25s}: {size:6.2f} MB")
    
    # Calculate compression ratios
    if 'Keras Model' in sizes:
        keras_size = sizes['Keras Model']
        print("\nCompression Ratios (compared to Keras):")
        for model_type, size in sizes.items():
            if model_type != 'Keras Model':
                ratio = (1 - size / keras_size) * 100
                print(f"  {model_type:25s}: {ratio:5.1f}% smaller")
    
    return sizes

def main():
    print("\n" + "="*60)
    print("TensorFlow Lite Model Conversion")
    print("="*60)
    
    try:
        # Load Keras model
        model = load_keras_model()
        
        # Convert to standard TFLite (Float32)
        print("\n" + "-"*60)
        size_float32 = convert_to_tflite(model, TFLITE_MODEL_PATH)
        test_tflite_model(TFLITE_MODEL_PATH)
        
        # Convert to Float16 (2x smaller, minimal accuracy loss)
        print("\n" + "-"*60)
        size_float16 = convert_to_tflite(model, TFLITE_FLOAT16_PATH, optimization='float16')
        test_tflite_model(TFLITE_FLOAT16_PATH)
        
        # Convert to INT8 Quantized (4x smaller, may have accuracy loss)
        print("\n" + "-"*60)
        try:
            size_quantized = convert_to_tflite(model, TFLITE_QUANTIZED_PATH, optimization='quantization')
            test_tflite_model(TFLITE_QUANTIZED_PATH)
        except Exception as e:
            print(f"  Warning: Quantization failed: {e}")
            print("  This is normal if representative dataset is not available")
        
        # Get model information
        print("\n" + "-"*60)
        print("Model Information:")
        model_info = get_model_info(TFLITE_MODEL_PATH)
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Compare sizes
        sizes = compare_model_sizes()
        
        # Save conversion info
        conversion_info = {
            "models_created": [
                "recyclable_items.tflite (Float32)",
                "recyclable_items_float16.tflite (Float16)",
                "recyclable_items_quantized.tflite (INT8) - if successful"
            ],
            "model_sizes_mb": sizes,
            "recommended_for_edge": "recyclable_items_float16.tflite (good balance of size and accuracy)"
        }
        
        with open(os.path.join(TFLITE_DIR, "conversion_info.json"), "w") as f:
            json.dump(conversion_info, f, indent=2)
        
        print("\n" + "="*60)
        print("✓ Conversion Complete!")
        print("="*60)
        print(f"\nTFLite models saved to: {TFLITE_DIR}/")
        print("\nNext step: Run 4_evaluate_tflite.py to test the TFLite model")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 2_train_model.py first to train a model.")
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

