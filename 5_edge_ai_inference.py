"""
Edge AI Inference Script
Simulates real-time inference on edge devices (Raspberry Pi simulation)
This script demonstrates how the TFLite model would be used in production
"""

import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse

# Paths
TFLITE_DIR = "tflite_models"
TFLITE_MODEL_PATH = os.path.join(TFLITE_DIR, "recyclable_items_float16.tflite")

# Class labels
CLASS_LABELS = ["plastic", "paper", "glass", "metal", "cardboard"]

class EdgeAIInference:
    """Edge AI Inference Engine for real-time classification"""
    
    def __init__(self, model_path, img_size=224):
        """
        Initialize the edge inference engine
        
        Args:
            model_path: Path to TFLite model file
            img_size: Input image size (default 224x224)
        """
        self.model_path = model_path
        self.img_size = img_size
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✓ Edge AI Model Loaded")
        print(f"  Model: {os.path.basename(model_path)}")
        print(f"  Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Input type: {self.input_details[0]['dtype']}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image array
        """
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path.convert('RGB')
        
        # Resize
        img = img.resize((self.img_size, self.img_size))
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Handle quantization if needed
        if self.input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            img_array = img_array / input_scale + input_zero_point
            img_array = img_array.astype(np.uint8)
        
        return img_array
    
    def predict(self, image_path, top_k=3):
        """
        Perform inference on a single image
        
        Args:
            image_path: Path to image file or PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and timing information
        """
        # Preprocess
        preprocess_start = time.time()
        input_data = self.preprocess_image(image_path)
        preprocess_time = time.time() - preprocess_start
        
        # Run inference
        inference_start = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = time.time() - inference_start
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Handle quantization if needed
        if self.output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        probabilities = output_data[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_predictions = [
            {
                'class': CLASS_LABELS[idx],
                'confidence': float(probabilities[idx])
            }
            for idx in top_indices
        ]
        
        total_time = preprocess_time + inference_time
        
        return {
            'predictions': top_predictions,
            'preprocessing_time_ms': preprocess_time * 1000,
            'inference_time_ms': inference_time * 1000,
            'total_time_ms': total_time * 1000,
            'fps': 1.0 / total_time if total_time > 0 else 0
        }
    
    def predict_batch(self, image_paths, batch_size=1):
        """
        Perform batch inference (simulated - TFLite processes one at a time)
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size (for simulation, actual batch processing requires model modification)
            
        Returns:
            List of prediction results
        """
        results = []
        
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        
        return results
    
    def benchmark(self, num_runs=100):
        """
        Benchmark inference performance
        
        Args:
            num_runs: Number of inference runs for benchmarking
            
        Returns:
            Dictionary with benchmark statistics
        """
        print(f"\nBenchmarking with {num_runs} runs...")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_image)
        
        inference_times = []
        total_times = []
        
        for i in range(num_runs):
            result = self.predict(dummy_image)
            inference_times.append(result['inference_time_ms'])
            total_times.append(result['total_time_ms'])
        
        return {
            'num_runs': num_runs,
            'avg_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'avg_total_time_ms': np.mean(total_times),
            'avg_fps': np.mean([1.0 / (t / 1000) for t in total_times]),
            'median_fps': np.median([1.0 / (t / 1000) for t in total_times])
        }

def simulate_realtime_stream(model_path, duration=10):
    """
    Simulate real-time streaming inference
    Simulates processing images at regular intervals
    
    Args:
        model_path: Path to TFLite model
        duration: Simulation duration in seconds
    """
    print(f"\n{'='*60}")
    print(f"Simulating Real-time Edge AI Inference")
    print(f"{'='*60}")
    print(f"Duration: {duration} seconds")
    print(f"Processing images continuously...")
    
    engine = EdgeAIInference(model_path)
    
    # Create dummy images
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_image)
    
    start_time = time.time()
    frame_count = 0
    inference_times = []
    
    while time.time() - start_time < duration:
        result = engine.predict(dummy_image)
        inference_times.append(result['inference_time_ms'])
        frame_count += 1
        
        if frame_count % 10 == 0:
            current_fps = frame_count / (time.time() - start_time)
            print(f"  Processed {frame_count} frames | Current FPS: {current_fps:.2f}")
        
        # Simulate real-time (wait for next frame interval)
        time.sleep(0.033)  # ~30 FPS
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    
    print(f"\n{'='*60}")
    print(f"Simulation Complete")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average inference time: {np.mean(inference_times):.2f} ms")
    print(f"Min inference time: {np.min(inference_times):.2f} ms")
    print(f"Max inference time: {np.max(inference_times):.2f} ms")

def main():
    parser = argparse.ArgumentParser(description='Edge AI Inference for Recyclable Items')
    parser.add_argument('--model', type=str, default=TFLITE_MODEL_PATH,
                       help='Path to TFLite model file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image file for inference')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark test')
    parser.add_argument('--simulate', action='store_true',
                       help='Simulate real-time streaming')
    parser.add_argument('--duration', type=int, default=10,
                       help='Simulation duration in seconds')
    
    args = parser.parse_args()
    
    try:
        engine = EdgeAIInference(args.model)
        
        # Single image inference
        if args.image:
            print(f"\n{'='*60}")
            print(f"Running Inference on: {args.image}")
            print(f"{'='*60}")
            
            result = engine.predict(args.image, top_k=3)
            
            print(f"\nTop Predictions:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"  {i}. {pred['class']:15s}: {pred['confidence']*100:6.2f}%")
            
            print(f"\nPerformance:")
            print(f"  Preprocessing: {result['preprocessing_time_ms']:.2f} ms")
            print(f"  Inference:     {result['inference_time_ms']:.2f} ms")
            print(f"  Total:         {result['total_time_ms']:.2f} ms")
            print(f"  FPS:           {result['fps']:.2f}")
        
        # Benchmark
        if args.benchmark:
            print(f"\n{'='*60}")
            print(f"Benchmarking Edge AI Model")
            print(f"{'='*60}")
            
            benchmark_results = engine.benchmark(num_runs=100)
            
            print(f"\nBenchmark Results:")
            print(f"  Runs:                {benchmark_results['num_runs']}")
            print(f"  Avg inference time:  {benchmark_results['avg_inference_time_ms']:.2f} ms")
            print(f"  Std deviation:       {benchmark_results['std_inference_time_ms']:.2f} ms")
            print(f"  Min inference time:  {benchmark_results['min_inference_time_ms']:.2f} ms")
            print(f"  Max inference time:  {benchmark_results['max_inference_time_ms']:.2f} ms")
            print(f"  Average FPS:         {benchmark_results['avg_fps']:.2f}")
            print(f"  Median FPS:          {benchmark_results['median_fps']:.2f}")
        
        # Real-time simulation
        if args.simulate:
            simulate_realtime_stream(args.model, duration=args.duration)
        
        # Default: show info if no specific action
        if not args.image and not args.benchmark and not args.simulate:
            print("\nUsage examples:")
            print("  Single image:  python 5_edge_ai_inference.py --image path/to/image.jpg")
            print("  Benchmark:     python 5_edge_ai_inference.py --benchmark")
            print("  Real-time sim: python 5_edge_ai_inference.py --simulate --duration 10")
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  1. Model is converted to TFLite format (run 3_convert_to_tflite.py)")
        print("  2. Model path is correct")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

