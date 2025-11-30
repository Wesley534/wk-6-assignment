# Edge AI Image Classification: Recyclable Items Recognition

This project implements a lightweight image classification model for recognizing recyclable items (plastic, paper, glass, metal, cardboard) using TensorFlow Lite for edge device deployment.

## Project Overview

The goal is to train a lightweight model optimized for edge devices (Raspberry Pi, mobile devices) that can perform real-time image classification of recyclable items. The solution includes:

1. Dataset preparation
2. Model training using transfer learning (MobileNetV2)
3. TensorFlow Lite conversion with optimizations
4. Model evaluation and benchmarking
5. Edge inference simulation

## Project Structure

```
wk-6-assignment/
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── REPORT.md                     # Comprehensive project report
├── 1_prepare_dataset.py          # Dataset preparation script
├── create_test_dataset.py        # Create synthetic test dataset
├── 2_train_model.py              # Model training script
├── 3_convert_to_tflite.py        # TFLite conversion script
├── 4_evaluate_tflite.py          # Model evaluation script
├── 5_edge_ai_inference.py        # Edge inference simulation
├── dataset/                       # Dataset directory (created after step 1)
│   ├── train/
│   ├── test/
│   └── val/
├── models/                        # Trained models (created after step 2)
├── tflite_models/                 # TFLite models (created after step 3)
└── results/                       # Evaluation results (created after step 4)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Option A: Create synthetic test dataset (for testing the pipeline)**
```bash
python 1_prepare_dataset.py  # Creates directory structure
python create_test_dataset.py  # Creates placeholder images
```

**Option B: Use real dataset**
```bash
python 1_prepare_dataset.py
```

Then populate with your images:
- Download a recyclable items dataset (e.g., TrashNet) or use your own collection
- Organize images into: `dataset/train/`, `dataset/test/`, `dataset/val/`
- Each directory should have subdirectories for each class: `plastic/`, `paper/`, `glass/`, `metal/`, `cardboard/`

### 3. Train the Model

```bash
python 2_train_model.py
```

This will:
- Load and preprocess the dataset
- Build a MobileNetV2-based model
- Train the model with data augmentation
- Save the best model to `models/best_model.h5`

### 4. Convert to TensorFlow Lite

```bash
python 3_convert_to_tflite.py
```

This creates optimized TFLite models:
- `recyclable_items.tflite` (Float32)
- `recyclable_items_float16.tflite` (Float16, recommended)
- `recyclable_items_quantized.tflite` (INT8, if quantization succeeds)

### 5. Evaluate the Model

```bash
python 4_evaluate_tflite.py
```

This evaluates the model performance and generates:
- Accuracy metrics
- Confusion matrices
- Model comparison report

### 6. Test Edge Inference

```bash
# Single image inference
python 5_edge_ai_inference.py --image path/to/image.jpg

# Benchmark performance
python 5_edge_ai_inference.py --benchmark

# Simulate real-time streaming
python 5_edge_ai_inference.py --simulate --duration 10
```

## Usage Examples

### Training a Model

```python
# The training script handles everything automatically
python 2_train_model.py
```

### Running Inference

```python
from 5_edge_ai_inference import EdgeAIInference

# Initialize
engine = EdgeAIInference("tflite_models/recyclable_items_float16.tflite")

# Predict
result = engine.predict("path/to/image.jpg")
print(f"Predicted: {result['predictions'][0]['class']}")
print(f"Confidence: {result['predictions'][0]['confidence']*100:.2f}%")
```

### Benchmarking

```python
benchmark_results = engine.benchmark(num_runs=100)
print(f"Average FPS: {benchmark_results['avg_fps']:.2f}")
```

## Model Architecture

- **Base Model**: MobileNetV2 (transfer learning)
- **Input Size**: 224x224x3
- **Classes**: 5 (plastic, paper, glass, metal, cardboard)
- **Optimization**: Optimized for edge devices with reduced size and latency

## Key Features

1. **Lightweight**: Model size < 10 MB (quantized)
2. **Fast Inference**: ~10-50ms per image on edge devices
3. **Optimized**: Multiple TFLite variants (Float32, Float16, INT8)
4. **Real-time Ready**: Can process 20+ FPS on Raspberry Pi
5. **Production Ready**: Includes evaluation, benchmarking, and deployment scripts

## Edge AI Benefits

See `REPORT.md` for detailed discussion on:
- Low latency inference
- Privacy and security
- Offline operation
- Reduced bandwidth
- Cost efficiency
- Real-time processing

## Results

After training and evaluation, check:
- `results/evaluation_results.json` - Detailed metrics
- `results/model_comparison.csv` - Model comparison
- `results/confusion_matrix_*.png` - Visualization
- `models/training_history.png` - Training curves

## Deployment to Raspberry Pi

1. Copy TFLite model to Raspberry Pi:
   ```bash
   scp tflite_models/recyclable_items_float16.tflite pi@raspberrypi:/home/pi/models/
   ```

2. Install dependencies on Raspberry Pi:
   ```bash
   pip install tensorflow-lite-runtime pillow numpy
   ```

3. Run inference:
   ```python
   python 5_edge_ai_inference.py --image image.jpg
   ```

## Notes

- For best results, use a balanced dataset with sufficient images per class
- Data augmentation helps improve generalization
- Float16 quantization provides best balance of size and accuracy
- INT8 quantization may require calibration dataset for best results

## Troubleshooting

**Issue**: Model not found
- **Solution**: Ensure you've run the training script first

**Issue**: Dataset not found
- **Solution**: Run `1_prepare_dataset.py` and populate with images

**Issue**: Low accuracy
- **Solution**: Ensure balanced dataset, increase training epochs, or use data augmentation

**Issue**: Slow inference
- **Solution**: Use Float16 or INT8 quantized model, optimize preprocessing

## License

This project is for educational purposes.

## Author

Created for AI for Software Engineering (Week 6 Assignment)

# wk-6-assignment
