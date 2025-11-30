# Quick Start Guide

Get up and running quickly with the Recyclable Items Classifier!

## Prerequisites

1. Python 3.8+ installed
2. pip package manager
3. (Optional) A dataset of recyclable item images

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Demo (Without Dataset)

If you don't have a dataset yet, you can create a synthetic test dataset:

```bash
# 1. Create dataset structure
python 1_prepare_dataset.py

# 2. Create synthetic test images (for testing the pipeline)
python create_test_dataset.py

# 3. Now you can train and test the full pipeline!
python 2_train_model.py
```

**Note**: The synthetic dataset is for testing the pipeline only. For real-world accuracy, use actual photos of recyclable items.

## Full Workflow (With Dataset)

### Step 1: Prepare Your Dataset

1. Organize your images into folders:
```
dataset/
├── train/
│   ├── plastic/
│   ├── paper/
│   ├── glass/
│   ├── metal/
│   └── cardboard/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

2. Run dataset preparation:
```bash
python 1_prepare_dataset.py
```

### Step 2: Train Model

```bash
python 2_train_model.py
```

This will:
- Load your dataset
- Train a MobileNetV2-based model
- Save the best model to `models/best_model.h5`

**Expected time**: 20-60 minutes (depending on dataset size and hardware)

### Step 3: Convert to TensorFlow Lite

```bash
python 3_convert_to_tflite.py
```

This creates optimized models in `tflite_models/`:
- `recyclable_items.tflite` (standard)
- `recyclable_items_float16.tflite` (recommended for edge)
- `recyclable_items_quantized.tflite` (smallest)

### Step 4: Evaluate Model

```bash
python 4_evaluate_tflite.py
```

Generates:
- Accuracy metrics
- Confusion matrices
- Model comparison report

### Step 5: Test Edge Inference

```bash
# Benchmark performance
python 5_edge_ai_inference.py --benchmark

# Test with an image
python 5_edge_ai_inference.py --image path/to/image.jpg

# Simulate real-time processing
python 5_edge_ai_inference.py --simulate --duration 10
```

## Using the Pipeline Script

Run everything automatically:

```bash
./run_complete_pipeline.sh
```

## Example: Testing with a Single Image

```python
from 5_edge_ai_inference import EdgeAIInference

# Load model
engine = EdgeAIInference("tflite_models/recyclable_items_float16.tflite")

# Predict
result = engine.predict("path/to/image.jpg")

# Display results
for pred in result['predictions']:
    print(f"{pred['class']}: {pred['confidence']*100:.2f}%")
```

## Common Workflows

### Workflow 1: Training from Scratch
```bash
python 1_prepare_dataset.py  # Setup structure
# Add your images to dataset/train/, dataset/val/, dataset/test/
python 2_train_model.py      # Train
python 3_convert_to_tflite.py # Convert
python 4_evaluate_tflite.py  # Evaluate
```

### Workflow 2: Using Pre-trained Model
```bash
# If you already have a trained model
python 3_convert_to_tflite.py  # Convert existing model
python 4_evaluate_tflite.py    # Evaluate
```

### Workflow 3: Edge Deployment
```bash
# On development machine
python 3_convert_to_tflite.py

# Transfer to Raspberry Pi
scp tflite_models/recyclable_items_float16.tflite pi@raspberrypi:~/models/

# On Raspberry Pi
python 5_edge_ai_inference.py --image test.jpg
```

## Expected Results

After training, you should see:
- Training accuracy: 85-95%
- Validation accuracy: 80-90%
- Test accuracy: 80-90%
- Inference time: 10-50ms per image
- Model size: 6MB (Float16) or 2MB (INT8)

## Troubleshooting

**"No module named 'tensorflow'"**
```bash
pip install tensorflow
```

**"Dataset directory not found"**
- Run `python 1_prepare_dataset.py` first
- Make sure images are in the correct subdirectories

**"Model file not found"**
- Train the model first: `python 2_train_model.py`
- Check that `models/best_model.h5` or `models/final_model.h5` exists

**Low accuracy**
- Ensure balanced dataset (similar number of images per class)
- Add more training images
- Increase training epochs
- Use data augmentation

## Next Steps

1. **Read the full REPORT.md** for detailed analysis
2. **Check DEPLOYMENT.md** for Raspberry Pi deployment
3. **Experiment** with different model configurations
4. **Collect more data** to improve accuracy

## Need Help?

- Check `README.md` for detailed documentation
- Review `REPORT.md` for technical details
- See `DEPLOYMENT.md` for edge deployment guide

## Dataset Recommendations

For best results, aim for:
- **Minimum**: 100 images per class in training set
- **Recommended**: 500+ images per class
- **Ideal**: 1000+ images per class

Sources for recyclable item datasets:
- TrashNet (GitHub)
- TACO dataset
- Custom collection using smartphone camera

Happy classifying! 🗑️♻️

