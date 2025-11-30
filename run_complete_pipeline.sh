#!/bin/bash
# Complete Pipeline Script for Recyclable Items Classification
# This script runs the entire workflow from dataset preparation to evaluation

set -e  # Exit on error

echo "=========================================="
echo "Recyclable Items Classification Pipeline"
echo "=========================================="
echo ""

# Step 1: Prepare Dataset
echo "Step 1/5: Preparing dataset structure..."
python 1_prepare_dataset.py
echo "✓ Dataset structure created"
echo ""

# Step 2: Train Model
echo "Step 2/5: Training model..."
echo "Note: This step requires a populated dataset. If you have images ready,"
echo "place them in dataset/train/, dataset/val/, and dataset/test/ directories."
echo ""
read -p "Do you want to proceed with training? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python 2_train_model.py
    echo "✓ Model training complete"
else
    echo "Skipping training. You can run 'python 2_train_model.py' later."
fi
echo ""

# Step 3: Convert to TFLite
echo "Step 3/5: Converting to TensorFlow Lite..."
if [ -f "models/best_model.h5" ] || [ -f "models/final_model.h5" ]; then
    python 3_convert_to_tflite.py
    echo "✓ TFLite conversion complete"
else
    echo "⚠ No trained model found. Please train the model first."
fi
echo ""

# Step 4: Evaluate Model
echo "Step 4/5: Evaluating TFLite model..."
if [ -f "tflite_models/recyclable_items.tflite" ]; then
    python 4_evaluate_tflite.py
    echo "✓ Model evaluation complete"
else
    echo "⚠ No TFLite model found. Please convert the model first."
fi
echo ""

# Step 5: Test Edge Inference
echo "Step 5/5: Testing edge inference..."
if [ -f "tflite_models/recyclable_items_float16.tflite" ]; then
    echo "Running benchmark..."
    python 5_edge_ai_inference.py --benchmark
    echo "✓ Edge inference test complete"
else
    echo "⚠ No TFLite model found. Please convert the model first."
fi
echo ""

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review results in the 'results/' directory"
echo "2. Check model files in 'tflite_models/' directory"
echo "3. Read REPORT.md for detailed analysis"
echo "4. Deploy to Raspberry Pi using the deployment guide"

