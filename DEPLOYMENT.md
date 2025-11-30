# Deployment Guide: Edge AI Recyclable Items Classifier

This guide provides step-by-step instructions for deploying the TensorFlow Lite model to edge devices, specifically Raspberry Pi.

## Prerequisites

### Hardware Requirements
- Raspberry Pi 3B+ or 4 (recommended: Pi 4 with 4GB RAM)
- MicroSD card (16GB+, Class 10 recommended)
- Power supply (5V, 3A for Pi 4)
- Optional: Camera module for real-time classification
- Optional: Cooling solution (heat sink/fan)

### Software Requirements
- Raspberry Pi OS (Raspbian) or Ubuntu
- Python 3.8 or higher
- Internet connection (for initial setup)

---

## Step-by-Step Deployment

### 1. Setup Raspberry Pi

#### 1.1 Install Operating System
1. Download Raspberry Pi Imager
2. Flash Raspberry Pi OS to SD card
3. Enable SSH and configure WiFi (or use Ethernet)
4. Boot Raspberry Pi

#### 1.2 Initial System Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv git
```

---

### 2. Install TensorFlow Lite Runtime

#### 2.1 Install TFLite Runtime (Raspberry Pi Optimized)

For Raspberry Pi, use the pre-built wheel:

```bash
# For Python 3.9 (default on latest Raspberry Pi OS)
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime

# Or install from source if wheel not available
pip3 install tensorflow-lite-runtime
```

**Alternative**: Use TensorFlow Lite Interpreter (lightweight)
```bash
pip3 install tflite-runtime
```

#### 2.2 Install Other Dependencies
```bash
pip3 install pillow numpy opencv-python-headless
```

**Note**: `opencv-python-headless` is lighter than full OpenCV and doesn't require GUI libraries.

---

### 3. Transfer Model to Raspberry Pi

#### Option A: Using SCP (from development machine)
```bash
# Transfer TFLite model
scp tflite_models/recyclable_items_float16.tflite pi@raspberrypi.local:~/models/

# Transfer inference script
scp 5_edge_ai_inference.py pi@raspberrypi.local:~/scripts/

# Create directories on Pi
ssh pi@raspberrypi.local "mkdir -p ~/models ~/scripts"
```

#### Option B: Using USB Drive
1. Copy model files to USB drive
2. Insert USB drive into Raspberry Pi
3. Mount and copy files:
```bash
sudo mkdir -p /media/usb
sudo mount /dev/sda1 /media/usb  # Adjust device name as needed
cp /media/usb/recyclable_items_float16.tflite ~/models/
```

#### Option C: Using Git
```bash
# Clone repository (if using Git)
git clone <repository-url>
cd wk-6-assignment
cp tflite_models/recyclable_items_float16.tflite ~/models/
```

---

### 4. Create Deployment Script

Create a simplified inference script for Raspberry Pi:

```python
# ~/scripts/recycle_inference.py
import os
import sys
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite
from PIL import Image

MODEL_PATH = os.path.expanduser('~/models/recyclable_items_float16.tflite')
CLASSES = ["plastic", "paper", "glass", "metal", "cardboard"]
IMG_SIZE = 224

def load_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(interpreter, image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_data = preprocess_image(image_path)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = output_data[0]
    
    top_idx = np.argmax(probabilities)
    return CLASSES[top_idx], probabilities[top_idx]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recycle_inference.py <image_path>")
        sys.exit(1)
    
    interpreter = load_model()
    image_path = sys.argv[1]
    
    class_name, confidence = predict(interpreter, image_path)
    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence*100:.2f}%")
```

Save this as `recycle_inference.py` on your Raspberry Pi.

---

### 5. Test Deployment

#### 5.1 Test with Sample Image
```bash
# Test inference
python3 ~/scripts/recycle_inference.py /path/to/test/image.jpg
```

#### 5.2 Benchmark Performance
```bash
# Run benchmark (if you copied the full script)
python3 ~/scripts/5_edge_ai_inference.py --benchmark
```

---

### 6. Real-Time Camera Integration (Optional)

For real-time classification using Raspberry Pi camera:

```python
# ~/scripts/camera_inference.py
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import time

MODEL_PATH = '~/models/recyclable_items_float16.tflite'
CLASSES = ["plastic", "paper", "glass", "metal", "cardboard"]

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((224, 224))
    img_array = np.array(img_pil, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Get prediction
    top_idx = np.argmax(output[0])
    class_name = CLASSES[top_idx]
    confidence = output[0][top_idx]
    
    # Display on frame
    text = f"{class_name}: {confidence*100:.1f}%"
    cv2.putText(frame, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Recyclable Items Classifier', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Note**: For headless operation (no display), use SSH with X11 forwarding or save results to files.

---

### 7. Production Deployment Options

#### Option 1: Standalone Service
Create a systemd service for continuous operation:

```bash
# /etc/systemd/system/recycle-classifier.service
[Unit]
Description=Recyclable Items Classifier Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/scripts
ExecStart=/usr/bin/python3 /home/pi/scripts/camera_inference.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable recycle-classifier
sudo systemctl start recycle-classifier
```

#### Option 2: Web API (Flask)
```python
# ~/scripts/api_server.py
from flask import Flask, request, jsonify
from recycle_inference import load_model, predict
import io
from PIL import Image

app = Flask(__name__)
interpreter = load_model()

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Save temporarily
    temp_path = '/tmp/temp_image.jpg'
    image.save(temp_path)
    
    class_name, confidence = predict(interpreter, temp_path)
    
    return jsonify({
        'class': class_name,
        'confidence': float(confidence),
        'confidence_percent': float(confidence * 100)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run API server:
```bash
pip3 install flask
python3 ~/scripts/api_server.py
```

Test with:
```bash
curl -X POST -F "image=@test_image.jpg" http://raspberrypi.local:5000/predict
```

---

### 8. Performance Optimization

#### 8.1 Enable Hardware Acceleration (Pi 4)
```bash
# Install GPU acceleration libraries (if available)
sudo apt install libedgetpu1-std
```

#### 8.2 Overclock Raspberry Pi (Optional)
Edit `/boot/config.txt`:
```
arm_freq=1750
gpu_freq=600
over_voltage=2
```

**Warning**: Overclocking may void warranty and cause instability.

#### 8.3 Reduce Background Processes
```bash
# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon
```

---

### 9. Monitoring and Logging

#### 9.1 Add Logging
```python
import logging

logging.basicConfig(
    filename='/home/pi/logs/recycle_classifier.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

#### 9.2 Monitor Performance
```bash
# Monitor CPU usage
top

# Monitor temperature
vcgencmd measure_temp

# Monitor memory
free -h
```

---

### 10. Troubleshooting

#### Issue: Import Error for tflite_runtime
**Solution**: Install TensorFlow Lite Runtime correctly
```bash
pip3 install --upgrade tflite-runtime
```

#### Issue: Out of Memory
**Solution**: Use INT8 quantized model (smaller) or increase swap
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### Issue: Slow Inference
**Solutions**:
1. Use INT8 quantized model
2. Reduce image resolution (if acceptable)
3. Enable hardware acceleration
4. Close unnecessary background processes

#### Issue: Camera Not Detected
**Solution**: Enable camera interface
```bash
sudo raspi-config
# Navigate to Interface Options > Camera > Enable
sudo reboot
```

---

### 11. Updating the Model

When you have an improved model:

1. Transfer new model to Raspberry Pi
2. Test with new model
3. Replace old model:
```bash
mv ~/models/recyclable_items_float16.tflite ~/models/recyclable_items_float16.tflite.backup
cp new_model.tflite ~/models/recyclable_items_float16.tflite
```
4. Restart service (if using systemd)

---

## Quick Start Checklist

- [ ] Raspberry Pi setup and OS installed
- [ ] TensorFlow Lite Runtime installed
- [ ] Model file transferred to Pi
- [ ] Inference script created and tested
- [ ] Basic inference working
- [ ] (Optional) Camera integration working
- [ ] (Optional) Service/API deployed
- [ ] Logging and monitoring configured

---

## Additional Resources

- [TensorFlow Lite Raspberry Pi Guide](https://www.tensorflow.org/lite/guide/python)
- [Raspberry Pi Documentation](https://www.raspberrypi.org/documentation/)
- [Coral TPU Integration](https://coral.ai/docs/accelerator/get-started/) (for even faster inference)

---

**Note**: This deployment guide assumes basic familiarity with Raspberry Pi and Linux. Adjust commands and paths according to your specific setup.

