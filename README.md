# Image-recognition
# CIFAR-10 Image Classification Project

A Python-based image classification project that uses a Convolutional Neural Network (CNN) to classify images into 10 categories from the CIFAR-10 dataset. The project includes model training capabilities and image prediction functionality.

## Project Structure

```
cifar10-image-classifier/
├── main.py                    # Training script and basic prediction
├── image_recognition_gui.py   # Full-featured GUI application
├── imageClassifier.keras      # Pre-trained CNN model
├── models/                    # Model directory (GUI expects model here)
│   └── imageClassifier.keras
├── deer.jpg                   # Sample test image
└── README.md                  # This file
```

## Features

- **🖼️ Professional GUI**: Beautiful, user-friendly interface with emojis and animations
- **📁 Easy Image Upload**: Drag-and-drop or browse for image files
- **🎯 Real-time Classification**: Instant predictions with confidence scores
- **📊 Detailed Results**: Shows all class probabilities with visual bars
- **🔄 Multi-threading**: Non-blocking image processing
- **⚡ Progress Indicators**: Loading animations and status updates
- **🎨 Modern Design**: Clean, responsive interface with proper styling
- **CIFAR-10 Classification**: Classifies images into 10 categories
- **CNN Architecture**: Uses Convolutional Neural Networks for image recognition
- **Model Training**: Complete training pipeline in main.py

## CIFAR-10 Categories

The model can classify images into these 10 categories:
1. **✈️ Airplane** - Aircraft and planes
2. **🚗 Car** - Automobiles and vehicles  
3. **🐦 Bird** - Various bird species
4. **🐱 Cat** - Cats and felines
5. **🦌 Deer** - Deer and similar animals
6. **🐕 Dog** - Dogs and canines
7. **🐸 Frog** - Frogs and amphibians
8. **🐎 Horse** - Horses and equines
9. **🚢 Ship** - Ships and boats
10. **🚛 Truck** - Trucks and large vehicles

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
tensorflow>=2.8.0
numpy>=1.19.0
matplotlib>=3.3.0
opencv-python>=4.5.0
pillow>=8.0.0
tkinter (usually included with Python)
```

## Installation

1. **Clone or download the project files**
   ```bash
   git clone <your-repository-url>
   cd cifar10-image-classifier
   ```

2. **Install required packages**
   ```bash
   pip install tensorflow numpy matplotlib opencv-python pillow
   ```

   Or create a requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up model directory (for GUI)**
   ```bash
   mkdir models
   cp imageClassifier.keras models/
   # GUI expects model in models/imageClassifier.keras
   ```

   **Note**: The GUI looks for the model in `models/imageClassifier.keras` while `main.py` expects it in the root directory.

## Usage

### GUI Application (Recommended)

1. **Launch the GUI**
   ```bash
   python image_recognition_gui.py
   ```

2. **Using the interface**
   - Click **"📁 Select Image"** to browse for an image file
   - Supported formats: JPG, JPEG, PNG, BMP, GIF, TIFF
   - The app will automatically:
     - Display your image (resized for viewing)
     - Show the top prediction with confidence percentage
     - Display all 10 class probabilities
     - Animate the confidence bar
   - Click **"🔄 Upload Another Image"** to classify more images

### Command Line Prediction (Basic)

1. **Place your test image in the project directory**
   - The script expects `deer.jpg` by default
   - Replace with your own image file

2. **Run the prediction script**
   ```bash
   python main.py
   ```

3. **View results**
   - Prediction printed to console
   - Image displayed using matplotlib

### Training a New Model

To train a new model, uncomment the training section in `main.py`:

1. **Uncomment the model building and training code** (lines 27-44)
2. **Run the training script**
   ```bash
   python main.py
   ```
3. **Wait for training to complete** (10 epochs with reduced dataset)
4. **The trained model will be saved as `imageClassifier.keras`**

## Model Architecture

The CNN model consists of:
- **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pooling
- **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation  
- **MaxPooling2D**: 2x2 pooling
- **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation
- **Flatten Layer**: Converts 2D to 1D
- **Dense Layer**: 64 neurons, ReLU activation
- **Dense Layer**: 10 neurons, Softmax activation (output)

**Input Shape**: 32x32x3 (RGB images)
**Output**: 10 class probabilities

## File Descriptions

### `main.py`
The main script that handles:
- **Data Loading**: CIFAR-10 dataset from Keras
- **Data Preprocessing**: Normalization (pixel values 0-1)
- **Model Architecture**: CNN definition (commented out)
- **Model Training**: Training loop with validation (commented out)
- **Model Loading**: Loads pre-trained model
- **Image Prediction**: Predicts class of test image
- **Visualization**: Displays image and prediction

### `imageClassifier.keras`
Pre-trained Keras model file (binary format) containing:
- **Trained CNN architecture** with Conv2D and Dense layers
- **Optimized weights** from training on CIFAR-10 dataset
- **Model configuration** and metadata
- **File size**: Approximately 1.2MB
- **Format**: HDF5-based Keras SavedModel format

**Important**: 
- GUI expects model at: `models/imageClassifier.keras`
- Main script expects model at: `imageClassifier.keras` (root directory)
- Copy the model to both locations or update paths as needed

### `image_recognition_gui.py`
Full-featured GUI application with:
- **Modern Interface**: Clean design with emojis and animations
- **ImageClassifierGUI Class**: Main application class
- **Threading**: Non-blocking image processing and model loading
- **Error Handling**: Graceful error messages and status updates
- **Image Display**: Resizes and displays uploaded images
- **Results Visualization**: Shows top prediction and all probabilities
- **Progress Indicators**: Loading bars and status messages
- **Reset Functionality**: Easy interface reset for multiple classifications

## Image Requirements

For best results with custom images:
- **Format**: JPG, PNG, or other common formats
- **Content**: Should contain objects from the 10 CIFAR-10 categories
- **Quality**: Clear, well-lit images work best
- **Size**: Any size (will be resized to 32x32 internally)

## Performance Notes

- **Training Data**: Uses 20,000/50,000 training images for faster training
- **Testing Data**: Uses 4,000/10,000 test images for validation
- **Model Size**: Lightweight CNN suitable for CPU inference
- **Accuracy**: Typically achieves 60-70% accuracy on CIFAR-10

## Troubleshooting

### Common Issues

1. **GUI Model not found**
   ```
   ⚠️ Model file not found. Please train the model first.
   ```
   - Copy `imageClassifier.keras` to `models/imageClassifier.keras`
   - Or update the model path in `image_recognition_gui.py` line 76:
     ```python
     model_path = 'imageClassifier.keras'  # Change this path
     ```

2. **Model loading errors**
   ```
   Error loading model: [TensorFlow error]
   ```
   - Ensure TensorFlow version compatibility (≥2.8.0)
   - Verify model file isn't corrupted
   - Check file permissions

3. **Missing image file (main.py)**
   ```
   Error: cv2.imread() returns None
   ```
   - Ensure `deer.jpg` exists in the project directory
   - Update filename in `main.py` line 48 if using different image

4. **GUI not starting**
   ```
   ModuleNotFoundError: No module named 'tkinter'
   ```
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`
   - On CentOS/RHEL: `sudo yum install tkinter` or `sudo dnf install python3-tkinter`
   - On macOS: tkinter should be included with Python

5. **Image upload fails in GUI**
   ```
   Error processing image: Could not load image
   ```
   - Check image file isn't corrupted
   - Ensure image format is supported (JPG, PNG, BMP, GIF, TIFF)
   - Try a different image file
   - Check file permissions

6. **Threading errors in GUI**
   ```
   RuntimeError: main thread is not in main loop
   ```
   - This is usually handled automatically by the GUI
   - If persistent, restart the application

7. **Memory errors during training**
   - Reduce dataset size further in `main.py` lines 19-23
   - Lower batch size in model.fit()
   - Close other applications to free up RAM

8. **Poor prediction accuracy**
   - CIFAR-10 is trained on 32x32 low-resolution images
   - Best results with simple, centered objects
   - Try images similar to CIFAR-10 style
   - Ensure image contains objects from the 10 categories

## Customization

### GUI Customization

**Change Model Path**
- Update line 76 in `image_recognition_gui.py`: 
  ```python
  model_path = 'your/path/to/model.keras'
  ```

**Quick Fix for Model Path**
- Copy model to expected location: `cp imageClassifier.keras models/`
- Or change GUI to use root directory model:
  ```python
  model_path = 'imageClassifier.keras'
  ```

**Modify Interface Colors**
- Colors are defined throughout the GUI setup
- Main colors: `#3498db` (blue), `#2c3e50` (dark), `#f0f0f0` (light)

**Add New Categories**
- Update `self.class_names` and `self.class_emojis` in the GUI class
- Retrain model with new categories

### Main Script Customization

**Using Different Images**
1. Replace `"deer.jpg"` with your image filename in line 48
2. Ensure the image contains objects from the 10 categories

**Modifying Training Parameters**
- **Epochs**: Change `epochs=10` in model.fit()
- **Dataset Size**: Modify lines 19-23 for different data amounts
- **Architecture**: Uncomment and modify layers in model building section

**Adding New Categories**
To classify different objects:
1. Replace CIFAR-10 dataset with your custom dataset
2. Update `class_names` list with your categories
3. Modify final Dense layer neurons to match category count
4. Retrain the model


## Contributing

Contributions welcome! Areas for improvement:
- **Model Optimization**: Better architectures, data augmentation
- **GUI Enhancements**: Batch processing, drag-and-drop support  
- **Additional Features**: Webcam input, result export, model comparison
- **Performance**: GPU acceleration, model quantization
- **Documentation**: More examples, troubleshooting guides

## Acknowledgments

- **CIFAR-10 Dataset**: Created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **TensorFlow/Keras**: For the deep learning framework
- **OpenCV**: For image processing capabilities

---

**Note**: This model works best with images similar to CIFAR-10 style - small, centered objects against simple backgrounds. For real-world images, consider using transfer learning with pre-trained models like ResNet or VGG.
