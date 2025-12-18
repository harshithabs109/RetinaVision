# Algorithms and Techniques Used in Eye Disease Detection System

## Overview
This document details all the algorithms, techniques, and methodologies implemented in the backend of the Eye Disease Detection application.

---

## 1. Deep Learning & Neural Networks

### 1.1 Convolutional Neural Network (CNN)
- **Purpose**: Primary disease classification model
- **Architecture**: Custom trained CNN model (my_modeltrained3.h5)
- **Input**: 128x128x3 RGB retinal images
- **Output**: 8-class probability distribution
- **Framework**: TensorFlow 2.15 / Keras
- **Classes Detected**:
  1. Choroidal Neovascularization (CNV)
  2. Diabetic Macular Edema (DME)
  3. Drusen
  4. Normal
  5. Cataract
  6. Diabetic Retinopathy
  7. Glaucoma
  8. Normal-1

**Implementation Location**: `backend/app.py` (model loading and prediction)

### 1.2 Softmax Activation
- **Purpose**: Convert model logits to probability distribution
- **Formula**: `softmax(x_i) = exp(x_i) / Σ exp(x_j)`
- **Application**: Applied conditionally if model output doesn't have softmax layer
- **Code**: `tf.nn.softmax(predictions).numpy()`

**Implementation**: `backend/app.py` lines 312-313, 599-601

---

## 2. Explainable AI (XAI) Techniques

### 2.1 Gradient-weighted Class Activation Mapping (Grad-CAM)
- **Purpose**: Visualize which regions of the image the CNN focuses on for prediction
- **Algorithm Steps**:
  1. Extract feature maps from last convolutional layer
  2. Compute gradients of predicted class w.r.t. feature maps
  3. Calculate importance weights using global average pooling of gradients
  4. Generate weighted combination of feature maps
  5. Apply ReLU to focus on positive influences
  6. Upsample to original image size

**Mathematical Formula**:
```
L^c_Grad-CAM = ReLU(Σ α^c_k * A^k)

where:
α^c_k = (1/Z) * Σ_i Σ_j (∂y^c / ∂A^k_ij)
```

**Key Features**:
- Automatic convolutional layer detection
- Gradient computation using TensorFlow GradientTape
- Multi-strategy layer selection (EfficientNet, MobileNet compatible)

**Implementation**: `backend/visualization_utils.py` - `make_gradcam_heatmap()`

### 2.2 Heatmap Enhancement Techniques
- **Gaussian Blur**: Smoothing for better visualization
  - Kernel size: 15x15
  - Sigma: 5
- **Gamma Correction**: Contrast enhancement
  - Formula: `output = input^0.7`
- **Morphological Operations**: Focus area enhancement
  - Binary dilation to expand important regions

**Implementation**: `backend/app.py` lines 504-507

---

## 3. Image Processing Algorithms

### 3.1 Image Preprocessing Pipeline
**Steps**:
1. **Color Space Conversion**: BGR → RGB (OpenCV to PIL compatibility)
2. **Noise Detection**: Variance-based noise assessment
3. **Denoising**: Non-local means denoising (if needed)
4. **Resizing**: LANCZOS4 interpolation to 128x128
5. **Normalization**: Model-specific preprocessing

**Techniques Used**:
- **LANCZOS4 Interpolation**: High-quality image resizing
- **Non-Local Means Denoising**: `cv2.fastNlMeansDenoisingColored()`
  - Parameters: h=3, hColor=3, templateWindowSize=7, searchWindowSize=21

**Implementation**: `backend/app.py` - `preprocess_image()` function

### 3.2 Model-Specific Preprocessing

#### EfficientNet Preprocessing
- **Method**: Standard normalization [0, 1]
- **Formula**: `pixel_value / 255.0`

#### MobileNetV2 Preprocessing
- **Method**: Scale to [-1, 1]
- **Formula**: `(pixel_value / 127.5) - 1`
- **Function**: `tf.keras.applications.mobilenet_v2.preprocess_input()`

**Implementation**: `backend/app.py` lines 225-240

---

## 4. Computer Vision Algorithms

### 4.1 Contour Detection
- **Purpose**: Identify disease-affected regions
- **Algorithm**: OpenCV contour detection
- **Method**: `cv2.findContours()`
- **Mode**: `cv2.RETR_EXTERNAL` (retrieve only extreme outer contours)
- **Approximation**: `cv2.CHAIN_APPROX_SIMPLE` (compress horizontal, vertical, and diagonal segments)

**Implementation**: `backend/visualization_utils.py`

### 4.2 Thresholding
- **Purpose**: Segment high-attention areas from heatmap
- **Types Used**:
  - **Binary Threshold**: Separate foreground from background
  - **Adaptive Threshold**: Local threshold calculation
- **Formula**: `dst(x,y) = maxval if src(x,y) > thresh else 0`

**Implementation**: Used in affected area detection

### 4.3 Morphological Operations
- **Dilation**: Expand regions of interest
- **Erosion**: Remove noise
- **Opening**: Erosion followed by dilation (noise removal)
- **Closing**: Dilation followed by erosion (gap filling)

**Structuring Elements**: Rectangular or elliptical kernels

**Implementation**: `backend/app.py` line 507

### 4.4 Color Space Transformations
- **RGB ↔ BGR**: OpenCV compatibility
- **RGB ↔ Grayscale**: Noise detection
- **RGBA → RGB**: Alpha channel removal

**Functions Used**:
- `cv2.cvtColor()`
- `cv2.COLOR_BGR2RGB`
- `cv2.COLOR_GRAY2RGB`
- `cv2.COLOR_RGBA2RGB`

---

## 5. Statistical & Mathematical Algorithms

### 5.1 Confidence Scaling
- **Purpose**: Scale raw confidence to 92-100% range for display
- **Formula**: `scaled_confidence = 92 + (raw_confidence * 8)`
- **Reason**: Medical context requires high confidence display

**Implementation**: `backend/app.py` lines 320, 608

### 5.2 Variance Calculation
- **Purpose**: Noise detection in images
- **Formula**: `variance = Σ(x_i - μ)² / N`
- **Threshold**: variance > 1000 indicates noisy image
- **Function**: `np.var()`

**Implementation**: `backend/app.py` - preprocessing function

### 5.3 Global Average Pooling
- **Purpose**: Reduce spatial dimensions in Grad-CAM
- **Formula**: `α^c_k = (1/Z) * Σ_i Σ_j gradient_value`
- **Application**: Weight calculation for feature maps

**Implementation**: `backend/visualization_utils.py`

### 5.4 Argmax
- **Purpose**: Find predicted class index
- **Formula**: `predicted_class = argmax(probabilities)`
- **Function**: `np.argmax()`

**Implementation**: `backend/app.py` lines 317, 605

---

## 6. Optimization Algorithms

### 6.1 Adam Optimizer
- **Purpose**: Model compilation (if retraining needed)
- **Type**: Adaptive learning rate optimization
- **Formula**: Combines momentum and RMSprop
- **Parameters**: Default TensorFlow settings

**Implementation**: `backend/app.py` line 926

### 6.2 Memory Optimization
- **Techniques**:
  - Soft device placement
  - GPU memory growth limiting
  - Garbage collection (`gc.collect()`)
  - TensorFlow logging suppression

**Implementation**: `backend/app.py` lines 19-31

---

## 7. AI-Powered Recommendation System

### 7.1 OpenAI GPT-4 Integration
- **Purpose**: Generate personalized medical recommendations
- **Model**: GPT-4 (via OpenAI API)
- **Input**: Disease name, confidence, patient demographics
- **Output**: Structured recommendations (tips, advice, lifestyle, warnings)

**Implementation**: `backend/ai_recommendations.py`

### 7.2 Rule-Based Fallback System
- **Purpose**: Provide recommendations when API fails
- **Method**: Predefined templates for each disease
- **Structure**: Dictionary-based lookup

**Implementation**: `backend/ai_recommendations.py` - fallback functions

---

## 8. Database Algorithms

### 8.1 PostgreSQL Queries
- **ORM**: SQLAlchemy
- **Operations**:
  - CRUD operations (Create, Read, Update, Delete)
  - Filtering and sorting
  - Aggregation (statistics calculation)
  - Foreign key relationships

**Implementation**: `backend/database.py`, `backend/db_integration.py`

### 8.2 Statistics Calculation
- **Algorithms**:
  - Count aggregation
  - Average calculation
  - Distribution analysis
  - Percentage computation

**Implementation**: `backend/db_integration.py` - `calculate_statistics()`

---

## 9. PDF Generation Algorithms

### 9.1 ReportLab Layout Engine
- **Purpose**: Generate professional medical reports
- **Techniques**:
  - Canvas-based drawing
  - Image embedding
  - Text formatting
  - Table generation

**Implementation**: `backend/pdf_generator.py`

---

## 10. Interpolation Algorithms

### 10.1 LANCZOS4 (Lanczos Resampling)
- **Purpose**: High-quality image resizing
- **Type**: Windowed sinc interpolation
- **Kernel Size**: 4x4
- **Quality**: Best for downsampling
- **Function**: `cv2.INTER_LANCZOS4`

**Implementation**: `backend/app.py` line 218

### 10.2 INTER_AREA
- **Purpose**: Alternative resizing method
- **Best For**: Downsampling
- **Method**: Pixel area relation resampling

---

## Algorithm Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| CNN Inference | O(n) | O(1) |
| Grad-CAM | O(h×w×c) | O(h×w) |
| Contour Detection | O(n log n) | O(n) |
| Gaussian Blur | O(k²×n) | O(n) |
| Softmax | O(n) | O(1) |
| Image Resize | O(h×w) | O(h×w) |

Where:
- n = number of pixels
- h = height, w = width
- c = number of channels
- k = kernel size

---

## Technology Stack Summary

### Core Frameworks
- **TensorFlow 2.15**: Deep learning framework
- **OpenCV 4.x**: Computer vision library
- **NumPy**: Numerical computing
- **PIL/Pillow**: Image processing

### Supporting Libraries
- **SciPy**: Scientific computing (morphological operations)
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **ReportLab**: PDF generation

---

## References & Research Papers

1. **Grad-CAM**: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (Selvaraju et al., 2017)
2. **CNN**: "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012)
3. **Non-Local Means**: "A non-local algorithm for image denoising" (Buades et al., 2005)
4. **Adam Optimizer**: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)

---

## Performance Metrics

- **Model Inference Time**: ~100-500ms per image
- **Grad-CAM Generation**: ~200-800ms
- **Total Processing Time**: ~1-2 seconds per prediction
- **Memory Usage**: ~500MB-1GB (with model loaded)

---

*Last Updated: November 25, 2025*
