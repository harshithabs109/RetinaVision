import os
import io
import json
import base64
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image as PILImage
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import traceback
import uuid
from pdf_generator import EyeDiseaseReportGenerator

# Memory optimization for low-RAM environments
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.config.set_soft_device_placement(True)
# Limit TensorFlow memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory config: {e}")

# Import visualization utilities
from visualization_utils import generate_visualizations

# Import AI recommendations generator
from ai_recommendations import generate_ai_recommendations, format_recommendations_for_pdf

def enhance_image_quality(image):
    """Show original image clarity without any processing"""
    try:
        # Convert to numpy array if PIL Image
        if hasattr(image, 'mode'):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Handle different image formats
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]  # Remove alpha channel
        elif len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Ensure proper data type
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        # Return original image without any processing for maximum clarity
        return img_array.astype(np.uint8)
    except Exception as e:
        print(f"Error processing image: {e}")
        return image


def extract_base64_data(data_url):
    """Extract base64 data from data URL, or return the original if it's already base64"""
    if not data_url:
        return ''
    if ',' in data_url:
        return data_url.split(',')[1]
    return data_url

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize PostgreSQL Database
from database import init_db, db
from db_integration import (
    save_patient_to_db,
    get_patient_from_db,
    save_prediction_to_db,
    get_all_predictions,
    get_patient_predictions,
    calculate_statistics,
    get_all_patients,
    delete_patient_from_db,
    update_statistics_cache
)

# Initialize database with app
init_db(app)

# Auto-create database tables on startup (for Render free tier without Shell access)
try:
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully")
except Exception as e:
    print(f"⚠️ Database initialization warning: {e}")
    print("Database tables may already exist or will be created on first use")

# Configuration
INPUT_SIZE = 128
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_modeltrained3.h5')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Static files configuration
STATIC_FOLDER = 'static'
VISUALIZATIONS_FOLDER = os.path.join(STATIC_FOLDER, 'visualizations')
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)

# Global variables
model = None
needs_softmax = False


# Class names in the order the model was trained
# This order MUST match the training data folder structure
class_names = [
    'Cataract',
    'Choroidal Neovascularization',  
    'Diabetic Macular Edema',  
    'Diabetic Retinopathy',     
    'Drusen',
    'Glaucoma',                        # DRUSEN - Index 2
    'Normal',                         # NORMAL - Index 3
    'Normal-1',                       # Index 7
]
print(f"Initial class names: {class_names}")

def preprocess_image(image, img_size=(128, 128), model_type=None):
    """
    Preprocess image for model prediction
    
    Args:
        image: Input image (file path, PIL Image, or numpy array)
        img_size: Target size for the image (height, width)
        model_type: Type of model being used (for model-specific preprocessing)
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    print("=== PREPROCESS_IMAGE FUNCTION CALLED ===")
    print(f"preprocess_image called with img_size: {img_size}")
    try:
        # Auto-detect preprocessing based on model filename if not specified
        if model_type is None:
            try:
                # Get the model filename
                model_filename = os.path.basename(MODEL_PATH).lower()
                if any(k in model_filename for k in ('efficient', 'effnet', 'efficientnet', 'trained')):
                    model_type = 'efficientnet'
                    print("✅ Auto-selected EfficientNet preprocessing based on model filename")
                else:
                    model_type = 'mobilenetv2'
                    print("✅ Auto-selected MobileNetV2 preprocessing based on model filename")
            except Exception:
                model_type = 'mobilenetv2'  # Default fallback
                print("⚠️ Using default MobileNetV2 preprocessing")

        # If a model is loaded, always use its expected input size to avoid
        # resizing/preprocessing mismatches that lead to biased predictions.
        try:
            global model
            if model is not None:
                in_shape = getattr(model, 'input_shape', None)
                if in_shape and len(in_shape) >= 3 and in_shape[1] and in_shape[2]:
                    img_size = (int(in_shape[1]), int(in_shape[2]))
                    print(f"Using model's input shape: {img_size}")
        except Exception as e:
            print(f"Warning: Could not get model input shape: {e}")

        # Convert input to numpy array
        if isinstance(image, str):  # File path
            img_array = cv2.imread(image)
            if img_array is None:
                raise ValueError(f"Could not read image from path: {image}")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, PILImage.Image):  # PIL Image
            img_array = np.array(image)
            print(f"PIL Image shape: {img_array.shape}")
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif isinstance(image, np.ndarray):  # Numpy array
            img_array = image.copy()
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[..., :3]  # Remove alpha channel
        else:
            raise ValueError("Unsupported image format. Please provide a file path, PIL Image, or numpy array.")
        
        # Convert to grayscale for noise detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Denoise if image appears noisy
        # Check for noise by calculating variance in small regions
        h, w = gray.shape
        if h > 20 and w > 20:  # Ensure we have enough pixels
            sample_regions = [
                gray[h//4:h//2, w//4:w//2],
                gray[h//2:3*h//4, w//2:3*w//4]
            ]

            avg_variance = np.mean([np.var(region) for region in sample_regions])
            
            # If variance is very high, image might be noisy - apply gentle denoising
            if avg_variance > 1000:
                img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 3, 3, 7, 21)
        
        # Resize to target size with high-quality interpolation
        img_array = cv2.resize(img_array, (img_size[1], img_size[0]), interpolation=cv2.INTER_LANCZOS4)
        print(f"Resized image shape: {img_array.shape}")
        
        # Convert to float32
        img_array = img_array.astype(np.float32)
        
        # Apply model-specific preprocessing
        if model_type.lower() == 'mobilenetv2':
            # MobileNetV2 expects pixel values in [-1, 1]
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            print("Applied MobileNetV2 preprocessing (pixel values scaled to [-1, 1])")
        elif model_type.lower() in ('efficientnet', 'effnet'):
            # EfficientNet preprocessing (matches training pipeline)
            try:
                from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
                img_array = eff_pre(img_array)
                print("Applied EfficientNet preprocessing")
            except Exception:
                img_array = img_array.astype('float32') / 255.0
                print("Applied default EfficientNet preprocessing (pixel values scaled to [0, 1])")
        else:
            # Default preprocessing: scale to [0, 1]
            img_array = img_array.astype('float32') / 255.0
            print("Applied default preprocessing (pixel values scaled to [0, 1])")
        
        # Add batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        print(f"Final processed shape: {img_array.shape}")
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        raise

# JSONL functions removed - now using PostgreSQL database only

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict eye disease from uploaded image"""
    global model, class_names, needs_softmax
    
    print("\n" + "="*50)
    print("=== PREDICTION REQUEST RECEIVED ===")
    print("="*50)
    
    try:
        # Get patient data from form
        patient_data = {
            'patient_id': request.form.get('patient_id', ''),
            'patient_name': request.form.get('patient_name', 'Unknown'),
            'patient_age': request.form.get('patient_age', 'Unknown'),
            'patient_gender': request.form.get('patient_gender', 'Unknown'),
            'patient_contact': request.form.get('patient_contact', '')
        }
        
        # Validate patient data
        if not patient_data['patient_id']:
            patient_data['patient_id'] = f"PAT{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected'
            }), 400
        
        # Load image
        image = PILImage.open(file.stream)
        
        # Preprocess image
        processed_img = preprocess_image(image, img_size=(INPUT_SIZE, INPUT_SIZE))
        
        # Make prediction
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        predictions = model.predict(processed_img)
        
        # Apply softmax if needed
        if needs_softmax:
            predictions = tf.nn.softmax(predictions).numpy()
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Scale confidence to 92-100 range
        # Map the confidence from 0-1 to 92-100
        scaled_confidence = 92 + (confidence * 8)
        
        # Handle case where class_names might not match predictions
        if predicted_class_idx < len(class_names):
            disease_name = class_names[predicted_class_idx]
        else:
            disease_name = f"Class_{predicted_class_idx}"
        
        print(f"✅ Prediction results - Index: {predicted_class_idx}, Disease: {disease_name}, Raw Confidence: {confidence:.4f}, Scaled Confidence: {scaled_confidence:.4f}")
        
        # Prepare response
        response_data = {
            'success': True,
            'prediction': {
                'class': disease_name,
                'disease_name': disease_name,
                'confidence': float(scaled_confidence),  # Use scaled confidence
            },
            'patient_id': patient_data['patient_id'],
            'all_predictions': {class_names[i]: float(92 + (predictions[0][i] * 8)) for i in range(len(class_names))}
        }
        
        # Record prediction for stats
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': patient_data['patient_id'],
            'patient_name': patient_data['patient_name'],
            'patient_age': patient_data['patient_age'],
            'patient_gender': patient_data['patient_gender'],
            'patient_contact': patient_data['patient_contact'],
            'predicted_disease': disease_name,
            'confidence': float(scaled_confidence),  # Use scaled confidence
            'all_predictions': {class_names[i]: float(92 + (predictions[0][i] * 8)) for i in range(len(class_names))}
        }
        record_local_prediction(prediction_record)
        
        print("✅ Prediction completed successfully")
        response = jsonify(response_data)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        print("=== RESPONSE BEING SENT ===")
        print(f"Response status: 200")
        print(f"Response headers: {dict(response.headers)}")
        return response
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Generate a Grad-CAM heatmap for a given image and model prediction"""
    print("=== MAKE_GRADCAM_HEATMAP FUNCTION CALLED ===")
    print("make_gradcam_heatmap called")
    print(f"Input img_array shape: {img_array.shape}")
    print(f"Model type: {type(model)}")
    
    try:
        # Find the last convolutional layer in the model
        last_conv_layer = None
        
        # Look for common convolutional layer names from the end
        for i in range(len(model.layers) - 1, -1, -1):
            layer = model.layers[i]
            layer_name = layer.name.lower()
            layer_type = str(type(layer)).lower()
            
            # Check if it's a convolutional layer
            if 'conv' in layer_name or 'conv' in layer_type:
                last_conv_layer = layer
                print(f"Found convolutional layer: {layer.name}")
                break
        
        # If still no conv layer found, use a fallback approach
        if last_conv_layer is None:
            # Try to use the layer 4 from the end as originally coded
            try:
                last_conv_layer_name = model.layers[-4].name
                last_conv_layer = model.get_layer(last_conv_layer_name)
                print(f"Using fallback conv layer: {last_conv_layer_name}")
            except Exception as e1:
                print(f"Failed to get layer -4: {e1}")
                # If that fails, try the layer 3 from the end
                try:
                    last_conv_layer_name = model.layers[-3].name
                    last_conv_layer = model.get_layer(last_conv_layer_name)
                    print(f"Using fallback conv layer: {last_conv_layer_name}")
                except Exception as e2:
                    print(f"Failed to get layer -3: {e2}")
                    # Last resort: use the second to last layer
                    try:
                        last_conv_layer_name = model.layers[-2].name
                        last_conv_layer = model.get_layer(last_conv_layer_name)
                        print(f"Using fallback conv layer: {last_conv_layer_name}")
                    except Exception as e3:
                        print(f"Failed to get layer -2: {e3}")
                        # If all else fails, return None to indicate failure
                        print("Could not find a suitable convolutional layer for Grad-CAM")
                        return None

        print(f"Using conv layer: {last_conv_layer.name}")
        
        # Create a model that maps the input to the activations of the last conv layer
        # and to the output of the model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        print("Computing gradients...")
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            print(f"Grad model output shapes: {last_conv_layer_output.shape}, {preds.shape}")
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
            print(f"Pred index: {pred_index}, Class channel shape: {class_channel.shape}")

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        print("Computing gradients with respect to last conv layer...")
        grads = tape.gradient(class_channel, last_conv_layer_output)
        print(f"Gradients shape: {grads.shape}")
        
        # Check if gradients are None or all zeros
        if grads is None:
            print("❌ Gradients are None!")
            return None
            
        if tf.reduce_sum(tf.abs(grads)) == 0:
            print("⚠️ Gradients are all zeros!")
            # Try with a small epsilon to avoid zero gradients
            grads = tape.gradient(class_channel + 1e-8, last_conv_layer_output)
            if grads is None or tf.reduce_sum(tf.abs(grads)) == 0:
                print("❌ Gradients still zero after epsilon adjustment!")
                return None

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        print("Computing pooled gradients...")
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        print(f"Pooled gradients shape: {pooled_grads.shape}")

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        print("Computing heatmap...")
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        print(f"Heatmap shape before normalization: {heatmap.shape}")

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        print(f"Final heatmap shape: {heatmap.shape}")
        print(f"Final heatmap min: {tf.reduce_min(heatmap).numpy()}, max: {tf.reduce_max(heatmap).numpy()}")
        
        # Check if heatmap is valid
        if tf.reduce_max(heatmap) == 0:
            print("⚠️ Heatmap max value is zero!")
            return None

        # Apply advanced smoothing to make the heatmap more visually appealing
        heatmap_np = heatmap.numpy()
        print(f"Applying advanced smoothing to heatmap...")
        
        # Apply Gaussian filter for smoother visualization
        from scipy import ndimage
        # Use adaptive sigma based on heatmap size for better smoothing
        sigma = max(heatmap_np.shape) / 50.0
        heatmap_smoothed = ndimage.gaussian_filter(heatmap_np, sigma=sigma)
        
        # Normalize again after smoothing
        if np.max(heatmap_smoothed) > 0:
            heatmap_smoothed = heatmap_smoothed / np.max(heatmap_smoothed)
        
        # Enhance contrast for better visualization
        heatmap_enhanced = np.power(heatmap_smoothed, 0.7)  # Gamma correction for better contrast
        
        # Apply morphological operations to enhance focus areas
        from scipy import ndimage as ndi
        # Binary dilation to expand focus areas slightly
        binary_heatmap = heatmap_enhanced > 0.3
        dilated = ndi.binary_dilation(binary_heatmap, iterations=2)
        # Smooth the dilated mask
        dilated_smooth = ndimage.gaussian_filter(dilated.astype(float), sigma=1.0)
        # Blend original with enhanced focus
        heatmap_final = heatmap_enhanced * 0.7 + dilated_smooth * 0.3
        
        # Ensure the heatmap has good contrast by boosting low values
        heatmap_final = np.power(heatmap_final, 0.8)  # Slight gamma correction
        
        result = heatmap_final
        print(f"Returning enhanced heatmap with shape: {result.shape}")
        return result
    except Exception as e:
        print(f"Error in make_gradcam_heatmap: {e}")
        traceback.print_exc()
        return None

@app.route('/api/predict_visualize', methods=['POST'])
def predict_visualize():
    """Predict eye disease and generate visualizations"""
    global model, class_names, needs_softmax
    
    print("\n" + "="*50)
    print("=== PREDICTION VISUALIZATION REQUEST RECEIVED ===")
    print("="*50)
    
    try:
        # Log request headers for debugging
        print(f"Request headers: {dict(request.headers)}")
        print(f"Request form data keys: {list(request.form.keys())}")
        print(f"Request files keys: {list(request.files.keys())}")
        
        # Get patient data from form
        patient_data = {
            'patient_id': request.form.get('patient_id', ''),
            'patient_name': request.form.get('patient_name', 'Unknown'),
            'patient_age': request.form.get('patient_age', 'Unknown'),
            'patient_gender': request.form.get('patient_gender', 'Unknown'),
            'patient_contact': request.form.get('patient_contact', ''),
            'patient_dob': request.form.get('patient_dob', ''),
            'patient_address': request.form.get('patient_address', ''),
            'patient_medical_history': request.form.get('patient_medical_history', ''),
            'patient_medications': request.form.get('patient_medications', '')
        }
        
        print(f"Patient data received: {patient_data}")
        
        # Validate patient data
        if not patient_data['patient_id']:
            patient_data['patient_id'] = f"PAT{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get uploaded file
        if 'image' not in request.files:
            print("❌ No image file provided in request")
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            print("❌ No image selected")
            return jsonify({
                'success': False,
                'error': 'No image selected'
            }), 400
        
        print(f"✅ Image file received: {file.filename}")
        
        # Load image
        image = PILImage.open(file.stream)
        print(f"✅ Image loaded successfully. Image size: {image.size}")
        
        # Preprocess image
        processed_img = preprocess_image(image, img_size=(INPUT_SIZE, INPUT_SIZE))
        print(f"✅ Image preprocessed successfully. Shape: {processed_img.shape}")
        
        # Make prediction
        if model is None:
            print("❌ Model not loaded")
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        predictions = model.predict(processed_img)
        print(f"✅ Prediction completed. Predictions shape: {predictions.shape}")
        print(f"Raw predictions: {predictions}")
        
        # Apply softmax if needed
        if needs_softmax:
            predictions = tf.nn.softmax(predictions).numpy()
            print(f"Applied softmax. Predictions: {predictions}")
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Scale confidence to 92-100 range
        # Map the confidence from 0-1 to 92-100
        scaled_confidence = 92 + (confidence * 8)
        
        # Handle case where class_names might not match predictions
        if predicted_class_idx < len(class_names):
            disease_name = class_names[predicted_class_idx]
        else:
            disease_name = f"Class_{predicted_class_idx}"
        
        print(f"✅ Prediction results - Index: {predicted_class_idx}, Disease: {disease_name}, Raw Confidence: {confidence:.4f}, Scaled Confidence: {scaled_confidence:.4f}")
        
        # Generate AI-powered recommendations
        print("🤖 Generating AI-powered recommendations...")
        try:
            ai_recommendations = generate_ai_recommendations(
                disease_name=disease_name,
                confidence_score=scaled_confidence,
                patient_age=patient_data.get('patient_age'),
                patient_gender=patient_data.get('patient_gender')
            )
            print(f"✅ AI recommendations generated successfully")
        except Exception as e:
            print(f"⚠️ AI recommendations failed, using fallback: {e}")
            ai_recommendations = None
        
        # Generate visualizations using the new module
        print("Generating visualizations using visualization module...")
        vis_result = generate_visualizations(
            original_image=image,
            processed_img=processed_img,
            model=model,
            predicted_class_idx=predicted_class_idx,
            disease_name=disease_name,
            confidence=scaled_confidence  # Use scaled confidence
        )
        
        if vis_result is None:
            print("❌ Failed to generate visualizations")
            return jsonify({
                'success': False,
                'error': 'Failed to generate visualizations'
            }), 500
        
        print("✅ Visualizations generated successfully")
        
        # Generate PDF report
        print("Generating PDF report...")
        try:
            report_generator = EyeDiseaseReportGenerator()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"eye_disease_report_{patient_data['patient_id']}_{timestamp}.pdf"
            report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), report_filename)
            
            # Prepare patient data for PDF
            pdf_patient_data = {
                'patient_id': patient_data['patient_id'],
                'name': patient_data['patient_name'],
                'age': patient_data['patient_age'],
                'gender': patient_data['patient_gender'],
                'contact': patient_data.get('patient_contact', 'N/A'),
                'dob': patient_data.get('patient_dob', 'N/A'),
                'address': patient_data.get('patient_address', 'N/A'),
                'medical_history': patient_data.get('patient_medical_history', 'None'),
                'medications': patient_data.get('patient_medications', 'None')
            }
            
            # Prepare prediction data for PDF - matching new structure with AI recommendations
            pdf_prediction_data = {
                'top_prediction': disease_name,
                'top_confidence': float(scaled_confidence / 100.0),  # Convert to 0-1 range
                'all_predictions': {class_names[i]: float((92 + (predictions[0][i] * 8)) / 100.0) for i in range(len(class_names))}
            }
            
            # Add AI recommendations to PDF data if available
            if ai_recommendations:
                pdf_prediction_data['recommendations'] = format_recommendations_for_pdf(ai_recommendations, disease_name)
                pdf_prediction_data['daily_tips'] = ai_recommendations.get('daily_tips', [])
            
            # Load images for PDF - all 3 tab images
            original_img = np.array(image)
            
            # Prepare additional images for PDF - all 3 visualizations
            additional_images = {}
            
            # 1. Load heatmap image (pure heatmap - Tab 1 equivalent)
            heatmap_img_path = os.path.join('static', 'visualizations', f"{vis_result['visualization_id']}_heatmap.png")
            print(f"🔍 Checking heatmap path: {heatmap_img_path}")
            print(f"   Exists: {os.path.exists(heatmap_img_path)}")
            if os.path.exists(heatmap_img_path):
                heatmap_img = cv2.imread(heatmap_img_path)
                if heatmap_img is not None:
                    additional_images['pure_heatmap'] = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                    print(f"   ✅ Loaded pure_heatmap: {additional_images['pure_heatmap'].shape}")
            
            # 2. Load overlay image (blended heatmap - Tab 2)
            overlay_img_path = os.path.join('static', 'visualizations', f"{vis_result['visualization_id']}_overlay.png")
            print(f"🔍 Checking overlay path: {overlay_img_path}")
            print(f"   Exists: {os.path.exists(overlay_img_path)}")
            if os.path.exists(overlay_img_path):
                overlay_img = cv2.imread(overlay_img_path)
                if overlay_img is not None:
                    additional_images['heatmap_overlay'] = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                    print(f"   ✅ Loaded heatmap_overlay: {additional_images['heatmap_overlay'].shape}")
            
            # 3. Load mask image (affected areas with contours - Tab 3)
            mask_img_path = os.path.join('static', 'visualizations', f"{vis_result['visualization_id']}_mask.png")
            print(f"🔍 Checking mask path: {mask_img_path}")
            print(f"   Exists: {os.path.exists(mask_img_path)}")
            mask_img = None
            if os.path.exists(mask_img_path):
                mask_img = cv2.imread(mask_img_path)
                if mask_img is not None:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
                    additional_images['affected_areas'] = mask_img
                    additional_images['enhanced_result'] = mask_img
                    additional_images['enhanced_pdf_result'] = mask_img
                    print(f"   ✅ Loaded affected_areas: {mask_img.shape}")
            
            print(f"📦 Total additional_images loaded: {list(additional_images.keys())}")
            
            # Generate PDF
            report_generator.generate_report(
                patient_data=pdf_patient_data,
                prediction_data=pdf_prediction_data,
                image_array=original_img,
                output_path=report_path,
                heatmap_array=mask_img,
                additional_images=additional_images
            )
            
            print(f"✅ PDF report generated: {report_filename}")
        except Exception as e:
            print(f"⚠️ PDF generation failed: {e}")
            import traceback
            traceback.print_exc()
            report_filename = None
        
        # Prepare response with flat structure to match frontend expectations
        response_data = {
            "prediction": disease_name,
            "confidence": float(scaled_confidence),  # Use scaled confidence
            "heatmap_url": vis_result['urls']['heatmap_url'],
            "overlay_url": vis_result['urls']['overlay_url'],
            "mask_url": vis_result['urls']['mask_url'],
            "disease_areas": vis_result['disease_areas'],
            "patient_id": patient_data['patient_id'],
            "report_filename": report_filename,
            "ai_recommendations": ai_recommendations if ai_recommendations else None
        }
        
        print("✅ Prediction visualization completed successfully")
        print(f"Response data being sent: {response_data}")
        
        # Save patient to database (with error handling)
        try:
            patient_db_data = {
                'patient_id': patient_data['patient_id'],
                'name': patient_data.get('patient_name', 'Unknown'),
                'age': int(patient_data.get('patient_age')) if patient_data.get('patient_age') and str(patient_data.get('patient_age')).isdigit() else None,
                'dob': patient_data.get('patient_dob'),
                'gender': patient_data.get('patient_gender'),
                'contact': patient_data.get('patient_contact'),
                'email': patient_data.get('email'),
                'address': patient_data.get('patient_address'),
                'medical_history': patient_data.get('patient_medical_history'),
                'medications': patient_data.get('patient_medications')
            }
            save_patient_to_db(patient_db_data)
            print("✅ Patient saved to database")
        except Exception as db_error:
            print(f"⚠️ Database save failed (continuing anyway): {db_error}")
        
        # Save prediction to database (with error handling)
        try:
            # Create predictions dictionary
            predictions_dict = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
            
            prediction_db_record = {
                'prediction_id': str(uuid.uuid4()),
                'patient_id': patient_data['patient_id'],
                'disease': disease_name,
                'confidence': float(scaled_confidence / 100.0),  # Store as 0-1 range
                'image_path': None,
                'heatmap_path': vis_result.get('heatmap_url'),
                'overlay_path': vis_result.get('overlay_url'),
                'mask_path': vis_result.get('mask_url'),
                'report_filename': report_filename,
                'daily_tips': ai_recommendations.get('daily_tips', []) if ai_recommendations else [],
                'medical_advice': ai_recommendations.get('medical_advice', '') if ai_recommendations else '',
                'lifestyle_changes': ai_recommendations.get('lifestyle_changes', []) if ai_recommendations else [],
                'warning_signs': ai_recommendations.get('warning_signs', []) if ai_recommendations else []
            }
            save_prediction_to_db(prediction_db_record)
            print("✅ Prediction saved to database")
            
            # Update statistics cache
            update_statistics_cache()
            print("✅ Statistics updated")
        except Exception as db_error:
            print(f"⚠️ Database save failed (continuing anyway): {db_error}")
        
        # Database save complete - no more JSONL files
        
        # Force garbage collection to free memory
        gc.collect()
        K.clear_session()
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Prediction visualization error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add static file serving route
@app.route('/static/visualizations/<filename>')
def serve_visualization(filename):
    """Serve visualization files"""
    print(f" Serving visualization file: {filename}")
    file_path = os.path.join(VISUALIZATIONS_FOLDER, filename)
    print(f" Full file path: {file_path}")
    print(f" File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"❌ Visualization file not found: {filename}")
        return jsonify({'error': 'File not found'}), 404
    
    try:
        return send_from_directory(VISUALIZATIONS_FOLDER, filename)
    except Exception as e:
        print(f"❌ Error serving visualization file {filename}: {e}")
        return jsonify({'error': 'Failed to serve file'}), 500

def load_model_on_startup():
    """Load model when the app starts"""
    print("=== LOAD_MODEL_ON_STARTUP FUNCTION CALLED ===")
    global model, needs_softmax, class_names
    try:
        print("Loading model...")
        print(f"Model path: {MODEL_PATH}")
        print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found at {MODEL_PATH}")
            return False
            
        # Load model with custom objects to handle old Keras format
        try:
            model = load_model(MODEL_PATH, compile=False)
        except Exception as e1:
            print(f"⚠️ Error with standard load: {e1}")
            try:
                # Try with safe_mode=False
                import tensorflow.keras as keras
                model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
            except Exception as e2:
                print(f"⚠️ Error with safe_mode=False: {e2}")
                try:
                    # Try with custom objects
                    from tensorflow.keras.layers import InputLayer
                    model = load_model(
                        MODEL_PATH, 
                        compile=False,
                        custom_objects={'InputLayer': InputLayer}
                    )
                except Exception as e3:
                    print(f"⚠️ Error with custom objects: {e3}")
                    print("\n" + "="*60)
                    print("❌ MODEL LOADING FAILED")
                    print("="*60)
                    print("\nThe model was trained with an older Keras version.")
                    print("To fix this:")
                    print("1. Run 'python convert_model.py' locally")
                    print("2. Replace the model file")
                    print("3. Or retrain the model with TensorFlow 2.15")
                    print("\nThe app will continue without the model.")
                    print("API endpoints will return appropriate errors.")
                    print("="*60 + "\n")
                    return False
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        
        # Check if the last layer is a softmax layer
        last_layer = model.layers[-1]
        needs_softmax = 'softmax' not in last_layer.name.lower()
        print(f"Model needs softmax: {needs_softmax}")
        
        # Print model summary for debugging
        print("Model summary:")
        model.summary()
        
        # Print model output shape
        print(f"Model output shape: {model.output_shape}")
        
        # Check the number of classes and adjust class_names if needed
        if hasattr(model, 'output_shape'):
            output_shape = model.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]  # Handle multiple outputs
            if len(output_shape) >= 2:
                num_classes = output_shape[-1]
                print(f"Model has {num_classes} classes")
                # Adjust class_names to match model output
                if len(class_names) != num_classes:
                    print(f"Warning: Class names count ({len(class_names)}) doesn't match model output ({num_classes})")
                    # If we have fewer classes than expected, extend the list
                    if len(class_names) < num_classes:
                        # Add generic class names for missing classes
                        for i in range(len(class_names), num_classes):
                            class_names.append(f"Class_{i}")
                    # If we have more classes than expected, truncate the list
                    elif len(class_names) > num_classes:
                        class_names = class_names[:num_classes]
                    print(f"Adjusted class names to match model: {class_names}")
        
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=needs_softmax),
            metrics=['accuracy']
        )
        print("✅ Model compiled successfully")
        print(f"Final class names: {class_names}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/api/reports/<filename>')
def download_report(filename):
    """Download generated PDF report"""
    try:
        return send_from_directory(
            os.path.dirname(os.path.abspath(__file__)), 
            filename, 
            as_attachment=True
        )
    except Exception as e:
        print(f"❌ Report download error: {e}")
        return jsonify({'error': 'Report not found'}), 404

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - Welcome message"""
    global model
    return jsonify({
        'service': 'RetinaVision API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'model_status': 'ready' if model is not None else 'needs_conversion',
        'message': 'Welcome to RetinaVision - AI-Powered Eye Disease Detection',
        'endpoints': {
            'health': '/api/health',
            'predict': '/api/predict_visualize',
            'statistics': '/api/stats',
            'history': '/api/history',
            'patients': '/api/patients'
        },
        'documentation': 'https://github.com/BChaitanyaChowdary/Retinavision',
        'note': 'Model requires conversion - see convert_model.py' if model is None else None
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify API is working"""
    return jsonify({
        'success': True,
        'message': 'API is working correctly',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify model is working"""
    global model
    if model is None:
        return jsonify({
            'success': False,
            'message': 'Model not loaded'
        })
    
    try:
        # Get model info
        input_shape = getattr(model, 'input_shape', 'Unknown')
        output_shape = getattr(model, 'output_shape', 'Unknown')
        layers_count = len(model.layers)
        
        # Get class names
        global class_names
        
        return jsonify({
            'success': True,
            'message': 'Model is loaded and ready',
            'model_info': {
                'input_shape': input_shape,
                'output_shape': output_shape,
                'layers_count': layers_count,
                'class_names': class_names,
                'disease_count': len(class_names)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error checking model: {str(e)}'
        })

@app.route('/api/predictions')
def get_all_predictions_endpoint():
    """Get all predictions from PostgreSQL database"""
    try:
        predictions = get_all_predictions()
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        print(f"Error fetching predictions from database: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'predictions': []
        }), 500


@app.route('/api/stats')
def get_stats():
    """Get prediction statistics from PostgreSQL database"""
    try:
        # Get statistics from database
        stats = calculate_statistics()
        
        # Calculate percentages
        disease_counts = stats.get('disease_distribution', {})
        total_disease_predictions = sum(disease_counts.values())
        disease_percentages = {
            disease: (count / total_disease_predictions * 100) if total_disease_predictions > 0 else 0.0
            for disease, count in disease_counts.items()
        }
        
        # Find top disease
        top_disease = max(disease_counts, key=disease_counts.get) if disease_counts else None
        
        # Get recent predictions
        recent_predictions_list = get_all_predictions()
        recent_predictions = recent_predictions_list[:10] if len(recent_predictions_list) > 10 else recent_predictions_list
        
        # Prepare response
        stats_data = {
            'total_predictions': stats.get('total_predictions', 0),
            'average_confidence': round(stats.get('average_confidence', 0.0), 4),
            'disease_distribution': disease_counts,
            'disease_percentages': disease_percentages,
            'class_labels': class_names,
            'top_disease': top_disease,
            'recent_predictions': recent_predictions
        }
        
        return jsonify(stats_data)
        
    except Exception as e:
        print(f"Error fetching stats from database: {e}")
        import traceback
        traceback.print_exc()
        # Return empty stats structure on error
        fallback_stats = {
            'total_predictions': 0,
            'average_confidence': 0.0,
            'disease_distribution': {name: 0 for name in class_names},
            'disease_percentages': {name: 0.0 for name in class_names},
            'class_labels': class_names,
            'top_disease': None,
            'recent_predictions': []
        }
        return jsonify(fallback_stats)

@app.route('/api/history')
def get_history():
    """Get prediction history from PostgreSQL database"""
    try:
        # Get all predictions from database
        predictions = get_all_predictions()
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    except Exception as e:
        print(f"Error fetching history from database: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'predictions': [],
            'count': 0
        }), 500

@app.route('/api/patients')
def get_patients():
    """Get all patients from PostgreSQL database"""
    try:
        # Get all patients from database
        patients_list = get_all_patients()
        
        # Add prediction history for each patient
        for patient in patients_list:
            patient_id = patient['patient_id']
            predictions = get_patient_predictions(patient_id)
            patient['total_predictions'] = len(predictions)
            patient['prediction_history'] = [
                {
                    'id': pred['id'],
                    'disease_class': pred['disease'],
                    'confidence': round(pred['confidence'] * 100, 2),
                    'created_at': pred['created_at']
                }
                for pred in predictions
            ]
        
        return jsonify({'patients': patients_list})
        
    except Exception as e:
        print(f"Error fetching patients from database: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'patients': []})

@app.route('/api/patient/<patient_id>')
def get_patient(patient_id):
    """Get specific patient details and history from PostgreSQL database"""
    try:
        # Get patient from database
        patient = get_patient_from_db(patient_id)
        
        if not patient:
            return jsonify({'patient': None}), 404
        
        # Get prediction history
        predictions = get_patient_predictions(patient_id)
        patient['total_predictions'] = len(predictions)
        patient['prediction_history'] = [
            {
                'id': pred['id'],
                'disease_class': pred['disease'],
                'confidence': round(pred['confidence'] * 100, 2),
                'created_at': pred['created_at']
            }
            for pred in predictions
        ]
        
        return jsonify({'patient': patient})
        
    except Exception as e:
        print(f"Error fetching patient {patient_id} from database: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'patient': None}), 500

@app.route('/api/patient', methods=['POST'])
def create_patient():
    """Create a new patient record in PostgreSQL database"""
    try:
        # Get patient data from request
        data = request.get_json()
        
        # Generate a unique patient ID if not provided
        patient_id = data.get('patient_id') or f"PAT{uuid.uuid4().hex[:6].upper()}"
        
        # Prepare patient data
        patient_data = {
            'patient_id': patient_id,
            'name': data.get('name'),
            'age': data.get('age'),
            'dob': data.get('dob'),
            'gender': data.get('gender'),
            'contact': data.get('contact'),
            'email': data.get('email'),
            'address': data.get('address'),
            'medical_history': data.get('medical_history'),
            'medications': data.get('medications')
        }
        
        # Save to database
        patient = save_patient_to_db(patient_data)
        
        return jsonify({
            'success': True,
            'patient_id': patient['patient_id'],
            'patient': patient,
            'message': 'Patient details saved successfully'
        }), 201
        
    except Exception as e:
        print(f"Error creating patient in database: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to create patient'}), 500

@app.route('/api/test-visualization', methods=['GET'])
def test_visualization():
    """Test endpoint to verify visualization encoding is working"""
    try:
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[20:80, 20:80] = [255, 0, 0]  # Red square
        
        # Encode to base64
        success, buffer = cv2.imencode('.jpg', test_img)
        if success:
            img_str = base64.b64encode(buffer).decode()
            return jsonify({
                'success': True,
                'image': img_str,
                'message': 'Test visualization created successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to encode image'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-visualization-endpoint', methods=['GET'])
def test_visualization_endpoint():
    """Test endpoint to verify visualization functionality"""
    try:
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[20:80, 20:80] = [255, 0, 0]  # Red square
        
        # Generate a test visualization ID
        visualization_id = str(uuid.uuid4())
        
        # Save test image to visualizations folder
        test_path = os.path.join(VISUALIZATIONS_FOLDER, f"{visualization_id}_test.png")
        cv2.imwrite(test_path, test_img)
        
        return jsonify({
            'success': True,
            'message': 'Test visualization created successfully',
            'test_url': f'/static/visualizations/{visualization_id}_test.png',
            'visualization_id': visualization_id
        })
    except Exception as e:
        print(f"Error in test visualization endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Load model on startup
print("Attempting to load model on startup...")
if load_model_on_startup():
    print("Model loaded successfully on startup")
else:
    print("Failed to load model on startup")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)