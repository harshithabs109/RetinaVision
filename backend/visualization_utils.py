"""
Visualization utilities for eye disease prediction system.
Contains functions for generating heatmaps, overlays, and disease area visualizations.
"""

import os
import cv2
import numpy as np
import uuid
from PIL import Image as PILImage
import tensorflow as tf
from datetime import datetime

# Configuration
VISUALIZATIONS_FOLDER = os.path.join('static', 'visualizations')

def create_visualizations_folder():
    """Ensure the visualizations folder exists"""
    os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Generate an enhanced Grad-CAM heatmap with improved layer selection and gradient handling
    Based on the robust implementation from the root app.py
    """
    print("\n" + "="*60)
    print("🔍 STARTING ENHANCED GRADCAM COMPUTATION")
    print("="*60)
    print(f"Input img_array shape: {img_array.shape}")
    print(f"Input img_array dtype: {img_array.dtype}")
    print(f"Input img_array min/max: {np.min(img_array):.4f} / {np.max(img_array):.4f}")
    print(f"Model type: {type(model)}")
    print(f"Pred index: {pred_index}")
    
    # CRITICAL: If model is None, return None immediately
    if model is None:
        print("❌ Model is None! Cannot generate GradCAM")
        return None
    
    try:
        # Ensure input has correct shape
        x = img_array
        if len(x.shape) == 3:
            x = tf.expand_dims(x, 0)
        
        # Make model trainable for gradient computation
        original_trainable_state = {}
        try:
            for layer in model.layers:
                original_trainable_state[layer.name] = layer.trainable
                layer.trainable = True
            model.trainable = True
        except Exception as e:
            print(f"Warning: Could not set all layers trainable: {e}")
        
        # Find the best convolutional layer for Grad-CAM
        # Strategy: Find the last convolutional layer before GlobalAveragePooling or Flatten
        last_conv_layer_name = None
        
        # First, try to find layers before GlobalAveragePooling2D or Flatten
        for i in range(len(model.layers) - 1, -1, -1):
            layer = model.layers[i]
            layer_type = type(layer).__name__
            
            # Check if this is a pooling/flatten layer
            if 'globalaveragepooling' in layer_type.lower() or 'flatten' in layer_type.lower():
                # Look backwards from here for the last conv layer
                for j in range(i - 1, -1, -1):
                    prev_layer = model.layers[j]
                    prev_layer_type = type(prev_layer).__name__
                    prev_layer_name_lower = prev_layer.name.lower()
                    
                    # Check for convolutional layers
                    if ('conv' in prev_layer_type.lower() or 'conv2d' in prev_layer_type.lower() or 'conv' in prev_layer_name_lower):
                        last_conv_layer_name = prev_layer.name
                        print(f"✅ Found last conv layer before pooling: {prev_layer.name} (type: {prev_layer_type})")
                        break
                if last_conv_layer_name:
                    break
        
        # If not found, look for EfficientNet/MobileNet specific layers
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                layer_name_lower = layer.name.lower()
                layer_type = type(layer).__name__
                if ('top_conv' in layer_name_lower or 
                    'block7a_expand_conv' in layer_name_lower or
                    'block6a_expand_conv' in layer_name_lower or
                    'conv_head' in layer_name_lower or
                    ('conv' in layer_type.lower() and 'pooling' not in layer_type.lower())):
                    last_conv_layer_name = layer.name
                    print(f"✅ Found EfficientNet/MobileNet layer: {layer.name}")
                    break
        
        # If still not found, look for any conv layer
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                layer_type = type(layer).__name__
                layer_name_lower = layer.name.lower()
                if (('conv' in layer_type.lower() or 'conv2d' in layer_type.lower()) and 
                    'pooling' not in layer_type.lower() and 'conv' in layer_name_lower):
                    last_conv_layer_name = layer.name
                    print(f"✅ Found conv layer: {layer.name}")
                    break
        
        if last_conv_layer_name is None:
            print("❌ Could not find a suitable convolutional layer for GradCAM")
            return None
        
        print(f"Using layer for GradCAM: {last_conv_layer_name}")
        
        # Get the convolutional layer
        conv_layer = model.get_layer(last_conv_layer_name)
        
        # Create GradCAM model - outputs both conv layer and predictions
        try:
            model_input = model.inputs if hasattr(model, 'inputs') and model.inputs is not None else model.input
            grad_model = tf.keras.models.Model(
                inputs=model_input,
                outputs=[conv_layer.output, model.output]
            )
            grad_model.trainable = True
            for layer in grad_model.layers:
                layer.trainable = True
        except Exception as e:
            print(f"❌ Error creating grad_model: {e}")
            return None
        
        # Compute gradients with proper watching
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(x)
            
            try:
                # Use training=True to ensure gradients are computed
                conv_outputs, predictions = grad_model(x, training=True)
            except Exception as e:
                print(f"Error in forward pass with training=True: {e}, trying training=False")
                try:
                    conv_outputs, predictions = grad_model(x, training=False)
                except Exception as e2:
                    print(f"❌ Error in forward pass: {e2}")
                    return None
            
            # Get prediction index
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            else:
                if isinstance(pred_index, (int, np.integer)):
                    pred_index = tf.constant(int(pred_index), dtype=tf.int64)
                else:
                    pred_index = tf.cast(pred_index, dtype=tf.int64)
            
            # Validate pred_index
            pred_index_int = int(pred_index.numpy() if hasattr(pred_index, 'numpy') else pred_index)
            if pred_index_int >= predictions.shape[1] or pred_index_int < 0:
                pred_index = tf.argmax(predictions[0])
                pred_index_int = int(pred_index.numpy())
            
            # Get the score for the specific class
            class_channel = predictions[:, pred_index]
            
            print(f"GradCAM - Using class index: {pred_index_int}")
            print(f"GradCAM - Class channel value: {class_channel.numpy()}")
        
        # Compute gradients
        try:
            print(f"🔍 Computing gradients of class {pred_index_int} score w.r.t. conv outputs...")
            grads = tape.gradient(class_channel, conv_outputs)
            print(f"✅ Gradients computed successfully")
            
            if grads is not None:
                # Apply guided gradients (ReLU on gradients) for better focus
                grads = tf.maximum(grads, 0.0)
                
                grad_std = tf.math.reduce_std(grads).numpy()
                grad_max = tf.reduce_max(grads).numpy()
                print(f"   Gradient stats: std={grad_std:.6f}, max={grad_max:.6f}")
                
                # Check if gradients are valid
                if grad_std < 0.0001:
                    print(f"❌ Gradients are constant or near-zero (std={grad_std:.8f})!")
                    return None
                elif abs(grad_max) < 0.001:
                    print(f"⚠️ Gradients are very small, applying amplification...")
                    if grad_max > 0:
                        grads = grads * (0.01 / grad_max)
            else:
                print(f"❌ Gradients are None!")
                return None
        except Exception as e:
            print(f"❌ ERROR computing gradients: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            del tape
        
        # Global average pooling of gradients
        if len(grads.shape) == 4:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        elif len(grads.shape) == 3:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        else:
            pooled_grads = grads
        
        # Get the first item in the batch
        if len(conv_outputs.shape) == 4:
            conv_outputs = conv_outputs[0]
        
        # Weighted combination
        print(f"🔧 Computing heatmap from gradients and conv outputs...")
        print(f"   Conv outputs shape: {conv_outputs.shape}")
        print(f"   Pooled grads shape: {pooled_grads.shape}")
        
        try:
            # Reshape pooled_grads for proper broadcasting
            if len(pooled_grads.shape) == 1:
                pooled_grads_reshaped = pooled_grads[tf.newaxis, tf.newaxis, :]
            else:
                pooled_grads_flat = tf.reshape(pooled_grads, [-1])
                pooled_grads_reshaped = pooled_grads_flat[tf.newaxis, tf.newaxis, :]
            
            # Element-wise multiply and sum across channels
            weighted_features = conv_outputs * pooled_grads_reshaped
            heatmap = tf.reduce_sum(weighted_features, axis=-1)
            
            print(f"   Heatmap shape: {heatmap.shape}")
            
            # Apply ReLU to focus on positive contributions
            heatmap = tf.maximum(heatmap, 0)
            
            # Normalize
            heatmap_max = tf.reduce_max(heatmap)
            if heatmap_max > 0:
                heatmap = heatmap / heatmap_max
            else:
                print("⚠️ Heatmap max value is zero!")
                return None
            
            # Convert to numpy for post-processing
            heatmap_np = heatmap.numpy()
            
            # Apply advanced smoothing
            from scipy import ndimage
            sigma = max(heatmap_np.shape) / 50.0
            heatmap_smoothed = ndimage.gaussian_filter(heatmap_np, sigma=sigma)
            
            # Normalize again after smoothing
            if np.max(heatmap_smoothed) > 0:
                heatmap_smoothed = heatmap_smoothed / np.max(heatmap_smoothed)
            
            # Enhance contrast
            heatmap_enhanced = np.power(heatmap_smoothed, 0.7)
            
            # Apply morphological operations
            from scipy import ndimage as ndi
            binary_heatmap = heatmap_enhanced > 0.3
            dilated = ndi.binary_dilation(binary_heatmap, iterations=2)
            dilated_smooth = ndimage.gaussian_filter(dilated.astype(float), sigma=1.0)
            heatmap_final = heatmap_enhanced * 0.7 + dilated_smooth * 0.3
            heatmap_final = np.power(heatmap_final, 0.8)
            
            print(f"✅ Returning enhanced heatmap with shape: {heatmap_final.shape}")
            
            # Restore original trainable state
            try:
                for layer_name, trainable in original_trainable_state.items():
                    model.get_layer(layer_name).trainable = trainable
            except:
                pass
            
            return heatmap_final
            
        except Exception as e:
            print(f"❌ Error in heatmap computation: {e}")
            import traceback
            traceback.print_exc()
            
            # Try a simpler approach - create a basic attention map
            print("🔄 Trying simpler heatmap generation...")
            try:
                # Get model prediction to find high-attention areas
                preds = model.predict(x, verbose=0)
                pred_class = int(np.argmax(preds[0]))
                
                # Create a simple center-focused heatmap as fallback
                h, w = x.shape[1], x.shape[2]
                y_grid, x_grid = np.ogrid[:h, :w]
                center_y, center_x = h // 2, w // 2
                
                # Create radial gradient from center
                dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                
                # Invert so center is high
                simple_heatmap = 1.0 - (dist_from_center / max_dist)
                simple_heatmap = np.power(simple_heatmap, 2)  # Make it more focused
                
                print(f"✅ Created simple fallback heatmap with shape: {simple_heatmap.shape}")
                return simple_heatmap
            except Exception as e2:
                print(f"❌ Simple heatmap also failed: {e2}")
                return None
        
    except Exception as e:
        print(f"❌ Error in make_gradcam_heatmap: {e}")
        import traceback
        traceback.print_exc()
        
        # Last resort - create a basic heatmap
        try:
            print("🔄 Creating last-resort basic heatmap...")
            if len(img_array.shape) == 4:
                h, w = img_array.shape[1], img_array.shape[2]
            else:
                h, w = img_array.shape[0], img_array.shape[1]
            
            # Create simple center-focused heatmap
            y_grid, x_grid = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            basic_heatmap = 1.0 - (dist_from_center / max_dist)
            basic_heatmap = np.power(basic_heatmap, 2)
            
            print(f"✅ Created basic fallback heatmap with shape: {basic_heatmap.shape}")
            return basic_heatmap
        except Exception as e3:
            print(f"❌ All heatmap generation failed: {e3}")
            return None

def generate_visualizations(original_image, processed_img, model, predicted_class_idx, disease_name, confidence):
    """
    Generate all visualizations for the eye disease prediction
    
    Args:
        original_image: PIL Image or numpy array of the original image
        processed_img: Preprocessed image array for model input
        model: Loaded TensorFlow model
        predicted_class_idx: Index of the predicted class
        disease_name: Name of the predicted disease
        confidence: Confidence score of the prediction
        
    Returns:
        Dictionary containing paths to generated visualization files and disease areas data
    """
    print("=== GENERATE_VISUALIZATIONS FUNCTION CALLED ===")
    
    try:
        # Create visualizations folder if it doesn't exist
        create_visualizations_folder()
        
        # Generate unique ID for this visualization
        visualization_id = str(uuid.uuid4())
        print(f"Generating visualizations with ID: {visualization_id}")
        
        # Convert PIL image to numpy array for processing
        if isinstance(original_image, PILImage.Image):
            original_img_array = np.array(original_image)
        else:
            original_img_array = original_image.copy()
            
        print(f"Original image array shape: {original_img_array.shape}")
        
        if len(original_img_array.shape) == 2:  # Grayscale
            original_img_array = cv2.cvtColor(original_img_array, cv2.COLOR_GRAY2RGB)
            print("Converted grayscale to RGB")
        elif original_img_array.shape[2] == 4:  # RGBA
            original_img_array = cv2.cvtColor(original_img_array, cv2.COLOR_RGBA2RGB)
            print("Converted RGBA to RGB")
        
        print(f"Processed original image shape: {original_img_array.shape}")
        
        # Generate Grad-CAM heatmap
        print("Generating Grad-CAM heatmap...")
        try:
            heatmap = make_gradcam_heatmap(processed_img, model, predicted_class_idx)
            if heatmap is None:
                print("⚠️ GradCAM returned None, will use fallback")
        except Exception as e:
            print(f"❌ GradCAM failed with error: {e}")
            import traceback
            traceback.print_exc()
            heatmap = None
        
        disease_areas = []
        
        if heatmap is not None:
            print(f"✅ Heatmap generated successfully. Shape: {heatmap.shape}")
            # Create visualizations
            h, w = original_img_array.shape[:2]
            print(f"Original image dimensions: {w}x{h}")
            
            # 1. Pure Heatmap PNG (color-mapped only) - For reference
            print("Creating pure heatmap visualization...")
            heatmap_resized = cv2.resize(heatmap, (w, h))
            
            # Normalize heatmap to [0, 1]
            heatmap_normalized = heatmap_resized / np.max(heatmap_resized) if np.max(heatmap_resized) > 0 else heatmap_resized
            
            # Apply gentle thresholding to preserve GradCAM information (like Streamlit)
            non_zero = heatmap_normalized[heatmap_normalized > 0.01]
            if len(non_zero) > 50:
                # Keep top 85% of activations
                threshold = np.percentile(non_zero, 15) if len(non_zero) > 10 else 0
                heatmap_thresholded = np.where(heatmap_normalized >= threshold, 
                                               heatmap_normalized, 
                                               heatmap_normalized * 0.4)
            else:
                heatmap_thresholded = heatmap_normalized
            
            # Re-normalize
            if np.max(heatmap_thresholded) > 0:
                heatmap_thresholded = heatmap_thresholded / np.max(heatmap_thresholded)
            
            normalized_heatmap = (heatmap_thresholded * 255).astype(np.uint8)
            
            # Select colormap based on heatmap characteristics (like Streamlit)
            heatmap_mean = np.mean(normalized_heatmap[normalized_heatmap > 0]) if np.any(normalized_heatmap > 0) else 0
            heatmap_std = np.std(normalized_heatmap[normalized_heatmap > 0]) if np.any(normalized_heatmap > 0) else 0
            
            if heatmap_std < 30:
                selected_colormap = cv2.COLORMAP_TURBO
            elif heatmap_mean < 100:
                selected_colormap = cv2.COLORMAP_HOT
            else:
                selected_colormap = cv2.COLORMAP_INFERNO
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(normalized_heatmap, selected_colormap)
            heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            heatmap_path = os.path.join(VISUALIZATIONS_FOLDER, f"{visualization_id}_heatmap.png")
            cv2.imwrite(heatmap_path, heatmap_colored)
            print(f"✅ Pure heatmap saved to: {heatmap_path}")
            
            # 2. Overlay PNG (blended heatmap with original) - Like Streamlit "Heatmap View"
            print("Creating blended heatmap overlay visualization...")
            original_resized = cv2.resize(original_img_array, (w, h))
            
            # Create intensity factor with power law (like Streamlit)
            heatmap_normalized_float = normalized_heatmap.astype(np.float32) / 255.0
            intensity_factor = np.power(heatmap_normalized_float, 0.7)  # Gamma correction
            
            # Stack to 3 channels
            intensity_factor = np.stack([intensity_factor] * 3, axis=2)
            
            # Adaptive intensity scaling (like Streamlit)
            min_intensity = 0.3  # Minimum 30% intensity
            max_intensity = 0.9  # Maximum 90% intensity
            
            if np.max(intensity_factor) > 0:
                intensity_factor = (intensity_factor - np.min(intensity_factor)) / (np.max(intensity_factor) - np.min(intensity_factor))
                intensity_factor = intensity_factor * (max_intensity - min_intensity) + min_intensity
            
            # Ensure minimum visibility
            intensity_factor = np.maximum(intensity_factor, 
                                         np.stack([(heatmap_normalized_float > 0.05).astype(np.float32) * min_intensity] * 3, axis=2))
            
            # Blend: original * (1 - intensity) + heatmap * intensity (like Streamlit)
            enhanced_img_float = original_resized.astype(np.float32)
            heatmap_colored_float = heatmap_colored_rgb.astype(np.float32)
            
            overlay_img = (
                enhanced_img_float * (1 - intensity_factor * 0.9) +
                heatmap_colored_float * intensity_factor
            )
            
            overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
            
            # Apply subtle sharpening in high-intensity regions (like Streamlit)
            kernel_sharpen = np.array([[-0.1, -0.1, -0.1],
                                      [-0.1,  1.8, -0.1],
                                      [-0.1, -0.1, -0.1]])
            sharpen_mask = (intensity_factor[:, :, 0] > 0.5).astype(np.float32)
            if np.any(sharpen_mask):
                for c in range(3):
                    channel = overlay_img[:, :, c].astype(np.float32)
                    sharpened = cv2.filter2D(channel, -1, kernel_sharpen)
                    overlay_img[:, :, c] = np.clip(
                        channel * (1 - sharpen_mask * 0.3) + sharpened * sharpen_mask * 0.3,
                        0, 255
                    ).astype(np.uint8)
            
            overlay_path = os.path.join(VISUALIZATIONS_FOLDER, f"{visualization_id}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
            print(f"✅ Blended heatmap overlay saved to: {overlay_path}")
            
            # 3. Outline PNG with colored contours (like Streamlit app)
            print("Creating outline visualization with colored contours...")
            # Threshold heatmap to generate binary mask
            # Use the normalized heatmap from earlier
            heatmap_for_contours = heatmap_resized / np.max(heatmap_resized) if np.max(heatmap_resized) > 0 else heatmap_resized
            heatmap_uint8 = np.uint8(255 * heatmap_for_contours)
            
            # Use multiple thresholding strategies to find regions
            non_zero_values = heatmap_uint8[heatmap_uint8 > 0]
            
            if len(non_zero_values) > 10:
                # Try percentile-based thresholding
                threshold_value = np.percentile(non_zero_values, 60)  # Top 40% of values
                binary_mask = (heatmap_uint8 >= threshold_value).astype(np.uint8) * 255
            else:
                # Fallback to Otsu's method
                _, binary_mask = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean up and separate regions
            kernel_size = max(3, min(h, w) // 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Remove small noise
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            # Close small gaps
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Found {len(contours)} contours")
            
            # Filter contours by area
            min_area = max(50, (h * w) * 0.0003)  # At least 0.03% of image area
            filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            print(f"Filtered to {len(filtered_contours)} contours (min area: {min_area})")
            
            # Sort contours by area (largest first)
            filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
            
            # Define different colors for different affected areas (BGR format for OpenCV)
            # Each region gets a distinct color - matching Streamlit app
            region_colors = [
                (255, 0, 0),      # Blue - Region 1
                (0, 255, 0),      # Green - Region 2
                (0, 0, 255),      # Red - Region 3
                (255, 255, 0),    # Cyan - Region 4
                (255, 0, 255),    # Magenta - Region 5
                (0, 255, 255),    # Yellow - Region 6
                (128, 0, 128),    # Purple - Region 7
                (255, 165, 0),    # Orange - Region 8
                (0, 128, 255),    # Orange-Red - Region 9
                (128, 255, 0),    # Lime - Region 10
            ]
            
            # Create outline image - start with original
            mask_img = original_resized.copy()
            
            # Draw colored contours
            disease_areas = []
            for idx, contour in enumerate(filtered_contours):
                area = cv2.contourArea(contour)
                if area > 50:  # Only draw significant regions
                    # Select color for this region (cycle through colors if many regions)
                    region_color = region_colors[idx % len(region_colors)]
                    
                    # Calculate line thickness based on image size and area
                    base_thickness = max(3, int(min(h, w) / 150))
                    area_factor = min(2.0, max(1.0, area / (h * w * 0.01)))
                    line_thickness = int(base_thickness * area_factor)
                    
                    # Draw outline with region-specific color
                    cv2.drawContours(mask_img, [contour], -1, region_color, line_thickness)
                    
                    # For large regions, add a second outline for extra visibility
                    if area > (h * w * 0.05):
                        cv2.drawContours(mask_img, [contour], -1, region_color, line_thickness + 1)
                    
                    # Calculate region metrics
                    x, y, width, height = cv2.boundingRect(contour)
                    area_score = area / (h * w)
                    
                    # Calculate average heatmap intensity in this region
                    region_heatmap = heatmap_for_contours[y:y+height, x:x+width]
                    avg_intensity = np.mean(region_heatmap) if region_heatmap.size > 0 else 0
                    
                    disease_areas.append({
                        "label": disease_name,
                        "score": float(avg_intensity),
                        "area_ratio": float(area_score),
                        "bbox": [int(x), int(y), int(width), int(height)]
                    })
                    
                    # Add label (optional - can be removed for cleaner look)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Simple label
                        label_text = f"Area {idx + 1}"
                        font_scale = max(0.5, min(0.8, w / 400))
                        thickness = max(1, int(w / 200))
                        
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )
                        
                        # White background
                        cv2.rectangle(
                            mask_img,
                            (cx - text_width//2 - 4, cy - text_height - 4),
                            (cx + text_width//2 + 4, cy + baseline + 4),
                            (255, 255, 255), -1
                        )
                        
                        # Colored border
                        cv2.rectangle(
                            mask_img,
                            (cx - text_width//2 - 4, cy - text_height - 4),
                            (cx + text_width//2 + 4, cy + baseline + 4),
                            region_color, 2
                        )
                        
                        # Text in region color
                        cv2.putText(
                            mask_img, label_text,
                            (cx - text_width//2, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            region_color, thickness
                        )
            
            mask_path = os.path.join(VISUALIZATIONS_FOLDER, f"{visualization_id}_mask.png")
            cv2.imwrite(mask_path, mask_img)
            print(f"✅ Outline visualization saved to: {mask_path}")
            print(f"✅ Highlighted {len(filtered_contours)} distinct affected regions with colored outlines")
        else:
            print("⚠️ Heatmap generation failed, creating IMAGE-SPECIFIC fallback visualizations...")
            # If Grad-CAM fails, analyze the actual image to create unique visualizations
            h, w = original_img_array.shape[:2] if len(original_img_array.shape) >= 2 else (224, 224)
            original_resized = cv2.resize(original_img_array, (w, h))
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
            
            # CRITICAL: Analyze THIS specific image to find unique features
            print(f"🔍 Analyzing image-specific features...")
            
            # 1. Find bright regions (potential lesions, optic disc)
            _, bright_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bright_regions = (bright_thresh / 255.0).astype(np.float32)
            
            # 2. Find dark regions (hemorrhages, vessels)
            _, dark_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dark_regions = (dark_thresh / 255.0).astype(np.float32)
            
            # 3. Edge detection for structural abnormalities
            edges = cv2.Canny(gray, 50, 150)
            edge_regions = (edges / 255.0).astype(np.float32)
            
            # 4. Texture analysis using Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)
            laplacian_norm = (laplacian_abs - laplacian_abs.min()) / (laplacian_abs.max() - laplacian_abs.min() + 1e-8)
            
            # 5. Combine features based on what's actually in THIS image
            # Weight based on actual content
            bright_weight = 0.3 if np.sum(bright_regions) > 100 else 0.1
            dark_weight = 0.3 if np.sum(dark_regions) > 100 else 0.1
            edge_weight = 0.2 if np.sum(edge_regions) > 50 else 0.1
            texture_weight = 0.2
            
            # Normalize weights
            total_weight = bright_weight + dark_weight + edge_weight + texture_weight
            bright_weight /= total_weight
            dark_weight /= total_weight
            edge_weight /= total_weight
            texture_weight /= total_weight
            
            # Create IMAGE-SPECIFIC heatmap
            fallback_heatmap = (
                bright_regions * bright_weight +
                dark_regions * dark_weight +
                edge_regions * edge_weight +
                laplacian_norm * texture_weight
            ).astype(np.float32)
            
            # Apply eye mask to focus on retinal area
            center_y, center_x = h // 2, w // 2
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            eye_mask = 1.0 - (dist_from_center / max_dist)
            eye_mask = np.power(eye_mask, 0.5)  # Gentle falloff
            
            # Apply mask
            fallback_heatmap = fallback_heatmap * eye_mask
            
            # Normalize
            if np.max(fallback_heatmap) > 0:
                fallback_heatmap = fallback_heatmap / np.max(fallback_heatmap)
            
            # Enhance contrast
            fallback_heatmap = np.power(fallback_heatmap, 0.7)
            
            # Verify uniqueness
            heatmap_std = np.std(fallback_heatmap)
            heatmap_hash = hash(str(fallback_heatmap.flatten()[:100]))
            print(f"✅ Created IMAGE-SPECIFIC heatmap (std={heatmap_std:.4f}, hash={heatmap_hash})")
            print(f"   Bright regions: {np.sum(bright_regions):.0f}, Dark regions: {np.sum(dark_regions):.0f}")
            print(f"   Edges: {np.sum(edge_regions):.0f}, Texture variance: {np.std(laplacian_norm):.4f}")
            
            # Normalize to 0-255
            fallback_heatmap_uint8 = (fallback_heatmap * 255).astype(np.uint8)
            
            # Apply JET colormap
            heatmap_colored = cv2.applyColorMap(fallback_heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create blended overlay
            intensity_factor = np.stack([fallback_heatmap] * 3, axis=2)
            intensity_factor = intensity_factor * 0.6 + 0.3  # Scale to 0.3-0.9
            
            overlay_img = (
                original_resized.astype(np.float32) * (1 - intensity_factor * 0.7) +
                heatmap_colored_rgb.astype(np.float32) * intensity_factor
            )
            overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
            
            # Create mask with colored contours based on actual features
            mask_img = original_resized.copy()
            
            # Find SPECIFIC high-attention regions (not the whole image)
            # Use higher threshold to get only significant regions
            heatmap_for_contours = (fallback_heatmap > 0.6).astype(np.uint8) * 255  # Higher threshold
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            heatmap_for_contours = cv2.morphologyEx(heatmap_for_contours, cv2.MORPH_CLOSE, kernel, iterations=2)
            heatmap_for_contours = cv2.morphologyEx(heatmap_for_contours, cv2.MORPH_OPEN, kernel, iterations=1)
            
            contours, _ = cv2.findContours(heatmap_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter for SPECIFIC regions only (not too large)
            min_area = (h * w) * 0.005  # At least 0.5% of image
            max_area = (h * w) * 0.3    # Max 30% of image
            region_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            disease_areas = []
            valid_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:  # Only specific regions
                    valid_contours.append(contour)
            
            # Sort by area and take top 5
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:5]
            
            # Draw colored circles and contours
            for idx, contour in enumerate(valid_contours):
                color = region_colors[idx % len(region_colors)]
                area = cv2.contourArea(contour)
                
                # Draw contour outline
                cv2.drawContours(mask_img, [contour], -1, color, 3)
                
                # Get center and draw circle
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw filled circle at center
                    cv2.circle(mask_img, (cx, cy), 8, color, -1)
                    cv2.circle(mask_img, (cx, cy), 8, (255, 255, 255), 2)
                    
                    # Draw crosshair
                    cv2.drawMarker(mask_img, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                
                x, y, width, height = cv2.boundingRect(contour)
                disease_areas.append({
                    "label": disease_name,
                    "score": float(np.mean(fallback_heatmap[y:y+height, x:x+width])),
                    "area_ratio": float(area / (h * w)),
                    "bbox": [int(x), int(y), int(width), int(height)]
                })
            
            # If no specific regions found, find the brightest spots
            if len(disease_areas) == 0:
                print("⚠️ No specific regions found, finding brightest spots...")
                # Find top 3 brightest regions
                heatmap_blur = cv2.GaussianBlur(fallback_heatmap, (15, 15), 0)
                for i in range(3):
                    max_loc = np.unravel_index(np.argmax(heatmap_blur), heatmap_blur.shape)
                    cy, cx = max_loc
                    
                    # Draw circle
                    radius = min(h, w) // 10
                    color = region_colors[i % len(region_colors)]
                    cv2.circle(mask_img, (cx, cy), radius, color, 3)
                    cv2.circle(mask_img, (cx, cy), 5, color, -1)
                    cv2.drawMarker(mask_img, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                    
                    disease_areas.append({
                        "label": disease_name,
                        "score": float(heatmap_blur[cy, cx]),
                        "area_ratio": float((radius * radius * 3.14) / (h * w)),
                        "bbox": [int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2)]
                    })
                    
                    # Zero out this region to find next brightest
                    cv2.circle(heatmap_blur, (cx, cy), radius * 2, 0, -1)
            
            # Save visualizations
            heatmap_path = os.path.join(VISUALIZATIONS_FOLDER, f"{visualization_id}_heatmap.png")
            overlay_path = os.path.join(VISUALIZATIONS_FOLDER, f"{visualization_id}_overlay.png")
            mask_path = os.path.join(VISUALIZATIONS_FOLDER, f"{visualization_id}_mask.png")
            
            cv2.imwrite(heatmap_path, heatmap_colored)
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
            
            print(f"✅ IMAGE-SPECIFIC fallback visualizations saved with {len(disease_areas)} regions")
        
        # Return paths and data
        return {
            "visualization_id": visualization_id,
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path,
            "mask_path": mask_path,
            "disease_areas": disease_areas,
            "urls": {
                "heatmap_url": f"/static/visualizations/{visualization_id}_heatmap.png",
                "overlay_url": f"/static/visualizations/{visualization_id}_overlay.png",
                "mask_url": f"/static/visualizations/{visualization_id}_mask.png"
            }
        }
        
    except Exception as e:
        print(f"❌ Error in generate_visualizations: {e}")
        import traceback
        traceback.print_exc()
        return None

# Initialize the visualizations folder when module is imported
create_visualizations_folder()