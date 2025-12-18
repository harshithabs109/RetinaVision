#!/usr/bin/env python3
"""
Beautiful PDF Report Generator for Eye Disease Classification
Generates stunning professional medical reports with modern design
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor
from datetime import datetime
import os
import io
from PIL import Image as PILImage
import numpy as np

class EyeDiseaseReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup ultra-professional medical report styles"""
        # Ultra-professional medical color palette
        medical_navy = HexColor('#0c4a6e')     # Medical navy
        professional_blue = HexColor('#1e3a8a') # Professional blue
        clinical_blue = HexColor('#2563eb')     # Clinical blue
        medical_green = HexColor('#059669')     # Medical green
        diagnostic_amber = HexColor('#d97706')  # Diagnostic amber
        alert_red = HexColor('#dc2626')         # Alert red
        pure_white = HexColor('#ffffff')        # Pure white
        medical_gray = HexColor('#374151')      # Medical gray
        secondary_gray = HexColor('#6b7280')    # Secondary gray
        border_gray = HexColor('#d1d5db')       # Border gray
        accent_teal = HexColor('#0d9488')       # Accent teal
        
        # Ultra-professional main title
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            spaceAfter=6,
            spaceBefore=3,
            alignment=TA_CENTER,
            textColor=medical_navy,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderPadding=0
        ))
        
        # Professional subtitle
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=8,
            alignment=TA_CENTER,
            textColor=secondary_gray,
            fontName='Helvetica'
        ))
        
        # Professional section headers
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=4,
            spaceBefore=8,
            textColor=medical_navy,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderPadding=0,
            leftIndent=0,
            rightIndent=0
        ))
        
        # Professional card header
        self.styles.add(ParagraphStyle(
            name='CardHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=3,
            spaceBefore=4,
            textColor=medical_navy,
            fontName='Helvetica-Bold'
        ))
        
        # Professional patient info
        self.styles.add(ParagraphStyle(
            name='PatientInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            leftIndent=12,
            textColor=medical_gray,
            fontName='Helvetica'
        ))
        
        # Professional prediction result
        self.styles.add(ParagraphStyle(
            name='PredictionResult',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=4,
            alignment=TA_CENTER,
            textColor=medical_green,
            fontName='Helvetica-Bold'
        ))
        
        # Professional confidence score
        self.styles.add(ParagraphStyle(
            name='ConfidenceScore',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=3,
            alignment=TA_CENTER,
            textColor=clinical_blue,
            fontName='Helvetica-Bold'
        ))
        
        # Professional recommendation
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=2,
            leftIndent=12,
            textColor=medical_gray,
            fontName='Helvetica'
        ))
        
        # Professional tip
        self.styles.add(ParagraphStyle(
            name='Tip',
            parent=self.styles['Normal'],
            fontSize=9,
            spaceAfter=3,
            leftIndent=16,
            textColor=secondary_gray,
            fontName='Helvetica'
        ))
        
        # Professional disclaimer
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            spaceAfter=5,
            textColor=secondary_gray,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Professional footer
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=7,
            spaceAfter=5,
            textColor=secondary_gray,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
    
    def create_header_drawing(self):
        """Create a professional medical header"""
        drawing = Drawing(600, 60)
        
        # Professional header background
        drawing.add(Rect(0, 0, 600, 60, fillColor=HexColor('#0c4a6e'), strokeColor=HexColor('#0c4a6e')))
        
        # Professional accent lines
        drawing.add(Line(50, 15, 550, 15, strokeColor=HexColor('#2563eb'), strokeWidth=2))
        drawing.add(Line(50, 45, 550, 45, strokeColor=HexColor('#2563eb'), strokeWidth=1))
        
        # Professional corner accents
        drawing.add(Rect(50, 20, 8, 8, fillColor=HexColor('#2563eb'), strokeColor=HexColor('#2563eb')))
        drawing.add(Rect(542, 20, 8, 8, fillColor=HexColor('#2563eb'), strokeColor=HexColor('#2563eb')))
        
        return drawing
    
    def create_patient_info_table(self, patient_data):
        """Create a beautiful modern table with patient information"""
        # Split into two columns for better layout
        left_data = [
            ['Patient ID', patient_data.get('patient_id', 'N/A')],
            ['Name', patient_data.get('name', 'N/A')],
            ['Age', patient_data.get('age', 'N/A')],
            ['Gender', patient_data.get('gender', 'N/A')],
            ['Date of Birth', patient_data.get('dob', 'N/A')]
        ]
        
        right_data = [
            ['Phone', patient_data.get('phone', 'N/A')],
            ['Email', patient_data.get('email', 'N/A')],
            ['Address', patient_data.get('address', 'N/A')],
            ['Medical History', patient_data.get('medical_history', 'None')],
            ['Medications', patient_data.get('medications', 'None')]
        ]
        
        # Create professional left table
        left_table = Table(left_data, colWidths=[1.5*inch, 2.5*inch])
        left_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#f7fafc')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#374151')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#0c4a6e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f7fafc')])
        ]))
        
        # Create professional right table
        right_table = Table(right_data, colWidths=[1.5*inch, 2.5*inch])
        right_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#f7fafc')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#374151')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#0c4a6e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f7fafc')])
        ]))
        
        # Combine tables side by side
        combined_table = Table([[left_table, right_table]], colWidths=[4*inch, 4*inch])
        combined_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0)
        ]))
        
        return combined_table
    
    def create_prediction_results_table(self, prediction_data):
        """Create a beautiful modern table with prediction results"""
        class_names = [
            'Cataract',
            'Choroidal Neovascularization',
            'Diabetic Macular Edema',
            'Diabetic Retinopathy',
            'Drusen',
            'Glaucoma',
            'Normal',
            'Normal-1'
        ]
        
        data = [['Disease Type', 'Confidence Score', 'Percentage']]
        
        # Handle case where predictions might be a dictionary or list
        if isinstance(prediction_data.get('predictions'), dict):
            # If predictions is a dictionary, use the keys as class names
            predictions = prediction_data['predictions']
            # Create a list with confidence values in the correct order
            predictions_list = []
            for class_name in class_names:
                confidence = predictions.get(class_name, 0.0)
                predictions_list.append(confidence)
        elif isinstance(prediction_data.get('predictions'), list):
            # If predictions is already a list, use it directly
            predictions_list = prediction_data['predictions']
        else:
            # Fallback to empty list
            predictions_list = [0.0] * len(class_names)
        
        # Add all classes with their confidence scores
        for i, class_name in enumerate(class_names):
            if i < len(predictions_list):
                confidence = float(predictions_list[i])
            else:
                confidence = 0.0
            percentage = f"{confidence*100:.2f}%"
            data.append([class_name, f"{confidence:.4f}", percentage])
        
        # Sort by confidence (highest first)
        data[1:] = sorted(data[1:], key=lambda x: float(x[1]), reverse=True)
        
        table = Table(data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            # Professional header styling
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#0c4a6e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            # Professional data row styling
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('LEFTPADDING', (0, 1), (-1, -1), 6),
            ('RIGHTPADDING', (0, 1), (-1, -1), 6),
            
            # Professional grid and borders
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
            
            # Professional highlight for top prediction
            ('BACKGROUND', (0, 1), (-1, 1), HexColor('#f0fff4')),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 1), (-1, 1), HexColor('#059669')),
            
            # Professional alternating row colors
            ('ROWBACKGROUNDS', (0, 2), (-1, -1), [HexColor('#ffffff'), HexColor('#f7fafc')])
        ]))
        
        return table
    
    def save_image_to_buffer(self, image_array, max_width=4*inch, max_height=4*inch, quality=95):
        """Convert numpy array to PIL Image and save to high-quality buffer"""
        # Import at function level to ensure availability
        from PIL import Image as PILImage
        import base64
        
        try:
            print(f"   🖼️  Processing image for PDF: shape={image_array.shape if hasattr(image_array, 'shape') else 'unknown'}")
            
            # Handle base64 string input
            if isinstance(image_array, str):
                # Decode base64 string
                img_data = base64.b64decode(image_array)
                img_buffer = io.BytesIO(img_data)
                pil_image = PILImage.open(img_buffer)
                # Convert to numpy array
                image_array = np.array(pil_image)
            
            # Ensure image_array is a numpy array
            if not isinstance(image_array, np.ndarray):
                raise ValueError("image_array must be a numpy array or base64 string")
            
            # Handle different image formats
            if len(image_array.shape) == 3:
                # RGB image - ensure proper data type and format
                if image_array.dtype != np.uint8:
                    if image_array.max() <= 1.0:
                        image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
                    else:
                        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                
                # Ensure RGB format (not BGR)
                if image_array.shape[2] == 3:
                    # Check if it's BGR (OpenCV format) - convert to RGB
                    # For now, assume it's already RGB from our processing
                    pil_image = PILImage.fromarray(image_array, mode='RGB')
                elif image_array.shape[2] == 4:
                    # RGBA - convert to RGB
                    pil_image = PILImage.fromarray(image_array, mode='RGBA')
                    pil_image = pil_image.convert('RGB')
                else:
                    pil_image = PILImage.fromarray(image_array, mode='RGB')
            elif len(image_array.shape) == 2:
                # Grayscale image
                if image_array.dtype != np.uint8:
                    if image_array.max() <= 1.0:
                        image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
                    else:
                        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                pil_image = PILImage.fromarray(image_array, mode='L')
                pil_image = pil_image.convert('RGB')
            else:
                raise ValueError(f"Unsupported image shape: {image_array.shape}")
            
            print(f"   ✅ PIL Image created: size={pil_image.size}, mode={pil_image.mode}")
            
            # Calculate aspect ratio and resize while maintaining it
            original_width, original_height = pil_image.size
            aspect_ratio = original_width / original_height
            
            # Convert max dimensions from reportlab units to pixels
            max_width_px = int(max_width * 72 / inch)  # Convert to pixels
            max_height_px = int(max_height * 72 / inch)
            
            if aspect_ratio > 1:
                # Landscape
                new_width = min(max_width_px, original_width)
                new_height = int(new_width / aspect_ratio)
            else:
                # Portrait
                new_height = min(max_height_px, original_height)
                new_width = int(new_height * aspect_ratio)
            
            # Resize with high quality
            pil_image = pil_image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
            print(f"   ✅ Image resized to: {pil_image.size}")
            
            # Save to buffer with high quality PNG
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG', optimize=False)
            img_buffer.seek(0)
            print(f"   ✅ Image saved to buffer: {len(img_buffer.getvalue())} bytes")
            
            return img_buffer
        except Exception as e:
            print(f"   ❌ Error saving image to buffer: {e}")
            import traceback
            traceback.print_exc()
            # Return a blank image buffer as fallback
            from PIL import Image as PILImage
            blank_image = PILImage.new('RGB', (200, 200), color='white')
            img_buffer = io.BytesIO()
            blank_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            return img_buffer
    
    def generate_report(self, patient_data, prediction_data, image_array=None, output_path="eye_disease_report.pdf", heatmap_array=None, additional_images=None):
        """Generate a stunning beautiful PDF report"""
        print("\n" + "="*60)
        print("📄 PDF GENERATION STARTED")
        print("="*60)
        print(f"Image array provided: {image_array is not None}")
        print(f"Heatmap array provided: {heatmap_array is not None}")
        print(f"Additional images provided: {additional_images is not None}")
        if additional_images:
            print(f"Additional images keys: {list(additional_images.keys())}")
        print("="*60 + "\n")
        
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                              rightMargin=50, leftMargin=50, 
                              topMargin=40, bottomMargin=40)
        story = []
        
        # Beautiful header with decorative elements
        header_drawing = self.create_header_drawing()
        story.append(header_drawing)
        story.append(Spacer(1, 4))
        
        # Professional title
        story.append(Paragraph("EYE DISEASE CLASSIFICATION REPORT", self.styles['MainTitle']))
        story.append(Paragraph("AI-Powered Medical Image Analysis", self.styles['Subtitle']))
        
        # Report metadata with modern styling
        current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"Generated on {current_time}", self.styles['Footer']))
        story.append(Spacer(1, 6))
        
        # Patient Information Section
        story.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))
        story.append(self.create_patient_info_table(patient_data))
        story.append(Spacer(1, 6))
        
        # Image Analysis Section with beautiful layout
        if image_array is not None:
            story.append(Paragraph("IMAGE ANALYSIS", self.styles['SectionHeader']))
            try:
                # Save original enhanced image with optimized size
                img_buffer = self.save_image_to_buffer(image_array, max_width=3.5*inch, max_height=3.5*inch)
                
                # Get image dimensions from buffer
                img_buffer.seek(0)
                temp_img = PILImage.open(img_buffer)
                img_width, img_height = temp_img.size
                aspect_ratio = img_width / img_height
                
                # Calculate display size maintaining aspect ratio - bigger for better visibility
                max_display_width = 3.5 * inch
                max_display_height = 3.5 * inch
                if aspect_ratio > 1:
                    display_width = max_display_width
                    display_height = max_display_width / aspect_ratio
                else:
                    display_height = max_display_height
                    display_width = max_display_height * aspect_ratio
                
                img_buffer.seek(0)
                try:
                    img = Image(img_buffer, width=display_width, height=display_height)
                    print(f"   ✅ Created ReportLab Image object for original image")
                except Exception as e:
                    print(f"   ❌ Failed to create ReportLab Image: {e}")
                    img = None
                
                if heatmap_array is not None:
                    # Use the enhanced PDF result for the main display (shows disease area more clearly)
                    enhanced_gradcam = additional_images.get('enhanced_pdf_result', additional_images.get('enhanced_result', heatmap_array)) if additional_images else heatmap_array
                    
                    # Save enhanced image with affected areas - optimized size
                    heatmap_buffer = self.save_image_to_buffer(enhanced_gradcam, max_width=3.5*inch, max_height=3.5*inch)
                    
                    # Get image dimensions from buffer
                    heatmap_buffer.seek(0)
                    temp_heatmap = PILImage.open(heatmap_buffer)
                    heatmap_width, heatmap_height = temp_heatmap.size
                    heatmap_aspect = heatmap_width / heatmap_height
                    
                    # Calculate display size maintaining aspect ratio
                    if heatmap_aspect > 1:
                        heatmap_display_width = max_display_width
                        heatmap_display_height = max_display_width / heatmap_aspect
                    else:
                        heatmap_display_height = max_display_height
                        heatmap_display_width = max_display_height * heatmap_aspect
                    
                    heatmap_buffer.seek(0)
                    try:
                        heatmap_img = Image(heatmap_buffer, width=heatmap_display_width, height=heatmap_display_height)
                        print(f"   ✅ Created ReportLab Image object for heatmap")
                    except Exception as e:
                        print(f"   ❌ Failed to create heatmap Image: {e}")
                        heatmap_img = None
                    
                    # Create side-by-side layout with bigger images
                    if img and heatmap_img:
                        image_table = Table([[img, heatmap_img]], colWidths=[3.5*inch, 3.5*inch])
                    elif img:
                        image_table = Table([[img]], colWidths=[7*inch])
                    elif heatmap_img:
                        image_table = Table([[heatmap_img]], colWidths=[7*inch])
                    else:
                        print("   ⚠️  No images to display in table")
                        image_table = None
                    
                    if image_table:
                        image_table.setStyle(TableStyle([
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 3),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                            ('TOPPADDING', (0, 0), (-1, -1), 3),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f8fafc')),
                            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e5e7eb'))
                        ]))
                        story.append(image_table)
                        print(f"   ✅ Added image table to PDF")
                        
                        # Image captions
                        if img and heatmap_img:
                            captions = Table([
                                [Paragraph('🔍 Original Image', self.styles['CardHeader']), 
                                 Paragraph('🎯 Affected Areas', self.styles['CardHeader'])]
                            ], colWidths=[3.5*inch, 3.5*inch])
                        elif img:
                            captions = Table([
                                [Paragraph('🔍 Original Image', self.styles['CardHeader'])]
                            ], colWidths=[7*inch])
                        else:
                            captions = Table([
                                [Paragraph('🎯 Affected Areas', self.styles['CardHeader'])]
                            ], colWidths=[7*inch])
                        
                        captions.setStyle(TableStyle([
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 3),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                            ('TOPPADDING', (0, 0), (-1, -1), 2),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 2)
                        ]))
                        story.append(captions)
                    else:
                        story.append(Paragraph("⚠️ Images could not be loaded", self.styles['PatientInfo']))
                    
                    # NO PAGE BREAK - Continue on same page
                    story.append(Spacer(1, 6))
                    
                    # Add detailed analysis section with ALL 3 TAB IMAGES
                    story.append(Paragraph("DETAILED VISUALIZATION ANALYSIS", self.styles['SectionHeader']))
                    
                    # Create a comprehensive analysis section showing all 3 tab images
                    if additional_images:
                        print(f"📊 Processing additional images for PDF...")
                        print(f"   Available keys: {list(additional_images.keys())}")
                        # Display all 3 visualization modes from the dashboard tabs
                        
                        # TAB 1: Pure Heatmap (if available)
                        if 'pure_heatmap' in additional_images:
                            print(f"   ✅ Adding pure_heatmap to PDF")
                            story.append(Paragraph("📷 Tab 1: AI Heatmap", self.styles['CardHeader']))
                            heatmap_buffer = self.save_image_to_buffer(additional_images['pure_heatmap'], max_width=3.5*inch, max_height=3.5*inch)
                            heatmap_buffer.seek(0)
                            temp_heatmap = PILImage.open(heatmap_buffer)
                            hm_width, hm_height = temp_heatmap.size
                            hm_aspect = hm_width / hm_height
                            if hm_aspect > 1:
                                hm_display_width = 3.5 * inch
                                hm_display_height = (3.5 * inch) / hm_aspect
                            else:
                                hm_display_height = 3.5 * inch
                                hm_display_width = (3.5 * inch) * hm_aspect
                            heatmap_buffer.seek(0)
                            heatmap_img = Image(heatmap_buffer, width=hm_display_width, height=hm_display_height)
                            story.append(Spacer(1, 2))
                            story.append(heatmap_img)
                            story.append(Spacer(1, 2))
                            explanation1 = Paragraph(
                                "AI attention heatmap - Red/yellow areas show disease patterns.",
                                self.styles['PatientInfo']
                            )
                            story.append(explanation1)
                        
                        # TAB 2: Heatmap Overlay (blended with original)
                        if 'heatmap_overlay' in additional_images:
                            print(f"   ✅ Adding heatmap_overlay to PDF")
                            story.append(Spacer(1, 4))
                            story.append(Paragraph("🔥 Tab 2: Heatmap Overlay", self.styles['CardHeader']))
                            overlay_buffer = self.save_image_to_buffer(additional_images['heatmap_overlay'], max_width=3.5*inch, max_height=3.5*inch)
                            overlay_buffer.seek(0)
                            temp_overlay = PILImage.open(overlay_buffer)
                            ov_width, ov_height = temp_overlay.size
                            ov_aspect = ov_width / ov_height
                            if ov_aspect > 1:
                                ov_display_width = 3.5 * inch
                                ov_display_height = (3.5 * inch) / ov_aspect
                            else:
                                ov_display_height = 3.5 * inch
                                ov_display_width = (3.5 * inch) * ov_aspect
                            overlay_buffer.seek(0)
                            overlay_img = Image(overlay_buffer, width=ov_display_width, height=ov_display_height)
                            story.append(Spacer(1, 2))
                            story.append(overlay_img)
                            story.append(Spacer(1, 2))
                            explanation2 = Paragraph(
                                "Blended heatmap overlay - Shows disease patterns in anatomical context.",
                                self.styles['PatientInfo']
                            )
                            story.append(explanation2)
                        
                        # TAB 3: Affected Areas with Colored Contours (most important for patients)
                        if 'affected_areas' in additional_images or 'enhanced_pdf_result' in additional_images:
                            print(f"   ✅ Adding affected_areas to PDF")
                            affected_img = additional_images.get('affected_areas', additional_images.get('enhanced_pdf_result', additional_images.get('enhanced_result', heatmap_array)))
                            story.append(Spacer(1, 4))
                            story.append(Paragraph("🎯 Tab 3: Affected Areas", self.styles['CardHeader']))
                            affected_buffer = self.save_image_to_buffer(affected_img, max_width=3.5*inch, max_height=3.5*inch)
                            affected_buffer.seek(0)
                            temp_affected = PILImage.open(affected_buffer)
                            af_width, af_height = temp_affected.size
                            af_aspect = af_width / af_height
                            if af_aspect > 1:
                                af_display_width = 3.5 * inch
                                af_display_height = (3.5 * inch) / af_aspect
                            else:
                                af_display_height = 3.5 * inch
                                af_display_width = (3.5 * inch) * af_aspect
                            affected_buffer.seek(0)
                            affected_img_obj = Image(affected_buffer, width=af_display_width, height=af_display_height)
                            story.append(Spacer(1, 2))
                            story.append(affected_img_obj)
                            story.append(Spacer(1, 2))
                            explanation3 = Paragraph(
                                "Colored contours mark specific disease-affected regions for easy identification.",
                                self.styles['PatientInfo']
                            )
                            story.append(explanation3)
                        
                        story.append(Spacer(1, 4))
                    else:
                        # Fallback: show the main heatmap result prominently
                        story.append(Paragraph("🎯 AI Model Focus - Affected Area Detection", self.styles['CardHeader']))
                        story.append(Spacer(1, 3))
                        if heatmap_array is not None:
                            heatmap_buffer = self.save_image_to_buffer(heatmap_array, max_width=5*inch, max_height=5*inch)
                            heatmap_buffer.seek(0)
                            temp_hm = PILImage.open(heatmap_buffer)
                            hm_w, hm_h = temp_hm.size
                            hm_asp = hm_w / hm_h
                            if hm_asp > 1:
                                hm_dw = 5 * inch
                                hm_dh = (5 * inch) / hm_asp
                            else:
                                hm_dh = 5 * inch
                                hm_dw = (5 * inch) * hm_asp
                            heatmap_buffer.seek(0)
                            heatmap_img_obj = Image(heatmap_buffer, width=hm_dw, height=hm_dh)
                            story.append(heatmap_img_obj)
                        story.append(Spacer(1, 4))
                else:
                    story.append(img)
                    story.append(Paragraph('🔍 Original Fundus Image', self.styles['CardHeader']))
                
                story.append(Spacer(1, 4))
            except Exception as e:
                story.append(Paragraph(f"⚠️ Error processing images: {str(e)}", self.styles['PatientInfo']))
                story.append(Spacer(1, 4))
        
        # Primary Diagnosis with emphasis
        top_prediction = prediction_data['top_prediction']
        top_confidence = prediction_data['top_confidence']
        
        # Check if eye is healthy (Normal or Normal-1)
        is_healthy = top_prediction in ['Normal', 'Normal-1']
        
        # Enhanced diagnosis with GradCAM metrics if available
        if additional_images and 'gradcam_metrics' in additional_images and additional_images['gradcam_metrics'] is not None:
            gradcam_metrics = additional_images['gradcam_metrics']
            diagnosis_data = [
                ['Primary Diagnosis', top_prediction],
                ['Confidence Level', f"{top_confidence:.1%}"],
                ['Affected Area', f"{gradcam_metrics.get('area_percentage', 0):.2f}% of image"],
                ['Severity Score', f"{gradcam_metrics.get('severity_score', 0):.1f}"],
                ['Intensity Level', f"{gradcam_metrics.get('mean_intensity', 0):.3f}"]
            ]
        else:
            diagnosis_data = [
                ['Primary Diagnosis', top_prediction],
                ['Confidence Level', f"{top_confidence:.1%}"]
            ]
        diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
        diagnosis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#f8fafc')),
            ('BACKGROUND', (1, 0), (1, 0), HexColor('#dcfce7')),
            ('BACKGROUND', (1, 1), (1, 1), HexColor('#dbeafe')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#374151')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e5e7eb'))
        ]))
        story.append(diagnosis_table)
        story.append(Spacer(1, 4))
        
        # Enhanced GradCAM Analysis Section (only show for diseases, not for healthy eyes)
        if not is_healthy and additional_images and 'gradcam_metrics' in additional_images and additional_images['gradcam_metrics'] is not None:
            gradcam_metrics = additional_images['gradcam_metrics']
            story.append(Paragraph("GRADCAM ANALYSIS", self.styles['SectionHeader']))
            
            # Create detailed GradCAM analysis table with comprehensive metrics
            gradcam_data = [
                ['Analysis Metric', 'Value', 'Interpretation'],
                ['Affected Area Size', f"{gradcam_metrics.get('area_pixels', 0):.0f} pixels", f"{gradcam_metrics.get('area_percentage', 0):.2f}% of image"],
                ['Number of Regions', f"{gradcam_metrics.get('num_regions', 0)}", 'Distinct disease-affected areas detected'],
                ['Largest Region Area', f"{gradcam_metrics.get('largest_region_area', 0):.0f} pixels", 'Size of largest affected region'],
                ['Mean Intensity', f"{gradcam_metrics.get('mean_intensity', 0):.4f}", 'Average heatmap intensity in affected region'],
                ['Peak Intensity', f"{gradcam_metrics.get('max_intensity', 0):.4f}", 'Highest heatmap intensity detected'],
                ['Min Intensity', f"{gradcam_metrics.get('min_intensity', 0):.4f}", 'Lowest heatmap intensity in affected areas'],
                ['Severity Score', f"{gradcam_metrics.get('severity_score', 0):.2f}", 'Combined severity assessment (0-100)'],
                ['Focus Score', f"{gradcam_metrics.get('focus_score', 0):.2f}%", 'Model focus concentration (higher = more focused)'],
                ['Heatmap Variance', f"{gradcam_metrics.get('heatmap_variance', 0):.4f}", 'Variability in heatmap values']
            ]
            
            gradcam_table = Table(gradcam_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
            gradcam_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1e3a8a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#e5e7eb')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f8fafc')])
            ]))
            story.append(gradcam_table)
            story.append(Spacer(1, 3))
        
        # AI-Powered Recommendations (different titles for healthy vs diseased eyes)
        if is_healthy:
            story.append(Paragraph("👁️ EYE HEALTH MAINTENANCE TIPS", self.styles['SectionHeader']))
        else:
            story.append(Paragraph("🤖 AI-POWERED DAILY CARE RECOMMENDATIONS", self.styles['SectionHeader']))
        
        tips = prediction_data.get('daily_tips', [])
        if tips:
            for i, tip in enumerate(tips, 1):
                story.append(Paragraph(f"{i}. {tip}", self.styles['Recommendation']))
        else:
            # Fallback tips based on health status
            if is_healthy:
                fallback_tips = [
                    "Continue annual comprehensive eye examinations",
                    "Wear UV-protective sunglasses when outdoors",
                    "Follow the 20-20-20 rule for screen time",
                    "Maintain a balanced diet rich in eye-healthy nutrients"
                ]
            else:
                fallback_tips = [
                    "Maintain regular checkups and adhere to prescribed treatments",
                    "Protect eyes from UV exposure and reduce glare",
                    "Follow balanced diet and adequate hydration",
                    "Avoid smoking; limit alcohol; manage systemic risks"
                ]
            for i, tip in enumerate(fallback_tips, 1):
                story.append(Paragraph(f"{i}. {tip}", self.styles['Recommendation']))
        story.append(Spacer(1, 3))

        # Lifestyle Changes (if available from AI)
        if 'recommendations' in prediction_data and prediction_data['recommendations']:
            ai_recs = prediction_data['recommendations']
            if 'lifestyle_changes' in ai_recs and ai_recs['lifestyle_changes']:
                if is_healthy:
                    story.append(Paragraph("HEALTHY LIFESTYLE HABITS", self.styles['SectionHeader']))
                else:
                    story.append(Paragraph("LIFESTYLE MODIFICATIONS", self.styles['SectionHeader']))
                for i, change in enumerate(ai_recs['lifestyle_changes'], 1):
                    story.append(Paragraph(f"{i}. {change}", self.styles['Recommendation']))
                story.append(Spacer(1, 3))
            
            # Warning Signs (if available from AI)
            if 'warning_signs' in ai_recs and ai_recs['warning_signs']:
                if is_healthy:
                    story.append(Paragraph("⚠️ WHEN TO SEEK MEDICAL ATTENTION", self.styles['SectionHeader']))
                else:
                    story.append(Paragraph("⚠️ WARNING SIGNS TO WATCH FOR", self.styles['SectionHeader']))
                for i, sign in enumerate(ai_recs['warning_signs'], 1):
                    story.append(Paragraph(f"{i}. {sign}", self.styles['Recommendation']))
                story.append(Spacer(1, 3))
        
        # Medical Recommendations (different title for healthy eyes)
        if is_healthy:
            story.append(Paragraph("PREVENTIVE CARE RECOMMENDATIONS", self.styles['SectionHeader']))
        else:
            story.append(Paragraph("PROFESSIONAL MEDICAL ADVICE", self.styles['SectionHeader']))
        
        # Use AI-based recommendations if available
        if 'recommendations' in prediction_data and prediction_data['recommendations']:
            ai_recommendations = prediction_data['recommendations']
            if isinstance(ai_recommendations, dict):
                # New format with description and recommendations
                if 'description' in ai_recommendations:
                    story.append(Paragraph(ai_recommendations['description'], self.styles['PatientInfo']))
                    story.append(Spacer(1, 3))
                
                if 'recommendations' in ai_recommendations and isinstance(ai_recommendations['recommendations'], list):
                    for i, rec in enumerate(ai_recommendations['recommendations'], 1):
                        story.append(Paragraph(f"{i}. {rec}", self.styles['Recommendation']))
                else:
                    # Fallback to disease-based recommendations
                    recommendations = {
                        'Cataract': "Consult with an ophthalmologist for treatment options including surgery. Regular follow-ups are recommended.",
                        'Choroidal Neovascularization': "Immediate consultation with a retina specialist is recommended. Anti-VEGF treatments may be necessary.",
                        'Diabetic Macular Edema': "Urgent consultation with a retina specialist is recommended. Blood sugar control and possible anti-VEGF therapy.",
                        'Diabetic Retinopathy': "Immediate consultation with a retina specialist is recommended. Monitor blood sugar levels closely.",
                        'Drusen': "Regular monitoring by an ophthalmologist is recommended. This may indicate early age-related macular degeneration.",
                        'Glaucoma': "Urgent consultation with a glaucoma specialist is recommended. Regular eye pressure monitoring is essential.",
                        'Normal': "No signs of eye disease detected. Continue regular eye checkups and maintain good eye health practices.",
                        'Normal-1': "No signs of eye disease detected. Continue regular eye checkups and maintain good eye health practices."
                    }
                    recommendation = recommendations.get(top_prediction, "Please consult with a healthcare professional for proper evaluation.")
                    story.append(Paragraph(f"• {recommendation}", self.styles['Recommendation']))
            else:
                # Old format - just a string
                story.append(Paragraph(f"• {ai_recommendations}", self.styles['Recommendation']))
        else:
            # Fallback to disease-based recommendations
            recommendations = {
                'Cataract': "Consult with an ophthalmologist for treatment options including surgery. Regular follow-ups are recommended.",
                'Choroidal Neovascularization': "Immediate consultation with a retina specialist is recommended. Anti-VEGF treatments may be necessary.",
                'Diabetic Macular Edema': "Urgent consultation with a retina specialist is recommended. Blood sugar control and possible anti-VEGF therapy.",
                'Diabetic Retinopathy': "Immediate consultation with a retina specialist is recommended. Monitor blood sugar levels closely.",
                'Drusen': "Regular monitoring by an ophthalmologist is recommended. This may indicate early age-related macular degeneration.",
                'Glaucoma': "Urgent consultation with a glaucoma specialist is recommended. Regular eye pressure monitoring is essential.",
                'Normal': "No signs of eye disease detected. Continue regular eye checkups and maintain good eye health practices.",
                'Normal-1': "No signs of eye disease detected. Continue regular eye checkups and maintain good eye health practices."
            }
            recommendation = recommendations.get(top_prediction, "Please consult with a healthcare professional for proper evaluation.")
            story.append(Paragraph(f"• {recommendation}", self.styles['Recommendation']))
        
        # Additional Notes (customized based on health status)
        story.append(Paragraph("ADDITIONAL NOTES", self.styles['SectionHeader']))
        if is_healthy:
            notes = [
                "This AI screening shows no signs of eye disease - your retinal examination appears healthy",
                "Continue regular preventive eye care with annual comprehensive eye exams",
                "This report is a screening tool and does not replace professional eye examinations",
                "Report any sudden changes in vision to your eye care provider immediately"
            ]
        else:
            notes = [
                "This report is generated using advanced AI-based image analysis",
                "Results should be interpreted by qualified medical professionals",
                "Early detection and treatment are crucial for preserving vision",
                "Image quality affects classification accuracy"
            ]
        for note in notes:
            story.append(Paragraph(f"• {note}", self.styles['PatientInfo']))
        story.append(Spacer(1, 6))
        
        # Disclaimer section
        story.append(Paragraph("MEDICAL DISCLAIMER", self.styles['SectionHeader']))
        disclaimer_text = """
        This report is generated for educational and research purposes only. 
        The AI-based classification results should not be used as a substitute for professional medical diagnosis, 
        treatment, or advice. Always consult with qualified healthcare professionals for medical decisions. 
        The accuracy of the classification depends on image quality and other factors.
        """
        story.append(Paragraph(disclaimer_text, self.styles['Disclaimer']))
        story.append(Spacer(1, 4))
        
        # Professional footer
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#d1d5db')))
        story.append(Spacer(1, 3))
        story.append(Paragraph("Generated by AI Eye Disease Classification System", self.styles['Footer']))
        story.append(Paragraph("Powered by Deep Learning Technology", self.styles['Footer']))
        
        # Build PDF
        print(f"\n📄 Building PDF with {len(story)} elements...")
        doc.build(story)
        print(f"✅ PDF generated successfully: {output_path}\n")
        return output_path

def create_sample_report():
    """Create a sample report for testing"""
    generator = EyeDiseaseReportGenerator()
    
    # Sample patient data
    patient_data = {
        'patient_id': 'P001',
        'name': 'John Doe',
        'age': '45',
        'gender': 'Male',
        'dob': '1978-05-15',
        'phone': '+1-555-0123',
        'email': 'john.doe@email.com',
        'address': '123 Main St, City, State 12345',
        'medical_history': 'Diabetes Type 2, Hypertension',
        'medications': 'Metformin, Lisinopril'
    }
    
    # Sample prediction data
    prediction_data = {
        'predictions': [0.1, 0.05, 0.05, 0.7, 0.02, 0.05, 0.02, 0.01],  # Diabetic Retinopathy highest
        'top_prediction': 'Diabetic Retinopathy',
        'top_confidence': 0.7
    }
    
    # Generate report
    output_path = generator.generate_report(patient_data, prediction_data)
    print(f"Sample report generated: {output_path}")

if __name__ == "__main__":
    create_sample_report()
