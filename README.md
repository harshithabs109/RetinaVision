# RetinaVision: Training a CNN for Eye Disease Prediction

## Abstract
Retinal diseases such as Diabetic Retinopathy, Glaucoma, Cataract, Diabetic Macular Edema (DME), Drusen, and Choroidal Neovascularization (CNV) are major causes of vision impairment worldwide. Early and accurate diagnosis is crucial to prevent irreversible vision loss. In clinical environments, retinal images are captured using specialized imaging equipment such as fundus cameras and Optical Coherence Tomography (OCT) devices.

This project, RetinaVision, presents a deep learning–based system that uses Convolutional Neural Networks (CNNs) to automatically analyze retinal images acquired using standard clinical imaging hardware. The system classifies retinal images into multiple disease categories, including Diabetic Retinopathy, Glaucoma, Cataract, DME, Drusen, CNV, and Normal retina. The CNN model learns disease-specific patterns such as lesions, retinal swelling, and abnormal vascular structures from labeled datasets.

To enhance clinical reliability and transparency, the system integrates Grad-CAM–based explainable AI, which highlights the important regions of the retina that influence the model’s predictions. The trained model is deployed through a web-based interface that allows healthcare professionals to upload retinal images and receive instant predictions along with confidence scores and visual explanations. RetinaVision acts as a clinical decision-support tool, complementing existing hospital imaging infrastructure.

---

## Problem Statement
Manual diagnosis of retinal diseases is time-consuming, requires expert ophthalmologists, and is dependent on specialized clinical infrastructure. Early screening becomes difficult in resource-constrained environments. There is a need for an AI-based system that can assist ophthalmologists by providing fast, accurate, and explainable disease predictions using retinal images.

---

## Objectives
- To develop a CNN-based system for multi-class retinal disease classification  
- To support early detection of eye diseases using medical images  
- To provide explainable AI outputs using Grad-CAM  
- To design a web-based interface for easy clinical usage  
- To complement existing hospital imaging systems  

---

## Diseases Classified
- Diabetic Retinopathy  
- Glaucoma  
- Cataract  
- Diabetic Macular Edema (DME)  
- Drusen  
- Choroidal Neovascularization (CNV)  
- Normal Retina  

---

## System Architecture
- Frontend: React.js  
- Backend: Flask (Python)  
- Model: Convolutional Neural Network (CNN)  
- Explainability: Grad-CAM  
- Database: PostgreSQL (optional)  

---

## Steps to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/retinavision.git
cd retinavision

### Step 2: Backend Setup

cd backend
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

Place the trained model file in the backend folder:
mymodeltrained3.h5

Run the backend server:
python app.py

Backend URL:
http://localhost:5000

### Step 3: Frontend Setup

cd frontend
npm install
npm start

Frontend URL:
http://localhost:3000

## How to Use
- Upload a retinal fundus or OCT image
- Click on Analyze
- View predicted disease and confidence score
- Analyze Grad-CAM heatmap visualization
- Download diagnostic report (optional)

---

## Project Screenshots
Screenshots of the application are available in the `screenshots/` directory:
- Dashboard
- Image Upload
- Prediction Output
- Grad-CAM Heatmap

---

## Demo Video
Project demonstration video link:
https://youtu.be/bZd-QWnk-9E?si=r-ukzwQQLGunB_Fj


---

## Technologies Used

### Frontend
- React.js
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask
- Flask-CORS

### Machine Learning
- TensorFlow
- Keras
- Convolutional Neural Networks (CNN)

### Image Processing
- OpenCV
- Pillow (PIL)
- NumPy

### Database
- PostgreSQL
- SQLAlchemy

### Tools
- Git
- GitHub
- VS Code
- Node.js

---

## Model Performance
- Accuracy: ~92%

### Metrics Used
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Applications
- Clinical decision support
- Early eye disease screening
- Ophthalmology assistance systems
- Medical image analysis research

---

## Future Enhancements
- Disease severity grading
- Mobile application support
- Cloud deployment
- Multimodal fusion (Fundus + OCT)
- Doctor/Admin dashboard

---

## License
This project is developed for academic and educational purposes.

