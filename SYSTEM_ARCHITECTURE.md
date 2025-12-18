# System Architecture - Eye Disease Detection Application

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER (Browser)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    React Frontend (Port 3000)                     │  │
│  │                                                                    │  │
│  │  ├─ Dashboard.js          (Main container)                       │  │
│  │  ├─ PatientPanel.js       (Patient info management)              │  │
│  │  ├─ PredictionPanel.js    (Upload & prediction display)          │  │
│  │  ├─ ImageViewer.js        (Visualization tabs)                   │  │
│  │  └─ HistoryPanel.js       (Prediction history)                   │  │
│  │                                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│                                    │ HTTP/REST API                       │
│                                    ▼                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER (Backend)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                  Flask Backend (Port 5000)                        │  │
│  │                         app.py                                    │  │
│  │                                                                    │  │
│  │  API Endpoints:                                                   │  │
│  │  ├─ POST /api/predict-visualize  (Main prediction endpoint)      │  │
│  │  ├─ POST /api/predict             (Simple prediction)            │  │
│  │  ├─ GET  /api/patients            (Get all patients)             │  │
│  │  ├─ POST /api/patients            (Create/update patient)        │  │
│  │  ├─ GET  /api/predictions         (Get all predictions)          │  │
│  │  ├─ GET  /api/statistics          (Get statistics)               │  │
│  │  └─ GET  /api/reports/<filename>  (Download PDF reports)         │  │
│  │                                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│                    ┌───────────────┼───────────────┐                    │
│                    ▼               ▼               ▼                    │
│  ┌──────────────────────┐ ┌──────────────┐ ┌──────────────────────┐   │
│  │  ML Processing       │ │  Database    │ │  External Services   │   │
│  │                      │ │  Integration │ │                      │   │
│  │  ├─ Model Loading    │ │              │ │  ├─ OpenAI API      │   │
│  │  ├─ Preprocessing    │ │  database.py │ │  │   (AI Recommend) │   │
│  │  ├─ Prediction       │ │  db_integ.py │ │  │                  │   │
│  │  ├─ GradCAM          │ │              │ │  └─ ai_recommend.py │   │
│  │  ├─ Visualization    │ │              │ │                      │   │
│  │  └─ PDF Generation   │ │              │ │                      │   │
│  │                      │ │              │ │                      │   │
│  │  visualization_      │ │              │ │                      │   │
│  │  utils.py            │ │              │ │                      │   │
│  │  pdf_generator.py    │ │              │ │                      │   │
│  └──────────────────────┘ └──────────────┘ └──────────────────────┘   │
│                                    │                                     │
└────────────────────────────────────┼─────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────┐ │
│  │  PostgreSQL Database │  │  File Storage        │  │  ML Model    │ │
│  │  (Render/Railway)    │  │                      │  │              │ │
│  │                      │  │  ├─ Uploaded Images  │  │  my_model    │ │
│  │  Tables:             │  │  ├─ Visualizations   │  │  trained3.h5 │ │
│  │  ├─ patients         │  │  └─ PDF Reports      │  │              │ │
│  │  ├─ predictions      │  │                      │  │  (TensorFlow │ │
│  │  └─ statistics       │  │  backend/static/     │  │   Keras)     │ │
│  │                      │  │  backend/uploads/    │  │              │ │
│  └──────────────────────┘  └──────────────────────┘  └──────────────┘ │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow - Prediction Process

```
┌─────────────┐
│   User      │
│  Uploads    │
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Frontend (PredictionPanel.js)                                 │
│    - Validates image                                             │
│    - Collects patient data                                       │
│    - Sends POST request to /api/predict-visualize               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Backend (app.py)                                              │
│    - Receives image and patient data                            │
│    - Saves/updates patient in database                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Image Preprocessing (preprocess_image)                       │
│    - Resize to 128x128                                          │
│    - Convert to RGB                                             │
│    - Denoise if needed                                          │
│    - Normalize (EfficientNet: [0,1] or MobileNetV2: [-1,1])    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. ML Model Prediction                                           │
│    - Load model (my_modeltrained3.h5)                           │
│    - Run inference                                              │
│    - Get probabilities for 8 disease classes                    │
│    - Apply softmax if needed                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Visualization Generation (visualization_utils.py)            │
│    - Generate GradCAM heatmap                                   │
│    - Create overlay visualization                               │
│    - Detect affected areas with contours                        │
│    - Save visualizations to static/visualizations/              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. AI Recommendations (ai_recommendations.py)                   │
│    - Call OpenAI API with disease info                          │
│    - Generate personalized recommendations                      │
│    - Fallback to rule-based if API fails                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. PDF Report Generation (pdf_generator.py)                     │
│    - Create professional PDF report                             │
│    - Include patient info, predictions, visualizations          │
│    - Add AI recommendations                                     │
│    - Save to backend directory                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Database Storage (db_integration.py)                         │
│    - Save prediction to predictions table                       │
│    - Link to patient record                                     │
│    - Update statistics cache                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. Response to Frontend                                          │
│    - Return prediction results                                  │
│    - Include visualization URLs                                 │
│    - Include PDF report filename                                │
│    - Include AI recommendations                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. Frontend Display (ImageViewer.js)                           │
│     - Show prediction with confidence                           │
│     - Display visualizations in tabs                            │
│     - Show AI recommendations                                   │
│     - Provide PDF download link                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Frontend
- **Framework**: React 18
- **Styling**: CSS3 with custom styles
- **HTTP Client**: Fetch API
- **State Management**: React Hooks (useState, useEffect)

### Backend
- **Framework**: Flask 3.0
- **ML Framework**: TensorFlow 2.15 / Keras
- **Image Processing**: OpenCV, PIL
- **PDF Generation**: ReportLab
- **Database ORM**: SQLAlchemy
- **CORS**: Flask-CORS

### Database
- **Type**: PostgreSQL
- **Hosting**: Render.com or Railway.app
- **Tables**: patients, predictions, statistics

### External Services
- **AI Recommendations**: OpenAI GPT-4 API
- **Deployment**: Render.com / Railway.app

### ML Model
- **Architecture**: Custom CNN (128x128x3 input, 8 classes output)
- **Format**: Keras H5
- **Classes**: 
  1. Choroidal Neovascularization (CNV)
  2. Diabetic Macular Edema (DME)
  3. Drusen
  4. Normal
  5. Cataract
  6. Diabetic Retinopathy
  7. Glaucoma
  8. Normal-1

## Database Schema

```
┌─────────────────────────────────────────────────────────────┐
│                      patients                                │
├─────────────────────────────────────────────────────────────┤
│ id                 INTEGER PRIMARY KEY                       │
│ patient_id         VARCHAR(50) UNIQUE NOT NULL               │
│ name               VARCHAR(200) NOT NULL                     │
│ age                INTEGER                                   │
│ dob                VARCHAR(50)                               │
│ gender             VARCHAR(20)                               │
│ contact            VARCHAR(100)                              │
│ email              VARCHAR(200)                              │
│ address            TEXT                                      │
│ medical_history    TEXT                                      │
│ medications        TEXT                                      │
│ created_at         TIMESTAMP                                 │
│ updated_at         TIMESTAMP                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    predictions                               │
├─────────────────────────────────────────────────────────────┤
│ id                 INTEGER PRIMARY KEY                       │
│ prediction_id      VARCHAR(50) UNIQUE NOT NULL               │
│ patient_id         VARCHAR(50) FOREIGN KEY → patients        │
│ disease            VARCHAR(200) NOT NULL                     │
│ confidence         FLOAT NOT NULL                            │
│ predictions_json   TEXT (JSON)                               │
│ image_path         VARCHAR(500)                              │
│ heatmap_path       VARCHAR(500)                              │
│ overlay_path       VARCHAR(500)                              │
│ mask_path          VARCHAR(500)                              │
│ report_filename    VARCHAR(500)                              │
│ daily_tips         TEXT (JSON array)                         │
│ medical_advice     TEXT                                      │
│ lifestyle_changes  TEXT (JSON array)                         │
│ warning_signs      TEXT (JSON array)                         │
│ created_at         TIMESTAMP                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    statistics                                │
├─────────────────────────────────────────────────────────────┤
│ id                    INTEGER PRIMARY KEY                    │
│ total_predictions     INTEGER                                │
│ average_confidence    FLOAT                                  │
│ disease_distribution  TEXT (JSON)                            │
│ last_updated          TIMESTAMP                              │
└─────────────────────────────────────────────────────────────┘
```

## Security Features

- **CORS**: Configured for cross-origin requests
- **Input Validation**: File type and size validation
- **Environment Variables**: Sensitive data in .env files
- **Database**: Connection pooling and timeout handling
- **API Keys**: OpenAI API key stored securely

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Environment                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Frontend        │         │  Backend         │          │
│  │  (Static Host)   │◄────────┤  (Railway/Render)│          │
│  │                  │  CORS   │                  │          │
│  │  - React Build   │         │  - Flask App     │          │
│  │  - Nginx/CDN     │         │  - Gunicorn      │          │
│  └──────────────────┘         │  - 1GB RAM       │          │
│                                └────────┬─────────┘          │
│                                         │                    │
│                                         ▼                    │
│                                ┌──────────────────┐          │
│                                │  PostgreSQL DB   │          │
│                                │  (Managed)       │          │
│                                └──────────────────┘          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Performance Considerations

- **Model Loading**: Model loaded once at startup
- **Image Processing**: Optimized with OpenCV
- **Database**: Connection pooling, indexed queries
- **Caching**: Statistics cached in database
- **File Storage**: Static files served efficiently
- **Memory**: 1GB RAM recommended for ML model

## Future Enhancements

- [ ] Real-time prediction streaming
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Advanced analytics dashboard
- [ ] Model versioning and A/B testing
- [ ] Batch prediction processing
- [ ] Integration with hospital systems (HL7/FHIR)
