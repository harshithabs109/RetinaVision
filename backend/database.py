"""
PostgreSQL Database Configuration and Models
"""
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize SQLAlchemy
db = SQLAlchemy()

def init_db(app):
    """Initialize database with Flask app"""
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    # If no DATABASE_URL is set, use SQLite for local development
    if not database_url:
        # Use SQLite database file in the backend directory
        db_path = os.path.join(os.path.dirname(__file__), 'retinavision.db')
        database_url = f'sqlite:///{db_path}'
        print(f"📁 Using SQLite database: {db_path}")
    
    # Fix for Render PostgreSQL URL (postgres:// -> postgresql://)
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Only set engine options for PostgreSQL (not SQLite)
    if database_url.startswith('postgresql://'):
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_size': 10,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'connect_args': {
                'connect_timeout': 10,
                'options': '-c timezone=utc'
            }
        }
    
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully")

# Patient Model
class Patient(db.Model):
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer)
    dob = db.Column(db.String(50))
    gender = db.Column(db.String(20))
    contact = db.Column(db.String(100))
    email = db.Column(db.String(200))
    address = db.Column(db.Text)
    medical_history = db.Column(db.Text)
    medications = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with predictions
    predictions = db.relationship('Prediction', backref='patient', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'name': self.name,
            'age': self.age,
            'dob': self.dob,
            'gender': self.gender,
            'contact': self.contact,
            'email': self.email,
            'address': self.address,
            'medical_history': self.medical_history,
            'medications': self.medications,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Prediction Model
class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    patient_id = db.Column(db.String(50), db.ForeignKey('patients.patient_id'), nullable=False, index=True)
    
    # Prediction details
    disease = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    predictions_json = db.Column(db.Text)  # JSON string of all class probabilities
    
    # Image paths
    image_path = db.Column(db.String(500))
    heatmap_path = db.Column(db.String(500))
    overlay_path = db.Column(db.String(500))
    mask_path = db.Column(db.String(500))
    
    # Report
    report_filename = db.Column(db.String(500))
    
    # AI Recommendations
    daily_tips = db.Column(db.Text)  # JSON array
    medical_advice = db.Column(db.Text)
    lifestyle_changes = db.Column(db.Text)  # JSON array
    warning_signs = db.Column(db.Text)  # JSON array
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        import json
        return {
            'id': self.id,
            'prediction_id': self.prediction_id,
            'patient_id': self.patient_id,
            'disease': self.disease,
            'confidence': self.confidence,
            'predictions': json.loads(self.predictions_json) if self.predictions_json else {},
            'image_path': self.image_path,
            'heatmap_path': self.heatmap_path,
            'overlay_path': self.overlay_path,
            'mask_path': self.mask_path,
            'report_filename': self.report_filename,
            'daily_tips': json.loads(self.daily_tips) if self.daily_tips else [],
            'medical_advice': self.medical_advice,
            'lifestyle_changes': json.loads(self.lifestyle_changes) if self.lifestyle_changes else [],
            'warning_signs': json.loads(self.warning_signs) if self.warning_signs else [],
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Statistics Model (for caching)
class Statistics(db.Model):
    __tablename__ = 'statistics'
    
    id = db.Column(db.Integer, primary_key=True)
    total_predictions = db.Column(db.Integer, default=0)
    average_confidence = db.Column(db.Float, default=0.0)
    disease_distribution = db.Column(db.Text)  # JSON string
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        import json
        return {
            'total_predictions': self.total_predictions,
            'average_confidence': self.average_confidence,
            'disease_distribution': json.loads(self.disease_distribution) if self.disease_distribution else {},
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
