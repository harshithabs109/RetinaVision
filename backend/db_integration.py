"""
Database Integration Helper Functions
Provides functions to interact with PostgreSQL database
"""
import json
from datetime import datetime
from database import db, Patient, Prediction, Statistics

def save_patient_to_db(patient_data):
    """Save or update patient in database"""
    try:
        patient_id = patient_data.get('patient_id')
        
        # Check if patient exists
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        
        if patient:
            # Update existing patient
            patient.name = patient_data.get('name', patient.name)
            patient.age = patient_data.get('age', patient.age)
            patient.dob = patient_data.get('dob', patient.dob)
            patient.gender = patient_data.get('gender', patient.gender)
            patient.contact = patient_data.get('contact', patient.contact)
            patient.email = patient_data.get('email', patient.email)
            patient.address = patient_data.get('address', patient.address)
            patient.medical_history = patient_data.get('medical_history', patient.medical_history)
            patient.medications = patient_data.get('medications', patient.medications)
            patient.updated_at = datetime.utcnow()
        else:
            # Create new patient
            patient = Patient(
                patient_id=patient_id,
                name=patient_data.get('name'),
                age=patient_data.get('age'),
                dob=patient_data.get('dob'),
                gender=patient_data.get('gender'),
                contact=patient_data.get('contact'),
                email=patient_data.get('email'),
                address=patient_data.get('address'),
                medical_history=patient_data.get('medical_history'),
                medications=patient_data.get('medications')
            )
            db.session.add(patient)
        
        db.session.commit()
        return patient.to_dict()
    except Exception as e:
        db.session.rollback()
        print(f"Error saving patient: {e}")
        raise

def get_patient_from_db(patient_id):
    """Get patient from database"""
    try:
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        return patient.to_dict() if patient else None
    except Exception as e:
        print(f"Error getting patient: {e}")
        return None

def save_prediction_to_db(prediction_data):
    """Save prediction to database"""
    try:
        prediction = Prediction(
            prediction_id=prediction_data.get('prediction_id'),
            patient_id=prediction_data.get('patient_id'),
            disease=prediction_data.get('disease'),
            confidence=prediction_data.get('confidence'),
            predictions_json=json.dumps(prediction_data.get('predictions', {})),
            image_path=prediction_data.get('image_path'),
            heatmap_path=prediction_data.get('heatmap_path'),
            overlay_path=prediction_data.get('overlay_path'),
            mask_path=prediction_data.get('mask_path'),
            report_filename=prediction_data.get('report_filename'),
            daily_tips=json.dumps(prediction_data.get('daily_tips', [])),
            medical_advice=prediction_data.get('medical_advice'),
            lifestyle_changes=json.dumps(prediction_data.get('lifestyle_changes', [])),
            warning_signs=json.dumps(prediction_data.get('warning_signs', []))
        )
        
        db.session.add(prediction)
        db.session.commit()
        return prediction.to_dict()
    except Exception as e:
        db.session.rollback()
        print(f"Error saving prediction: {e}")
        raise

def get_all_predictions():
    """Get all predictions from database"""
    try:
        predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
        return [p.to_dict() for p in predictions]
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return []

def get_patient_predictions(patient_id):
    """Get all predictions for a specific patient"""
    try:
        predictions = Prediction.query.filter_by(patient_id=patient_id).order_by(Prediction.created_at.desc()).all()
        return [p.to_dict() for p in predictions]
    except Exception as e:
        print(f"Error getting patient predictions: {e}")
        return []

def calculate_statistics():
    """Calculate statistics from database"""
    try:
        # Get all predictions
        all_predictions = Prediction.query.all()
        
        total_predictions = len(all_predictions)
        
        # Calculate average confidence
        if total_predictions > 0:
            avg_confidence = sum(p.confidence for p in all_predictions) / total_predictions
        else:
            avg_confidence = 0.0
        
        # Calculate disease distribution (including Normal and Normal-1)
        disease_dist = {}
        for pred in all_predictions:
            disease = pred.disease
            disease_dist[disease] = disease_dist.get(disease, 0) + 1
        
        return {
            'total_predictions': total_predictions,
            'average_confidence': avg_confidence,
            'disease_distribution': disease_dist
        }
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return {
            'total_predictions': 0,
            'average_confidence': 0.0,
            'disease_distribution': {}
        }

def get_all_patients():
    """Get all patients from database"""
    try:
        patients = Patient.query.order_by(Patient.created_at.desc()).all()
        return [p.to_dict() for p in patients]
    except Exception as e:
        print(f"Error getting patients: {e}")
        return []

def delete_patient_from_db(patient_id):
    """Delete patient and all their predictions"""
    try:
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if patient:
            db.session.delete(patient)
            db.session.commit()
            return True
        return False
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting patient: {e}")
        return False

def update_statistics_cache():
    """Update cached statistics in database"""
    try:
        stats = calculate_statistics()
        
        # Get or create statistics record
        stat_record = Statistics.query.first()
        if stat_record:
            stat_record.total_predictions = stats['total_predictions']
            stat_record.average_confidence = stats['average_confidence']
            stat_record.disease_distribution = json.dumps(stats['disease_distribution'])
            stat_record.last_updated = datetime.utcnow()
        else:
            stat_record = Statistics(
                total_predictions=stats['total_predictions'],
                average_confidence=stats['average_confidence'],
                disease_distribution=json.dumps(stats['disease_distribution'])
            )
            db.session.add(stat_record)
        
        db.session.commit()
        return stats
    except Exception as e:
        db.session.rollback()
        print(f"Error updating statistics cache: {e}")
        return None

def get_cached_statistics():
    """Get cached statistics from database"""
    try:
        stat_record = Statistics.query.first()
        if stat_record:
            return stat_record.to_dict()
        else:
            # Calculate and cache if not exists
            return update_statistics_cache()
    except Exception as e:
        print(f"Error getting cached statistics: {e}")
        return calculate_statistics()
