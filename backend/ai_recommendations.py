"""
AI-Powered Recommendations Generator using OpenAI GPT
Generates personalized day-to-day recommendations for eye disease patients
"""

import os
from openai import OpenAI

# Initialize OpenAI client - Load from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def generate_ai_recommendations(disease_name, confidence_score, patient_age=None, patient_gender=None):
    """
    Generate AI-powered personalized recommendations using OpenAI GPT
    
    Args:
        disease_name: Detected eye disease
        confidence_score: Model confidence (0-100)
        patient_age: Patient age (optional)
        patient_gender: Patient gender (optional)
    
    Returns:
        dict: {
            'daily_tips': list of day-to-day care tips,
            'medical_advice': professional medical recommendations,
            'lifestyle_changes': suggested lifestyle modifications,
            'warning_signs': symptoms to watch for
        }
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Build context-aware prompt
        patient_context = ""
        if patient_age:
            patient_context += f"Patient Age: {patient_age} years old. "
        if patient_gender:
            patient_context += f"Gender: {patient_gender}. "
        
        prompt = f"""You are an expert ophthalmologist AI assistant. Generate personalized recommendations for a patient diagnosed with {disease_name} (AI confidence: {confidence_score:.1f}%).

{patient_context}

Please provide:
1. 5 practical day-to-day care tips
2. Medical recommendations and when to seek immediate care
3. Lifestyle changes to manage the condition
4. Warning signs to watch for

Format your response as JSON with these keys:
- daily_tips: array of 5 practical tips
- medical_advice: string with professional recommendations
- lifestyle_changes: array of 3-4 lifestyle modifications
- warning_signs: array of 3-4 symptoms to watch for

Keep recommendations practical, clear, and patient-friendly. Focus on actionable advice."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert ophthalmologist providing patient care recommendations. Always provide accurate, helpful medical guidance while reminding patients to consult healthcare professionals."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        import json
        recommendations = json.loads(response.choices[0].message.content)
        
        print(f"✅ AI Recommendations generated for {disease_name}")
        return recommendations
        
    except Exception as e:
        print(f"❌ Error generating AI recommendations: {e}")
        # Fallback to basic recommendations
        return get_fallback_recommendations(disease_name)


def get_fallback_recommendations(disease_name):
    """Enhanced disease-specific recommendations when AI API is unavailable"""
    
    # Disease-specific recommendations database
    disease_recommendations = {
        'Cataract': {
            'daily_tips': [
                "Use brighter lighting when reading or doing close work",
                "Wear anti-glare sunglasses outdoors to reduce light sensitivity",
                "Update your eyeglass prescription regularly as vision changes",
                "Use magnifying lenses for detailed tasks if needed",
                "Avoid driving at night if you experience glare or halos around lights"
            ],
            'medical_advice': "Consult with an ophthalmologist to discuss cataract surgery options. Surgery is highly effective and typically recommended when cataracts interfere with daily activities. Regular monitoring is essential to determine the optimal timing for intervention.",
            'lifestyle_changes': [
                "Maintain a diet rich in antioxidants (leafy greens, colorful fruits)",
                "Protect eyes from UV radiation with quality sunglasses",
                "Manage diabetes and blood pressure if applicable",
                "Quit smoking as it accelerates cataract progression"
            ],
            'warning_signs': [
                "Rapid deterioration in vision quality",
                "Difficulty performing daily activities due to vision",
                "Double vision in one eye",
                "Significant glare or halos affecting safety"
            ]
        },
        'Diabetic Retinopathy': {
            'daily_tips': [
                "Monitor and maintain blood sugar levels within target range daily",
                "Take all prescribed diabetes medications exactly as directed",
                "Check your feet and eyes daily for any changes",
                "Keep a log of blood sugar readings to share with your doctor",
                "Attend all scheduled retinal examinations without delay"
            ],
            'medical_advice': "Immediate consultation with a retina specialist is critical. Diabetic retinopathy requires prompt treatment to prevent vision loss. Treatment options may include laser therapy, anti-VEGF injections, or vitrectomy. Continue close coordination with your endocrinologist for diabetes management.",
            'lifestyle_changes': [
                "Follow a low-glycemic diet to stabilize blood sugar",
                "Exercise for 30 minutes daily to improve circulation",
                "Maintain blood pressure below 130/80 mmHg",
                "Quit smoking immediately - it doubles retinopathy risk"
            ],
            'warning_signs': [
                "Sudden vision loss or significant blurring",
                "New floaters or dark spots in vision",
                "Flashes of light or curtain-like shadow",
                "Difficulty seeing at night or in dim light"
            ]
        },
        'Glaucoma': {
            'daily_tips': [
                "Use prescribed eye drops at the same time each day",
                "Avoid activities that increase eye pressure (heavy lifting, inverted positions)",
                "Sleep with your head elevated on 2-3 pillows",
                "Drink fluids slowly throughout the day, not all at once",
                "Wear eye protection during sports and physical activities"
            ],
            'medical_advice': "Urgent consultation with a glaucoma specialist is essential. Glaucoma requires lifelong management to prevent irreversible vision loss. Treatment typically involves daily eye drops, laser procedures, or surgery to lower eye pressure. Regular monitoring of intraocular pressure and visual field testing are crucial.",
            'lifestyle_changes': [
                "Exercise regularly (walking, swimming) to lower eye pressure",
                "Limit caffeine intake as it can raise eye pressure",
                "Practice stress-reduction techniques like meditation",
                "Maintain a healthy weight and blood pressure"
            ],
            'warning_signs': [
                "Severe eye pain with nausea or vomiting",
                "Sudden vision loss or tunnel vision",
                "Seeing halos around lights",
                "Red eye with cloudy cornea"
            ]
        },
        'Diabetic Macular Edema': {
            'daily_tips': [
                "Strictly control blood sugar levels - check multiple times daily",
                "Take anti-VEGF injections as scheduled by your retina specialist",
                "Use an Amsler grid daily to monitor central vision changes",
                "Avoid activities that cause eye strain or increase blood pressure",
                "Keep all follow-up appointments for OCT scans and monitoring"
            ],
            'medical_advice': "Urgent consultation with a retina specialist is required. DME needs aggressive treatment with anti-VEGF injections, laser therapy, or steroid implants. Tight blood sugar control is essential. Work closely with both your ophthalmologist and endocrinologist for comprehensive management.",
            'lifestyle_changes': [
                "Follow a strict diabetic diet with consistent carbohydrate intake",
                "Monitor blood pressure daily - keep below 130/80",
                "Reduce sodium intake to prevent fluid retention",
                "Engage in moderate exercise approved by your doctor"
            ],
            'warning_signs': [
                "Distortion or wavy lines in central vision",
                "Difficulty reading or recognizing faces",
                "Dark or empty area in center of vision",
                "Sudden worsening of vision"
            ]
        },
        'Choroidal Neovascularization': {
            'daily_tips': [
                "Monitor vision daily with an Amsler grid - report any changes immediately",
                "Attend all scheduled anti-VEGF injection appointments",
                "Protect eyes from bright light with quality sunglasses",
                "Avoid smoking and secondhand smoke exposure",
                "Take AREDS2 vitamins as recommended by your doctor"
            ],
            'medical_advice': "Immediate consultation with a retina specialist is critical. CNV requires prompt treatment with anti-VEGF injections to prevent permanent vision loss. Regular OCT imaging is necessary to monitor treatment response. Early and aggressive treatment offers the best outcomes.",
            'lifestyle_changes': [
                "Eat a diet rich in omega-3 fatty acids (fish, nuts)",
                "Consume dark leafy greens daily (kale, spinach)",
                "Maintain healthy blood pressure and cholesterol",
                "Quit smoking - it significantly worsens CNV"
            ],
            'warning_signs': [
                "Sudden distortion of straight lines",
                "Dark spot in central vision",
                "Rapid decrease in reading ability",
                "Colors appearing less vivid or washed out"
            ]
        },
        'Drusen': {
            'daily_tips': [
                "Take AREDS2 vitamin supplements daily as recommended",
                "Monitor vision with an Amsler grid weekly",
                "Schedule comprehensive eye exams every 6-12 months",
                "Protect eyes from UV light with wraparound sunglasses",
                "Maintain good lighting for reading and close work"
            ],
            'medical_advice': "Regular monitoring by an ophthalmologist is essential as drusen may indicate early age-related macular degeneration. While drusen themselves don't require treatment, they increase risk of AMD progression. AREDS2 vitamins may slow progression. Annual dilated eye exams and OCT imaging are recommended.",
            'lifestyle_changes': [
                "Eat a Mediterranean-style diet rich in fish and vegetables",
                "Exercise regularly to improve cardiovascular health",
                "Maintain healthy weight and blood pressure",
                "Quit smoking - it doubles AMD progression risk"
            ],
            'warning_signs': [
                "New distortion or wavy lines in vision",
                "Difficulty adapting to low light",
                "Blurred or decreased central vision",
                "Need for brighter light when reading"
            ]
        },
        'Normal': {
            'daily_tips': [
                "Continue annual comprehensive eye examinations",
                "Wear UV-protective sunglasses when outdoors",
                "Follow the 20-20-20 rule: every 20 minutes, look 20 feet away for 20 seconds",
                "Maintain a balanced diet rich in eye-healthy nutrients",
                "Stay hydrated and get 7-8 hours of quality sleep"
            ],
            'medical_advice': "No signs of eye disease detected. Continue preventive care with annual eye exams. Maintain healthy lifestyle habits to preserve vision. Report any sudden changes in vision to your eye care provider promptly.",
            'lifestyle_changes': [
                "Eat foods rich in vitamins A, C, E, and omega-3 fatty acids",
                "Exercise regularly to maintain good circulation",
                "Manage screen time and take regular breaks",
                "Avoid smoking and limit alcohol consumption"
            ],
            'warning_signs': [
                "Sudden vision changes or loss",
                "Persistent eye pain or discomfort",
                "New floaters or flashes of light",
                "Unusual sensitivity to light"
            ]
        },
        'Normal-1': {
            'daily_tips': [
                "Maintain regular eye examination schedule",
                "Practice good eye hygiene and avoid eye rubbing",
                "Use proper lighting for reading and computer work",
                "Stay physically active to promote overall eye health",
                "Keep chronic conditions like diabetes and hypertension well-controlled"
            ],
            'medical_advice': "Your retinal examination appears healthy. Continue preventive eye care with regular checkups. Maintain healthy lifestyle habits and protect your eyes from injury and UV exposure. Contact your eye care provider if you notice any vision changes.",
            'lifestyle_changes': [
                "Consume a diet high in antioxidants and omega-3s",
                "Maintain healthy blood pressure and blood sugar",
                "Protect eyes during sports and hazardous activities",
                "Limit screen time and practice good digital eye hygiene"
            ],
            'warning_signs': [
                'NO wARNING'
            ]
        }
    }
    
    # Return disease-specific recommendations or default
    return disease_recommendations.get(disease_name, disease_recommendations['Normal'])


def format_recommendations_for_display(recommendations):
    """Format recommendations for frontend display"""
    return {
        'daily_tips': recommendations.get('daily_tips', []),
        'medical_advice': recommendations.get('medical_advice', ''),
        'lifestyle_changes': recommendations.get('lifestyle_changes', []),
        'warning_signs': recommendations.get('warning_signs', []),
        'ai_generated': True
    }


def format_recommendations_for_pdf(recommendations, disease_name):
    """Format recommendations for PDF report"""
    formatted = {
        'description': f"AI-Generated Personalized Recommendations for {disease_name}",
        'daily_tips': recommendations.get('daily_tips', []),
        'medical_recommendations': recommendations.get('medical_advice', ''),
        'lifestyle_changes': recommendations.get('lifestyle_changes', []),
        'warning_signs': recommendations.get('warning_signs', [])
    }
    return formatted
