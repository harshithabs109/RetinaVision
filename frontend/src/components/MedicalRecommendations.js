import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  Divider
} from '@mui/material';
import {
  Warning,
  LocalHospital,
  AccessTime,
  Healing,
  FitnessCenter,
  CheckCircle
} from '@mui/icons-material';

const medicalData = {
  'Cataract': {
    urgency: 'Moderate',
    urgencyColor: 'warning',
    recommendation: 'Schedule consultation with an ophthalmologist within 2-4 weeks for comprehensive evaluation and treatment planning.',
    treatments: [
      'Surgical removal (phacoemulsification)',
      'Intraocular lens implantation',
      'Conservative management for early stages'
    ],
    followUp: 'Regular monitoring every 6-12 months',
    lifestyle: [
      'Wear UV-protective sunglasses',
      'Use brighter lighting for reading',
      'Avoid night driving if glare is significant'
    ],
    warningsSigns: [
      'Rapid vision changes',
      'Severe glare sensitivity',
      'Difficulty with daily activities'
    ]
  },
  'Diabetic Retinopathy': {
    urgency: 'High',
    urgencyColor: 'error',
    recommendation: 'Immediate consultation with a retina specialist within 1-2 weeks. This condition requires urgent attention.',
    treatments: [
      'Laser photocoagulation',
      'Intravitreal injections (anti-VEGF)',
      'Vitrectomy surgery (advanced cases)'
    ],
    followUp: 'Frequent monitoring every 3-6 months',
    lifestyle: [
      'Strict blood glucose control',
      'Blood pressure management',
      'Regular HbA1c monitoring'
    ],
    warningsSigns: [
      'Sudden vision loss',
      'Floaters or flashes',
      'Blurred or distorted vision'
    ]
  },
  'Glaucoma': {
    urgency: 'High',
    urgencyColor: 'error',
    recommendation: 'Urgent consultation with a glaucoma specialist within 1 week. Early intervention is crucial to prevent vision loss.',
    treatments: [
      'Prescription eye drops',
      'Laser trabeculoplasty',
      'Surgical procedures (trabeculectomy, drainage devices)'
    ],
    followUp: 'Regular monitoring every 3-6 months',
    lifestyle: [
      'Regular exercise (moderate intensity)',
      'Avoid activities that increase eye pressure',
      'Maintain healthy blood pressure'
    ],
    warningsSigns: [
      'Severe eye pain',
      'Nausea and vomiting',
      'Sudden vision loss',
      'Halos around lights'
    ]
  },
  'Choroidal Neovascularization': {
    urgency: 'High',
    urgencyColor: 'error',
    recommendation: 'Consult a retina specialist within 1 week for comprehensive evaluation and anti-VEGF treatment planning.',
    treatments: [
      'Intravitreal anti-VEGF injections',
      'Photodynamic therapy (select cases)',
      'Nutritional support with AREDS2 formulation'
    ],
    followUp: 'Monthly monitoring until lesion stabilization',
    lifestyle: [
      'Stop smoking immediately',
      'Maintain healthy blood pressure and cholesterol',
      'Use home Amsler grid monitoring'
    ],
    warningsSigns: [
      'Sudden distortion of vision',
      'Central blind spots',
      'Rapid decline in visual acuity'
    ]
  },
  'Diabetic Macular Edema': {
    urgency: 'High',
    urgencyColor: 'error',
    recommendation: 'Arrange urgent appointment with a retina specialist within 1 week for macular edema management.',
    treatments: [
      'Intravitreal anti-VEGF or steroid injections',
      'Focal/grid laser therapy',
      'Systemic diabetic control optimization'
    ],
    followUp: 'Monitoring every 1-3 months depending on treatment response',
    lifestyle: [
      'Tight glycemic control',
      'Blood pressure and lipid management',
      'Regular exercise as approved by physician'
    ],
    warningsSigns: [
      'Increasing blurriness',
      'Difficulty reading',
      'New floaters or flashes'
    ]
  },
  'Drusen': {
    urgency: 'Moderate',
    urgencyColor: 'warning',
    recommendation: 'Schedule ophthalmology follow-up within 4-6 weeks to assess risk of age-related macular degeneration progression.',
    treatments: [
      'AREDS2 nutritional supplementation',
      'Regular macular OCT monitoring',
      'Low vision aids if needed'
    ],
    followUp: 'Every 6-12 months or sooner if symptoms change',
    lifestyle: [
      'Adopt Mediterranean-style diet',
      'Wear UV and blue-light protective eyewear',
      'Monitor vision weekly with Amsler grid'
    ],
    warningsSigns: [
      'Wavy or distorted central vision',
      'Difficulty adapting to low light',
      'Dark or empty area in central vision'
    ]
  },
  'Normal': {
    urgency: 'Low',
    urgencyColor: 'success',
    recommendation: 'No immediate action required. Continue regular eye examinations as recommended by your healthcare provider.',
    treatments: [
      'Routine eye care',
      'Preventive measures'
    ],
    followUp: 'Annual comprehensive eye examination',
    lifestyle: [
      'Maintain healthy diet rich in antioxidants',
      'Protect eyes from UV radiation',
      'Practice good eye hygiene'
    ],
    warningsSigns: [
      'Any changes in vision',
      'Eye pain or discomfort',
      'Persistent headaches'
    ]
  },
  'Normal-1': {
    urgency: 'Low',
    urgencyColor: 'success',
    recommendation: 'No urgent findings. Consider follow-up to monitor subtle variations noted in the retina.',
    treatments: [
      'Routine observation',
      'Preventive eye health measures'
    ],
    followUp: 'Annual routine examination',
    lifestyle: [
      'Maintain healthy diet',
      'Regular eye protection',
      'Monitor for changes'
    ],
    warningsSigns: [
      'Vision changes',
      'Eye discomfort',
      'Visual disturbances'
    ]
  }
};

const MedicalRecommendations = ({ disease, confidence }) => {
  const data = medicalData[disease] || medicalData['Normal'];

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          <LocalHospital sx={{ mr: 1, verticalAlign: 'middle' }} />
          Medical Recommendations
        </Typography>
        <Chip 
          label={`${data.urgency} Urgency`}
          color={data.urgencyColor}
          size="small"
        />
      </Box>

      <Alert severity={data.urgencyColor === 'error' ? 'error' : data.urgencyColor === 'warning' ? 'warning' : 'info'} sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>Recommendation:</strong> {data.recommendation}
        </Typography>
      </Alert>

      <Divider sx={{ mb: 2 }} />

      {/* Treatment Options */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <Healing sx={{ mr: 1, fontSize: 20 }} />
          Treatment Options
        </Typography>
        <List dense>
          {data.treatments.map((treatment, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                <CheckCircle fontSize="small" color="primary" />
              </ListItemIcon>
              <ListItemText primary={treatment} />
            </ListItem>
          ))}
        </List>
      </Box>

      {/* Follow-up */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <AccessTime sx={{ mr: 1, fontSize: 20 }} />
          Follow-up Schedule
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {data.followUp}
        </Typography>
      </Box>

      {/* Lifestyle Recommendations */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <FitnessCenter sx={{ mr: 1, fontSize: 20 }} />
          Lifestyle Recommendations
        </Typography>
        <List dense>
          {data.lifestyle.map((item, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                <CheckCircle fontSize="small" color="success" />
              </ListItemIcon>
              <ListItemText primary={item} />
            </ListItem>
          ))}
        </List>
      </Box>

      {/* Warning Signs */}
      <Box>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <Warning sx={{ mr: 1, fontSize: 20, color: 'error.main' }} />
          Warning Signs - Seek Immediate Care
        </Typography>
        <List dense>
          {data.warningsSigns.map((sign, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                <Warning fontSize="small" color="error" />
              </ListItemIcon>
              <ListItemText primary={sign} />
            </ListItem>
          ))}
        </List>
      </Box>

      <Alert severity="warning" sx={{ mt: 3 }}>
        <Typography variant="body2">
          <strong>Disclaimer:</strong> This AI analysis is a screening tool and not a substitute for professional medical diagnosis. 
          Please consult with a qualified ophthalmologist for definitive diagnosis and treatment planning.
        </Typography>
      </Alert>
    </Paper>
  );
};

export default MedicalRecommendations;
