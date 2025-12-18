import React from 'react';
import {
  Paper,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Grid,
  Card,
  CardContent,
  Divider,
  Avatar
} from '@mui/material';
import {
  TrendingUp,
  Assessment,
  Info,
  MedicalServices,
  Speed,
  LocalHospital
} from '@mui/icons-material';

const PredictionResult = ({ prediction }) => {
  // Very simple debugging - alert-style console logs
  console.log("!!! PREDICTION RESULT COMPONENT RENDERING !!!");
  console.log("!!! Received prediction:", prediction);
  
  // Check if prediction is valid
  if (!prediction) {
    console.log("!!! ERROR: No prediction data received !!!");
    return (
      <Paper elevation={4} sx={{ p: 3, mb: 3, borderRadius: 3 }}>
        <Typography variant="h6" color="error">
          Error: No prediction data received
        </Typography>
        <Typography variant="body1">
          Please check the console for debugging information
        </Typography>
      </Paper>
    );
  }
  
  // More robust extraction with detailed debugging
  console.log("!!! Type of prediction:", typeof prediction);
  console.log("!!! Prediction keys:", Object.keys(prediction));
  
  // Extract data with more robust error handling
  let diseaseName = 'Unknown Disease';
  let confidence = 0;
  
  // Handle different possible structures
  if (typeof prediction === 'object' && prediction !== null) {
    // Check if it's the direct structure from backend
    if (prediction.prediction) {
      diseaseName = prediction.prediction;
      console.log("!!! Found direct prediction.prediction:", prediction.prediction);
    }
    
    if (prediction.confidence !== undefined) {
      confidence = parseFloat(prediction.confidence) || 0;
      console.log("!!! Found direct prediction.confidence:", prediction.confidence);
    }
    
    // Fallback to nested structure if direct access didn't work
    if (diseaseName === 'Unknown Disease' && prediction.data && prediction.data.prediction) {
      diseaseName = prediction.data.prediction;
      console.log("!!! Found nested prediction.data.prediction:", prediction.data.prediction);
    }
    
    if (confidence === 0 && prediction.data && prediction.data.confidence !== undefined) {
      confidence = parseFloat(prediction.data.confidence) || 0;
      console.log("!!! Found nested prediction.data.confidence:", prediction.data.confidence);
    }
  }
  
  console.log("!!! Final extracted diseaseName:", diseaseName);
  console.log("!!! Final extracted confidence:", confidence);
  
  // If we don't have valid data, show a clear message
  if (typeof prediction === 'object' && Object.keys(prediction).length === 0) {
    console.log("!!! ERROR: Empty prediction object received !!!");
    return (
      <Paper elevation={4} sx={{ p: 3, mb: 3, borderRadius: 3 }}>
        <Typography variant="h6" color="error">
          Error: Empty prediction data received
        </Typography>
        <Typography variant="body1">
          Please check the console for debugging information
        </Typography>
      </Paper>
    );
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'success';
    if (confidence >= 75) return 'warning';
    return 'error';
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 97) return 'Very High';
    if (confidence >= 90) return 'High';
    if (confidence >= 75) return 'Moderate';
    return 'Low';
  };

  const getDiseaseIcon = (disease) => {
    const diseaseIcons = {
      'Cataract': '👁️',
      'Choroidal Neovascularization': '🔴',
      'Diabetic Macular Edema': '🟡',
      'Diabetic Retinopathy': '🟠',
      'Drusen': '🟣',
      'Glaucoma': '🟢',
      'Normal': '✅',
      'Normal-1': '✅'
    };
    return diseaseIcons[disease] || '🩺';
  };

  const getDiseaseColor = (disease) => {
    const diseaseColors = {
      'Cataract': '#ef4444',
      'Choroidal Neovascularization': '#dc2626',
      'Diabetic Macular Edema': '#f59e0b',
      'Diabetic Retinopathy': '#f97316',
      'Drusen': '#8b5cf6',
      'Glaucoma': '#10b981',
      'Normal': '#22c55e',
      'Normal-1': '#22c55e'
    };
    return diseaseColors[disease] || '#0ea5e9';
  };

  // Sort predictions by confidence - handle both structures
  const allPredictions = prediction.all_predictions || {};
  const sortedPredictions = Object.entries(allPredictions)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5); // Top 5

  return (
    <Paper elevation={4} sx={{ p: 3, mb: 3, borderRadius: 3 }}>
      {/* Very visible debugging indicator */}
      <div style={{ 
        position: 'fixed', 
        top: '50%', 
        left: '50%', 
        transform: 'translate(-50%, -50%)',
        background: '#ff0000', 
        color: '#ffffff', 
        padding: '20px', 
        borderRadius: '8px', 
        fontSize: '24px',
        fontWeight: 'bold',
        zIndex: 9999,
        border: '3px solid #000'
      }}>
        DEBUG: Component Rendering<br/>
        Disease: {diseaseName}<br/>
        Confidence: {confidence.toFixed(1)}%
      </div>
      
      {/* Original visible indicator */}
      <div style={{ 
        position: 'absolute', 
        top: 10, 
        right: 10, 
        background: '#ffeb3b', 
        color: '#000', 
        padding: '5px 10px', 
        borderRadius: '4px', 
        fontSize: '12px',
        zIndex: 1000
      }}>
        RENDERING: {diseaseName} ({confidence.toFixed(1)}%)
      </div>
      
      {/* Raw data display for debugging */}
      <div style={{ 
        background: '#f0f0f0', 
        border: '2px solid #333',
        padding: '15px',
        margin: '10px 0',
        borderRadius: '5px'
      }}>
        <h3>Raw Prediction Data (Debug View)</h3>
        <pre style={{ 
          background: '#fff', 
          padding: '10px', 
          borderRadius: '3px',
          maxHeight: '200px',
          overflow: 'auto',
          fontSize: '12px'
        }}>
          {JSON.stringify(prediction, null, 2)}
        </pre>
      </div>
      
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Avatar sx={{ 
          bgcolor: getDiseaseColor(diseaseName),
          width: 56, 
          height: 56,
          mr: 2
        }}>
          <MedicalServices sx={{ fontSize: 32 }} />
        </Avatar>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
            <LocalHospital sx={{ mr: 1, color: getDiseaseColor(diseaseName) }} />
            Diagnosis Result
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            AI-Powered Eye Disease Classification
          </Typography>
        </Box>
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Top Prediction */}
      <Card 
        sx={{ 
          mb: 3, 
          background: confidence >= 90 
            ? 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)'
            : confidence >= 75
            ? 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)'
            : 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
          borderLeft: `5px solid ${getDiseaseColor(diseaseName)}`,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
        }}
      >
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Box>
              <Typography variant="h4" sx={{ fontWeight: 800, mb: 1 }}>
                {getDiseaseIcon(diseaseName)} {diseaseName}
              </Typography>
              <Chip 
                icon={<Speed />}
                label={`${confidence.toFixed(1)}% Confidence`}
                color={getConfidenceColor(confidence)}
                size="medium"
                sx={{ 
                  fontWeight: 'bold',
                  fontSize: '1rem',
                  py: 1,
                  px: 2
                }}
              />
            </Box>
            <Chip 
              label={getConfidenceLevel(confidence)}
              color={getConfidenceColor(confidence)}
              size="large"
              sx={{ 
                fontWeight: 'bold', 
                fontSize: '1.1rem',
                py: 2,
                px: 3,
                height: 'auto'
              }}
            />
          </Box>
          
          <LinearProgress 
            variant="determinate" 
            value={confidence}
            color={getConfidenceColor(confidence)}
            sx={{ 
              height: 12, 
              borderRadius: 6,
              '& .MuiLinearProgress-bar': {
                borderRadius: 6
              }
            }}
          />
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              Confidence Level
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {confidence.toFixed(1)}%
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Alternative Predictions */}
      <Box sx={{ mt: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', fontWeight: 700, mb: 2 }}>
          <TrendingUp sx={{ mr: 1, fontSize: 28, color: '#0ea5e9' }} />
          Top 5 Predictions
        </Typography>
        <Grid container spacing={2}>
          {sortedPredictions.map(([disease, conf], index) => (
            <Grid item xs={12} key={disease}>
              <Box 
                sx={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  p: 2,
                  bgcolor: index === 0 ? 'action.selected' : 'background.paper',
                  borderRadius: 2,
                  border: index === 0 ? '2px solid' : '1px solid',
                  borderColor: index === 0 ? getDiseaseColor(disease) : 'divider',
                  boxShadow: index === 0 ? '0 4px 12px rgba(0, 0, 0, 0.1)' : '0 2px 4px rgba(0, 0, 0, 0.05)',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: '0 6px 16px rgba(0, 0, 0, 0.15)'
                  }
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Avatar 
                    sx={{ 
                      bgcolor: getDiseaseColor(disease),
                      width: 40, 
                      height: 40,
                      mr: 2,
                      fontSize: '1rem'
                    }}
                  >
                    {getDiseaseIcon(disease)}
                  </Avatar>
                  <Box>
                    <Typography variant="body1" sx={{ fontWeight: index === 0 ? 700 : 600 }}>
                      {index + 1}. {disease}
                    </Typography>
                    {index === 0 && (
                      <Chip 
                        label="Most Likely" 
                        size="small" 
                        sx={{ 
                          ml: 1, 
                          background: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
                          color: 'white',
                          fontWeight: 'bold'
                        }} 
                      />
                    )}
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={conf} 
                    color={getConfidenceColor(conf)}
                    sx={{ width: 120, height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="body1" sx={{ minWidth: 60, textAlign: 'right', fontWeight: 600 }}>
                    {conf.toFixed(1)}%
                  </Typography>
                </Box>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Info */}
      <Box sx={{ mt: 4, p: 2, bgcolor: 'info.light', borderRadius: 2, border: '1px solid', borderColor: 'info.main' }}>
        <Typography variant="body2" sx={{ display: 'flex', alignItems: 'flex-start' }}>
          <Info sx={{ fontSize: 20, mr: 1, flexShrink: 0 }} />
          <span>
            <strong>Analysis completed:</strong> {new Date().toLocaleString('en-US', {
              year: 'numeric',
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
              hour12: true
            })} | 
            <strong> Patient ID:</strong> {prediction.patient_id || 'N/A'} | 
            <strong> Report:</strong> {prediction.report_filename || 'Generating...'}
          </span>
        </Typography>
      </Box>
    </Paper>
  );
};

export default PredictionResult;