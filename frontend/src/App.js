import React, { useState } from 'react';
import {
  Container,
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardMedia,
  LinearProgress,
  Alert,
  Chip,
  Divider,
  AppBar,
  Toolbar,
  Avatar,
  CssBaseline
} from '@mui/material';
import {
  CloudUpload,
  Visibility,
  LocalHospital,
  MedicalServices,
  Speed
} from '@mui/icons-material';
import axios from 'axios';
import './App.css';

// Import components
import ImageUploader from './components/ImageUploader';
import ImageViewer from './components/ImageViewer';
import PredictionResult from './components/PredictionResult';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setImagePreview(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      // Use the predict_visualize endpoint for enhanced visualizations
      const response = await axios.post(`${API_BASE_URL}/api/predict_visualize`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Debug logging
      console.log("=== App.js API Response Debug ===");
      console.log("Full response:", response);
      console.log("Response data:", response.data);
      
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to process image');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
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

  return (
    <>
      <CssBaseline />
      <div className="App">
        <AppBar position="static" sx={{ background: 'linear-gradient(90deg, #0c4a6e 0%, #2563eb 100%)' }}>
          <Toolbar>
            <Avatar sx={{ 
              bgcolor: 'white', 
              mr: 2,
              width: 40,
              height: 40
            }}>
              <LocalHospital sx={{ color: '#0c4a6e' }} />
            </Avatar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 700 }}>
              AI-Powered Eye Disease Classification System
            </Typography>
            <Chip 
              icon={<MedicalServices />} 
              label="Medical AI" 
              color="secondary" 
              sx={{ 
                fontWeight: 'bold',
                background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                color: '#0c4a6e'
              }} 
            />
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
          <Grid container spacing={3}>
            {/* Left Panel - Image Upload */}
            <Grid item xs={12} md={5}>
              <Paper elevation={4} sx={{ p: 3, height: '100%', borderRadius: 3, background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ 
                    bgcolor: '#0ea5e9', 
                    mr: 2,
                    width: 48,
                    height: 48
                  }}>
                    <CloudUpload sx={{ color: 'white' }} />
                  </Avatar>
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#0c4a6e' }}>
                    Upload Retinal Image
                  </Typography>
                </Box>
                <Divider sx={{ mb: 3, backgroundColor: '#0ea5e9' }} />

                <ImageUploader onImageSelect={handleImageSelect} />

                {imagePreview && (
                  <Box sx={{ mt: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Avatar sx={{ 
                        bgcolor: '#10b981', 
                        mr: 1,
                        width: 32,
                        height: 32
                      }}>
                        <Visibility sx={{ fontSize: 20, color: 'white' }} />
                      </Avatar>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, color: '#0c4a6e' }}>
                        Original Image Preview:
                      </Typography>
                    </Box>
                    <Card sx={{ 
                      borderRadius: 2, 
                      overflow: 'hidden', 
                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                      border: '2px solid #bae6fd'
                    }}>
                      <CardMedia
                        component="img"
                        image={imagePreview}
                        alt="Uploaded eye image"
                        sx={{ maxHeight: 350, objectFit: 'contain' }}
                      />
                    </Card>

                    <Button
                      fullWidth
                      variant="contained"
                      size="large"
                      onClick={handlePredict}
                      disabled={loading}
                      sx={{ 
                        mt: 3, 
                        py: 1.5, 
                        borderRadius: 2,
                        fontWeight: 700,
                        fontSize: '1.1rem',
                        background: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
                        boxShadow: '0 4px 16px rgba(14, 165, 233, 0.4)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
                          boxShadow: '0 6px 20px rgba(14, 165, 233, 0.6)'
                        }
                      }}
                      startIcon={loading ? <LinearProgress sx={{ width: 20, color: 'inherit' }} /> : <Speed />}
                    >
                      {loading ? 'Analyzing with AI...' : 'Analyze Image'}
                    </Button>

                    {loading && (
                      <Box sx={{ mt: 3 }}>
                        <LinearProgress 
                          sx={{ 
                            height: 10, 
                            borderRadius: 5,
                            '& .MuiLinearProgress-bar': {
                              borderRadius: 5,
                              background: 'linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%)'
                            }
                          }} 
                        />
                        <Typography variant="body1" align="center" sx={{ mt: 1, fontWeight: 600, color: '#0c4a6e' }}>
                          Processing with medical-grade AI model...
                        </Typography>
                        <Typography variant="body2" align="center" color="text.secondary" sx={{ mt: 0.5 }}>
                          This may take 10-30 seconds
                        </Typography>
                      </Box>
                    )}
                  </Box>
                )}

                {error && (
                  <Alert 
                    severity="error" 
                    sx={{ 
                      mt: 3, 
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
                      color: '#7f1d1d'
                    }}
                  >
                    {error}
                  </Alert>
                )}
              </Paper>
            </Grid>

            {/* Right Panel - Results */}
            <Grid item xs={12} md={7}>
              {prediction ? (
                <Box className="fade-in">
                  {/* Prediction Result */}
                  <PredictionResult prediction={prediction} />
                  
                  {/* Image Visualizations */}
                  <ImageViewer prediction={prediction} imagePreview={imagePreview} />

                </Box>
              ) : (
                <Paper 
                  elevation={4} 
                  sx={{ 
                    p: 5, 
                    textAlign: 'center', 
                    height: '100%',
                    borderRadius: 3,
                    background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)'
                  }}
                >
                  <Box sx={{ 
                    display: 'flex', 
                    justifyContent: 'center', 
                    mb: 3
                  }}>
                    <Avatar sx={{ 
                      bgcolor: '#0ea5e9', 
                      width: 80, 
                      height: 80,
                      mb: 2
                    }}>
                      <Visibility sx={{ fontSize: 40, color: 'white' }} />
                    </Avatar>
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: '#0c4a6e', mb: 2 }}>
                    Ready for Analysis
                  </Typography>
                  <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
                    Upload a retinal image to begin AI-powered diagnosis
                  </Typography>
                  <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
                    Our advanced AI will detect and classify eye diseases with medical-grade accuracy
                  </Typography>
                  <Chip 
                    label="8 Disease Classes Supported" 
                    sx={{ 
                      mt: 3, 
                      background: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
                      color: 'white',
                      fontWeight: 'bold',
                      py: 1,
                      px: 2
                    }} 
                  />
                </Paper>
              )}
            </Grid>
          </Grid>
        </Container>
      </div>
    </>
  );
}

export default App;
