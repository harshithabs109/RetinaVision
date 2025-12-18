import React, { useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardMedia,
  LinearProgress,
  Alert,
  Chip,
  Divider,
  Tabs,
  Tab,
  CircularProgress,
  TextField
} from '@mui/material';
import { CloudUpload,
  Visibility,
  CheckCircle,
  Warning,
  Search,
  Person,
  PictureAsPdf,
  MedicalServices
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const PredictionPanel = ({ onPredictionComplete }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeImageTab, setActiveImageTab] = useState(0);
  
  // Patient details
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('');
  const [patientContact, setPatientContact] = useState('');
  const [patientDOB, setPatientDOB] = useState('');
  const [patientAddress, setPatientAddress] = useState('');
  const [patientMedicalHistory, setPatientMedicalHistory] = useState('');
  const [patientMedications, setPatientMedications] = useState('');
  const [searchPatientId, setSearchPatientId] = useState(() => {
    try {
      return localStorage.getItem('patient_id') || '';
    } catch (e) {
      return '';
    }
  });
  const [foundPatient, setFoundPatient] = useState(null);
  const [searchError, setSearchError] = useState(null);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState(null);

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    if (!patientName || !patientAge || !patientGender) {
      setError('Please fill in all patient details');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null); // Clear previous prediction to show loading state

    const formData = new FormData();
    formData.append('image', selectedImage);
    formData.append('patient_name', patientName);
    formData.append('patient_age', patientAge);
    formData.append('patient_gender', patientGender);
    formData.append('patient_contact', patientContact);
    formData.append('patient_dob', patientDOB);
    formData.append('patient_address', patientAddress);
    formData.append('patient_medical_history', patientMedicalHistory);
    formData.append('patient_medications', patientMedications);

    if (searchPatientId) {
      formData.append('patient_id', searchPatientId);
      console.log('📋 Using existing patient ID:', searchPatientId);
    }

    try {
      // Use the predict_visualize endpoint for enhanced visualizations
      const response = await axios.post(`${API_BASE_URL}/api/predict_visualize`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      console.log('🔍 Backend response:', response.data);
      console.log('🔍 Heatmap URL:', response.data.heatmap_url);
      console.log('🔍 Overlay URL:', response.data.overlay_url);
      console.log('🔍 Mask URL:', response.data.mask_url);

      setPrediction(response.data);
      // Persist patient id returned by backend so UI can reuse it
      const returnedPid = response.data.patient_id || response.data.prediction?.patient_id || '';
      if (returnedPid) {
        setSearchPatientId(returnedPid);
        console.log('📋 Received patient_id from backend:', returnedPid);
      }

      console.log('✅ Prediction successful - calling onPredictionComplete');
      if (onPredictionComplete) onPredictionComplete();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to process image');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    multiple: false
  });

  const handleSearchPatient = async () => {
    if (!searchPatientId) {
      setSearchError('Please enter a Patient ID');
      return;
    }

    try {
      const response = await axios.get(`${API_BASE_URL}/api/patient/${searchPatientId}`);
      const patient = response.data?.patient || null;
      if (patient) {
        setFoundPatient(patient);
        setPatientName(patient.name || '');
        setPatientAge(patient.age != null ? String(patient.age) : '');
        setPatientGender(patient.gender || '');
        setPatientContact(patient.contact || '');
        setPatientDOB(patient.dob || '');
        setPatientAddress(patient.address || '');
        setPatientMedicalHistory(patient.medical_history || '');
        setPatientMedications(patient.medications || '');
        setSearchError(null);
      } else {
        setSearchError(response.data?.error || 'Patient not found');
        setFoundPatient(null);
      }
    } catch (err) {
      setSearchError(err.response?.data?.error || 'Patient not found');
      setFoundPatient(null);
    }
  };

  const calculateAge = (dob) => {
    if (!dob) return '';
    const today = new Date();
    const birthDate = new Date(dob);
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    return age.toString();
  };

  const handleDOBChange = (e) => {
    const dob = e.target.value;
    setPatientDOB(dob);
    const calculatedAge = calculateAge(dob);
    if (calculatedAge) {
      setPatientAge(calculatedAge);
    }
  };

  const handleSavePatient = async () => {
    if (!patientName || !patientAge || !patientGender) {
      setSaveError('Please fill in Name, Age/DOB, and Gender to save patient');
      return;
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/api/patient`, {
        name: patientName,
        age: parseInt(patientAge),
        dob: patientDOB,
        gender: patientGender,
        contact: patientContact,
        address: patientAddress,
        medical_history: patientMedicalHistory,
        medications: patientMedications
      });

      if (response.data.patient_id) {
        setSaveSuccess(true);
        setSaveError(null);
        // backend returns `patient_id` - fall back to other keys if present
        const returnedId = response.data.patient_id || response.data.patient_unique_id || '';
        setSearchPatientId(returnedId);
        
        // Clear success message after 5 seconds
        setTimeout(() => {
          setSaveSuccess(false);
        }, 5000);
      }
    } catch (err) {
      setSaveError(err.response?.data?.error || 'Failed to save patient');
      setSaveSuccess(false);
    }
  };

  // persist patient id to localStorage so it survives reloads
  React.useEffect(() => {
    try {
      if (searchPatientId) localStorage.setItem('patient_id', searchPatientId);
      else localStorage.removeItem('patient_id');
    } catch (e) {
      // ignore
    }
  }, [searchPatientId]);

  // Helper function to get confidence level text
  const getConfidenceLevel = (confidence) => {
    if (confidence >= 90) return 'High Confidence';
    if (confidence >= 75) return 'Moderate Confidence';
    return 'Low Confidence';
  };

  return (
    <Grid container spacing={3}>
      {/* Left: Upload Section - Made more compact */}
      <Grid item xs={12} md={6}>
        <Paper elevation={4} sx={{ p: 3, borderRadius: 3, background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)' }}>
          <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', fontWeight: 600, mb: 2, color: '#0c4a6e' }}>
            <Person sx={{ mr: 1.5, color: '#0ea5e9' }} />
            Patient Information
          </Typography>
          <Divider sx={{ mb: 2, backgroundColor: '#0ea5e9' }} />

          {/* Patient ID Search - More compact */}
          <Box sx={{ mb: 2, p: 2, bgcolor: '#f0f9ff', borderRadius: 2, border: '1px solid #bae6fd' }}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                label="Patient ID"
                value={searchPatientId}
                onChange={(e) => setSearchPatientId(e.target.value.toUpperCase())}
                size="small"
                placeholder="PAT0001"
                variant="outlined"
                sx={{ 
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 1.5,
                    '&.Mui-focused fieldset': {
                      borderColor: '#0ea5e9',
                    },
                  }
                }}
              />
              <Button
                variant="contained"
                onClick={handleSearchPatient}
                startIcon={<Search />}
                sx={{ 
                  minWidth: '80px',
                  py: 1,
                  borderRadius: 1.5,
                  fontWeight: 600,
                  background: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
                  }
                }}
              >
                Search
              </Button>
            </Box>
            {searchError && (
              <Alert severity="error" sx={{ mt: 1, borderRadius: 1.5, py: 0.5, px: 1 }}>
                {searchError}
              </Alert>
            )}
            {foundPatient && (
              <Alert 
                severity="success" 
                sx={{ 
                  mt: 1, 
                  borderRadius: 1.5, 
                  py: 0.5, 
                  px: 1,
                  background: 'linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)',
                  color: '#14532d'
                }}
              >
                Patient: {foundPatient.name}
              </Alert>
            )}
          </Box>

          <Divider sx={{ mb: 2 }}>
            <Chip label="OR New Patient" size="small" sx={{ background: 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)', color: '#1e40af' }} />
          </Divider>

          <Grid container spacing={1.5}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Name"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                required
                size="small"
                variant="outlined"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: '#0ea5e9',
                    },
                  }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={3}>
              <TextField
                fullWidth
                label="Age"
                type="number"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                required
                size="small"
                variant="outlined"
                InputProps={{
                  readOnly: true,
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: '#0ea5e9',
                    },
                  }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={3}>
              <TextField
                fullWidth
                label="Gender"
                value={patientGender}
                onChange={(e) => setPatientGender(e.target.value)}
                required
                size="small"
                variant="outlined"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: '#0ea5e9',
                    },
                  }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="DOB"
                type="date"
                value={patientDOB}
                onChange={handleDOBChange}
                InputLabelProps={{ shrink: true }}
                size="small"
                variant="outlined"
                required
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: '#0ea5e9',
                    },
                  }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Contact"
                value={patientContact}
                onChange={(e) => setPatientContact(e.target.value)}
                size="small"
                variant="outlined"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: '#0ea5e9',
                    },
                  }
                }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Address"
                value={patientAddress}
                onChange={(e) => setPatientAddress(e.target.value)}
                size="small"
                variant="outlined"
                multiline
                rows={2}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: '#0ea5e9',
                    },
                  }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Medical History"
                value={patientMedicalHistory}
                onChange={(e) => setPatientMedicalHistory(e.target.value)}
                size="small"
                variant="outlined"
                multiline
                rows={3}
                placeholder="Previous conditions, surgeries, etc."
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: '#0ea5e9',
                    },
                  }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Current Medications"
                value={patientMedications}
                onChange={(e) => setPatientMedications(e.target.value)}
                size="small"
                variant="outlined"
                multiline
                rows={3}
                placeholder="List current medications"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: '#0ea5e9',
                    },
                  }
                }}
              />
            </Grid>
          </Grid>

          {/* Save Patient Button */}
          <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
            <Button
              fullWidth
              variant="outlined"
              color="primary"
              onClick={handleSavePatient}
              disabled={!patientName || !patientAge || !patientGender}
              startIcon={<Person />}
              sx={{ 
                py: 1, 
                borderRadius: 1.5,
                fontWeight: 600,
                fontSize: '0.9rem',
                borderColor: '#0ea5e9',
                color: '#0c4a6e',
                '&:hover': {
                  borderColor: '#0284c7',
                  backgroundColor: '#f0f9ff'
                }
              }}
            >
              Save
            </Button>
          </Box>

          {saveSuccess && (
            <Alert 
              severity="success" 
              icon={<CheckCircle fontSize="large" />}
              sx={{ 
                mt: 2, 
                borderRadius: 2, 
                py: 1.5,
                px: 2,
                background: 'linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)',
                color: '#14532d',
                border: '2px solid #22c55e',
                boxShadow: '0 4px 12px rgba(34, 197, 94, 0.3)',
                '& .MuiAlert-icon': {
                  fontSize: '2rem'
                }
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 0.5 }}>
                ✅ Patient Details Saved Successfully!
              </Typography>
              <Typography variant="body1" sx={{ fontWeight: 600 }}>
                Patient ID: {searchPatientId}
              </Typography>
              <Typography variant="body2" sx={{ mt: 0.5, opacity: 0.9 }}>
                You can now proceed with image analysis
              </Typography>
            </Alert>
          )}
          {saveError && (
            <Alert 
              severity="error" 
              sx={{ 
                mt: 2, 
                borderRadius: 2, 
                py: 1.5,
                px: 2,
                background: 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
                color: '#7f1d1d',
                border: '2px solid #ef4444',
                boxShadow: '0 4px 12px rgba(239, 68, 68, 0.3)'
              }}
            >
              <Typography variant="body1" sx={{ fontWeight: 600 }}>
                {saveError}
              </Typography>
            </Alert>
          )}
        </Paper>

        {/* Moved Upload Section Here - Reduced Top Margin */}
        <Paper elevation={4} sx={{ p: 3, mt: 1.5, borderRadius: 3, background: 'linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)' }}>
          <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', fontWeight: 600, mb: 2, color: '#92400e' }}>
            <CloudUpload sx={{ mr: 1.5, color: '#f59e0b' }} />
            Upload Retinal Image
          </Typography>
          <Divider sx={{ mb: 2, backgroundColor: '#f59e0b' }} />

          <Paper
            {...getRootProps()}
            sx={{
              p: 3,
              textAlign: 'center',
              cursor: 'pointer',
              border: '2px dashed',
              borderColor: isDragActive ? '#f59e0b' : '#fbbf24',
              bgcolor: isDragActive ? '#ffedd5' : '#fef3c7',
              transition: 'all 0.3s ease',
              borderRadius: 2,
              minHeight: 200,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              '&:hover': {
                borderColor: '#f59e0b',
                bgcolor: '#ffedd5'
              }
            }}
            elevation={0}
          >
            <input {...getInputProps()} />
            <CloudUpload sx={{ fontSize: 50, color: isDragActive ? '#f59e0b' : '#d97706', mb: 1 }} />
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: isDragActive ? '#92400e' : '#92400e' }}>
              {isDragActive ? 'Drop image here' : 'Drag & drop image'}
            </Typography>
            <Typography variant="body2" color="#92400e">
              or click to select file
            </Typography>
            <Typography variant="caption" sx={{ color: '#92400e', mt: 1 }}>
              Supports: PNG, JPG, JPEG
            </Typography>
          </Paper>

          {imagePreview && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600, mb: 1, color: '#92400e' }}>
                Selected Image:
              </Typography>
              <Card sx={{ borderRadius: 1.5, overflow: 'hidden', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)', border: '1px solid #fbbf24' }}>
                <CardMedia
                  component="img"
                  image={imagePreview}
                  alt="Uploaded eye image"
                  sx={{ maxHeight: 300, objectFit: 'contain' }}
                />
              </Card>

              <Button
                fullWidth
                variant="contained"
                size="large"
                onClick={handlePredict}
                disabled={loading}
                sx={{ 
                  mt: 2, 
                  py: 1.2, 
                  borderRadius: 1.5,
                  fontWeight: 600,
                  fontSize: '1rem',
                  background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                  boxShadow: '0 4px 12px rgba(245, 158, 11, 0.3)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #d97706 0%, #b45309 100%)',
                    boxShadow: '0 6px 20px rgba(245, 158, 11, 0.4)'
                  }
                }}
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <Visibility />}
              >
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </Button>

              {loading && (
                <Box 
                  className="fade-in"
                  sx={{ 
                    mt: 3,
                    p: 4,
                    background: 'linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%)',
                    borderRadius: 4,
                    border: '2px solid #a78bfa',
                    textAlign: 'center'
                  }}
                >
                  {/* Animated Medical Icon */}
                  <Box sx={{ position: 'relative', display: 'inline-block', mb: 2 }}>
                    <MedicalServices 
                      sx={{ 
                        fontSize: 80,
                        color: '#7c3aed',
                        animation: 'pulse 1.5s ease-in-out infinite'
                      }} 
                    />
                    <CircularProgress 
                      size={100}
                      thickness={2}
                      sx={{ 
                        position: 'absolute',
                        top: -10,
                        left: -10,
                        color: '#a78bfa',
                        animation: 'spin 2s linear infinite'
                      }}
                    />
                  </Box>
                  
                  <Typography 
                    variant="h6" 
                    sx={{ 
                      fontWeight: 700, 
                      color: '#5b21b6',
                      mb: 1,
                      animation: 'pulse 2s ease-in-out infinite'
                    }}
                  >
                    🔬 AI Analysis in Progress
                  </Typography>
                  
                  <Typography variant="body2" sx={{ color: '#6d28d9', mb: 2 }}>
                    Our advanced neural network is analyzing the retinal image...
                  </Typography>
                  
                  <LinearProgress 
                    sx={{ 
                      height: 8, 
                      borderRadius: 4,
                      background: '#e9d5ff',
                      '& .MuiLinearProgress-bar': {
                        borderRadius: 4,
                        background: 'linear-gradient(90deg, #7c3aed 0%, #a78bfa 50%, #7c3aed 100%)',
                        backgroundSize: '200% 100%',
                        animation: 'shimmer 2s infinite'
                      }
                    }} 
                  />
                  
                  <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 1, flexWrap: 'wrap' }}>
                    <Chip 
                      label="✓ Image Loaded" 
                      size="small"
                      sx={{ bgcolor: '#dcfce7', color: '#14532d', fontWeight: 600 }}
                    />
                    <Chip 
                      label="⚡ Processing..." 
                      size="small"
                      sx={{ 
                        bgcolor: '#fef3c7', 
                        color: '#92400e', 
                        fontWeight: 600,
                        animation: 'pulse 1s ease-in-out infinite'
                      }}
                    />
                    <Chip 
                      label="🎯 Generating Heatmaps" 
                      size="small"
                      sx={{ bgcolor: '#dbeafe', color: '#1e40af', fontWeight: 600 }}
                    />
                  </Box>
                </Box>
              )}
            </Box>
          )}

          {error && (
            <Alert 
              severity="error" 
              sx={{ 
                mt: 2, 
                borderRadius: 1.5,
                background: 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
                color: '#7f1d1d'
              }}
            >
              {error}
            </Alert>
          )}
        </Paper>
      </Grid>

      {/* Right: Results Section */}
      <Grid item xs={12} md={6}>
        {prediction ? (
          <Box className="fade-in">
            {/* Diagnosis Result - Enhanced Attractive Design */}
            <Paper 
              elevation={6} 
              sx={{ 
                p: 3, 
                mb: 2,
                borderRadius: 4,
                // Vibrant gradient backgrounds
                background: (Number(prediction.confidence||0) >= 90) 
                  ? 'linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%)'
                  : (Number(prediction.confidence||0) >= 75)
                  ? 'linear-gradient(135deg, #f59e0b 0%, #d97706 50%, #b45309 100%)'
                  : 'linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%)',
                borderLeft: `6px solid ${
                  (Number(prediction.confidence||0) >= 90) ? '#065f46' :
                  (Number(prediction.confidence||0) >= 75) ? '#92400e' : '#7f1d1d'
                }`,
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)',
                color: 'white',
                position: 'relative',
                overflow: 'hidden',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'radial-gradient(circle at top right, rgba(255,255,255,0.1) 0%, transparent 60%)',
                  pointerEvents: 'none'
                }
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2, position: 'relative', zIndex: 1 }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 800, mb: 1, color: 'white', textShadow: '2px 2px 4px rgba(0,0,0,0.3)' }}>
                    🎯 {prediction.prediction || 'Unknown'}
                  </Typography>
                  {prediction.patient_id && (
                    <Chip 
                      label={`Patient ID: ${prediction.patient_id}`}
                      sx={{ 
                        fontWeight: 'bold', 
                        fontSize: '0.85rem', 
                        py: 1.5, 
                        px: 1.5,
                        background: 'rgba(255, 255, 255, 0.95)',
                        color: '#1e40af',
                        backdropFilter: 'blur(10px)',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                      }}
                    />
                  )}
                </Box>
                <Chip 
                  icon={Number(prediction.confidence||0) >= 90 ? <CheckCircle /> : <Warning />}
                  label={getConfidenceLevel(Number(prediction.confidence||0))}
                  sx={{ 
                    fontWeight: 'bold', 
                    fontSize: '1rem', 
                    py: 2, 
                    px: 2,
                    height: 'auto',
                    background: 'rgba(255, 255, 255, 0.95)',
                    color: (Number(prediction.confidence||0) >= 90) 
                      ? '#065f46'
                      : (Number(prediction.confidence||0) >= 75)
                      ? '#92400e'
                      : '#7f1d1d',
                    backdropFilter: 'blur(10px)',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                  }}
                />
              </Box>
              <Box sx={{ mt: 1, mb: 1, position: 'relative', zIndex: 1 }}>
                {(() => {
                  const display = parseFloat(String(prediction.confidence || 0)) || 0;
                  return (
                    <>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="h6" sx={{ fontWeight: 700, color: 'white', textShadow: '1px 1px 2px rgba(0,0,0,0.3)' }}>
                          Confidence: {display.toFixed(1)}%
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={display} 
                        sx={{ 
                          height: 14, 
                          borderRadius: 7,
                          backgroundColor: 'rgba(255, 255, 255, 0.3)',
                          '& .MuiLinearProgress-bar': {
                            borderRadius: 7,
                            background: 'rgba(255, 255, 255, 0.9)',
                            boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                          }
                        }}
                      />
                      <Typography variant="body1" sx={{ mt: 1, fontWeight: 600, color: 'white', textShadow: '1px 1px 2px rgba(0,0,0,0.3)' }}>
                        {display >= 90
                          ? '✓ Strong diagnostic confidence'
                          : display >= 75
                          ? '⚠ Moderate confidence'
                          : '⚠ Low confidence'
                        }
                      </Typography>
                    </>
                  );
                })()}
              </Box>
              <Divider sx={{ my: 2, backgroundColor: 'rgba(255, 255, 255, 0.3)' }} />
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center', position: 'relative', zIndex: 1 }}>
                <Chip 
                  label={new Date().toLocaleString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: true
                  })}
                  sx={{ 
                    fontWeight: 'bold', 
                    fontSize: '0.8rem',
                    background: 'rgba(255, 255, 255, 0.9)',
                    color: '#1e40af',
                    backdropFilter: 'blur(10px)'
                  }}
                />
                {prediction.report_filename && (
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={<PictureAsPdf />}
                    onClick={() => {
                      window.open(`${API_BASE_URL}/api/reports/${prediction.report_filename}`, '_blank');
                    }}
                    sx={{ 
                      borderRadius: 2,
                      fontWeight: 600,
                      background: 'rgba(255, 255, 255, 0.9)',
                      color: '#dc2626',
                      backdropFilter: 'blur(10px)',
                      '&:hover': {
                        background: 'rgba(255, 255, 255, 1)',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
                      },
                      transition: 'all 0.3s ease'
                    }}
                  >
                    Download PDF Report
                  </Button>
                )}
              </Box>
            </Paper>

            {/* Disease Affected Areas - Heatmap Analysis (Tabs) */}
            <Paper elevation={6} sx={{ p: 3, borderRadius: 4, mb: 2, background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)', position: 'relative', overflow: 'hidden' }}>
              <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'radial-gradient(circle at bottom left, rgba(59, 130, 246, 0.1) 0%, transparent 50%)', pointerEvents: 'none' }} />
              <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', fontWeight: 700, mb: 3, color: 'white', textShadow: '2px 2px 4px rgba(0,0,0,0.3)', position: 'relative', zIndex: 1 }}>
                <Visibility sx={{ mr: 1.5, color: '#3b82f6', fontSize: 32 }} />
                Disease Affected Areas - Heatmap Analysis
              </Typography>
              <Divider sx={{ mb: 3, backgroundColor: 'rgba(59, 130, 246, 0.3)', height: 2 }} />

              <Tabs 
                value={activeImageTab} 
                onChange={(e, newValue) => setActiveImageTab(newValue)}
                variant="fullWidth"
                sx={{ 
                  mb: 3,
                  position: 'relative',
                  zIndex: 1,
                  '& .MuiTab-root': {
                    py: 1.5,
                    fontWeight: 700,
                    fontSize: '1rem',
                    color: 'rgba(255, 255, 255, 0.6)',
                    textTransform: 'none',
                    transition: 'all 0.3s ease',
                    '&.Mui-selected': {
                      color: 'white',
                      background: 'rgba(59, 130, 246, 0.2)',
                      borderRadius: 2
                    },
                    '&:hover': {
                      background: 'rgba(255, 255, 255, 0.05)',
                      borderRadius: 2
                    }
                  },
                  '& .MuiTabs-indicator': {
                    backgroundColor: '#3b82f6',
                    height: 3,
                    borderRadius: 2
                  }
                }}
              >
                <Tab label="📷 Original Image" sx={{ borderRadius: 2 }} />
                <Tab label="🔥 Heatmap View" sx={{ borderRadius: 2 }} />
                <Tab label="🎯 Affected Areas" sx={{ borderRadius: 2 }} />
              </Tabs>

              <Card sx={{ 
                borderRadius: 3, 
                overflow: 'hidden', 
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)',
                border: '3px solid rgba(59, 130, 246, 0.5)',
                position: 'relative',
                zIndex: 1,
                bgcolor: '#000',
                transition: 'all 0.3s ease',
                '&:hover': {
                  boxShadow: '0 12px 32px rgba(0, 0, 0, 0.4)',
                  border: '3px solid rgba(59, 130, 246, 0.7)'
                }
              }}>
                <CardMedia
                  component="img"
                  image={
                    activeImageTab === 0 
                      ? imagePreview
                      : activeImageTab === 1
                      ? (prediction.overlay_url ? `${API_BASE_URL}${prediction.overlay_url}` : imagePreview)
                      : (prediction.mask_url ? `${API_BASE_URL}${prediction.mask_url}` : imagePreview)
                  }
                  alt={
                    activeImageTab === 0 ? 'Original Eye Image' : 
                    activeImageTab === 1 ? 'GradCAM Heatmap Overlay' : 
                    'Disease Regions Outlined'
                  }
                  sx={{ 
                    height: 450, 
                    objectFit: 'contain',
                    backgroundColor: '#000',
                    imageRendering: 'crisp-edges'
                  }}
                />
              </Card>

              <Box sx={{ 
                mt: 2, 
                p: 2.5, 
                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.15) 100%)', 
                borderRadius: 3, 
                border: '2px solid rgba(59, 130, 246, 0.4)', 
                position: 'relative', 
                zIndex: 1,
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}>
                <Typography 
                  variant="body1" 
                  align="center"
                  sx={{ 
                    fontWeight: 700, 
                    color: 'white', 
                    textShadow: '2px 2px 4px rgba(0,0,0,0.5)',
                    fontSize: '1.05rem',
                    lineHeight: 1.6
                  }}
                >
                  {activeImageTab === 0 && '📷 Original Retinal Image - High-resolution view for detailed analysis'}
                  {activeImageTab === 1 && '🔥 AI Heatmap Overlay - Red/yellow areas show where the AI detected disease patterns'}
                  {activeImageTab === 2 && '🎯 Disease-Affected Areas - Colored outlines highlight specific regions with abnormalities'}
                </Typography>
              </Box>
            </Paper>

            {/* AI-Powered Day-to-Day Recommendations */}
            <Paper elevation={6} sx={{ p: 3, borderRadius: 4, background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)', border: '2px solid #f59e0b' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <MedicalServices sx={{ color: '#d97706', fontSize: 32 }} />
                <Typography variant="h5" sx={{ fontWeight: 700, color: '#92400e' }}>
                  🤖 AI-Powered Recommendations
                </Typography>
              </Box>
              <Divider sx={{ mb: 2, backgroundColor: '#f59e0b', height: 2 }} />
              
              {prediction.ai_recommendations ? (
                <>
                  <Typography 
                    variant="body1" 
                    component="div" 
                    sx={{ mb: 2, lineHeight: 1.8, color: '#92400e', fontWeight: 500 }}
                  >
                    Personalized AI-generated recommendations for <strong>{prediction.prediction || 'Unknown'}</strong>:
                  </Typography>
                  
                  {/* Daily Tips */}
                  <Typography variant="h6" sx={{ fontWeight: 700, color: '#92400e', mt: 2, mb: 1 }}>
                    📋 Daily Care Tips:
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, color: '#92400e', '& li': { mb: 1 } }}>
                    {prediction.ai_recommendations.daily_tips?.map((tip, index) => (
                      <li key={index}><strong>{tip}</strong></li>
                    ))}
                  </Box>
                  
                  {/* Medical Advice */}
                  {prediction.ai_recommendations.medical_advice && (
                    <>
                      <Typography variant="h6" sx={{ fontWeight: 700, color: '#92400e', mt: 2, mb: 1 }}>
                        🏥 Medical Advice:
                      </Typography>
                      <Typography variant="body1" sx={{ color: '#92400e', fontWeight: 500, mb: 2 }}>
                        {prediction.ai_recommendations.medical_advice}
                      </Typography>
                    </>
                  )}
                  
                  {/* Lifestyle Changes */}
                  {prediction.ai_recommendations.lifestyle_changes && prediction.ai_recommendations.lifestyle_changes.length > 0 && (
                    <>
                      <Typography variant="h6" sx={{ fontWeight: 700, color: '#92400e', mt: 2, mb: 1 }}>
                        🌟 Lifestyle Changes:
                      </Typography>
                      <Box component="ul" sx={{ pl: 3, color: '#92400e', '& li': { mb: 1 } }}>
                        {prediction.ai_recommendations.lifestyle_changes.map((change, index) => (
                          <li key={index}>{change}</li>
                        ))}
                      </Box>
                    </>
                  )}
                  
                  {/* Warning Signs */}
                  {prediction.ai_recommendations.warning_signs && prediction.ai_recommendations.warning_signs.length > 0 && (
                    <>
                      <Typography variant="h6" sx={{ fontWeight: 700, color: '#dc2626', mt: 2, mb: 1 }}>
                        ⚠️ Warning Signs to Watch For:
                      </Typography>
                      <Box component="ul" sx={{ pl: 3, color: '#92400e', '& li': { mb: 1, fontWeight: 600 } }}>
                        {prediction.ai_recommendations.warning_signs.map((sign, index) => (
                          <li key={index}>{sign}</li>
                        ))}
                      </Box>
                    </>
                  )}
                </>
              ) : (
                <>
                  <Typography 
                    variant="body1" 
                    component="div" 
                    sx={{ mb: 2, lineHeight: 1.8, color: '#92400e', fontWeight: 500 }}
                  >
                    Based on the diagnosis of <strong>{prediction.prediction || 'Unknown'}</strong>, please follow these recommendations:
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, color: '#92400e', '& li': { mb: 1 } }}>
                    <li><strong>Consult an ophthalmologist</strong> for professional medical advice and treatment</li>
                    <li><strong>Regular eye checkups</strong> to monitor disease progression</li>
                    <li><strong>Maintain healthy lifestyle</strong> with proper diet and exercise</li>
                    <li><strong>Protect your eyes</strong> from excessive screen time and UV exposure</li>
                    <li><strong>Follow prescribed medications</strong> and treatment plans</li>
                  </Box>
                </>
              )}
              <Alert 
                severity="warning" 
                sx={{ 
                  mt: 2, 
                  borderRadius: 2,
                  background: 'rgba(255, 255, 255, 0.8)',
                  color: '#92400e',
                  border: '2px solid #f59e0b',
                  fontWeight: 600
                }}
              >
                <strong>⚠️ Disclaimer:</strong> This AI analysis is for educational purposes only. 
                Always consult a qualified ophthalmologist for medical advice and treatment.
              </Alert>
              
              {/* Download PDF Button */}
              {prediction.report_filename && (
                <Box sx={{ mt: 3 }}>
                  <Alert 
                    severity="info" 
                    sx={{ 
                      mb: 2, 
                      borderRadius: 2,
                      background: 'rgba(59, 130, 246, 0.1)',
                      color: '#1e40af',
                      border: '2px solid #3b82f6',
                      fontWeight: 600
                    }}
                  >
                    <strong>📋 Complete Medical Report Available:</strong> Download a comprehensive PDF report with patient details, diagnosis, visualizations, and recommendations.
                  </Alert>
                  <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                    <Button
                      variant="contained"
                      size="large"
                      startIcon={<PictureAsPdf />}
                      onClick={() => {
                        window.open(`${API_BASE_URL}/api/reports/${prediction.report_filename}`, '_blank');
                      }}
                      sx={{ 
                        minWidth: 280, 
                        py: 1.8,
                        borderRadius: 3,
                        fontWeight: 700,
                        fontSize: '1.1rem',
                        background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                        boxShadow: '0 8px 24px rgba(239, 68, 68, 0.4)',
                        textTransform: 'none',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)',
                          boxShadow: '0 10px 32px rgba(239, 68, 68, 0.5)',
                          transform: 'translateY(-3px)'
                        },
                        transition: 'all 0.3s ease'
                      }}
                    >
                      📄 Download Complete PDF Report
                    </Button>
                  </Box>
                </Box>
              )}
            </Paper>
          </Box>
        ) : (
          <Paper elevation={4} sx={{ p: 4, textAlign: 'center', height: '100%', borderRadius: 3, background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)' }}>
            <Box sx={{ 
              width: 100, 
              height: 100, 
              borderRadius: '50%', 
              background: '#dbeafe',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1.5rem',
              border: '2px dashed #0ea5e9'
            }}>
              <Visibility sx={{ fontSize: 50, color: '#0ea5e9' }} />
            </Box>
            <Typography variant="h5" sx={{ fontWeight: 600, mb: 1, color: '#0c4a6e' }}>
              Upload an Image to Begin
            </Typography>
            <Typography variant="body1" sx={{ fontWeight: 400, maxWidth: 500, margin: '0 auto', color: '#0c4a6e' }}>
              Our AI will detect and classify eye diseases with medical-grade accuracy
            </Typography>
            <Typography variant="body2" sx={{ mt: 2, fontSize: '0.9rem', color: '#0c4a6e' }}>
              Select a retinal image to get started
            </Typography>
          </Paper>
        )}
      </Grid>
    </Grid>
  );
};

export default PredictionPanel;
