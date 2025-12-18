import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Tabs,
  Tab,
  Card,
  CardMedia,
  Alert,
  Box,
  Chip,
  Grid,
  Divider
} from '@mui/material';
import { Visibility, Thermostat, MyLocation, MedicalServices, Image as ImageIcon } from '@mui/icons-material';

const ImageViewer = ({ prediction, imagePreview }) => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Check if we have visualization data from prediction prop
  const visualizationData = prediction || null;
  
  if (!visualizationData) {
    return (
      <Paper elevation={3} sx={{ p: 3, mb: 3, borderRadius: 3 }}>
        <Alert severity="info">
          No visualization data available
        </Alert>
      </Paper>
    );
  }

  // Get API base URL from environment or default
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  return (
    <Paper 
      elevation={6} 
      sx={{ 
        p: 4, 
        mb: 3, 
        borderRadius: 4,
        background: 'white',
        border: '1px solid #e5e7eb'
      }}
    >
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, color: '#1e293b', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Visibility sx={{ color: '#0ea5e9', fontSize: 32 }} />
          AI Visualization Analysis
        </Typography>
        <Typography variant="body2" sx={{ color: '#64748b' }}>
          Three-view analysis showing original image, AI attention heatmap, and detected affected areas
        </Typography>
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Tab Navigation */}
      <Tabs 
        value={activeTab} 
        onChange={handleTabChange}
        variant="fullWidth"
        sx={{ 
          mb: 3,
          '& .MuiTab-root': {
            fontWeight: 600,
            fontSize: '1rem',
            py: 2,
            textTransform: 'none'
          },
          '& .MuiTabs-indicator': {
            height: 4,
            borderRadius: 2
          }
        }}
      >
        <Tab 
          icon={<ImageIcon />} 
          label="Original Image" 
          iconPosition="start"
        />
        <Tab 
          icon={<Thermostat />} 
          label="AI Heatmap" 
          iconPosition="start"
        />
        <Tab 
          icon={<MyLocation />} 
          label="Affected Areas" 
          iconPosition="start"
        />
      </Tabs>

      {/* Tab Content - Large Clear Images */}
      <Box sx={{ minHeight: 500 }}>
        {/* Tab 1: Original Image */}
        {activeTab === 0 && (
          <Box className="fade-in">
            <Card sx={{ 
              borderRadius: 3, 
              overflow: 'hidden', 
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
              border: '3px solid #0ea5e9',
              bgcolor: '#000'
            }}>
              <CardMedia
                component="img"
                image={imagePreview || visualizationData.image_preview}
                alt="Original Eye Image"
                sx={{ 
                  width: '100%',
                  height: 500, 
                  objectFit: 'contain'
                }}
              />
            </Card>
            <Box sx={{ mt: 2, p: 2, bgcolor: '#f0f9ff', borderRadius: 2, border: '1px solid #bae6fd' }}>
              <Typography variant="h6" sx={{ fontWeight: 600, color: '#0c4a6e', mb: 1 }}>
                📷 Original Retinal Image
              </Typography>
              <Typography variant="body2" sx={{ color: '#0369a1' }}>
                This is the original uploaded fundus image showing the retina. The AI model analyzes this image to detect eye diseases.
              </Typography>
            </Box>
          </Box>
        )}

        {/* Tab 2: Heatmap */}
        {activeTab === 1 && (
          <Box className="fade-in">
            <Card sx={{ 
              borderRadius: 3, 
              overflow: 'hidden', 
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
              border: '3px solid #ef4444',
              bgcolor: '#000'
            }}>
              <CardMedia
                component="img"
                image={visualizationData.heatmap_url ? `${API_BASE_URL}${visualizationData.heatmap_url}` : imagePreview}
                alt="GradCAM Heatmap"
                sx={{ 
                  width: '100%',
                  height: 500, 
                  objectFit: 'contain'
                }}
              />
            </Card>
            <Box sx={{ mt: 2, p: 2, bgcolor: '#fef2f2', borderRadius: 2, border: '1px solid #fecaca' }}>
              <Typography variant="h6" sx={{ fontWeight: 600, color: '#7f1d1d', mb: 1 }}>
                🔥 AI Attention Heatmap (GradCAM)
              </Typography>
              <Typography variant="body2" sx={{ color: '#991b1b', mb: 1 }}>
                Color-coded visualization showing where the AI model is focusing its attention:
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mt: 1 }}>
                <Chip label="🔴 Red = High Attention" sx={{ bgcolor: '#fee2e2', color: '#7f1d1d', fontWeight: 600 }} />
                <Chip label="🟡 Yellow = Medium Attention" sx={{ bgcolor: '#fef3c7', color: '#92400e', fontWeight: 600 }} />
                <Chip label="🟢 Green/Blue = Low Attention" sx={{ bgcolor: '#dcfce7', color: '#14532d', fontWeight: 600 }} />
              </Box>
            </Box>
          </Box>
        )}

        {/* Tab 3: Affected Areas */}
        {activeTab === 2 && (
          <Box className="fade-in">
            <Card sx={{ 
              borderRadius: 3, 
              overflow: 'hidden', 
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
              border: '3px solid #10b981',
              bgcolor: '#000'
            }}>
              <CardMedia
                component="img"
                image={visualizationData.mask_url ? `${API_BASE_URL}${visualizationData.mask_url}` : imagePreview}
                alt="Affected Areas"
                sx={{ 
                  width: '100%',
                  height: 500, 
                  objectFit: 'contain'
                }}
              />
            </Card>
            <Box sx={{ mt: 2, p: 2, bgcolor: '#f0fdf4', borderRadius: 2, border: '1px solid #bbf7d0' }}>
              <Typography variant="h6" sx={{ fontWeight: 600, color: '#14532d', mb: 1 }}>
                🎯 Detected Affected Areas
              </Typography>
              <Typography variant="body2" sx={{ color: '#15803d', mb: 1 }}>
                Colored contours and circles mark the specific regions where disease patterns are detected.
              </Typography>
              {visualizationData.disease_areas && visualizationData.disease_areas.length > 0 && (
                <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Chip 
                    label={`${visualizationData.disease_areas.length} Region${visualizationData.disease_areas.length > 1 ? 's' : ''} Detected`}
                    sx={{ bgcolor: '#dcfce7', color: '#14532d', fontWeight: 700, fontSize: '0.9rem' }}
                  />
                  <Chip 
                    label={`${(visualizationData.disease_areas.reduce((sum, area) => sum + (area.area_ratio || 0), 0) * 100).toFixed(1)}% Area Affected`}
                    sx={{ bgcolor: '#dbeafe', color: '#1e40af', fontWeight: 700, fontSize: '0.9rem' }}
                  />
                </Box>
              )}
            </Box>
          </Box>
        )}
      </Box>

      {/* Analysis Summary */}
      {visualizationData.disease_areas && visualizationData.disease_areas.length > 0 && (
        <Box sx={{ mt: 3, p: 3, bgcolor: '#f8fafc', borderRadius: 3, border: '2px solid #cbd5e1' }}>
          <Typography variant="h6" sx={{ fontWeight: 700, color: '#1e293b', mb: 2 }}>
            📊 Detailed Analysis Summary
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={4}>
              <Box sx={{ p: 2, bgcolor: 'white', borderRadius: 2, border: '1px solid #e2e8f0' }}>
                <Typography variant="body2" sx={{ color: '#64748b', mb: 0.5 }}>
                  Affected Regions
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, color: '#0ea5e9' }}>
                  {visualizationData.disease_areas.length}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box sx={{ p: 2, bgcolor: 'white', borderRadius: 2, border: '1px solid #e2e8f0' }}>
                <Typography variant="body2" sx={{ color: '#64748b', mb: 0.5 }}>
                  Total Affected Area
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, color: '#f59e0b' }}>
                  {(visualizationData.disease_areas.reduce((sum, area) => sum + (area.area_ratio || 0), 0) * 100).toFixed(1)}%
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box sx={{ p: 2, bgcolor: 'white', borderRadius: 2, border: '1px solid #e2e8f0' }}>
                <Typography variant="body2" sx={{ color: '#64748b', mb: 0.5 }}>
                  Average Intensity
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, color: '#10b981' }}>
                  {(visualizationData.disease_areas.reduce((sum, area) => sum + (area.score || 0), 0) / visualizationData.disease_areas.length * 100).toFixed(0)}%
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Box>
      )}
    </Paper>
  );
};

export default ImageViewer;