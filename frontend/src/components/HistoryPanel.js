import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Chip,
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Card,
  CardMedia,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip,
  TextField,
  InputAdornment,
  Grid
} from '@mui/material';
import {
  Visibility,
  Refresh,
  Search,
  Clear,
  PictureAsPdf,
  History,
  BarChart
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const HistoryPanel = ({ refreshKey }) => {
  const [history, setHistory] = useState([]);
  const [filteredHistory, setFilteredHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [imageTab, setImageTab] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    console.log('📊 HistoryPanel: refreshKey changed to', refreshKey, '- fetching history');
    fetchHistory();
  }, [refreshKey]);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      // Use new database endpoint
      const response = await axios.get(`${API_BASE_URL}/api/history`);
      const predictions = response.data.predictions || [];
      // Normalize records to expected history shape
      const preds = predictions.map((pred) => ({
        id: pred.id,
        disease_class: pred.disease,
        confidence: pred.confidence * 100, // Convert to percentage
        created_at: pred.created_at,
        patient_id: pred.patient_id,
        report_filename: pred.report_filename
      }));
      setHistory(preds);
      setFilteredHistory(preds);
      setError(null);
    } catch (err) {
      setError('Failed to fetch history from database');
      console.error('History error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (query) => {
    setSearchQuery(query);
    
    if (!query.trim()) {
      // If no search query, show all history
      setFilteredHistory(history);
      return;
    }

    // Client-side filtering
    const lowercaseQuery = query.toLowerCase();
    let filtered = history.filter((item) => {
      // Search by ID
      if (item.id.toString().includes(query)) return true;
      
      // Search by disease class
      if (item.disease_class.toLowerCase().includes(lowercaseQuery)) return true;
      
      // Search by date
      if (formatDate(item.created_at).toLowerCase().includes(lowercaseQuery)) return true;
      
      // Search by confidence
      if (item.confidence.toString().includes(query)) return true;
      
      // Search by patient ID
      if (item.patient_id && item.patient_id.toLowerCase().includes(lowercaseQuery)) return true;
      
      return false;
    });
    
    setFilteredHistory(filtered);
  };

  const handleClearSearch = async () => {
    setSearchQuery('');
    
    // Apply date filter if set
    if (startDate && endDate) {
      await handleDateFilter();
    } else {
      // Reload all history
      await fetchHistory();
    }
  };

  const fetchPredictionDetail = (id) => {
    try {
      setDetailLoading(true);
      // Find prediction in current history
      const prediction = history.find(p => p.id === id);
      if (prediction) {
        setSelectedPrediction(prediction);
        setDetailOpen(true);
      } else {
        alert('Prediction not found');
      }
    } catch (err) {
      console.error('Detail fetch error:', err);
      alert('Failed to fetch prediction details');
    } finally {
      setDetailLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'success';
    if (confidence >= 75) return 'warning';
    return 'error';
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    
    try {
      // Parse the date string (handles both ISO format and other formats)
      const date = new Date(dateString);
      
      // Check if date is valid
      if (isNaN(date.getTime())) return 'Invalid Date';
      
      // Format to local timezone with readable format
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
      });
    } catch (error) {
      console.error('Date formatting error:', error);
      return 'Invalid Date';
    }
  };

  const handleClose = () => {
    setDetailOpen(false);
    setSelectedPrediction(null);
    setImageTab(0);
  };

  const handleDateFilter = () => {
    if (!startDate || !endDate) {
      alert('Please select both start and end dates');
      return;
    }

    // Client-side date filtering
    const start = new Date(startDate);
    const end = new Date(endDate);
    end.setHours(23, 59, 59, 999); // Include the entire end date
    
    const filtered = history.filter((item) => {
      const itemDate = new Date(item.created_at);
      return itemDate >= start && itemDate <= end;
    });
    
    setFilteredHistory(filtered);
  };

  const clearDateFilter = () => {
    setStartDate('');
    setEndDate('');
    setFilteredHistory(history);
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  return (
    <Paper elevation={4} sx={{ p: 4, borderRadius: 3 }} className="fade-in">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
          📚 Prediction History
        </Typography>
        <Tooltip title="Refresh">
          <IconButton 
            onClick={fetchHistory} 
            color="primary"
            sx={{
              background: '#eff6ff',
              '&:hover': {
                background: '#dbeafe'
              }
            }}
          >
            <Refresh />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Search Bar and Filters */}
      <Box sx={{ mb: 4 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              placeholder="Search by ID, Disease, Date, or Confidence..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              variant="outlined"
              size="medium"
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                  bgcolor: 'white'
                }
              }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search sx={{ color: '#94a3b8' }} />
                  </InputAdornment>
                ),
                endAdornment: searchQuery && (
                  <InputAdornment position="end">
                    <IconButton 
                      size="small" 
                      onClick={() => setSearchQuery('')}
                      sx={{ color: '#94a3b8' }}
                    >
                      <Clear />
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                label="Start Date"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                InputLabelProps={{ shrink: true }}
                variant="outlined"
                size="medium"
                sx={{ flex: 1 }}
              />
              <TextField
                label="End Date"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                InputLabelProps={{ shrink: true }}
                variant="outlined"
                size="medium"
                sx={{ flex: 1 }}
              />
              <Button
                variant="contained"
                onClick={handleDateFilter}
                sx={{ 
                  py: 1.5,
                  px: 3,
                  borderRadius: 2,
                  fontWeight: 600,
                  background: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)'
                  }
                }}
              >
                Filter
              </Button>
              <Button
                variant="outlined"
                onClick={clearDateFilter}
                sx={{ 
                  py: 1.5,
                  px: 3,
                  borderRadius: 2,
                  fontWeight: 600,
                  borderColor: '#0ea5e9',
                  color: '#0ea5e9',
                  '&:hover': {
                    borderColor: '#0369a1',
                    color: '#0369a1'
                  }
                }}
              >
                Clear
              </Button>
            </Box>
          </Grid>
        </Grid>
        {(searchQuery || (startDate && endDate)) && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', fontWeight: 500 }}>
            Found {filteredHistory.length} result{filteredHistory.length !== 1 ? 's' : ''}
          </Typography>
        )}
      </Box>

      <TableContainer sx={{ borderRadius: 2, boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)' }}>
        <Table>
          <TableHead>
            <TableRow sx={{ bgcolor: '#f1f5f9' }}>
              <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                <strong>ID</strong>
              </TableCell>
              <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                <strong>Disease</strong>
              </TableCell>
              <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                <strong>Confidence</strong>
              </TableCell>
              <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                <strong>Date & Time</strong>
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                <strong>Actions</strong>
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredHistory.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} align="center">
                  <Box sx={{ py: 6, textAlign: 'center' }}>
                    <Box sx={{ 
                      width: 80, 
                      height: 80, 
                      borderRadius: '50%', 
                      background: '#f1f5f9',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto 1.5rem'
                    }}>
                      <History sx={{ fontSize: 40, color: '#94a3b8' }} />
                    </Box>
                    <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
                      {searchQuery ? 'No results found' : 'No predictions yet'}
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
                      {searchQuery ? 'Try adjusting your search criteria' : 'Upload an image to get started.'}
                    </Typography>
                  </Box>
                </TableCell>
              </TableRow>
            ) : (
              filteredHistory.map((item) => (
                <TableRow 
                  key={item.id}
                  sx={{ 
                    '&:hover': { 
                      bgcolor: 'action.hover',
                      transform: 'scale(1.01)'
                    },
                    transition: 'all 0.2s ease'
                  }}
                >
                  <TableCell>
                    <Typography variant="body1" sx={{ fontWeight: 600 }}>
                      #{item.id}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body1" fontWeight="medium">
                      {item.disease_class}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip 
                      label={`${item.confidence}%`}
                      color={getConfidenceColor(item.confidence)}
                      size="medium"
                      sx={{ 
                        fontWeight: 600,
                        py: 1,
                        px: 1.5
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body1" sx={{ fontWeight: 500 }}>
                      {formatDate(item.created_at)}
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Button
                      size="medium"
                      startIcon={<Visibility />}
                      onClick={() => fetchPredictionDetail(item.id)}
                      variant="outlined"
                      sx={{ 
                        mr: 1.5,
                        py: 1,
                        px: 2,
                        borderRadius: 2,
                        fontWeight: 600,
                        borderWidth: 2
                      }}
                    >
                      View
                    </Button>
                    <Button
                      size="medium"
                      startIcon={<PictureAsPdf />}
                      onClick={() => window.open(`${API_BASE_URL}/api/generate-pdf/${item.id}`, '_blank')}
                      variant="contained"
                      color="success"
                      sx={{ 
                        py: 1,
                        px: 2,
                        borderRadius: 2,
                        fontWeight: 600,
                        boxShadow: '0 4px 12px rgba(34, 197, 94, 0.3)',
                        '&:hover': {
                          boxShadow: '0 6px 20px rgba(34, 197, 94, 0.4)'
                        }
                      }}
                    >
                      PDF
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Detail Dialog */}
      <Dialog 
        open={detailOpen} 
        onClose={handleClose}
        maxWidth="lg"
        fullWidth
        sx={{
          '& .MuiDialog-paper': {
            borderRadius: 3
          }
        }}
      >
        <DialogTitle sx={{ 
          fontWeight: 700, 
          fontSize: '1.5rem',
          pb: 2,
          borderBottom: '1px solid #e2e8f0'
        }}>
          📋 Prediction Details - ID #{selectedPrediction?.id}
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          {detailLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 8 }}>
              <CircularProgress size={60} thickness={4} />
            </Box>
          ) : selectedPrediction && (
            <Box>
              {/* Diagnosis Info */}
              <Paper sx={{ 
                p: 3, 
                mb: 4, 
                bgcolor: 'grey.50',
                borderRadius: 2,
                borderLeft: '5px solid #2563eb'
              }}>
                <Typography variant="h5" gutterBottom sx={{ fontWeight: 700, mb: 2 }}>
                  Diagnosis: {selectedPrediction.disease_class || selectedPrediction.predicted_disease || selectedPrediction.disease_name || 'Unknown'}
                </Typography>
                <Chip 
                  label={`Confidence: ${selectedPrediction.confidence}%`}
                  color={getConfidenceColor(selectedPrediction.confidence)}
                  size="medium"
                  sx={{ 
                    fontWeight: 600,
                    py: 1.5,
                    px: 2,
                    fontSize: '1.1rem'
                  }}
                />
                <Typography variant="body1" sx={{ mt: 2, fontWeight: 500 }}>
                  Analyzed: {formatDate(selectedPrediction.created_at)}
                </Typography>
              </Paper>

              {/* All Predictions */}
              <Paper sx={{ p: 3, mb: 4, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3, display: 'flex', alignItems: 'center' }}>
                  <BarChart sx={{ mr: 1.5, color: '#2563eb' }} />
                  All Class Probabilities
                </Typography>
                {Object.entries(selectedPrediction.all_predictions || selectedPrediction.predictions || {}).map(([name, conf]) => (
                  <Box key={name} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>{name}</Typography>
                      <Typography variant="body1" sx={{ fontWeight: 600 }}>
                        {typeof conf === 'object' ? conf.confidence?.toFixed(2) : conf.toFixed(2)}%
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: '#e2e8f0', borderRadius: 2, overflow: 'hidden', height: 12 }}>
                      <Box 
                        sx={{ 
                          bgcolor: (typeof conf === 'object' ? conf.confidence : conf) > 50 ? '#10b981' : '#2563eb', 
                          height: '100%',
                          width: `${typeof conf === 'object' ? conf.confidence : conf}%`,
                          borderRadius: 2,
                          transition: 'width 0.5s ease'
                        }} 
                      />
                    </Box>
                  </Box>
                ))}
              </Paper>

              {/* Images */}
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3, display: 'flex', alignItems: 'center' }}>
                  <Visibility sx={{ mr: 1.5, color: '#2563eb' }} />
                  AI Visualization
                </Typography>
                <Tabs 
                  value={imageTab} 
                  onChange={(e, newValue) => setImageTab(newValue)}
                  variant="fullWidth"
                  sx={{ 
                    mb: 3,
                    '& .MuiTab-root': {
                      py: 1.5,
                      fontWeight: 600,
                      fontSize: '1rem'
                    }
                  }}
                >
                  <Tab label="Original" sx={{ borderRadius: 1 }} />
                  <Tab label="Heatmap" sx={{ borderRadius: 1 }} />
                  <Tab label="Affected Areas" sx={{ borderRadius: 1 }} />
                </Tabs>

                <Card sx={{ 
                  borderRadius: 2, 
                  overflow: 'hidden', 
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                  border: '1px solid #e2e8f0'
                }}>
                  <CardMedia
                    component="img"
                    image={
                      imageTab === 0 
                        ? selectedPrediction.original_image || selectedPrediction.image_url || '/placeholder-image.png'
                        : imageTab === 1
                        ? selectedPrediction.heatmap_visualization || selectedPrediction.heatmap_url || '/placeholder-heatmap.png'
                        : selectedPrediction.disease_affected_areas_visualization || selectedPrediction.mask_url || selectedPrediction.affected_areas_visualization || selectedPrediction.overlay_url || '/placeholder-affected.png'
                    }
                    alt="Prediction visualization"
                    sx={{ maxHeight: 500, objectFit: 'contain', bgcolor: '#000' }}
                  />
                </Card>
              </Paper>
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ p: 3, borderTop: '1px solid #e2e8f0' }}>
          {selectedPrediction && (
            <Button
              startIcon={<PictureAsPdf />}
              variant="contained"
              color="success"
              onClick={() => {
                // Check if we have a report filename
                const reportFilename = selectedPrediction.report_filename || selectedPrediction.pdf_filename;
                if (reportFilename) {
                  window.open(`${API_BASE_URL}/api/reports/${reportFilename}`, '_blank');
                } else {
                  // Fallback to generating a new report
                  const visualizationId = selectedPrediction.visualization_id || selectedPrediction.id;
                  if (visualizationId) {
                    window.open(`${API_BASE_URL}/api/generate-pdf/${visualizationId}`, '_blank');
                  } else {
                    alert('No report available for this prediction');
                  }
                }
              }}
              sx={{ 
                py: 1.5,
                px: 3,
                borderRadius: 2,
                fontWeight: 600,
                fontSize: '1.1rem',
                boxShadow: '0 4px 12px rgba(34, 197, 94, 0.3)',
                '&:hover': {
                  boxShadow: '0 6px 20px rgba(34, 197, 94, 0.4)'
                }
              }}
            >
              Download PDF Report
            </Button>
          )}
          <Button 
            onClick={handleClose}
            variant="outlined"
            sx={{ 
              py: 1.5,
              px: 3,
              borderRadius: 2,
              fontWeight: 600,
              fontSize: '1.1rem',
              borderWidth: 2
            }}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default HistoryPanel;
