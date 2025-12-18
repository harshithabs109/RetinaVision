import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Box,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  InputAdornment,
  IconButton,
  Divider,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import { Person, Save, Search, Clear, Refresh, History, Close } from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const PatientPanel = ({ refreshKey }) => {
  const [formData, setFormData] = useState({
    name: '',
    age: '',
    dob: '',
    gender: '',
    contact: '',
    address: '',
    medical_history: '',
    medications: ''
  });
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState(null);
  const [createdPatientId, setCreatedPatientId] = useState(null);
  const [patients, setPatients] = useState([]);
  const [filteredPatients, setFilteredPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    console.log('👥 PatientPanel: refreshKey changed to', refreshKey, '- fetching patients');
    fetchPatients();
  }, [refreshKey]);

  const fetchPatients = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/patients`);
      setPatients(response.data.patients || []);
      setFilteredPatients(response.data.patients || []);
    } catch (err) {
      console.error('Failed to fetch patients:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (query) => {
    setSearchQuery(query);
    
    if (!query.trim()) {
      setFilteredPatients(patients);
      return;
    }

    const lowercaseQuery = query.toLowerCase();
    const filtered = patients.filter((patient) => {
      return (
        patient.patient_id.toLowerCase().includes(lowercaseQuery) ||
        patient.name.toLowerCase().includes(lowercaseQuery) ||
        patient.gender?.toLowerCase().includes(lowercaseQuery) ||
        patient.contact?.toLowerCase().includes(lowercaseQuery)
      );
    });
    
    setFilteredPatients(filtered);
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setFilteredPatients(patients);
  };

  const handleViewHistory = async (patientId) => {
    try {
      setHistoryLoading(true);
      setHistoryOpen(true);
      const response = await axios.get(`${API_BASE_URL}/api/patient/${patientId}`);
      setSelectedPatient(response.data.patient);
    } catch (err) {
      console.error('Failed to fetch patient history:', err);
      alert('Failed to load patient history');
      setHistoryOpen(false);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleCloseHistory = () => {
    setHistoryOpen(false);
    setSelectedPatient(null);
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

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'success';
    if (confidence >= 75) return 'warning';
    return 'error';
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
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
    return age;
  };

  const handleDOBChange = (e) => {
    const dob = e.target.value;
    const calculatedAge = calculateAge(dob);
    setFormData({
      ...formData,
      dob: dob,
      age: calculatedAge
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(`${API_BASE_URL}/api/patient`, formData);
      setSuccess(true);
      // backend returns `patient_id` key
      const pid = response.data.patient_id || response.data.patient_unique_id || null;
      setCreatedPatientId(pid);
      setError(null);
      setFormData({ name: '', age: '', dob: '', gender: '', contact: '', address: '', medical_history: '', medications: '' });
      
      // Refresh patient list
      fetchPatients();
      
      setTimeout(() => {
        setSuccess(false);
        setCreatedPatientId(null);
      }, 5000);
    } catch (err) {
      setError('Failed to create patient record');
      console.error('Patient creation error:', err);
    }
  };

  return (
    <Box className="fade-in">
      {/* Patient List - Removed "Add New Patient" section */}
    <Paper elevation={4} sx={{ p: 4, borderRadius: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
          👥 All Patients
        </Typography>
        <IconButton 
          onClick={fetchPatients} 
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
      </Box>

      {/* Search Bar */}
      <TextField
        fullWidth
        placeholder="Search by Patient ID, Name, Gender, or Contact..."
        value={searchQuery}
        onChange={(e) => handleSearch(e.target.value)}
        variant="outlined"
        size="medium"
        sx={{ 
          mb: 3,
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
                onClick={handleClearSearch}
                sx={{ color: '#94a3b8' }}
              >
                <Clear />
              </IconButton>
            </InputAdornment>
          ),
        }}
      />
      {searchQuery && (
        <Typography variant="caption" color="text.secondary" sx={{ mb: 3, display: 'block', fontWeight: 500 }}>
          Found {filteredPatients.length} result{filteredPatients.length !== 1 ? 's' : ''}
        </Typography>
      )}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 8 }}>
          <CircularProgress size={60} thickness={4} />
        </Box>
      ) : (
        <TableContainer sx={{ borderRadius: 2, boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)' }}>
          <Table>
            <TableHead>
              <TableRow sx={{ bgcolor: '#f1f5f9' }}>
                <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                  <strong>Patient ID</strong>
                </TableCell>
                <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                  <strong>Name</strong>
                </TableCell>
                <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                  <strong>Age</strong>
                </TableCell>
                <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                  <strong>Gender</strong>
                </TableCell>
                <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                  <strong>Contact</strong>
                </TableCell>
                <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                  <strong>Visits</strong>
                </TableCell>
                <TableCell align="center" sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                  <strong>Actions</strong>
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredPatients.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} align="center">
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
                        <Person sx={{ fontSize: 40, color: '#94a3b8' }} />
                      </Box>
                      <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
                        {searchQuery ? 'No patients found' : 'No patients yet'}
                      </Typography>
                      <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
                        {searchQuery ? 'Try adjusting your search criteria' : 'Add a new patient to get started.'}
                      </Typography>
                    </Box>
                  </TableCell>
                </TableRow>
              ) : (
                filteredPatients.map((patient) => (
                  <TableRow 
                    key={patient.patient_id}
                    sx={{ 
                      '&:hover': { 
                        bgcolor: 'action.hover',
                        transform: 'scale(1.01)'
                      },
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <TableCell>
                      <Chip 
                        label={patient.patient_id} 
                        color="primary" 
                        size="medium" 
                        variant="filled"
                        sx={{ 
                          fontWeight: 600,
                          py: 1,
                          px: 1.5
                        }}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>
                        {patient.name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body1">
                        {patient.age}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body1">
                        {patient.gender}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body1">
                        {patient.contact}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={patient.total_predictions} 
                        color="secondary" 
                        size="medium"
                        sx={{ 
                          fontWeight: 600,
                          py: 1,
                          px: 1.5
                        }}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Button
                        size="medium"
                        startIcon={<History />}
                        onClick={() => handleViewHistory(patient.patient_id)}
                        variant="outlined"
                        sx={{ 
                          py: 1,
                          px: 2,
                          borderRadius: 2,
                          fontWeight: 600,
                          borderWidth: 2
                        }}
                      >
                        View History
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>

    {/* Patient History Dialog */}
    <Dialog 
      open={historyOpen} 
      onClose={handleCloseHistory}
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
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
            📚 Patient Visit History
          </Typography>
          <IconButton 
            onClick={handleCloseHistory} 
            size="large"
            sx={{
              background: '#f1f5f9',
              '&:hover': {
                background: '#e2e8f0'
              }
            }}
          >
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent sx={{ pt: 3 }}>
        {historyLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 8 }}>
            <CircularProgress size={60} thickness={4} />
          </Box>
        ) : selectedPatient && (
          <Box>
            {/* Patient Information */}
            <Paper sx={{ 
              p: 3, 
              mb: 4, 
              bgcolor: '#eff6ff',
              borderRadius: 2,
              borderLeft: '5px solid #2563eb'
            }}>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 700, mb: 2 }}>
                Patient Information
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    Patient ID
                  </Typography>
                  <Typography variant="body1" fontWeight="bold" sx={{ fontSize: '1.1rem' }}>
                    {selectedPatient.patient_id}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    Name
                  </Typography>
                  <Typography variant="body1" fontWeight="bold" sx={{ fontSize: '1.1rem' }}>
                    {selectedPatient.name}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    Age
                  </Typography>
                  <Typography variant="body1" sx={{ fontSize: '1.1rem' }}>
                    {selectedPatient.age}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    Gender
                  </Typography>
                  <Typography variant="body1" sx={{ fontSize: '1.1rem' }}>
                    {selectedPatient.gender}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    Contact
                  </Typography>
                  <Typography variant="body1" sx={{ fontSize: '1.1rem' }}>
                    {selectedPatient.contact}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    Address
                  </Typography>
                  <Typography variant="body1" sx={{ fontSize: '1.1rem' }}>
                    {selectedPatient.address || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    Medical History
                  </Typography>
                  <Typography variant="body1" sx={{ fontSize: '1.1rem' }}>
                    {selectedPatient.medical_history || 'None'}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    Current Medications
                  </Typography>
                  <Typography variant="body1" sx={{ fontSize: '1.1rem' }}>
                    {selectedPatient.medications || 'None'}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    Total Visits
                  </Typography>
                  <Chip 
                    label={`${selectedPatient.total_predictions} visits`} 
                    color="primary" 
                    size="medium"
                    sx={{ 
                      fontWeight: 600,
                      py: 1.5,
                      px: 2,
                      fontSize: '1.1rem'
                    }}
                  />
                </Grid>
              </Grid>
            </Paper>

            {/* Visit History */}
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 700, mb: 3, display: 'flex', alignItems: 'center' }}>
              <History sx={{ mr: 1.5, color: '#2563eb' }} />
              Visit History
            </Typography>
            <Divider sx={{ mb: 3 }} />

            {selectedPatient.prediction_history && selectedPatient.prediction_history.length === 0 ? (
              <Alert 
                severity="info" 
                sx={{ 
                  borderRadius: 2,
                  fontWeight: 500
                }}
              >
                No visit history available for this patient.
              </Alert>
            ) : (
              <TableContainer sx={{ borderRadius: 2, boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)' }}>
                <Table>
                  <TableHead>
                    <TableRow sx={{ bgcolor: '#f1f5f9' }}>
                      <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                        <strong>Visit #</strong>
                      </TableCell>
                      <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                        <strong>Date & Time</strong>
                      </TableCell>
                      <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                        <strong>Diagnosis</strong>
                      </TableCell>
                      <TableCell sx={{ fontWeight: 600, fontSize: '1.1rem', py: 2 }}>
                        <strong>Confidence</strong>
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {selectedPatient.prediction_history?.map((visit, index) => (
                      <TableRow 
                        key={visit.id}
                        sx={{ 
                          '&:hover': { 
                            bgcolor: 'action.hover'
                          }
                        }}
                      >
                        <TableCell>
                          <Typography variant="body1" sx={{ fontWeight: 600 }}>
                            #{selectedPatient.prediction_history.length - index}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body1" sx={{ fontWeight: 500 }}>
                            {formatDate(visit.created_at)}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body1" fontWeight="medium">
                            {visit.disease_class}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={`${visit.confidence}%`}
                            color={getConfidenceColor(visit.confidence)}
                            size="medium"
                            sx={{ 
                              fontWeight: 600,
                              py: 1,
                              px: 1.5
                            }}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Box>
        )}
      </DialogContent>
      <DialogActions sx={{ p: 3, borderTop: '1px solid #e2e8f0' }}>
        <Button 
          onClick={handleCloseHistory}
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
  </Box>
  );
};

export default PatientPanel;
