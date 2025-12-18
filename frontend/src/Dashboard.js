import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Box,
  Tabs,
  Tab,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  BarChart,
  CloudUpload,
  History,
  Person,
  CheckCircle,
  MedicalServices
} from '@mui/icons-material';
import axios from 'axios';
import PredictionPanel from './components/PredictionPanel';
import HistoryPanel from './components/HistoryPanel';
import StatsPanel from './components/StatsPanel';
import PatientPanel from './components/PatientPanel';
import './Dashboard.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function Dashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    fetchStats();
  }, [refreshKey]); // Re-fetch stats when refreshKey changes

  const fetchStats = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/stats`);
      setStats(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch statistics');
      console.error('Stats error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const refreshData = () => {
    console.log('🔄 Refreshing data - refreshKey will increment from', refreshKey);
    fetchStats();
    // Trigger refresh in child components
    const newRefreshKey = refreshKey + 1;
    console.log('🔄 New refreshKey will be:', newRefreshKey);
    setRefreshKey(newRefreshKey);
  };

  return (
    <div className="dashboard">
      {/* Header - Enhanced Attractive Design */}
      <Box sx={{ 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)',
        color: 'white',
        p: 5,
        mb: 4,
        borderRadius: '0 0 30px 30px',
        boxShadow: '0 10px 40px rgba(102, 126, 234, 0.3)',
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: '-50%',
          right: '-10%',
          width: '40%',
          height: '200%',
          background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)',
          animation: 'pulse 4s ease-in-out infinite'
        }
      }}>
        <Container maxWidth="xl" sx={{ position: 'relative', zIndex: 1 }}>
          <Typography variant="h2" gutterBottom sx={{ fontWeight: 800, mb: 1, textShadow: '2px 2px 8px rgba(0,0,0,0.2)' }}>
            🏥 Eye Disease Analysis Dashboard
          </Typography>
          <Typography variant="h5" sx={{ opacity: 0.95, fontWeight: 500, textShadow: '1px 1px 4px rgba(0,0,0,0.2)' }}>
            AI-Powered Medical Diagnosis System with Advanced Visualization
          </Typography>
        </Container>
      </Box>

      <Container maxWidth="xl">
        {/* Statistics Overview */}
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 8 }}>
            <CircularProgress size={60} thickness={4} />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 4, borderRadius: 3 }}>{error}</Alert>
        ) : (
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {/* Total Predictions Card */}
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ 
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                height: '100%',
                borderRadius: 4,
                boxShadow: '0 10px 30px rgba(102, 126, 234, 0.3)',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'radial-gradient(circle at top right, rgba(255,255,255,0.2) 0%, transparent 60%)',
                  pointerEvents: 'none'
                },
                '&:hover': {
                  transform: 'translateY(-8px) scale(1.02)',
                  boxShadow: '0 15px 40px rgba(102, 126, 234, 0.4)'
                }
              }}>
                <CardContent sx={{ position: 'relative', zIndex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ 
                      width: 56, 
                      height: 56, 
                      borderRadius: '50%', 
                      background: 'rgba(255, 255, 255, 0.2)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 2,
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                    }}>
                      <History sx={{ fontSize: 32 }} />
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Total Predictions
                    </Typography>
                  </Box>
                  <Typography variant="h2" sx={{ fontWeight: 700, mb: 1, textShadow: '2px 2px 4px rgba(0,0,0,0.2)' }}>
                    {stats?.total_predictions || 0}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9, fontWeight: 500 }}>
                    All time analyses
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Average Confidence Card */}
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ 
                background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                color: 'white',
                height: '100%',
                borderRadius: 4,
                boxShadow: '0 10px 30px rgba(240, 147, 251, 0.3)',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'radial-gradient(circle at top right, rgba(255,255,255,0.2) 0%, transparent 60%)',
                  pointerEvents: 'none'
                },
                '&:hover': {
                  transform: 'translateY(-8px) scale(1.02)',
                  boxShadow: '0 15px 40px rgba(240, 147, 251, 0.4)'
                }
              }}>
                <CardContent sx={{ position: 'relative', zIndex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ 
                      width: 56, 
                      height: 56, 
                      borderRadius: '50%', 
                      background: 'rgba(255, 255, 255, 0.2)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 2,
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                    }}>
                      <CheckCircle sx={{ fontSize: 32 }} />
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Avg Confidence
                    </Typography>
                  </Box>
                  <Typography variant="h2" sx={{ fontWeight: 700, mb: 1, textShadow: '2px 2px 4px rgba(0,0,0,0.2)' }}>
                    {stats?.average_confidence ? (stats.average_confidence * 100).toFixed(2) : '0.00'}%
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9, fontWeight: 500 }}>
                    Model accuracy
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Most Common Disease Card */}
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ 
                background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                color: 'white',
                height: '100%',
                borderRadius: 4,
                boxShadow: '0 10px 30px rgba(79, 172, 254, 0.3)',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'radial-gradient(circle at top right, rgba(255,255,255,0.2) 0%, transparent 60%)',
                  pointerEvents: 'none'
                },
                '&:hover': {
                  transform: 'translateY(-8px) scale(1.02)',
                  boxShadow: '0 15px 40px rgba(79, 172, 254, 0.4)'
                }
              }}>
                <CardContent sx={{ position: 'relative', zIndex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ 
                      width: 56, 
                      height: 56, 
                      borderRadius: '50%', 
                      background: 'rgba(255, 255, 255, 0.2)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 2,
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                    }}>
                      <MedicalServices sx={{ fontSize: 32 }} />
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Top Disease
                    </Typography>
                  </Box>
                  <Typography variant="h5" sx={{ fontWeight: 700, mb: 1, minHeight: 40, textShadow: '2px 2px 4px rgba(0,0,0,0.2)' }}>
                    {(stats?.disease_distribution && Object.entries(stats.disease_distribution)
                      .sort(([,a], [,b]) => b - a)[0]?.[0]) || 'N/A'}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9, fontWeight: 500 }}>
                    Most detected
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* System Status Card */}
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ 
                background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
                color: 'white',
                height: '100%',
                borderRadius: 4,
                boxShadow: '0 10px 30px rgba(67, 233, 123, 0.3)',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'radial-gradient(circle at top right, rgba(255,255,255,0.2) 0%, transparent 60%)',
                  pointerEvents: 'none'
                },
                '&:hover': {
                  transform: 'translateY(-8px) scale(1.02)',
                  boxShadow: '0 15px 40px rgba(67, 233, 123, 0.4)'
                }
              }}>
                <CardContent sx={{ position: 'relative', zIndex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ 
                      width: 56, 
                      height: 56, 
                      borderRadius: '50%', 
                      background: 'rgba(255, 255, 255, 0.2)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 2,
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                    }}>
                      <CheckCircle sx={{ fontSize: 32 }} />
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      System Status
                    </Typography>
                  </Box>
                  <Typography variant="h2" sx={{ fontWeight: 700, mb: 1, textShadow: '2px 2px 4px rgba(0,0,0,0.2)' }}>
                    ✓
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9, fontWeight: 500 }}>
                    Operational
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        {/* Main Dashboard Tabs */}
        <Paper elevation={4} sx={{ mb: 4, borderRadius: 3, overflow: 'hidden' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange}
            variant="fullWidth"
            sx={{ 
              borderBottom: 1, 
              borderColor: 'divider',
              '& .MuiTab-root': {
                py: 2,
                fontWeight: 600,
                fontSize: '1.1rem'
              },
              '& .MuiTabs-indicator': {
                height: 4,
                borderRadius: 2
              }
            }}
          >
            <Tab 
              icon={<CloudUpload />} 
              label="New Prediction" 
              iconPosition="start"
            />
            <Tab 
              icon={<BarChart />} 
              label="Statistics" 
              iconPosition="start"
            />
            <Tab 
              icon={<History />} 
              label="History" 
              iconPosition="start"
            />
            <Tab 
              icon={<Person />} 
              label="Patients" 
              iconPosition="start"
            />
          </Tabs>
        </Paper>

        {/* Tab Panels */}
        <Box className="fade-in">
          {activeTab === 0 && <PredictionPanel onPredictionComplete={refreshData} />}
          {activeTab === 1 && <StatsPanel stats={stats} refreshKey={refreshKey} onRefresh={fetchStats} />}
          {activeTab === 2 && <HistoryPanel refreshKey={refreshKey} />}
          {activeTab === 3 && <PatientPanel refreshKey={refreshKey} />}
        </Box>
      </Container>
    </div>
  );
}

export default Dashboard;
