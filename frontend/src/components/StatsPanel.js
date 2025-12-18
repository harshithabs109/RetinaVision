import React, { useEffect } from 'react';
import {
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Divider,
  CircularProgress,
  IconButton,
  Tooltip as MuiTooltip
} from '@mui/material';
import { Refresh } from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d', '#ffc658', '#ff7c7c'];

const StatsPanel = ({ stats, refreshKey, onRefresh }) => {
  useEffect(() => {
    console.log('📊 StatsPanel: refreshKey changed to', refreshKey);
    // Trigger refresh when refreshKey changes
    if (refreshKey > 0 && onRefresh) {
      console.log('📊 StatsPanel: calling onRefresh');
      onRefresh();
    }
  }, [refreshKey, onRefresh]);

  if (!stats) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
        <CircularProgress />
      </Box>
    );
  }

  // Define the 8 specific diseases in capital letters
  const specificDiseases = [
    'CATARACT',
    'CHOROIDAL NEOVASCULARIZATION',
    'DIABETIC MACULAR EDEMA',
    'DIABETIC RETINOPATHY',
    'DRUSEN',
    'GLAUCOMA',
    'NORMAL',
    'NORMAL-1'
  ];

  // Prepare data for charts - only show the 8 specific diseases
  const diseaseData = Object.entries(stats.disease_distribution || {})
    .filter(([name, count]) => specificDiseases.includes(name.toUpperCase()) && count > 0)
    .map(([name, count]) => ({
      name: name.toUpperCase(),
      count
    }));

  const pieData = diseaseData.map(item => ({
    name: item.name,
    value: item.count
  }));

  return (
    <Box className="fade-in">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
          📊 Detailed Statistics & Analytics
        </Typography>
        {onRefresh && (
          <MuiTooltip title="Refresh Statistics">
            <IconButton 
              onClick={onRefresh} 
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
          </MuiTooltip>
        )}
      </Box>
      
      <Grid container spacing={4}>
      {/* Overview Cards */}
      <Grid item xs={12} md={4}>
        <Card sx={{ 
          height: '100%', 
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
          color: 'white',
          borderRadius: 3,
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 12px 24px rgba(0, 0, 0, 0.15)'
          }
        }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Box sx={{ 
                width: 56, 
                height: 56, 
                borderRadius: '50%', 
                background: 'rgba(255, 255, 255, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mr: 2
              }}>
                <BarChart sx={{ fontSize: 32 }} />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                📊 Total Predictions
              </Typography>
            </Box>
            <Typography variant="h2" sx={{ fontWeight: 700, mb: 1 }}>
              {stats.total_predictions}
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9, fontWeight: 500 }}>
              All time medical analyses
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={4}>
        <Card sx={{ 
          height: '100%', 
          background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', 
          color: 'white',
          borderRadius: 3,
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 12px 24px rgba(0, 0, 0, 0.15)'
          }
        }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Box sx={{ 
                width: 56, 
                height: 56, 
                borderRadius: '50%', 
                background: 'rgba(255, 255, 255, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mr: 2
              }}>
                <BarChart sx={{ fontSize: 32 }} />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                🎯 Average Confidence
              </Typography>
            </Box>
            <Typography variant="h2" sx={{ fontWeight: 700, mb: 1 }}>
              {(stats.average_confidence * 100).toFixed(2)}%
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9, fontWeight: 500 }}>
              Model prediction accuracy
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={4}>
        <Card sx={{ 
          height: '100%', 
          background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', 
          color: 'white',
          borderRadius: 3,
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 12px 24px rgba(0, 0, 0, 0.15)'
          }
        }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Box sx={{ 
                width: 56, 
                height: 56, 
                borderRadius: '50%', 
                background: 'rgba(255, 255, 255, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mr: 2
              }}>
                <BarChart sx={{ fontSize: 32 }} />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                🏥 Disease Classes
              </Typography>
            </Box>
            <Typography variant="h2" sx={{ fontWeight: 700, mb: 1 }}>
              {Object.keys(stats.disease_distribution || {}).length}
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9, fontWeight: 500 }}>
              Detectable conditions
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* Disease Distribution Bar Chart */}
      <Grid item xs={12} md={8}>
        <Paper elevation={4} sx={{ p: 4, height: 550, borderRadius: 3 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3, display: 'flex', alignItems: 'center' }}>
            <BarChart sx={{ mr: 1.5, color: '#2563eb' }} />
            Disease Distribution Analysis
          </Typography>
          <Divider sx={{ mb: 3 }} />
          <ResponsiveContainer width="100%" height="85%">
            <BarChart data={diseaseData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis 
                dataKey="name" 
                angle={-45}
                textAnchor="end"
                height={120}
                interval={0}
                tick={{ fontSize: 12, fill: '#475569' }}
              />
              <YAxis tick={{ fill: '#475569' }} />
              <Tooltip 
                contentStyle={{ 
                  borderRadius: 12, 
                  border: '1px solid #e2e8f0',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Legend />
              <Bar dataKey="count" fill="#2563eb" name="Number of Cases" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Disease Distribution Pie Chart */}
      <Grid item xs={12} md={4}>
        <Paper elevation={4} sx={{ p: 4, height: 550, borderRadius: 3 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3, display: 'flex', alignItems: 'center' }}>
            <BarChart sx={{ mr: 1.5, color: '#2563eb' }} />
            Disease Proportion
          </Typography>
          <Divider sx={{ mb: 3 }} />
          <ResponsiveContainer width="100%" height="85%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={true}
                label={({ name, percent }) => `${name.substring(0, 15)}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                paddingAngle={2}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  borderRadius: 12, 
                  border: '1px solid #e2e8f0',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Disease Details Table */}
      <Grid item xs={12}>
        <Paper elevation={4} sx={{ p: 4, borderRadius: 3 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3, display: 'flex', alignItems: 'center' }}>
            <BarChart sx={{ mr: 1.5, color: '#2563eb' }} />
            Detailed Disease Statistics
          </Typography>
          <Divider sx={{ mb: 3 }} />
          <Grid container spacing={3}>
            {diseaseData.map((disease, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card sx={{ 
                  borderLeft: `5px solid ${COLORS[index % COLORS.length]}`,
                  height: '100%',
                  borderRadius: 2,
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-3px)',
                    boxShadow: '0 8px 16px rgba(0, 0, 0, 0.12)'
                  }
                }}>
                  <CardContent>
                    <Typography variant="subtitle1" color="text.secondary" gutterBottom sx={{ fontWeight: 600 }}>
                      {disease.name}
                    </Typography>
                    <Typography variant="h3" sx={{ fontWeight: 700, my: 1 }}>
                      {disease.count}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
                      {stats.total_predictions > 0 
                        ? `${((disease.count / stats.total_predictions) * 100).toFixed(1)}% of total`
                        : '0% of total'
                      }
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>
    </Grid>
    </Box>
  );
};

export default StatsPanel;
