import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Typography, Paper, Box, Chip } from '@mui/material';
import { CloudUpload, Image, PhotoCamera } from '@mui/icons-material';

const ImageUploader = ({ onImageSelect }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      onImageSelect(acceptedFiles[0]);
    }
  }, [onImageSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: false
  });

  return (
    <Paper
      {...getRootProps()}
      sx={{
        p: 4,
        textAlign: 'center',
        cursor: 'pointer',
        border: '3px dashed',
        borderColor: isDragActive ? '#0ea5e9' : '#94a3b8',
        bgcolor: isDragActive ? '#f0f9ff' : '#f8fafc',
        transition: 'all 0.3s ease',
        borderRadius: 3,
        boxShadow: isDragActive ? '0 10px 25px rgba(14, 165, 233, 0.2)' : '0 4px 12px rgba(0, 0, 0, 0.05)',
        '&:hover': {
          borderColor: '#0ea5e9',
          bgcolor: '#f0f9ff',
          boxShadow: '0 8px 20px rgba(14, 165, 233, 0.15)'
        }
      }}
      elevation={isDragActive ? 4 : 1}
    >
      <input {...getInputProps()} />
      
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        mb: 2,
        animation: isDragActive ? 'pulse 1.5s infinite' : 'none'
      }}>
        <CloudUpload sx={{ 
          fontSize: 70, 
          color: isDragActive ? '#0ea5e9' : '#64748b',
          transition: 'all 0.3s ease'
        }} />
      </Box>
      
      <Typography 
        variant="h5" 
        gutterBottom 
        sx={{ 
          fontWeight: 700, 
          color: isDragActive ? '#0c4a6e' : '#334155',
          mb: 1
        }}
      >
        {isDragActive ? 'Drop the Image Here' : 'Upload Retinal Image'}
      </Typography>
      
      <Typography 
        variant="body1" 
        color="text.secondary" 
        sx={{ mb: 2 }}
      >
        {isDragActive ? 'Release to upload' : 'Drag & drop or click to select file'}
      </Typography>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, flexWrap: 'wrap', mb: 2 }}>
        <Chip 
          icon={<Image />} 
          label="PNG" 
          size="small" 
          sx={{ 
            background: 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)',
            color: '#1e40af',
            fontWeight: 'bold'
          }} 
        />
        <Chip 
          icon={<Image />} 
          label="JPG" 
          size="small" 
          sx={{ 
            background: 'linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)',
            color: '#14532d',
            fontWeight: 'bold'
          }} 
        />
        <Chip 
          icon={<Image />} 
          label="JPEG" 
          size="small" 
          sx={{ 
            background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
            color: '#92400e',
            fontWeight: 'bold'
          }} 
        />
      </Box>
      
      <Typography 
        variant="caption" 
        display="block" 
        sx={{ 
          mt: 1, 
          color: isDragActive ? '#0c4a6e' : '#64748b',
          fontWeight: 500
        }}
      >
        <PhotoCamera sx={{ fontSize: 16, verticalAlign: 'middle', mr: 0.5 }} />
        High-quality retinal images recommended for best results
      </Typography>
    </Paper>
  );
};

export default ImageUploader;