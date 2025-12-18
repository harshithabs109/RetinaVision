# 🚀 Deploy RetinaVision to Render

This guide will help you deploy your RetinaVision application to Render with both backend and frontend services.

## 📋 Prerequisites

1. GitHub account with your code pushed to repository
2. Render account (sign up at https://render.com)
3. OpenAI API key
4. PostgreSQL database (will be created on Render)

## 🗄️ Step 1: Create PostgreSQL Database

1. Go to Render Dashboard: https://dashboard.render.com
2. Click **"New +"** → **"PostgreSQL"**
3. Configure database:
   - **Name**: `retinavision-db`
   - **Database**: `retinavision`
   - **User**: `retinavision_user`
   - **Region**: Singapore (or closest to you)
   - **Plan**: Free (or paid for better performance)
4. Click **"Create Database"**
5. Wait for database to be created
6. **Copy the Internal Database URL** (starts with `postgresql://`)

## 🔧 Step 2: Deploy Backend API

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Configure service:
   - **Name**: `retinavision-backend`
   - **Region**: Singapore
   - **Branch**: `main`
   - **Root Directory**: Leave empty
   - **Runtime**: Python 3
   - **Build Command**: 
     ```bash
     cd backend && pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     cd backend && gunicorn app:app
     ```
   - **Plan**: Free (or paid for better performance)

4. **Add Environment Variables**:
   Click "Advanced" → "Add Environment Variable"
   
   Add these variables:
   ```
   DATABASE_URL = [paste your PostgreSQL Internal URL from Step 1]
   OPENAI_API_KEY = [your OpenAI API key]
   FLASK_ENV = production
   FLASK_DEBUG = False
   PYTHON_VERSION = 3.9.0
   ```

5. Click **"Create Web Service"**
6. Wait for deployment (5-10 minutes for first deploy)
7. **Copy your backend URL** (e.g., `https://retinavision-backend.onrender.com`)

## 🎨 Step 3: Deploy Frontend

1. Click **"New +"** → **"Static Site"**
2. Connect your GitHub repository
3. Configure service:
   - **Name**: `retinavision-frontend`
   - **Region**: Singapore
   - **Branch**: `main`
   - **Root Directory**: Leave empty
   - **Build Command**: 
     ```bash
     cd frontend && npm install && npm run build
     ```
   - **Publish Directory**: 
     ```
     frontend/build
     ```

4. **Add Environment Variable**:
   ```
   https://retinavision.onrender.com
   ```
   Example: `https://retinavision-backend.onrender.com`

5. Click **"Create Static Site"**
6. Wait for deployment (3-5 minutes)

## 🔄 Step 4: Initialize Database

After backend is deployed, initialize the database:

1. Go to your backend service on Render
2. Click **"Shell"** tab
3. Run these commands:
   ```bash
   cd backend
   python init_database.py
   ```

This will create all necessary tables in your PostgreSQL database.

## ✅ Step 5: Test Your Deployment

1. Open your frontend URL (e.g., `https://retinavision-frontend.onrender.com`)
2. Upload a test retinal image
3. Check if predictions work
4. Verify PDF generation
5. Test patient management features

## 🔍 Troubleshooting

### Backend Issues

**Check Logs**:
- Go to backend service → "Logs" tab
- Look for errors during startup or requests

**Common Issues**:

1. **Database Connection Error**:
   - Verify DATABASE_URL is correct
   - Check database is running
   - Ensure database is in same region

2. **Model Loading Error**:
   - Model file (`my_modeltrained3.h5`) must be in repository
   - Check file size (Render has limits)
   - Consider using external storage for large models

3. **OpenAI API Error**:
   - Verify OPENAI_API_KEY is set correctly
   - Check API key has credits
   - Ensure key has proper permissions

### Frontend Issues

**Check Build Logs**:
- Go to frontend service → "Logs" tab
- Look for build errors

**Common Issues**:

1. **API Connection Error**:
   - Verify REACT_APP_API_URL is correct
   - Must include `https://` prefix
   - No trailing slash

2. **CORS Error**:
   - Backend has CORS enabled
   - Check backend logs for CORS errors

3. **Build Fails**:
   - Check Node.js version compatibility
   - Verify all dependencies in package.json

## 📊 Performance Tips

### Free Tier Limitations

- **Backend**: Spins down after 15 minutes of inactivity
- **First request**: May take 30-60 seconds (cold start)
- **Database**: Limited connections and storage

### Upgrade Recommendations

For production use, consider:
- **Backend**: Starter plan ($7/month) - No spin down
- **Database**: Starter plan ($7/month) - Better performance
- **Frontend**: Free tier is sufficient

## 🔐 Security Best Practices

1. **Never commit .env files** - Already in .gitignore
2. **Rotate API keys** regularly
3. **Use environment variables** for all secrets
4. **Enable HTTPS** (automatic on Render)
5. **Monitor logs** for suspicious activity

## 🔄 Continuous Deployment

Render automatically deploys when you push to GitHub:

1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Your changes"
   git push origin main
   ```
3. Render automatically detects changes and redeploys

## 📱 Custom Domain (Optional)

To use your own domain:

1. Go to service settings
2. Click "Custom Domains"
3. Add your domain
4. Update DNS records as instructed
5. SSL certificate is automatic

## 🆘 Support

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **GitHub Issues**: Create issue in your repository

## 🎉 Success!

Your RetinaVision app is now live! Share your deployment URL with others.

**Example URLs**:
- Frontend: `https://retinavision-frontend.onrender.com`
- Backend API: `https://retinavision-backend.onrender.com/api/health`

---

**Note**: First deployment may take 10-15 minutes. Subsequent deployments are faster (2-5 minutes).
