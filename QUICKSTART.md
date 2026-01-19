# WebOptimizer AI - Quick Start Guide

## 🚀 Running the Application

### Step 1: Start the ML Server

**Option A: Using the batch file (Windows)**
```bash
# Double-click or run from terminal:
start_ml_server.bat
```

**Option B: Using Python directly**
```bash
# Activate virtual environment
source .venv/Scripts/activate  # Git Bash
# OR
.venv\Scripts\activate.bat  # CMD

# Start server
python src/api/ml_server.py
```

The server will start at `http://localhost:8000`

### Step 2: Start the Next.js Dev Server

In a **separate terminal**:

```bash
# Using yarn
yarn dev

# OR using npm
npm run dev
```

The app will be available at `http://localhost:3000`

## ✅ Testing the Setup

### 1. Test ML Server
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### 2. Test Website Analysis

1. Open `http://localhost:3000`
2. Enter any URL (e.g., `https://google.com`)
3. Click "Analyze Now"
4. Watch the progress bar
5. View results on dashboard

## 📊 Features

- **Progress Bar**: Shows real-time analysis progress (0-100%)
- **Real Lighthouse Audits**: Collects live Core Web Vitals for every run
- **ML Integration**: LightGBM model with 98.47% accuracy
- **Error Handling**: Automatic ML server health checks
- **Beautiful UI**: Modern gradient design with react-icons

## 🔧 Troubleshooting

### ML Server Not Running
**Error**: "ML server is not running"

**Fix**:
1. Check if port 8000 is available
2. Start the ML server: `python src/api/ml_server.py`
3. Verify with: `curl http://localhost:8000/health`

### Slow Analysis
Real Lighthouse audits run directly against Chrome. Ensure Google Chrome is installed and consider limiting concurrent requests. Installing Lighthouse globally is required:
```bash
npm install -g lighthouse
```

### Port Already in Use
```bash
# Kill process on port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

## 📁 Project Structure

```
src/
├── api/
│   └── ml_server.py         # ML server collecting live Lighthouse metrics
├── app/
│   ├── page.tsx             # Homepage with input
│   └── dashboard/
│       └── page.tsx         # Analysis dashboard
├── components/
│   ├── home/                # Homepage components
│   ├── dashboard/           # Dashboard components
│   │   ├── ProgressBar.tsx  # Progress indicator
│   │   ├── LoadingState.tsx # Loading screen
│   │   └── ...
│   └── shared/              # Shared components
├── store/
│   └── analysis.store.ts    # Zustand state management
└── lib/
    └── api/
        └── analysis.api.ts  # API client with progress
```

## 🎯 Tech Stack

- **Frontend**: Next.js 15, React, TypeScript, Tailwind CSS
- **State Management**: Zustand
- **Icons**: react-icons
- **HTTP Client**: Axios
- **Backend**: FastAPI (Python)
- **ML Model**: LightGBM (98.47% F1-score)

## 📝 Development Tips

1. **Progress Tracking**: Progress is simulated in steps (10% → 20% → 40% → 90% → 100%)
2. **Error Messages**: Clear error messages with actionable steps
3. **Code Quality**: All components are typed, optimized, and separated
4. **Performance**: Fast mock data for quick iterations during development

## 🔗 API Endpoints

### ML Server (Port 8000)

- `GET /` - Server info
- `GET /health` - Health check
- `POST /predict` - Analyze URL

### Next.js API (Port 3000)

- `POST /api/analyze` - Full analysis with recommendations

## 💡 Pro Tips

- Use the fast ML server during development
- Progress bar helps users understand wait time
- ML server health check prevents confusing errors
- All errors show helpful messages with solutions
