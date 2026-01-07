# Quick Fix Guide - Progress Not Showing

## Issue
Progress bar stuck at 0% with "Initializing..." message.

## Root Cause
ML server not running or not accessible.

## Solution

### Step 1: Start ML Server

**Windows (CMD or PowerShell):**
```cmd
start_ml_server.bat
```

**Git Bash:**
```bash
bash start_ml_server.sh
```

**Manual:**
```bash
cd f:/client/Optimizer/optimizer
.venv/Scripts/python.exe src/api/ml_server_fast.py
```

### Step 2: Verify Server is Running

Open another terminal:
```bash
# Test with curl
curl http://localhost:8000/health

# OR open in browser
http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### Step 3: Test with HTML
Open `test_ml_server.html` in your browser and click "Test Health"

### Step 4: Check Browser Console
1. Open Browser DevTools (F12)
2. Go to Console tab
3. Try analysis again
4. Look for error messages

## Common Issues

### Port 8000 Already in Use
```bash
# Windows - Kill process on port 8000
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### CORS Errors
If you see CORS errors in console, the ML server needs to restart.

### "No response from server"
1. ML server crashed - check terminal running ML server
2. Firewall blocking port 8000
3. Wrong URL (should be localhost:8000, not 127.0.0.1:8000)

## Debugging Steps

### 1. Check ML Server Logs
Look at the terminal where `start_ml_server.bat` is running.
Should show:
```
==================================================
  WebOptimizer ML Server Starting...
  Model: LightGBM (K-means) - 98.47% Accuracy
  Server: http://localhost:8000
==================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Check Next.js Server
Make sure Next.js dev server is running:
```bash
yarn dev
```

### 3. Check Browser Console
Should show progress logs:
```
ML Server Health: {status: 'healthy', ...}
Sending request to /api/analyze with URL: ...
Analysis response: {...}
```

### 4. Test API Directly
```bash
# Test health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://google.com"}'
```

## Still Not Working?

1. **Restart everything:**
   - Stop ML server (Ctrl+C)
   - Stop Next.js (Ctrl+C)
   - Start ML server: `start_ml_server.bat`
   - Start Next.js: `yarn dev`
   - Clear browser cache (Ctrl+Shift+Del)
   - Try again

2. **Check the test HTML:**
   - Open `test_ml_server.html` in browser
   - Click "Test Health" - should work
   - Click "Test Prediction" - should return data
   - If this works, issue is in Next.js app

3. **Enable verbose logging:**
   - Open browser DevTools (F12)
   - Go to Network tab
   - Try analysis
   - Check failed requests
   - Look at request/response details
