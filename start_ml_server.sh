#!/bin/bash
echo ""
echo "=========================================="
echo "  WebOptimizer ML Server"
echo "=========================================="
echo ""
echo "Starting Python ML Server..."
echo "Server will run at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
.venv/Scripts/python.exe src/api/ml_server_fast.py
