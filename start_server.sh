#!/bin/bash

echo "🚀 Starting MiniCPM-V-4_5 FastAPI Server"
echo "========================================"

# Activate virtual environment
source venv/bin/activate

# Check if model is available
echo "📋 Checking system requirements..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

echo ""
echo "🌐 Starting server on http://127.0.0.1:8000"
echo "📖 API Documentation: http://127.0.0.1:8000/docs"
echo "🧪 Test Interface: Open test_upload.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
