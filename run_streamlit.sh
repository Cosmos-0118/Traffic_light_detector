#!/bin/bash
# Run Streamlit App for Traffic Light Detection

echo "🚦 Starting Traffic Light Detection Streamlit App..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run Streamlit app
echo "🚀 Starting Streamlit app..."
echo "   Open your browser to: http://localhost:8501"
echo "   Press Ctrl+C to stop the app"
echo ""

streamlit run app.py
