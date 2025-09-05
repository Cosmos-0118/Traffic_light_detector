#!/bin/bash
# Run Streamlit App for Traffic Light Detection

echo "🚦 Starting Traffic Light Detection Streamlit App..."
echo "=================================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "[INFO] Script directory: ${SCRIPT_DIR}"

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed or not in PATH"
    echo "Please install Python 3.7+ from https://python.org"
    exit 1
fi

# Check if required files exist
if [ ! -f "${SCRIPT_DIR}/app.py" ]; then
    echo "[ERROR] app.py not found in ${SCRIPT_DIR}"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found in ${SCRIPT_DIR}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "${VENV_DIR}/bin/activate"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi

echo "[INFO] Python: $(python --version 2>&1)"

# Install dependencies
echo "📥 Installing dependencies..."
python -m pip install --upgrade pip --quiet
if [ $? -ne 0 ]; then
    echo "[WARNING] Failed to upgrade pip, continuing anyway..."
fi

pip install -r "${SCRIPT_DIR}/requirements.txt"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

# Test Streamlit installation
echo "[INFO] Testing Streamlit installation..."
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Streamlit installation failed"
    exit 1
fi

# Run Streamlit app
echo "🚀 Starting Streamlit app..."
echo "   Open your browser to: http://localhost:8501"
echo "   Press Ctrl+C to stop the app"
echo ""

cd "${SCRIPT_DIR}"
streamlit run app.py
