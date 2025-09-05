#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "[INFO] Traffic Light Detection Application"
echo "[INFO] Script directory: ${SCRIPT_DIR}"

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed or not in PATH"
    echo "Please install Python 3.7+ from https://python.org"
    exit 1
fi

# Check if required files exist
if [ ! -f "${SCRIPT_DIR}/simple_ui.py" ]; then
    echo "[ERROR] simple_ui.py not found in ${SCRIPT_DIR}"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found in ${SCRIPT_DIR}"
    exit 1
fi

# Ensure scripts are executable
chmod +x "${SCRIPT_DIR}/traffic_light_detector.py" 2>/dev/null || true
chmod +x "${SCRIPT_DIR}/simple_ui.py" 2>/dev/null || true

# Create venv if missing
if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] Creating virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi

echo "[INFO] Python: $(python --version 2>&1)"

# Install/Update dependencies
echo "[INFO] Installing/Updating dependencies"
python -m pip install --upgrade pip --quiet
if [ $? -ne 0 ]; then
    echo "[WARNING] Failed to upgrade pip, continuing anyway..."
fi

pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

# Test OpenCV installation
echo "[INFO] Testing OpenCV installation..."
python -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] OpenCV installation failed"
    exit 1
fi

echo "[INFO] Launching UI (press q in window to quit, d to toggle debug masks)"
PYTHONPATH="${SCRIPT_DIR}" python "${SCRIPT_DIR}/simple_ui.py" "$@"
