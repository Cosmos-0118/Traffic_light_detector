#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "[INFO] Traffic Light Detection Application"
echo "[INFO] Script directory: ${SCRIPT_DIR}"

# Ensure scripts are executable
chmod +x "${SCRIPT_DIR}/traffic_light_detector.py" || true
chmod +x "${SCRIPT_DIR}/simple_ui.py" || true

# Create venv if missing
if [ ! -d "${VENV_DIR}" ]; then
	echo "[INFO] Creating virtual environment at ${VENV_DIR}"
	python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
echo "[INFO] Python: $(python --version)"

echo "[INFO] Installing/Updating dependencies"
pip install -q --upgrade pip
pip install -q -r "${SCRIPT_DIR}/requirements.txt"

echo "[INFO] Launching UI (press q in window to quit, d to toggle debug masks)"
PYTHONPATH="${SCRIPT_DIR}" python "${SCRIPT_DIR}/simple_ui.py" "$@"
