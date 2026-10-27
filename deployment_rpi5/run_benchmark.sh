#!/usr/bin/env bash
# =============================================================================
# Automated Runner for Raspberry Pi 5 Zooplankton ONNX Benchmark
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "PlanktonAI — Raspberry Pi 5 Benchmark Runner"
echo "Host:       $(hostname 2>/dev/null || echo 'localhost')"
echo "Arch:       $(uname -m)"
echo "Date:       $(date)"
echo "============================================================"

# 1. Check or create Python virtual environment
VENV_DIR="${SCRIPT_DIR}/venv_rpi5"
if [ ! -d "${VENV_DIR}" ]; then
    echo ""
    echo "Creating Python virtual environment in ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# 2. Install / Verify Dependencies
echo ""
echo "Verifying edge dependencies..."
pip install -r requirements_rpi5.txt --prefer-binary --quiet

# 3. Verify models exist
MISSING_MODELS=0
for MODEL in models/yolo26n_detector.onnx models/binary_filter.onnx models/mobilenet_classifier.onnx; do
    if [ ! -f "${MODEL}" ]; then
        echo "ERROR: Model not found: ${MODEL}"
        MISSING_MODELS=1
    fi
done

if [ "${MISSING_MODELS}" -eq 1 ]; then
    echo "Aborting. Ensure all three ONNX models are placed in models/ before running."
    exit 1
fi

# 4. Check or generate test images
IMG_COUNT=$(find test_images -maxdepth 1 -name "*.jpg" -o -name "*.png" | wc -l)
if [ "${IMG_COUNT}" -eq 0 ]; then
    echo ""
    echo "No test images found in test_images/. Generating synthetic test frames..."
    python3 test_images/generate_synthetic.py --count 5 --out-dir test_images
fi

# 5. Execute benchmark
echo ""
echo "Starting benchmark harness..."
python3 benchmark_rpi5.py "$@"
