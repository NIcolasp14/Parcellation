#!/bin/bash
#
# Quick Test Script for Trained Model
# ====================================
# Usage:
#   bash test_model.sh [model_dir] [test_data_dir] [device]
#
# Examples:
#   bash test_model.sh  # Uses defaults
#   bash test_model.sh ../models/xyz_conditioned_v2_1view_7785764
#   bash test_model.sh ../models/xyz_conditioned_v2_1view_7785764 /path/to/testdata cuda:0
#

# Default values
DEFAULT_MODEL="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/models/xyz_conditioned_v2_1view_7785764"
DEFAULT_TEST_DATA="/autofs/space/ballarat_004/users/np341/mindboggle2"
DEFAULT_DEVICE="cuda:0"

# Use arguments or defaults
MODEL_DIR="${1:-$DEFAULT_MODEL}"
TEST_DATA="${2:-$DEFAULT_TEST_DATA}"
DEVICE="${3:-$DEFAULT_DEVICE}"

# Resolve relative paths
if [[ "$MODEL_DIR" != /* ]]; then
    MODEL_DIR="$(cd "$(dirname "$0")/$MODEL_DIR" && pwd)/$(basename "$MODEL_DIR")"
fi

echo "========================================================================"
echo "Testing Arbitrary-View Conditioned Model"
echo "========================================================================"
echo "Model directory: $MODEL_DIR"
echo "Test data:       $TEST_DATA"
echo "Device:          $DEVICE"
echo ""

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo ""
    echo "Available models:"
    ls -lhd /autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/models/xyz_* 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Check if test data exists
if [ ! -d "$TEST_DATA" ]; then
    echo "ERROR: Test data directory not found: $TEST_DATA"
    exit 1
fi

# Find checkpoint
if [ -f "$MODEL_DIR/last.ckpt" ]; then
    echo "Found checkpoint: last.ckpt"
elif ls "$MODEL_DIR"/*.ckpt 1> /dev/null 2>&1; then
    echo "Found checkpoints in model directory"
else
    echo "ERROR: No checkpoints found in $MODEL_DIR"
    exit 1
fi

echo ""
echo "Starting test..."
echo "========================================================================"
echo ""

# Change to script directory for imports
cd "$(dirname "$0")"

# Run test
python test_conditioned_model_v2.py \
    --model_dir "$MODEL_DIR" \
    --test_data "$TEST_DATA" \
    --device "$DEVICE"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Test completed with exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE
