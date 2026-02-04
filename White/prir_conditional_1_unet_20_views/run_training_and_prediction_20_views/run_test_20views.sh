#!/bin/bash
# Test script for 20-view conditioned model

# Configuration
MODEL_DIR="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/Models_WM_Conditioned_V2_20views_7785741"
TEST_DATA="/autofs/space/ballarat_004/users/np341/mindboggle2"
DEVICE="cuda:0"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="test_20views_${TIMESTAMP}.log"

echo "============================================================================"
echo "Testing 20-View Conditioned Model"
echo "============================================================================"
echo "Model: $MODEL_DIR"
echo "Test data: $TEST_DATA"
echo "Device: $DEVICE"
echo "Log file: $LOG_FILE"
echo "============================================================================"

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /autofs/space/ballarat_004/users/np341/conda_envs/prir_optix

# Run test script
python /autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_conditional_1_unet_20_views/run_training_and_prediction_20_views/test_conditioned_model_v2_20_views.py \
    --model_dir "$MODEL_DIR" \
    --test_data "$TEST_DATA" \
    --device "$DEVICE"

echo ""
echo "============================================================================"
echo "Testing completed at $(date)"
echo "============================================================================"
