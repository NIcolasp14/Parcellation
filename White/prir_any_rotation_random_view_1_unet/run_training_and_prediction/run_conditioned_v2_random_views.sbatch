#!/bin/bash
#SBATCH --account=mlsclemon
#SBATCH --partition=rtx8000
#SBATCH --cpus-per-task=3
#SBATCH --mem=22G
#SBATCH --time=5-00:00:00
#SBATCH --gpus=1
#SBATCH --job-name=xyz_random_views
#SBATCH --output=/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/logs/xyz_random_views_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --chdir=/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White

# ============================================================================
# View-Conditioned U-Net Training - ARBITRARY RANDOM ROTATIONS
# ============================================================================
#
# KEY FEATURES:
# 1. Completely random rotations (full sphere coverage)
# 2. Continuous camera pose conditioning (3 rotation angles)
# 3. Multiple random views per mesh per epoch for better coverage
# 4. True rotation invariance learning
#
# Training Strategy:
# - Each mesh generates N completely random views per epoch
# - Rotations sampled uniformly from full rotation space
# - No bias toward canonical views
# - Total samples per epoch = num_subjects × num_random_views
#
# This version learns TRUE ROTATION INVARIANCE
# Recommended for: Maximum view generalization, arbitrary camera angles
# ============================================================================

echo "============================================================================"
echo "VIEW-CONDITIONED U-NET V2 - ARBITRARY RANDOM ROTATIONS"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================================================"

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /autofs/space/ballarat_004/users/np341/conda_envs/prir_optix

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Set paths
INPUT_PATH="/autofs/vast/lemon/temp_stuff/nicolas/train_data/wm"
OUTPUT_PATH="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/models/xyz_random_views_${SLURM_JOB_ID}"

# Create logs and models directories if they don't exist
mkdir -p /autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/logs
mkdir -p /autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/models

# Training parameters - SAME AS 6-VIEW TRAINING
BATCH_SIZE=8
NUM_RANDOM_VIEWS=1  # 1 random view per mesh per epoch (same as 6-view training)
LEARNING_RATE=0.0001
NUM_EPOCHS=500
IMG_WIDTH=800
IMG_HEIGHT=800

echo "Configuration:"
echo "  Input: $INPUT_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  Batch size: $BATCH_SIZE"
echo "  Random views per mesh: $NUM_RANDOM_VIEWS (FULLY RANDOM ROTATIONS)"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Image size: ${IMG_WIDTH}x${IMG_HEIGHT}"
echo "  Mode: ARBITRARY RANDOM ROTATIONS (uniform sampling from full sphere)"
echo ""

# Change to training_code directory
cd /autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/training_code

# Run training with RANDOM VIEWS
python network_trainer_wm_conditioned_v2.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --num_random_views $NUM_RANDOM_VIEWS \
    --use_random_views \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --img_width $IMG_WIDTH \
    --img_height $IMG_HEIGHT \
    --recompute_normals \
    --num_workers 0

EXIT_CODE=$?

echo ""
echo "============================================================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================================"

exit $EXIT_CODE
