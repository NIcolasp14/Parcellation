#!/bin/bash
#SBATCH --account=mlsclemon
#SBATCH --partition=rtx8000
#SBATCH --cpus-per-task=3
#SBATCH --mem=22G
#SBATCH --time=4-00:00:00
#SBATCH --gpus=1
#SBATCH --job-name=conditioned_v2_1view
#SBATCH --output=/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/logs/conditioned_v2_1view_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --chdir=/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White

# ============================================================================
# View-Conditioned U-Net Training (V2) - 1 Random View per Epoch
# ============================================================================
#
# KEY IMPROVEMENTS:
# 1. One-hot view encoding (6 views) instead of continuous camera parameters
# 2. Fixed camera Z position (mesh-size independent)
# 3. Random view augmentation with 1 random view per mesh per epoch
# 4. Proper monitoring and per-view metrics
#
# Training Strategy:
# - Each mesh generates 1 random augmented view per epoch
# - View is randomly selected from {Front, Back, Left, Right, Top, Bottom}
# - Small augmentation (±7 degrees) applied to rotation
# - Total samples per epoch = num_subjects × 1
#
# This version is MORE DATA EFFICIENT but SLOWER convergence
# Recommended for: Long training runs, limited GPU memory
# ============================================================================

echo "============================================================================"
echo "VIEW-CONDITIONED U-NET V2 - 1 RANDOM VIEW PER EPOCH"
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
OUTPUT_PATH="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/Models_WM_Conditioned_V2_1view_${SLURM_JOB_ID}"

# Training parameters
BATCH_SIZE=8  # Restored to 8 after fixing mesh cache leak
NUM_RANDOM_VIEWS=1  # 1 random view per mesh per epoch
LEARNING_RATE=0.0001
NUM_EPOCHS=500
IMG_WIDTH=800
IMG_HEIGHT=800

echo "Configuration:"
echo "  Input: $INPUT_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  Batch size: $BATCH_SIZE"
echo "  Random views per mesh: $NUM_RANDOM_VIEWS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Image size: ${IMG_WIDTH}x${IMG_HEIGHT}"
echo ""

# Run training
python network_trainer_wm_conditioned_v2.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --num_random_views $NUM_RANDOM_VIEWS \
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

