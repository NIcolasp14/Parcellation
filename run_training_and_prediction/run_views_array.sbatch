#!/bin/bash
#SBATCH --account=mlsclemon
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --gpus=1
#SBATCH --array=1-6%6
#SBATCH --output=/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/logs/prir_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --chdir=/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White

set -eo pipefail

echo "Host: $(hostname)"; date
echo "JOB=$SLURM_JOB_ID  TASK=$SLURM_ARRAY_TASK_ID"

# Check if SLURM assigned a GPU
if [ -n "$SLURM_JOB_GPUS" ] || [ -n "$SLURM_STEP_GPUS" ]; then
    echo "SLURM assigned GPU(s): SLURM_JOB_GPUS=$SLURM_JOB_GPUS SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
else
    echo "WARNING: No SLURM GPU assignment detected"
fi

nvidia-smi -L || true

# conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /autofs/space/ballarat_004/users/np341/conda_envs/prir_optix
which python3; python3 -V

# CRITICAL: Force WandB to write to project directory, NOT home directory
export WANDB_DIR="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White"
export WANDB_CACHE_DIR="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/.wandb_cache"
export WANDB_CONFIG_DIR="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/.wandb_config"
echo "WandB directory set to: $WANDB_DIR"

views=(Front Back Right Left Top Bottom)
view="${views[$((SLURM_ARRAY_TASK_ID-1))]}"

PY=/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/network_trainer_wm.py

# ===========================
# Use SLURM_JOB_ID to create a unique checkpoint directory for this batch
# All 6 array tasks (views) will share the same directory
# ===========================
BASE_CHECKPOINT_DIR="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White"
CHECKPOINT_BASE_NAME="Models_WM_SLURM"

# Use SLURM_ARRAY_JOB_ID (same for all tasks in this array) to create ONE folder for all 6 views
ckpt_root="${BASE_CHECKPOINT_DIR}/${CHECKPOINT_BASE_NAME}_${SLURM_ARRAY_JOB_ID}"
outdir="${ckpt_root}/${view}"
mkdir -p "$outdir"

echo "=========================================================================="
echo "SLURM Training Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID} (${view})"
echo "Shared Checkpoint Root: ${ckpt_root}"
echo "This View's Dir: ${outdir}"
echo "=========================================================================="

# ---- Skip if already done (avoid duplicates) ----
# Define "done" as: any *.ckpt exists in the view's dir (adjust to your scheme)
if compgen -G "${outdir}/*.ckpt" > /dev/null; then
  echo "[SKIP] ${view}: checkpoint(s) already present in ${outdir}"
  exit 0
fi

# ---- Lock so two submits don't collide on the same view ----
lock="${outdir}/.lock_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if ! mkdir "$lock" 2>/dev/null; then
  echo "[SKIP] ${view}: another job holds the lock (${outdir}/.lock_*)"
  exit 0
fi
trap 'rmdir "'"$lock"'" || true' EXIT

# ---- Launch training ----
python3 "$PY" \
  --input_path "/autofs/vast/lemon/temp_stuff/nicolas/train_data/wm" \
  --view "$view" \
  --recompute_normals \
  --batch_size 8 \
  --epochs 500 \
  --checkpoint_dir "$outdir" \
  --raycasting "pytorch3d"
