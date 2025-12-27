#!/bin/bash
#===============================================================================
# Stage 1: The Captioner - "Anti-Gravity" Slurm Script for Big Red 200
#===============================================================================
# Strategy: Request exactly one A100 80GB node with reasonable time.
# This "Expedite" approach moves your job to the front of the queue compared
# to massive multi-node requests.
#===============================================================================

#SBATCH -J Phase1_Captioner           # Job name
#SBATCH -p gpu                         # GPU partition
#SBATCH --nodes=1                      # Single node request
#SBATCH --gpus-per-node=1              # One GPU per node
#SBATCH --constraint=a100_80gb         # The '80GB' tag is your priority secret
#SBATCH --time=10:00:00                # 10 hrs for ~6-8 hr task (safe buffer)
#SBATCH --mem=128G                     # High RAM ensures data loader never blocked
#SBATCH --mail-type=ALL                # Email on start, end, and fail
#SBATCH --mail-user=YOUR_EMAIL@iu.edu  # <-- UPDATE THIS
#SBATCH --output=phase1_log_%j.txt     # Output log with job ID
#SBATCH --error=phase1_err_%j.txt      # Error log with job ID

#===============================================================================
# Environment Setup
#===============================================================================

echo "=============================================="
echo "Stage 1: The Captioner - Training Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

# Defy gravity: Load precise CUDA environment for A100s
module purge
module load python/3.11.4
module load cuda/12.1

# Activate virtual environment
source ~/venvs/iu_masters_proj/bin/activate

# Verify environment
echo ""
echo "Environment Check:"
echo "  Python: $(which python)"
echo "  Python Version: $(python --version)"
echo "  CUDA: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TRANSFORMERS_CACHE=/N/scratch/$USER/.cache/huggingface
export HF_HOME=/N/scratch/$USER/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

# Create cache directory if needed
mkdir -p $TRANSFORMERS_CACHE

#===============================================================================
# Training Execution
#===============================================================================

echo "Starting Phase 1 Training..."
echo ""

cd $SLURM_SUBMIT_DIR

# Run the training script
python phase1_train.py \
    --epochs 5 \
    --batch-size 1 \
    --learning-rate 2e-4

#===============================================================================
# Completion
#===============================================================================

echo ""
echo "=============================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "=============================================="

# Copy results to permanent storage (optional)
# cp -r /N/scratch/$USER/phase1_output ~/phase1_results_$SLURM_JOB_ID
