#!/bin/bash
#SBATCH -J blip2_finetune
#SBATCH -p gpu
#SBATCH -A c01949
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o finetune_%j.out
#SBATCH -e finetune_%j.err

echo "========================================"
echo "BLIP-2 Fine-tuning on TextCaps"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

# Load modules
module load python/3.12.11
module load cudatoolkit/12.6

# Change to project directory
cd /N/scratch/madbala/OmniBridge

# Set HuggingFace cache directories to scratch (BEFORE any Python runs)
export HF_HOME=/N/scratch/madbala/.cache/huggingface
export HF_DATASETS_CACHE=/N/scratch/madbala/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/N/scratch/madbala/.cache/huggingface/hub
export XDG_CACHE_HOME=/N/scratch/madbala/.cache
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE

# Install dependencies
pip install --user peft datasets accelerate transformers torch pillow tqdm --quiet

# Set HuggingFace token (if needed)
export HF_TOKEN="your_token_here"

# Run training
echo "Starting fine-tuning..."
python finetune_blip2.py

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
