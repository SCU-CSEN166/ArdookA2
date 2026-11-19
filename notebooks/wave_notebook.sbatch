#!/bin/bash
# Execute a notebook end-to-end on an SCU Wave GPU node, saving outputs in place.
# Submit from the repo root:  sbatch notebooks/wave_notebook.sbatch notebooks/<name>.ipynb
# The PyTorch/2.11 (CUDA 13) module has no kernels for the gpu partition's V100s (compute
# capability 7.0), so request the hub partition's RTX PRO 6000 (Blackwell) instead.
#SBATCH --job-name=ardook-nb
#SBATCH --partition=hub
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=notebooks/wave_%x_%j.log

set -euo pipefail
NOTEBOOK="${1:?usage: sbatch notebooks/wave_notebook.sbatch notebooks/<name>.ipynb}"
module load PyTorch/2.11.0-Python-3.12
# Wave points XDG_CACHE_HOME at job-local scratch; pin the model cache that the login node filled.
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
# Weights too large for the 50 GB home quota (Qwen2.5-32B) live on the course scratch space.
export LARGE_MODEL_CACHE="/WAVE/scratch/CSEN-166-Fa26/$USER/hf/hub"
echo "notebook=$NOTEBOOK host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 "$NOTEBOOK"
