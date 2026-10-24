#!/bin/bash
# Kelvin2 SLURM job, from <scratch>/vmp/code:  sbatch scripts/submit_cluster.sh [stages...]
# One video only:  sbatch --export=ALL,VIDEO_IDS=<video_id> scripts/submit_cluster.sh level2
# A100 MIG 20 GB slice: enough for Level 1 and Level 2 (Qwen3.5-4B bf16). `sinfo -o "%P %G"` lists alternatives.
#SBATCH --job-name=vmp
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=../outputs/slurm-%x-%j.out
set -euo pipefail
unset PYTHONPATH PYTHONHOME          # the miniforge module points PYTHONPATH at PyPy, which breaks Python 3.11
export PATH=$HOME/.conda/envs/vmp/bin:$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv || true
STAGES=${*:-ingest level1 correction}
python -m video_metadata.cli ingest --config configs/cluster.yaml --manifest-only
if [ -n "${VIDEO_IDS:-}" ]; then
  python -m video_metadata.cli run --config configs/cluster.yaml --stages $STAGES --video-id $VIDEO_IDS
else
  python -m video_metadata.cli resume --config configs/cluster.yaml --stages $STAGES
fi
python -m video_metadata.cli status --config configs/cluster.yaml
