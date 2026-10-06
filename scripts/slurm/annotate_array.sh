#!/bin/bash
# Kelvin (live): one array task per shard.
#   N=$(ls -d $DATA_ROOT/shards/shard-* | wc -l)
#   sbatch --array=0-$((N-1)) scripts/slurm/annotate_array.sh
# TODO: confirm partition / GPU names and module versions with Kelvin docs.
#SBATCH --job-name=cvpr-annotate
#SBATCH --partition=k2-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/annotate_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load python/3.10   # TODO
# source .venv/bin/activate

DATA_ROOT=$(python -c "from src.common.config import load_config; print(load_config('live')['paths'].root)")
SHARD=$(ls -d "$DATA_ROOT"/shards/shard-* | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" | xargs basename)
echo "Task $SLURM_ARRAY_TASK_ID -> $SHARD"
python scripts/run_annotate.py --env live --shard "$SHARD"
