#!/bin/bash
# Kelvin2 SLURM job, from <scratch>/vmp/code:  sbatch scripts/submit_cluster.sh [stages...]
#   all stages, all videos:   sbatch scripts/submit_cluster.sh ingest level1 correction level2
#   one video:                sbatch --export=ALL,VIDEO_IDS=<video_id> scripts/submit_cluster.sh level2
# Stages run stage by stage over all videos. Level 2 runs in the vLLM env (scripts/prepare_vllm.sh) when it exists,
# batched; otherwise in the main env, one call at a time.
# A100 MIG 20 GB slice: enough for Level 1 and Level 2 (Qwen3.5-4B bf16). `sinfo -o "%P %G"` lists alternatives.
#SBATCH --job-name=vmp
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=../outputs/slurm-%x-%j.out
set -euo pipefail
unset PYTHONPATH PYTHONHOME          # the miniforge module points PYTHONPATH at PyPy, which breaks Python
MAIN_ENV=$HOME/.conda/envs/vmp
L2_ENV=${VMP_L2_ENV:-$HOME/.conda/envs/vmp-vllm}
[ -x "$L2_ENV/bin/python" ] || L2_ENV=$MAIN_ENV
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
STAGES=${*:-ingest level1 correction level2}
PRE=$(for s in $STAGES; do [ "$s" != level2 ] && echo -n "$s "; done || true)
run() {  # run <env> <stages...>
  local env=$1; shift
  if [ -n "${VIDEO_IDS:-}" ]; then
    "$env/bin/python" -m video_metadata.cli run --config configs/cluster.yaml --stages "$@" --video-id $VIDEO_IDS
  else
    "$env/bin/python" -m video_metadata.cli resume --config configs/cluster.yaml --stages "$@"
  fi
}
"$MAIN_ENV/bin/python" -m video_metadata.cli ingest --config configs/cluster.yaml --manifest-only
[ -n "$PRE" ] && run "$MAIN_ENV" $PRE
if [[ " $STAGES " == *" level2 "* ]]; then
  echo "level2 env: $L2_ENV"
  "$L2_ENV/bin/python" scripts/patch_vllm_mig.py || true
  run "$L2_ENV" level2
fi
"$MAIN_ENV/bin/python" -m video_metadata.cli status --config configs/cluster.yaml
