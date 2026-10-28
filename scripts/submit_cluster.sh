#!/bin/bash
# Kelvin2 SLURM job, from <scratch>/vmp/code:  sbatch scripts/submit_cluster.sh [stages...]
#   all stages, all videos:   sbatch scripts/submit_cluster.sh ingest level1 correction level2
#   one video:                sbatch --export=ALL,VIDEO_IDS=<video_id> scripts/submit_cluster.sh level2
#   split over 4 GPU slices:  sbatch --array=0-3 scripts/submit_cluster.sh   (each task gets ~1/4 of the hours)
# Stages run stage by stage over all videos. Level 2 runs in the vLLM env (scripts/prepare_vllm.sh) when it exists,
# batched; otherwise in the main env, one call at a time.
# A100 MIG 20 GB slice: enough for Level 1 and Level 2 (Qwen3.5-4B bf16). `sinfo -o "%P %G"` lists alternatives.
#SBATCH --job-name=vmp
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=../outputs/slurm-%x-%A_%a.out
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
mkdir -p ../outputs
# array tasks share the manifest: rebuild it one task at a time
flock ../outputs/.manifest.lock "$MAIN_ENV/bin/python" -m video_metadata.cli ingest --config configs/cluster.yaml --manifest-only
if [ -z "${VIDEO_IDS:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 ]; then
  SHARD=$(( SLURM_ARRAY_TASK_ID - ${SLURM_ARRAY_TASK_MIN:-0} ))
  VIDEO_IDS=$("$MAIN_ENV/bin/python" scripts/shard_videos.py --config configs/cluster.yaml --index $SHARD --count $SLURM_ARRAY_TASK_COUNT)
  echo "array shard $SHARD/$SLURM_ARRAY_TASK_COUNT: $(wc -w <<< "$VIDEO_IDS") videos"
  [ -n "$VIDEO_IDS" ] || { echo "empty shard"; exit 0; }
fi
[ -n "$PRE" ] && run "$MAIN_ENV" $PRE
if [[ " $STAGES " == *" level2 "* ]]; then
  echo "level2 env: $L2_ENV"
  "$L2_ENV/bin/python" scripts/patch_vllm_mig.py || true
  # vLLM compiles some kernels at start-up and needs nvcc: use the one installed in the vLLM env
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"  # no FlashInfer JIT sampler
  [ -x "$L2_ENV/bin/nvcc" ] && export CUDA_HOME="$L2_ENV" PATH="$L2_ENV/bin:$PATH"
  [ -n "${VMP_CUDA_MODULE:-}" ] && module load "$VMP_CUDA_MODULE"
  run "$L2_ENV" level2
fi
"$MAIN_ENV/bin/python" -m video_metadata.cli status --config configs/cluster.yaml
