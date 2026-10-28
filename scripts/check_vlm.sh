#!/bin/bash
# Short GPU job before a long run: sbatch scripts/check_vlm.sh   (log: ../outputs/slurm-vmp-check-<jobid>.out)
#SBATCH --job-name=vmp-check
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=../outputs/slurm-%x-%j.out
set -euo pipefail
unset PYTHONPATH PYTHONHOME
L2_ENV=${VMP_L2_ENV:-$HOME/.conda/envs/vmp-vllm}
[ -x "$L2_ENV/bin/python" ] || L2_ENV=$HOME/.conda/envs/vmp
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "env: $L2_ENV"
"$L2_ENV/bin/python" scripts/patch_vllm_mig.py || true
# vLLM compiles some kernels at start-up and needs nvcc: use the one installed in the vLLM env
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"  # no FlashInfer JIT sampler
[ -x "$L2_ENV/bin/nvcc" ] && export CUDA_HOME="$L2_ENV" PATH="$L2_ENV/bin:$PATH"
[ -n "${VMP_CUDA_MODULE:-}" ] && module load "$VMP_CUDA_MODULE"
"$L2_ENV/bin/python" -m video_metadata.cli check-vlm --config configs/cluster.yaml
