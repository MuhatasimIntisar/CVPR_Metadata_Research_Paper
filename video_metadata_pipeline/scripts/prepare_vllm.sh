#!/bin/bash
# Run ONCE on a Kelvin2 LOGIN node (internet), from <scratch>/vmp/code:  bash scripts/prepare_vllm.sh
# Separate conda env for Level 2 (vLLM brings its own PyTorch/CUDA build; keeping it apart protects the Level-1 env).
# Models are not downloaded again: Level 2 reads Qwen3.5-4B from ../models (download-models already did it).
set -euo pipefail
module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || echo "load your conda module manually (module avail)"
conda env list | grep -q "^vmp-vllm " || conda create -y -n vmp-vllm -c conda-forge python=3.12 ffmpeg
unset PYTHONPATH PYTHONHOME          # the miniforge module points PYTHONPATH at PyPy
export PATH=$HOME/.conda/envs/vmp-vllm/bin:$PATH
python -m pip install "vllm>=0.17"
python -m pip install -e ".[vllm,dev]"
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| CUDA build', torch.version.cuda)"
python -m pytest -q tests/test_batching.py tests/test_level2.py tests/test_metadata_v21.py
echo "ready: level2 runs in this env automatically (scripts/submit_cluster.sh)"
