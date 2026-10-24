#!/bin/bash
# Run ONCE on a Kelvin2 LOGIN node (has internet), from <scratch>/vmp/code:  bash scripts/prepare_cluster.sh
# Creates a conda env, installs the package and pre-downloads all models into ../models.
set -euo pipefail
module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || echo "load your conda module manually (module avail)"
conda env list | grep -q "^vmp " || conda create -y -n vmp -c conda-forge python=3.11 ffmpeg
unset PYTHONPATH PYTHONHOME          # the miniforge module points PYTHONPATH at PyPy
export PATH=$HOME/.conda/envs/vmp/bin:$PATH
# PyTorch with CUDA 12.x wheels (works on A100/H100 drivers >= 525); change cu124 if the node's driver is older
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[level1,dev]"
python -m video_metadata.cli download-models --config configs/cluster.yaml --include-vlm
python -m pytest -q
echo "ready: submit with  sbatch scripts/submit_cluster.sh"
