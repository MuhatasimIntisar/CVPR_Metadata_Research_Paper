# Running on Kelvin2 (Linux + SLURM)

Layout on scratch (all config paths are relative to `code/`, nothing machine-specific is hard-coded):
```
/mnt/scratch2/users/<user>/vmp/
  code/            this package
  data/pilot/      Internet Archive item folders (videos)
  data/segments/   earlier detector outputs (reused if present)
  data/level1/     earlier YOLO/CLIP/X-CLIP/VAD outputs (reused if present)
  models/          model cache (filled by download-models)
  outputs/         results, manifest, SLURM logs
```

## 1. Copy from the laptop (PowerShell, from C:\msc_ds_ai\CVPR_Metadata_Research_Paper)
Use your usual Kelvin2 ssh login (host/port as you normally connect):
```
ssh <kelvin-login> "mkdir -p /mnt/scratch2/users/<user>/vmp/data"
scp -r video_metadata_pipeline <kelvin-login>:/mnt/scratch2/users/<user>/vmp/code
scp -r data/pilot data/segments data/level1 <kelvin-login>:/mnt/scratch2/users/<user>/vmp/data/
```
(~1.5 GB; `outputs/` from the laptop is not needed.)

## 2. One-time setup on a LOGIN node (internet available)
```
cd /mnt/scratch2/users/<user>/vmp/code
bash scripts/prepare_cluster.sh
```
Creates a conda env `vmp` (Python 3.11 + ffmpeg), installs PyTorch (CUDA 12.4 wheels) and the package, downloads all
models into `../models` (Level 1 ~2 GB, Qwen3.5-4B ~10 GB) and runs the unit tests. If huggingface.co is blocked on
the login node, download the models elsewhere and copy `models/` across.

### Level-2 batching env (vLLM) - once, on the login node
```
bash scripts/prepare_vllm.sh
```
Creates a second conda env `vmp-vllm` (Python 3.12, vLLM >= 0.17 with its own PyTorch build) and installs this
package into it; Level 1 stays in `vmp`. No models are downloaded again (Level 2 reads Qwen3.5-4B from `../models`).
`submit_cluster.sh` then runs ingest/level1/correction in `vmp` and level2 in `vmp-vllm` (batched). Without the env,
level2 falls back to `vmp` and the transformers backend (one call at a time).
If the engine fails to start, the log shows `[vllm] engine start failed ... retrying` (prefix caching off, then
CUDA graphs off). If it still fails, set `VMP_L2_ENV=$HOME/.conda/envs/vmp` to use the transformers backend.

### Fast kernels for the transformers backend only
Without them transformers runs Qwen3.5's linear-attention layers in a slow reference implementation (the SLURM log
warns "`causal_conv1d` / `flash-linear-attention` is not installed ... much slower"). On the login node:
```
unset PYTHONPATH PYTHONHOME; export PATH=$HOME/.conda/envs/vmp/bin:$PATH
pip install flash-linear-attention
pip install causal-conv1d --no-build-isolation    # compiles CUDA code; skip if it fails, the first one still helps
```
Check the next SLURM log: the two warnings should be gone (or only the causal_conv1d one).

## 3. Submit
Edit the partition / GPU lines in `scripts/submit_cluster.sh` (`sinfo -o "%P %G"` lists them), then:
```
sbatch scripts/submit_cluster.sh                       # ingest + level1 + correction + level2 for every video
sbatch scripts/submit_cluster.sh level1 correction     # selected stages
squeue -u $USER ; tail -f ../outputs/slurm-vmp-<jobid>.out
python -m video_metadata.cli status --config configs/cluster.yaml
```
Resume after a time-out/interruption: submit again; completed stages are cached per video, failed ones retried.
GPU memory: Level 1 < 6 GB (any slice); Level 2 (Qwen3.5-4B, bf16) >= 20 GB.

## 4. Bring results back
`scp -r <kelvin-login>:/mnt/scratch2/users/<user>/vmp/outputs ./video_metadata_pipeline/outputs_kelvin`
