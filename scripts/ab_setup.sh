#!/bin/bash
# Model-size comparison for Level 2: a separate outputs root per model that reuses the existing ingest / Level 1 /
# correction results (symlinked, read-only) so only level2 + cleanup run again. The 4B results are never touched.
#   bash scripts/ab_setup.sh <tag> <video_id>...        e.g. bash scripts/ab_setup.sh q27b $AB_IDS
# then submit with (see docs/cluster_setup.md):
#   sbatch -p k2-gpu-a100 --gres=gpu:a100:1 --export=ALL,VMP_PATHS__OUTPUTS_ROOT=../outputs_ab/q27b,\
#     VMP_PATHS__METADATA_ROOT=../metadata_ab/q27b,VMP_LEVEL2__MODEL_ID=Qwen/Qwen3.5-27B,VIDEO_IDS="$AB_IDS",\
#     VMP_PATHS__CORPUS_ROOT=../data/batches/batch_01,VMP_PATHS__MANIFEST=../outputs/manifests/batch_01.json \
#     scripts/submit_cluster.sh level2 cleanup
set -euo pipefail
TAG=$1; shift
SRC=${VMP_AB_SOURCE:-../outputs}
DST=../outputs_ab/$TAG
mkdir -p "$DST"
for id in "$@"; do
  s=$(cd "$SRC/$id" && pwd)
  d="$DST/$id"
  mkdir -p "$d/logs" "$d/reports"
  for sub in source level1 correction; do
    [ -e "$d/$sub" ] || ln -s "$s/$sub" "$d/$sub"
  done
  [ -e "$d/pipeline_state.json" ] || cp "$s/pipeline_state.json" "$d/pipeline_state.json"
  echo "ready: $d"
done
