# video_metadata_pipeline

Portable, human-in-the-loop pipeline that turns Internet Archive community-TV videos into timestamped moments with
structured metadata, for a video corpus moment retrieval (VCMR) dataset.

```
IA item folders -> ingest -> Level 1 (detectors) -> correction (moment candidates) -> Level 2 (VLM) -> review -> gold
```

**Status:** ingest, level1, correction and level2 are implemented (level2 = Qwen3.5-4B semantic merging + metadata; control flow tested with a scripted stand-in, real inference runs on the GPU). The review CLI, feedback loop and gold export are checkpoint 3.

## 1. Install

```bash
cd video_metadata_pipeline
# PyTorch first, matching your CUDA (see docs/cluster_setup.md), then:
pip install -e ".[level1,dev]"        # detectors + tests
pip install -e .                      # core only (ingestion, correction, reports)
python -m video_metadata.cli --help
python -m pytest -q                   # unit tests (synthetic fixtures, no models)
```
FFmpeg (`ffmpeg`, `ffprobe`) must be on PATH.

## 2. Point it at the downloaded Internet Archive folders

Item folders can sit anywhere below one corpus folder (e.g. `../data/pilot/<category>/<item>/`). Each item folder
is expected to hold a video plus, optionally, `metadata.json`, `files.json`, `*.srt` / `*.vtt`, a thumbnail. Exact
names are not assumed; folders without a video are reported and skipped.

Paths live in YAML configs (`configs/default.yaml` + overrides). On the development machine `configs/local.yaml`
uses `../data/pilot` and reuses earlier research outputs (`../data/segments`, `../data/level1`).

## 3. Process

```bash
# scan + manifest + ingestion of every valid video
python -m video_metadata.cli ingest --config configs/local.yaml
# detectors + correction for one video
python -m video_metadata.cli run --config configs/local.yaml --stages level1 correction --video-id detvde-Chocolate_Ballerina_Founder_Chane_5b4f44
# all videos (each video is independent; a failure never stops the others)
python -m video_metadata.cli run --config configs/local.yaml --stages ingest level1 correction
python -m video_metadata.cli status --config configs/local.yaml
python -m video_metadata.cli resume --config configs/local.yaml --stages ingest level1 correction   # after an interruption
```
Completed stages are cached per video (`outputs/<id>/pipeline_state.json`); `--force` recomputes. Rerunning a stage
marks later stages `stale`.

## 4. Inspect results

```
outputs/<video_id>/
  source/      source_metadata.json  transcript.json  source_manifest.json  ingestion_report.json
  level1/      raw/ (unmodified detector outputs + provenance.json)  observations.jsonl  window_summaries.jsonl
  correction/  corrected_observations.jsonl  candidate_moments.json  boundary_evidence.json  correction_log.jsonl  correction_report.md
  reports/     level1_timeline.png  correction_timeline.png
  logs/        <stage>.log
```

## 5. Moving to another PC / the cluster
Copy this folder (no outputs, models or videos are bundled), install as above, write a config with that machine's
paths (see `configs/cluster.yaml`) and run the same commands. See `docs/cluster_setup.md`.

**Cluster (Kelvin2):** see `docs/cluster_setup.md` (copy, one-time setup with `scripts/prepare_cluster.sh`, then `sbatch scripts/submit_cluster.sh`).

Docs: `docs/architecture.md`, `docs/correction_rules.md`, `docs/cluster_setup.md`.
