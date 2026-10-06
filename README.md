# CVPR_Metadata_Research_Paper

Pipeline for building a compositional, metadata-based video retrieval dataset from real-world YouTube footage.
Short clips are annotated with 12 structured fields (location, time, action, cinematography × 3 levels).

## Layout

```
configs/      taxonomy.yaml (the 12 fields + search seeds) and one config per machine
src/
  discovery/  (1) taxonomy -> search queries -> candidate video IDs (no downloads)
  ingest/     (2) download -> shot segmentation -> cheap filters -> clips -> shard; raw video deleted
  annotate/   (3) one module per dimension + fuse.py (currently stubs that print + cite references)
  common/     config, manifest, shard I/O
scripts/      entry points; slurm/ for Kelvin
data/         gitignored; test_env data lives here
```

## Environments

| `--env`       | Machine            | Role                                          |
|---------------|--------------------|-----------------------------------------------|
| `test_env`    | laptop             | tiny batches to debug the full chain           |
| `storage_env` | professor's PC     | full download + ingest, long-term shard store  |
| `live`        | Kelvin cluster     | annotation + embeddings (SLURM array per shard)|

Code is identical everywhere; only `configs/<env>.yaml` (mainly `data_root`) differs.

## Data layout (under `data_root`)

```
manifest/videos.parquet      one row per video ID: status (candidate/ingested/failed), shard, counts
discovery/candidates.parquet search results from the last discovery run
shards/shard-00000/          immutable once _SUCCESS exists
    clips.tar                kept shots, 480p H.264 + mono AAC
    keyframes.tar            one 384px JPEG per kept shot
    shots.parquet            every detected shot, kept or discarded (with reason), caption text
    info.parquet             trimmed YouTube metadata (title, description, upload date, channel ...)
annotations/<annotator>/<version>/shard-00000.parquet
annotations/fused/<version>/shard-00000.parquet
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux: source .venv/bin/activate)
pip install -r requirements.txt
ffmpeg -version                   # must be on PATH
```

## Test run

```
python scripts/run_discovery.py --env test_env
python scripts/run_ingest.py    --env test_env
python scripts/run_annotate.py  --env test_env
python scripts/inspect_shard.py --env test_env --extract 5
```

Re-running is safe: ingest only takes `candidate` videos, annotation skips outputs that already exist (`--force` to redo).

## Troubleshooting

- **YouTube download errors / missing formats:** update yt-dlp (`pip install -U "yt-dlp[default]"`). Recent yt-dlp versions need a JavaScript runtime (e.g. Deno) for full YouTube support.
- **HTTP 429 / bot checks:** increase `sleep_between_downloads_s`; captions are fetched separately so caption rate-limits don't fail the video.
