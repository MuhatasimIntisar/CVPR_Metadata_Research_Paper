# CVPR_Metadata_Research_Paper

Pipeline for building a compositional, metadata-based video retrieval dataset from real-world YouTube footage.
Short clips are annotated with 12 structured fields (location, time, action, cinematography × 3 levels).

## Layout

```
configs/      taxonomy.yaml (fixed 12-field label schema), search_pools.yaml (search terms), one config per machine
src/
  discovery/  (1) coverage-balanced random queries -> candidate video IDs (no downloads)
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
discovery/query_log.parquet  every search result: batch, seed, query, combo, channel, added or not
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

## Discovery (query generation)

Search terms live in `configs/search_pools.yaml`, not the taxonomy. Each slot maps a canonical key
(e.g. `protesting`) to several phrasings (`protest`, `march`, `rally`). Discovery runs in batches:

1. Sample `action x place x location` combos, weighted toward combos with few videos so far
   (`1/(1+count)`; combos at `max_per_combo` are skipped) -> even coverage.
2. Pick random phrasings + a random style, search YouTube.
3. Apply caps: `results_per_query`, `max_per_channel`, `max_per_combo`; skip known IDs.
4. Append to the manifest and `query_log.parquet` (with batch id + seed, so it's reproducible).

Repeat until `--n-videos x oversample` new candidates. Interrupting is safe.
`--report` prints coverage without searching.

## Expanding search terms with a local model

`scripts/expand_search_pools.py` asks a local Ollama model for extra phrasings of each value in
`search_pools.yaml` (actions, place types, styles; cities are fixed and not expanded). It only adds
phrasings to existing keys, never new keys, and drops suggestions that mention other slots (e.g. a
city inside an action phrasing).

```
ollama pull qwen2.5:7b                       # or qwen2.5:3b if memory is tight
python scripts/expand_search_pools.py --dry-run
python scripts/expand_search_pools.py        # writes configs/search_pools.expanded.yaml
```

Review the output, delete bad phrasings, set `_provenance.reviewed: true`, then set
`search_pools: search_pools.expanded.yaml` in the env config. The baseline file is never modified.

## Test run

```
python scripts/run_discovery.py --env test_env --n-videos 10
python scripts/run_ingest.py    --env test_env
python scripts/run_annotate.py  --env test_env
python scripts/inspect_shard.py --env test_env --extract 5
```

Re-running is safe: ingest only takes `candidate` videos, annotation skips outputs that already exist (`--force` to redo).

## Troubleshooting

- **YouTube download errors / missing formats:** update yt-dlp (`pip install -U "yt-dlp[default]"`). Recent yt-dlp versions need a JavaScript runtime (e.g. Deno) for full YouTube support.
- **HTTP 429 / bot checks:** increase `sleep_between_downloads_s`; captions are fetched separately so caption rate-limits don't fail the video.
