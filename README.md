# TVM metadata pipeline

Turns Internet Archive community-TV videos into timestamped moments with structured, evidence-checked metadata,
for a video corpus moment retrieval (VCMR) dataset.

```
collection (pick + download) -> ingest -> Level 1 (detectors) -> correction (candidate moments)
  -> Level 2 (Qwen3.5-4B: merge decisions + moment metadata) -> review -> gold
```
Status: everything up to Level 2 is implemented (schema `vmp-2.1`, `docs/metadata_schema.md`); review / gold export
is the next checkpoint.

## Repository layout

| Folder | What it holds | In git |
|---|---|---|
| `src/video_metadata/` | the pipeline: `ingestion/`, `level1/`, `correction/`, `level2/`, `utils/`, `cli.py` | yes |
| `configs/` | `default.yaml` (every parameter), `local.yaml` (laptop), `cluster.yaml` (Kelvin2), `models.yaml` | yes |
| `scripts/` | Kelvin2: `prepare_cluster.sh`, `prepare_vllm.sh`, `check_vlm.sh`, `submit_cluster.sh` | yes |
| `tests/` | unit + integration tests (synthetic fixtures, no models): `python -m pytest -q` | yes |
| `docs/` | `architecture.md`, `correction_rules.md`, `metadata_schema.md`, `cluster_setup.md` | yes |
| `collection/` | `ia_download.py` (search -> pick -> download) and `rounds/` (candidate, pick and manifest CSVs per round) | yes |
| `corpus/` | downloaded videos, `<category>/<item>/`; laptop staging before copying to Kelvin | no |
| `outputs/` | per-video working outputs (full set lives on Kelvin; only inspected videos here) | no |
| `review/` | human edits per video, `<video_id>/overrides.json`; the input to gold | yes |
| `metadata/` | **the dataset**: `<original video name>_metadata.json` (+ readable `.md`) per video | yes |
| `models/` | YOLO weights for laptop runs | no |
| `archive/` | earlier phases (segmentation experiments, Level-1 prototypes, discovery tools, reference datasets, old runs); not used by the pipeline | code yes, data no |

## Workflow
1. **Pick + download** (laptop, off the university network - it blocks archive.org storage):
   `python collection/ia_download.py search` -> mark `x` in `collection/rounds/candidates.csv` ->
   `python collection/ia_download.py download collection/rounds/candidates.csv` (videos land in `corpus/`).
   Keep each round's CSVs in `collection/rounds/` (`roundN_*.csv`): they document how the corpus was chosen.
2. **Copy videos to Kelvin**: `scp -r corpus/* <kelvin>:/mnt/scratch2/users/<user>/vmp/data/pilot/`
3. **Code on Kelvin**: `git pull` in `vmp/code` (see `docs/cluster_setup.md`; first time: `git clone`).
4. **Run**: `sbatch scripts/check_vlm.sh` once after setup changes, then `sbatch scripts/submit_cluster.sh`.
5. **Bring back the dataset files**: `scp <kelvin>:/mnt/scratch2/users/<user>/vmp/metadata/* metadata/`, commit.
6. **Review**: human edits go to `review/<video_id>/`; re-running correction/Level 2 applies them (they always win).

## Branches
- `metadata-phase`: the working pipeline (what Kelvin runs).
- `scratch`: Claude's changes are pushed here; pull it, test, then merge into `metadata-phase`.

## Local use (laptop)
```bash
pip install -e ".[level1,dev]"     # PyTorch for your CUDA first; FFmpeg on PATH
python -m video_metadata.cli ingest --config configs/local.yaml
python -m video_metadata.cli run --config configs/local.yaml --stages level1 correction --video-id <id>
python -m video_metadata.cli status --config configs/local.yaml
```
Completed stages are cached per video (`outputs/<id>/pipeline_state.json`); `--force` recomputes; rerunning a stage
marks later stages stale. Stages run stage by stage over all videos; a failed video never stops the others.

Videos are never redistributed: the dataset is the metadata plus Internet Archive identifiers and URLs.
