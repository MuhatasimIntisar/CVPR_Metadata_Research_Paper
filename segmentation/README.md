# Training-free multimodal segmentation pilot

Compares PySceneDetect (M1), CLIP (M2), CLAP (M3), Sentence-BERT (M4) and late fusion (M5, mean and max)
plus ablations, on MovieChat-SSeg scene boundaries and AMI topic boundaries, evaluated separately.

```powershell
pip install -r requirements-segmentation.txt
python -m segmentation.run_experiment                      # validate -> scenedetect -> encode -> detect -> evaluate -> plot -> report
python -m segmentation.run_experiment --steps validate     # inputs and timestamp alignment only
python -m segmentation.run_experiment --steps detect evaluate plot report --config params.json   # re-run without inference
python -m segmentation.run_experiment --steps transcribe   # optional: Whisper transcripts for MovieChat (enables M4 there)
```

Default data folder: `data/AMI_ES2002_segmentation_reference` (`--data-dir` to change). Source files are
never modified. Outputs are written next to them:

| Folder | Contents |
|---|---|
| `cache/` | `<id>_clip.npz/.json`, `<id>_clap.npz/.json`, `<id>_text.npz/.json`: embeddings, timestamps, model id/revision, sampling settings, runtime |
| `predictions/` | `<id>__<method>.json` (boundaries in seconds, status, modalities used, runtime) and `<id>__curves.npz` (raw change scores) |
| `evaluation/` | `validation.json`, `per_video.csv`, `dataset_summary.csv`, `results.md`, `report.md`, `config_used.json`, `runs_log.jsonl` |
| `plots/` | `<id>_boundaries.png` (methods vs human boundaries), `<id>_scores.png` (per-modality and fused change scores) |

Pipeline (M2-M5): embeddings -> change score at each 2 s grid point (cosine distance between mean
embeddings of the context before and after) -> per-modality robust z-score (clipped ±5) -> availability-
aware fusion (missing evidence never counts as 0) -> moving average -> `scipy.signal.find_peaks`
(prominence, minimum segment 15 s). Parameters are in `segmentation/config.py`; override any field
with `--config file.json`. Reference boundaries are only read by evaluation and plotting.

Matching is one-to-one within the tolerance (Hungarian assignment). Dataset scores are micro-averaged
over videos; MovieChat and AMI are never pooled.

## Experiment A: temporal context and minimum segment duration

```powershell
python -m segmentation.run_experiment --steps encode --modalities text    # once: transcript chunk embeddings (seconds)
python -m segmentation.experiment_a                                         # MovieChat ctx 5/10/20 x min 15/30/45; AMI ctx 30/60/90 x min 60/90/120
```

Reuses cached CLIP / CLAP / transcript-chunk embeddings (no inference). Four independent parameters:
embedding chunk length (`text_chunk_s`, `text_chunk_hop_s`; CLIP 1 fps; CLAP window/hop), context window
(swept, applied to every modality), grid step (`grid_s`) and minimum segment (swept). Outputs in
`evaluation/experiment_A/`: `predictions.jsonl`, `per_video.csv`, `summary.csv`, `summary.md`,
`heatmap_<dataset>.png`, `params.json`. Each score has a chance F1 (same number of random boundaries) and the
number of reference boundaries reachable under that minimum segment.
