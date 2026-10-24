# Architecture

| Stage | Module | Input | Output (outputs/<id>/) | Models |
|---|---|---|---|---|
| ingest | `ingestion/` | IA item folder | `source/` | none |
| level1 | `level1/` | video, transcript | `level1/raw/*`, `observations.jsonl`, `window_summaries.jsonl` | YOLO11s+ByteTrack, YOLO11n-pose, CLIP B/32, X-CLIP B/32, Silero VAD, ECAPA diarisation, AST audio events, EasyOCR, PySceneDetect, optical flow |
| correction | `correction/` | observations | `correction/*` | none (deterministic rules) |
| level2 | `level2/` (checkpoint 2) | video frames, corrected evidence, transcript | `level2/*` | Qwen3.5-4B |
| review | `review/` (checkpoint 3) | Level-2 output + reviewer edits | `review/`, `gold/` | none |

Principles
- Source metadata (titles, tags, descriptions) is video-level context, never moment ground truth.
- Transcripts are timestamped linguistic evidence (YouTube automatic captions in this corpus), not ground truth.
- Level 1 keeps raw detector outputs unchanged and model scores alongside labels; CLIP/X-CLIP scores are relative
  within-head softmax values, not calibrated probabilities.
- Every stage is cached, resumable and isolated per video; reruns mark later stages stale.
- Automatic, corrected, reviewed and gold records stay separate (checkpoint 3).

Transcript policy: caption tracks are parsed (rolling YouTube captions de-duplicated, inline word times kept); the
primary track is the item's own language (other languages are likely machine translations), word-timed, highest
coverage; if none is adequate, an earlier Whisper output is reused (`ingestion.fallback_asr: legacy`) or Whisper is
run (`run`). Diarisation uses the selected transcript's word times (no ASR needed).
