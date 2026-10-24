"""Experiment configuration and dataset discovery.

All parameters live in Config; every run writes a snapshot of it next to its outputs, so any
change (tuning) is recorded. Nothing here reads reference boundaries except `reference_boundaries`,
which is only called by evaluation and plotting.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path("data") / "AMI_ES2002_segmentation_reference"


@dataclass
class Config:
    # models
    clip_model: str = "openai/clip-vit-base-patch32"
    clap_model: str = "laion/clap-htsat-unfused"          # LAION-CLAP (HTSAT), via transformers
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    # sampling
    visual_fps: float = 1.0
    audio_window_s: float = 5.0
    audio_hop_s: float = 2.0
    audio_sr: int = 48000                                 # CLAP's required rate
    text_chunk_s: float = 10.0                            # embedding chunk length (encoded once)
    text_chunk_hop_s: float = 5.0                         # chunk hop -> overlapping chunks
    text_min_words: int = 8                               # fewer words in a context side -> text unavailable there
    # change scores (context windows before/after each grid time)
    grid_s: float = 2.0
    visual_context_s: float = 8.0
    audio_context_s: float = 10.0
    text_context_s: float = 20.0                          # transcript aggregated on each side of a candidate time
    min_context_coverage: float = 0.5                     # share of context samples needed on each side
    # boundary detection (shared by M2-M5)
    normalise: str = "robust_z"                           # robust_z | z | minmax
    score_clip: float = 5.0                               # cap on |normalised score|
    smooth_window: int = 3                                # moving average over grid points (1 = off)
    peak_prominence: float = 1.0                          # in normalised units
    peak_height: float | None = None
    min_segment_s: float = 15.0
    # fusion
    fusion_weights: dict = field(default_factory=lambda: {"visual": 1.0, "audio": 1.0, "text": 1.0})
    # PySceneDetect (M1) - library defaults
    psd_threshold: float = 27.0
    psd_min_scene_len: int = 15
    # evaluation tolerances (seconds), per dataset
    tolerances: dict = field(default_factory=lambda: {"moviechat": [2.0, 5.0], "ami": [5.0, 10.0]})
    # parameter selection may only use these videos (none by default = no tuning)
    dev_videos: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_file(cls, path: Path | None):
        if path is None or not Path(path).exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


@dataclass
class VideoItem:
    video_id: str
    dataset: str                # "moviechat" | "ami"
    video: Path | None
    audio: Path | None          # separate audio file (AMI); None -> take audio from the video
    transcript: Path | None     # oracle transcript (AMI); None -> not available
    reference_csv: Path

    def to_dict(self):
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}


def _first(paths):
    paths = sorted(paths)
    return paths[0] if paths else None


def discover(data_dir: Path) -> list[VideoItem]:
    """MovieChat: <id>.mp4 + <id>_scene_boundaries.csv. AMI: <meeting>.{Overhead,Corner,...}.avi +
    <meeting>.Mix-Headset.wav + <meeting>_transcript.json + <meeting>_topic_boundaries.csv."""
    items = []
    for ref in sorted(data_dir.glob("*_scene_boundaries.csv")):
        vid = ref.name[: -len("_scene_boundaries.csv")]
        items.append(VideoItem(vid, "moviechat", _first(data_dir.glob(f"{vid}.mp4")), None, None, ref))
    for ref in sorted(data_dir.glob("*_topic_boundaries.csv")):
        mid = ref.name[: -len("_topic_boundaries.csv")]
        video = _first(p for p in data_dir.glob(f"{mid}.*.avi"))
        audio = _first(data_dir.glob(f"{mid}.Mix-Headset.wav"))
        tr = data_dir / f"{mid}_transcript.json"
        items.append(VideoItem(mid, "ami", video, audio, tr if tr.exists() else None, ref))
    return items


def reference_boundaries(item: VideoItem) -> list[float]:
    """Evaluation/plotting only. Never passed to any segmentation method."""
    with open(item.reference_csv, encoding="utf-8") as f:
        return sorted(float(r["time_seconds"]) for r in csv.DictReader(f))


def out_dirs(data_dir: Path) -> dict:
    d = {k: data_dir / k for k in ("cache", "predictions", "evaluation", "plots")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d
