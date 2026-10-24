"""Per-video, per-stage log files plus console output."""
from __future__ import annotations

import logging
from pathlib import Path


def stage_logger(log_dir: Path, video_id: str, stage: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"vmp.{video_id}.{stage}"
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        fh = logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(f"[{video_id[:28]}:{stage}] %(message)s"))
        lg.addHandler(fh)
        lg.addHandler(sh)
        lg.propagate = False
    return lg
