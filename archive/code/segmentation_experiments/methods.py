"""Method definitions. M2-M5 and ablations differ only in which cached change curves they use and
how they combine them; all share boundary_detection.detect()."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    id: str
    label: str
    modalities: tuple = ()
    fusion: str | None = None        # None (single modality) | "mean" | "max"
    require_all: bool = True         # ablations need every listed modality; M5 uses what is available
    group: str = "main"              # main | ablation


METHODS = [
    Method("M1_pyscenedetect", "PySceneDetect (ContentDetector)"),
    Method("M2_clip", "CLIP-only", ("visual",)),
    Method("M3_clap", "CLAP-only", ("audio",)),
    Method("M4_sbert", "Sentence-BERT (speech)", ("text",)),
    Method("M5_fusion_mean", "Fusion: CLIP+CLAP+SBERT (mean)", ("visual", "audio", "text"), "mean", False),
    Method("M5_fusion_max", "Fusion: CLIP+CLAP+SBERT (max)", ("visual", "audio", "text"), "max", False),
    Method("A_clip_clap", "Ablation: CLIP+CLAP (mean)", ("visual", "audio"), "mean", True, "ablation"),
    Method("A_clip_sbert", "Ablation: CLIP+SBERT (mean)", ("visual", "text"), "mean", True, "ablation"),
    Method("A_clip_clap_sbert", "Ablation: CLIP+CLAP+SBERT (mean)", ("visual", "audio", "text"), "mean", True,
           "ablation"),
]
BY_ID = {m.id: m for m in METHODS}
