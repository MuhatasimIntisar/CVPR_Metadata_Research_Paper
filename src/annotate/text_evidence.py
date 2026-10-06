from src.annotate.base import Annotator

# Reads the YouTube description/title and the per-shot caption text; contributes evidence
# mainly for named location and time estimate. Fusion decides how much to trust it per shot.
ANNOTATORS = [
    Annotator(
        name="text_evidence", version="0.0-stub", dimension="text",
        fields=["location.L3_named_location", "time.L3_time_estimate", "action.L2_specific_action"],
        inputs=["captions", "info"],
        references=["GLiNER — Zaratiana et al., 'GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer', NAACL 2024"],
    ),
]
