from src.annotate.base import Annotator

ANNOTATORS = [
    Annotator(
        name="time_illumination", version="0.0-stub", dimension="time",
        fields=["time.L1_illumination", "time.L2_daypart"], inputs=["keyframes"],
        references=["CLIP zero-shot — Radford et al., 'Learning Transferable Visual Models From Natural Language Supervision', ICML 2021"],
    ),
    Annotator(
        name="time_estimate", version="0.0-stub", dimension="time",
        fields=["time.L3_time_estimate"], inputs=["keyframes", "info"],
        references=["PaddleOCR — https://github.com/PaddlePaddle/PaddleOCR (visible clocks / timestamps)",
                    "Upload date from YouTube info.json as a prior"],
    ),
]
