from src.annotate.base import Annotator

ANNOTATORS = [
    Annotator(
        name="location_scene", version="0.0-stub", dimension="location",
        fields=["location.L1_environment", "location.L2_place_type"], inputs=["keyframes"],
        references=["Places365 — Zhou et al., 'Places: A 10 Million Image Database for Scene Recognition', TPAMI 2018"],
    ),
    Annotator(
        name="location_geo", version="0.0-stub", dimension="location",
        fields=["location.L3_named_location"], inputs=["keyframes"],
        references=["GeoCLIP — Vivanco Cepeda et al., 'GeoCLIP: Clip-Inspired Alignment between Locations and Images', NeurIPS 2023",
                    "PaddleOCR — https://github.com/PaddlePaddle/PaddleOCR (signage / on-screen text)"],
    ),
]
