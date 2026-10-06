from src.annotate.base import Annotator

ANNOTATORS = [
    Annotator(
        name="cine_shot_scale", version="0.0-stub", dimension="cinematography",
        fields=["cinematography.L1_shot_scale", "cinematography.L3_viewpoint"], inputs=["keyframes"],
        references=["Rao et al., 'A Unified Framework for Shot Type Classification Based on Subject Centric Lens', ECCV 2020"],
    ),
    Annotator(
        name="cine_camera_motion", version="0.0-stub", dimension="cinematography",
        fields=["cinematography.L2_camera_motion"], inputs=["clips"],
        references=["RAFT optical flow — Teed & Deng, 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow', ECCV 2020",
                    "CameraBench — Lin et al., 'Towards Understanding Camera Motions in Any Video', 2025"],
    ),
]
