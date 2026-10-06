from src.annotate.base import Annotator

ANNOTATORS = [
    Annotator(
        name="action_vlm", version="0.0-stub", dimension="action",
        fields=["action.L1_activity_family", "action.L2_specific_action", "action.L3_interaction"],
        inputs=["clips"],
        references=["Qwen2.5-VL — Bai et al., 'Qwen2.5-VL Technical Report', arXiv:2502.13923 (2025), fixed label vocabulary",
                    "VideoMAE V2 — Wang et al., 'VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking', CVPR 2023"],
    ),
]
