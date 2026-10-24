"""Level-2 control flow with a scripted stand-in model (unit tests only; real runs always use the VLM)."""
import json
import logging

import numpy as np
import pytest

from video_metadata.config import load_config
from video_metadata.level2.output_validator import ask, check_metadata, extract_json
from video_metadata.level2.semantic_merging import consolidate, decide, windows
from video_metadata.level2.vlm_backend import CachedVLM

cv2 = pytest.importorskip("cv2")
LOG = logging.getLogger("test")


class Scripted:
    """Answers from a function of the prompt text; counts calls."""
    load_s = 0.0

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def generate(self, content, max_new_tokens):
        self.calls += 1
        return self.fn(" ".join(c["text"] for c in content if c["type"] == "text"))

    def gpu_memory_gb(self):
        return None


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for k in range(400):  # 40 s
        w.write(np.full((48, 64, 3), (k // 100) * 60, np.uint8))
    w.release()
    return p


def cand(i, a, b):
    return {"interval_id": f"c{i:03d}", "start_s": a, "end_s": b, "duration_s": b - a, "status": "moment",
            "activities": [{"state": "dancing", "share_of_steps": 1.0}], "sub_events": [], "environment": [],
            "people_median": 2, "audio": {"speech_share": 0.5, "music_share": 0.0},
            "start_boundary": {"types": ["activity"], "score": 1.0, "reason": "accepted", "decided_by": "rules"},
            "end_boundary": {}}


def test_extract_json_repairs_fences_and_commas():
    assert extract_json('Sure!\n```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}


def test_windows_overlap():
    assert windows(5, 3) == [[0, 1, 2], [1, 2, 3], [2, 3, 4]] and windows(2, 3) == [[0, 1]] and windows(1, 3) == []


def test_reconciliation_conflict_becomes_uncertain(tmp_path, video):
    cfg = load_config()
    cands = [cand(0, 0, 10), cand(1, 10, 20), cand(2, 20, 30), cand(3, 30, 40)]
    state = {"n": 0}

    def fn(text):
        state["n"] += 1
        ids = [b for b in ("B1", "B2", "B3") if "Decide boundaries:" in text and b in text.split("Decide boundaries:")[1]]
        # window 0 says B2 merge, window 1 says B2 keep -> conflict
        d = {"B1": "merge", "B2": "merge" if state["n"] == 1 else "keep", "B3": "keep"}
        return json.dumps({"decisions": [{"boundary": b, "decision": d[b], "explanation": "frames at 9 s and 11 s"} for b in ids]})

    vlm = CachedVLM(cfg, "cpu", tmp_path / "cache", "h", Scripted(fn))
    D, audit = decide(vlm, cfg, video, cands, {"segments": []}, {}, LOG)
    by = {d["boundary"]: d for d in D}
    assert by["B1"]["decision"] == "merge" and not by["B1"]["conflict"]
    assert by["B2"]["decision"] == "uncertain" and by["B2"]["conflict"] and by["B2"]["needs_review"]
    assert len(by["B2"]["votes"]) == 2 and by["B3"]["decision"] == "keep"
    M = consolidate("v", cands, D)
    assert [m["source_intervals"] for m in M] == [["c000", "c001"], ["c002"], ["c003"]]
    assert M[0]["end_s"] == 20 and M[1]["start_boundary"]["decision"] == "uncertain"
    calls = vlm.backend.calls
    decide(vlm, cfg, video, cands, {"segments": []}, {}, LOG)  # cached: no new model calls
    assert vlm.backend.calls == calls


def test_human_boundary_never_sent_to_vlm(tmp_path, video):
    cfg = load_config()
    cands = [cand(0, 0, 10), cand(1, 10, 20)]
    cands[1]["start_boundary"]["decided_by"] = "human"
    vlm = CachedVLM(cfg, "cpu", tmp_path / "c", "h", Scripted(lambda t: "{}"))
    D, _ = decide(vlm, cfg, video, cands, {"segments": []}, {}, LOG)
    assert D[0]["decision"] == "keep" and D[0]["decided_by"] == "human" and vlm.backend.calls == 0


def test_invalid_output_bounded_retries_never_invents(tmp_path):
    cfg = load_config()
    vlm = CachedVLM(cfg, "cpu", tmp_path / "c", "h", Scripted(lambda t: "I cannot comply"))
    obj, attempts = ask(vlm, [{"type": "text", "text": "x"}], 50, check_metadata, 2, "t")
    assert obj is None and len(attempts) == 3 and vlm.backend.calls == 3


def test_metadata_schema_checks_entity_references():
    ok = {"description": "d", "tags": [], "entities": [{"id": "p1", "type": "person"}], "relationships": [], "context": {},
          "events": [{"type": "dance", "participants": ["p1"]}]}
    assert check_metadata(ok) == []
    bad = dict(ok, events=[{"type": "dance", "participants": ["ghost"]}])
    assert check_metadata(bad)
