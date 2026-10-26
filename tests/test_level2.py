"""Level-2 control flow with a scripted stand-in model (unit tests only; real runs always use the VLM)."""
import json
import logging

import numpy as np
import pytest

from video_metadata.config import load_config
from video_metadata.level2.output_validator import ask, check_metadata, extract_json
from video_metadata.level2.semantic_merging import consolidate, decide, reconcile
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


def test_one_call_per_boundary_and_consolidation(tmp_path, video):
    cfg = load_config()
    cands = [cand(0, 0, 10), cand(1, 10, 20), cand(2, 20, 30), cand(3, 30, 40)]

    def fn(text):
        t = float(text.split("THE CUT IS AT ")[1].split(" s")[0])
        return json.dumps({"explanation": f"frames around {t:g} s", "decision": "merge" if t == 10 else "keep"})

    vlm = CachedVLM(cfg, "cpu", tmp_path / "cache", "h", Scripted(fn))
    D, audit = decide(vlm, cfg, video, cands, {"segments": []}, {}, LOG)
    assert vlm.backend.calls == 3 and len(audit) == 3
    assert [d["decision"] for d in D] == ["merge", "keep", "keep"] and not any(d["conflict"] for d in D)
    M = consolidate("v", cands, D)
    assert [m["source_intervals"] for m in M] == [["c000", "c001"], ["c002"], ["c003"]]
    assert M[0]["end_s"] == 20 and M[1]["start_boundary"]["decision"] == "keep"
    decide(vlm, cfg, video, cands, {"segments": []}, {}, LOG)  # cached: no new model calls
    assert vlm.backend.calls == 3


def test_reconcile_disagreement_becomes_uncertain():
    b = {1: {"id": "B1", "t": 10, "left": "c000", "right": "c001", "evidence": {}}}
    v = {1: [{"decision": "merge", "explanation": "a"}, {"decision": "keep", "explanation": "b"}]}
    d = reconcile(b, v)[0]
    assert d["decision"] == "uncertain" and d["conflict"] and d["needs_review"] and len(d["votes"]) == 2


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


def test_metadata_v2_checks_and_normalisation():
    from video_metadata.level2.metadata_generator import normalise
    ok = {"description": "d", "tags": ["Ballet", "ballet", "stage_performance"], "context": {"setting": "stage", "location": "unknown"},
          "attributes": {"action": "dance", "time_of_day": "morning"},
          "entities": [{"id": "p1", "type": "person", "name": "Ann", "name_source": None, "evidence": "visual"},
                       {"id": "p2", "type": "person_group", "count": "6-10", "evidence": "visual"}],
          "actions": [{"type": "performing_ballet", "participants": ["p1", "p2"]}],
          "relationships": [{"subject": "p1", "predicate": "dances with", "object": "p2"},
                            {"subject": "p1", "predicate": "is", "object": "p2"},
                            {"subject": "p1", "predicate": "wears", "object": "white tutu"},
                            {"subject": "p1", "predicate": "watches", "object": "p1"}]}
    assert check_metadata(ok) == []
    n, rep = normalise(ok)
    assert n["tags"] == ["ballet", "stage performance"] and n["context"]["location"] is None
    assert "time_of_day" not in n["attributes"] and n["entities"][0]["name"] is None  # name without a source removed
    assert n["actions"][0]["type"] == "perform_ballet"
    assert n["relationships"] == [{"subject": "p1", "predicate": "dances_with", "object": "p2"}]
    assert len(rep) >= 6
    assert check_metadata(dict(ok, actions=[{"type": "x", "participants": ["ghost"]}]))
    assert check_metadata(dict(ok, entities=[{"id": "p1", "type": "robot", "evidence": "visual"}]))
