"""Batched Level-2 inference: same prompts/cache as one-by-one, retries only for invalid items, stage-major runner."""
import json
import logging

import pytest

from video_metadata import pipeline as P
from video_metadata.config import load_config
from video_metadata.level2.output_validator import ask_batch, check_single
from video_metadata.level2.vlm_backend import CachedVLM, VLLMBackend

cv2 = pytest.importorskip("cv2")
from test_level2 import cand, video  # noqa: E402,F401  (fixture)


class Batched:
    load_s = 0.0
    name = "test"

    def __init__(self, fn):
        self.fn, self.batches = fn, []

    def generate_batch(self, contents, ns):
        self.batches.append(len(contents))
        return [self.fn(" ".join(c["text"] for c in x if c["type"] == "text")) for x in contents]

    def gpu_memory_gb(self):
        return None


def answer(t):
    return json.dumps({"explanation": "same scene", "decision": "merge" if "c001" in t.split("RIGHT =")[-1] else "keep"})


def test_batched_decisions_equal_one_by_one(tmp_path, video):  # noqa: F811
    from video_metadata.level2.semantic_merging import decide
    cands = [cand(i, 10 * i, 10 * (i + 1)) for i in range(4)]
    out = {}
    for bs in (64, 1):
        cfg = load_config()
        cfg["level2"]["batch_size"] = bs
        be = Batched(answer)
        vlm = CachedVLM(cfg, "cpu", tmp_path / f"c{bs}", "h", be)
        D, _ = decide(vlm, cfg, video, cands, {}, {}, logging.getLogger("t"))
        out[bs] = ([d["decision"] for d in D], sorted(p.name for p in (tmp_path / f"c{bs}").iterdir()), be.batches)
    assert out[64][0] == out[1][0] and out[64][1] == out[1][1]  # same answers, same cache files
    assert out[64][2] == [3] and out[1][2] == [1, 1, 1]          # 3 boundaries: one batch vs three


def test_retry_round_only_resends_invalid(tmp_path):
    cfg = load_config()
    seen = {}

    def fn(t):
        seen[t] = seen.get(t, 0) + 1
        return "not json" if "bad" in t and "rejected" not in t else '{"explanation": "x", "decision": "keep"}'
    be = Batched(fn)
    vlm = CachedVLM(cfg, "cpu", tmp_path, "h", be)
    items = [([{"type": "text", "text": "good"}], 20, "a"), ([{"type": "text", "text": "bad"}], 20, "b")]
    res = ask_batch(vlm, items, check_single, 2)
    assert [r[0]["decision"] for r in res] == ["keep", "keep"] and be.batches == [2, 1]
    assert len(res[1][1]) == 2 and res[1][1][0]["errors"]


def test_vllm_prompt_keeps_image_order_and_checks_placeholders():
    class Tok:
        def apply_chat_template(self, msgs, **kw):
            assert kw["enable_thinking"] is False
            return "".join("<|image_pad|>" if c["type"] == "image" else c["text"] for c in msgs[0]["content"])
    b = object.__new__(VLLMBackend)
    b.tok, b.max_images = Tok(), 4
    r = b.prompt([{"type": "text", "text": "A"}, {"type": "image", "image": "i1"}, {"type": "text", "text": "B"},
                  {"type": "image", "image": "i2"}])
    assert r["prompt"] == "A<|image_pad|>B<|image_pad|>" and r["multi_modal_data"]["image"] == ["i1", "i2"]
    with pytest.raises(RuntimeError):
        b.prompt([{"type": "image", "image": x} for x in range(5)])


def test_runner_is_stage_major(tmp_path, monkeypatch):
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"videos": [{"video_id": v, "status": "ok"} for v in ("a", "b")]}))
    order = []
    monkeypatch.setattr(P, "run_stage", lambda cfg, m, e, s, f: order.append((s, e["video_id"])) or
                        ("failed" if (s, e["video_id"]) == ("level1", "a") else "done"))
    P.run({}, man, ["ingest", "level1", "correction", "level2"])
    assert order == [("ingest", "a"), ("ingest", "b"), ("level1", "a"), ("level1", "b"), ("correction", "b"), ("level2", "b")]
