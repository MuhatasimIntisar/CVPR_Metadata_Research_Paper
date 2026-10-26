import numpy as np

from video_metadata.config import load_config
from video_metadata.correction.moment_proposals import pool, topic_evidence
from video_metadata.level1.observations import topic_curve
from video_metadata.level2.pipeline import children

C = load_config()


def test_topic_curve_detects_change_and_needs_words(tmp_path):
    t = np.arange(2.5, 600, 5.0)
    e = np.where((t < 300)[:, None], np.eye(8)[0], np.eye(8)[1]).astype(np.float32)
    np.savez(tmp_path / "transcript_topics.npz", times=t.astype(np.float32), n_words=np.full(len(t), 10, np.int32), emb=e)
    grid = np.arange(1, 600, 2.0)
    cur = topic_curve(tmp_path, grid, C["level1"]["topics"])
    vals = {g: v[0] for g, v in zip(grid, cur)}
    assert vals[301.0] > 0.4 and vals[151.0] == 0.0
    assert vals[1.0] is None  # not enough words before t=1 s


def test_topic_peak_is_a_boundary_on_its_own():
    obs = []
    for k in range(300):
        t = 1 + 2 * k
        ch = 0.05 + (0.6 if 290 <= t <= 310 else 0.0) + 0.01 * np.sin(k)
        obs.append({"t": t, "topic": {"change": ch, "words_before": 50, "words_after": 50}})
    ev = topic_evidence(obs, C["correction"], 2.0, 600.0)
    top = max(ev, key=lambda e: e["z"])
    assert 290 <= top["t"] <= 310 and top["z"] >= 4.0  # the real change dominates; small wiggles stay low
    g = pool([top], C["correction"], {"activity_overrides": [], "forced_boundaries": [], "removed_boundaries": []}, 2.0)
    assert g[0]["accepted"]


def test_children_join_short_pieces():
    src = [{"interval_id": f"c{i}", "start_s": a, "end_s": b} for i, (a, b) in enumerate([(0, 100), (100, 110), (110, 300), (300, 305)])]
    ev = {"moment_id": "v_m000", "video_id": "v", "start_boundary": {}, "end_boundary": {}, "needs_review": False}
    kids = children(ev, src, {}, C["level2"])
    assert [k["source_intervals"] for k in kids] == [["c0"], ["c1", "c2", "c3"]]  # short piece joins the next; the last joins the previous
    assert kids[0]["moment_id"] == "v_m000.01" and kids[-1]["end_s"] == 305
