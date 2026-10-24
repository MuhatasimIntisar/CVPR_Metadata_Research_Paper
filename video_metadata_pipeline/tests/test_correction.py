from video_metadata.config import load_config
from video_metadata.correction.moment_proposals import activity_evidence, pool
from video_metadata.correction.observation_filter import step_state
from video_metadata.correction.temporal_consistency import certain_runs, denoise

C = load_config()["correction"]
NO = {"activity_overrides": [], "forced_boundaries": [], "removed_boundaries": []}


def obs(top1="dancing", rel=3.0, margin=2.0, black=False, persons=2):
    return {"black_frame": black, "people": {"n_frames": 4, "count_median": persons},
            "activity": {"top1": top1, "top1_rel": rel, "top1_margin_ratio": margin}}


def test_top1_rule():
    assert step_state(obs(), C)[0] == "dancing"
    assert step_state(obs(rel=1.5), C)[0] == "uncertain"
    assert step_state(obs(margin=1.05), C)[0] == "uncertain"
    assert step_state(obs(black=True), C)[0] == "unobservable"
    assert step_state(obs(persons=0), C)[0] == "no person visible"


def test_noise_floor_bridges_and_drops():
    log = []
    st = denoise(["a", "a", "b", "a", "a", "c", "d", "d"], 2, log, list(range(8)))
    assert st == ["a", "a", "a", "a", "a", "uncertain", "d", "d"]
    assert len(log) == 2


def test_uncertain_gap_does_not_split():
    R = certain_runs(["a", "a", "uncertain", "uncertain", "a", "a"])
    assert len(R) == 1


def test_persistence_rules():
    t = [k * 2 + 1 for k in range(12)]
    ev = activity_evidence(["a"] * 6 + ["b"] * 3 + ["a"] * 3, t, C, 2.0)
    assert ev[0]["type"] == "activity" and ev[0]["weight"] == 1.0          # 6 s -> alone is enough
    ev2 = activity_evidence(["a"] * 6 + ["b"] * 2 + ["a"] * 4, t, C, 2.0)
    assert ev2[0]["type"] == "activity_short" and ev2[0]["weight"] == 0.6  # 4 s -> needs support


def test_pooling_redundant_visual_evidence_and_threshold():
    ev = [{"t": 10.0, "type": "activity_short", "weight": 0.6},
          {"t": 10.5, "type": "visual_transition", "weight": 0.2, "shot_cut": True}]
    g = pool(ev, C, NO, 2.0)
    assert not g[0]["accepted"]                       # short activity + cut alone = 0.8 < 1
    ev.append({"t": 11.0, "type": "visual_transition", "weight": 0.4})
    g = pool(ev, C, NO, 2.0)
    assert g[0]["accepted"] and g[0]["score"] == 1.0  # cut and CLIP jump count once (0.4), not 0.6
    g = pool([{"t": 5.0, "type": "people", "weight": 0.4}, {"t": 5.5, "type": "visual_transition", "weight": 0.4}], C, NO, 2.0)
    assert not g[0]["accepted"]                       # reframing alone never ends a moment


def test_human_overrides_win():
    ev = [{"t": 10.0, "type": "activity", "weight": 1.0}]
    ov = {"activity_overrides": [], "forced_boundaries": [{"t": 30.0, "edit_id": "e1"}],
          "removed_boundaries": [{"t": 10.0, "edit_id": "e2"}]}
    g = {x["t"]: x for x in pool(ev, C, ov, 2.0)}
    assert not g[10.0]["accepted"] and g[10.0]["decided_by"] == "human"
    assert g[30.0]["accepted"] and g[30.0]["decided_by"] == "human"
