"""Schema v2.1 fixes: name rules, camera vocabularies, OCR branding, short-interval absorption, shot scale, review."""
from video_metadata.correction.moment_proposals import absorb_short
from video_metadata.level1.observations import shot_scale
from video_metadata.level2.metadata_generator import normalise
from video_metadata.level2.pipeline import ocr_filter, review_reasons, useful_text, verify_names, words

BASE = {"description": "d", "tags": [], "actions": [], "relationships": [], "context": {}}
SC = {"person_conf": 0.5, "kpt_conf": 0.5, "min_box_h": 0.08, "extreme_wide_h": 0.2}


def test_names_one_entity_proper_and_sourced():
    o = dict(BASE, attributes={}, entities=[
        {"id": "p1", "type": "person", "name": "Ann Lee", "name_source": "transcript"},
        {"id": "p2", "type": "person", "name": "ann lee", "name_source": "transcript"},
        {"id": "g1", "type": "person_group", "name": "dancers", "name_source": "transcript"},
        {"id": "p3", "type": "person", "name": "Bo", "name_source": "visual"},
        {"id": "p4", "type": "person", "name": "Cy", "name_source": "on_screen_text"}])
    n, rep = normalise(o)
    names = {e["id"]: e["name"] for e in n["entities"]}
    assert names == {"p1": None, "p2": None, "g1": None, "p3": None, "p4": "Cy"}
    assert sum(r.startswith("name removed") for r in rep) == 4


def test_camera_vocab_from_field_or_note():
    n, _ = normalise(dict(BASE, entities=[], attributes={"shot_scale": "Close-Up", "cinematography": "slow pan"}))
    assert n["attributes"]["shot_scale"] == "close_up" and n["attributes"]["camera_motion"] == "pan"
    n, _ = normalise(dict(BASE, entities=[], attributes={"cinematography": "medium wide shot, static camera"}))
    assert (n["attributes"]["shot_scale"], n["attributes"]["camera_motion"]) == ("medium_wide", "static")
    n, _ = normalise(dict(BASE, entities=[], attributes={"shot_scale": "dramatic"}))
    assert n["attributes"]["shot_scale"] is None


def test_ocr_branding_filter():
    items = lambda *ts: {"items": [{"text": t, "conf": 0.9} for t in ts]}  # noqa: E731
    f = ocr_filter({"static_overlay": ["live"], "samples": [items("MORNING", "DELAWARE")] * 5 + [items("THE SHOW")] * 5})
    assert f["brand_words"] == ("DELAWARE", "MORNING")
    assert not useful_text("MORANG", f) and not useful_text("DELA", f) and not useful_text("LIVB", f)
    assert useful_text("THE SHOW", f) and useful_text("APRIL 19TH", f) and useful_text("CHANEL HOLLAND", f)


def _obs(t, act):
    return {"t": t, "activity": {"scores_top5": [[act, 0.5]]}, "environment": {"top3": [["studio", 1.0]]}}


def test_short_interval_joins_more_similar_neighbour():
    obs = [_obs(t, "talk" if t < 20 else "dance") for t in range(1, 40, 2)]
    g = [{"t": 20.0, "accepted": True, "decided_by": "rules", "reason": "x", "score": 2.0},
         {"t": 22.0, "accepted": True, "decided_by": "rules", "reason": "x", "score": 1.0}]
    log = []
    absorb_short(g, obs, 40.0, {"min_interval_s": 5.0, "absorb_context_s": 6.0}, log)
    assert [x["accepted"] for x in g] == [True, False] and log[0]["joined"] == "next"  # 20-22 is dancing like 22-40
    g[1]["accepted"], g[1]["decided_by"] = True, "human"
    g[0]["accepted"] = True
    absorb_short(g, obs, 40.0, {"min_interval_s": 5.0, "absorb_context_s": 6.0}, [])
    assert g[1]["accepted"]  # reviewer boundaries are never removed


def _person(h, visible):
    k = [[0.5, 0.5, 0.0]] * 17
    for i in visible:
        k[i] = [0.5, 0.5, 0.9]
    return {"conf": 0.9, "box": [0.4, 0.0, 0.6, h], "kpts": k}


def test_shot_scale_from_keypoints():
    assert shot_scale([], SC)[0] == "no_person"
    assert shot_scale([_person(0.9, [0, 5, 6, 11, 13, 15])], SC)[0] == "wide"
    assert shot_scale([_person(0.15, [0, 5, 15])], SC)[0] == "extreme_wide"
    assert shot_scale([_person(1.0, [0, 5, 6, 11])], SC)[0] == "medium"
    assert shot_scale([_person(1.0, [0, 5, 6])], SC)[0] == "close_up"
    assert shot_scale([_person(1.0, [0, 1, 2])], SC)[0] == "extreme_close_up"


def test_review_reasons():
    rec = {"start_s": 0, "end_s": 3, "entities": [{"id": "p1", "type": "person", "name": "Zed", "name_source": "transcript"},
                                                 {"id": "g", "type": "person_group", "count": "6-10"}],
           "attributes": {"shot_scale": "extreme_close_up"},
           "observed": {"people_count_max": 2, "shot_scale_detector": [{"label": "wide", "share": 1.0}]}}
    R = review_reasons(rec, {}, ["name removed: x was given to 2 entities", "name removed: y no source"],
                       {"min_moment_s": 5, "people_excess": 1})
    assert any("short" in r for r in R) and any("people" in r for r in R)
    assert any("shot scale" in r for r in R) and sum(r.startswith("name removed") for r in R) == 1


def test_name_sources_are_checked():
    ents = [{"id": "a", "name": "Chanel Holland", "name_source": "transcript"},
            {"id": "b", "name": "Ivan", "name_source": "transcript"},
            {"id": "c", "name": "Vincenza", "name_source": "transcript"},
            {"id": "d", "name": "Bob", "name_source": "transcript"}]
    span = {"transcript": words("thank you chanell for coming"), "on_screen_text": words("BOB SMITH")}
    rep = []
    verify_names(ents, span, words("and ivan the host") | span["transcript"], rep)
    assert [(e["name"], e["name_source"]) for e in ents] == [
        ("Chanel Holland", "transcript"), ("Ivan", "elsewhere_in_video"), (None, None), ("Bob", "on_screen_text")]
    assert not any("only in source metadata" in r for r in review_reasons(
        {"start_s": 0, "end_s": 10, "entities": ents, "observed": {}}, {}, rep, {}))
