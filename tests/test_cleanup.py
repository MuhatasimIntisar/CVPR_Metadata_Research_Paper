"""Rule-based cleanup (no models): each rule on a small synthetic document."""
import copy

from video_metadata.cleanup import geo
from video_metadata.cleanup.rules import Ctx, OcrCleaner, apply_all

CFG = {"version": "clean-v1",
       "merge": {"min_description_sim": 0.75, "min_tag_jaccard": 0.5, "max_speech_share": 0.3, "max_merged_s": 180.0},
       "ocr": {"fuzzy": 0.85},
       "names": {"fuzzy": 0.8, "min_profile_mentions": 3, "conflict_score": 2.0, "conflict_margin": 1.0},
       "shot": {"min_person_share": 0.5, "min_dominant_share": 0.7, "max_model_share": 0.2, "max_people": 2},
       "speakers": {"min_s": 3.0, "min_share": 0.1},
       "review": {"crowd_min_people": 8}}
SRC = {"title": "Interview with Zoe Royer, Member of Parliament", "description": "she told interviewer Sher Sultani",
       "creator": "Skyrise Media Society", "subjects": ["British Columbia", "Coquitlam", "Skyrise Media Society"]}
TRANSCRIPT = {"segments": [{"text": "welcome zoe royer follow us for more"}]}
OCR = {"samples": [{"t": 1, "items": [{"text": "Sherry Soltani/ParvazTV"}, {"text": "Sherry SoltanilParvazTV"},
                                      {"text": "Royerie"}, {"text": "Pledge"}]}]}
ZOE, SHERRY = "woman with blonde hair wearing a pink blazer", "woman with glasses and an orange shirt"


def moment(k, ents, desc="Two women talk in a studio.", seg=None, speech=0.9, tags=("interview", "studio"), **kw):
    m = {"moment_id": f"v_s{seg if seg is not None else k:03d}_m01", "segment_id": f"v_s{seg if seg is not None else k:03d}",
         "start_s": 10.0 * k, "end_s": 10.0 * k + 10, "source_intervals": [f"c{k}"],
         "boundary": {"start": {}, "end": {}}, "description": desc, "tags": list(tags),
         "entities": [{"id": f"person_{i + 1}", "type": "person", "name": n, "name_source": "transcript" if n else None,
                       "description": d, "evidence": "visual"} for i, (n, d) in enumerate(ents)],
         "actions": [], "relationships": [], "context": {"setting": "studio", "topic": None},
         "attributes": {"shot_scale": "medium", "camera_motion": "static"},
         "observed": {"speech_share": speech, "speakers": [], "on_screen_text": [], "shot_scale_detector": []},
         "annotation": {"review_reasons": [], "normalisation_repairs": [], "evidence_provenance": {}}}
    m.update(kw)
    return m


def doc(moments):
    segs = {}
    for m in moments:
        segs.setdefault(m["segment_id"], []).append(m["moment_id"])
    return {"video": {}, "annotation_set": {}, "moments": moments,
            "segments": [{"segment_id": s, "children": c} for s, c in segs.items()]}


def run(d, speakers=None):
    ctx = Ctx(CFG, SRC, TRANSCRIPT, OCR, speakers)
    apply_all(d, ctx)
    return d, ctx


def test_ocr_lines():
    cl = OcrCleaner(Ctx(CFG, SRC, TRANSCRIPT, OCR, None))
    assert cl.line("Sherry SoltanilParvazTV") == "Sherry Soltani | ParvazTV"
    assert cl.line("Skyrise Media Societyl KTV") == "Skyrise Media Society | KTV"
    assert cl.line("FOLLOW US FOR MOREL") == "FOLLOW US FOR MORE"
    assert cl.line("2 0 2 6") == "2026" and cl.line("10 : 19 :12:4 4") == "10:19:12:44"
    assert cl.line("Royerie") is None          # single unknown word resembling a known one
    assert cl.line("Pledge") == "Pledge"        # single unknown word resembling nothing: kept
    assert cl.line("DOOMSDAY IS COMING") == "DOOMSDAY IS COMING"  # no dictionary: real words are never "corrected"


def test_names_spelling_variants_and_appearance():
    ms = [moment(0, [("Sherry Sultani", SHERRY), (None, ZOE)]),
          moment(1, [("Zoe Royer", ZOE), (None, SHERRY)]),
          moment(2, [("Zoe Royer", ZOE)]), moment(3, [("Zoe Royer", ZOE)]),
          moment(4, [("Sher Sultani", SHERRY)]), moment(5, [("Sherry Sultani", SHERRY)]),
          moment(6, [("Zoe Royer", SHERRY), (None, ZOE)]),                     # name on the wrong person -> moved
          moment(7, [("Sherry Sultani", ZOE), ("Zoe Royer", SHERRY)]),         # both swapped -> swapped back
          moment(8, [("Mercy Sherry", ZOE)]),                                  # ASR junk on Zoe's look -> removed
          moment(9, [("Sam", "a man in a suit")]), moment(10, [("Sam", "a woman in a red dress")])]
    d, _ = run(doc(ms))
    M = {m["moment_id"][:6]: m for m in d["moments"]}
    names = lambda m: [e.get("name") for e in m["entities"]]  # noqa: E731
    assert names(M["v_s000"]) == ["Sherry Soltani", None]       # on-screen spelling wins over ASR / metadata
    assert names(M["v_s004"]) == ["Sherry Soltani"]
    assert names(M["v_s006"]) == [None, "Zoe Royer"]
    assert names(M["v_s007"]) == ["Zoe Royer", "Sherry Soltani"]
    assert names(M["v_s008"]) == [None]
    assert M["v_s006"]["annotation"]["needs_review"] and M["v_s007"]["annotation"]["needs_review"]
    assert any("both a man and a woman" in r for r in M["v_s009"]["annotation"]["review_reasons"])
    assert {e["name_binding"] for m in d["moments"] for e in m["entities"] if e.get("name") == "Zoe Royer"} <= \
        {"appearance_consistent", "unverified"}


def test_vocab_tags_shot_review_location():
    m = moment(0, [(None, "conductor"), (None, "violinist")], tags=("woman", "orange shirt", "board", "board meeting", "music"),
               attributes={"shot_scale": "medium", "camera_motion": "static"},
               observed={"speech_share": 0.9, "speakers": [], "on_screen_text": [], "people_count_median": 1,
                         "shot_scale_detector": [{"label": "close_up", "share": 0.9}, {"label": "medium", "share": 0.1}]},
               actions=[{"id": "a1", "type": "talk", "participants": ["person_1"]},
                        {"id": "a2", "type": "give_interview", "description": "person_2 interviews person_1",
                         "participants": ["person_2", "person_1"]}],
               annotation={"normalisation_repairs": ["relationship person_1 conducts person_2 dropped: predicate 'conducts' not allowed",
                                                     "relationship person_1 sits_with person_2 dropped: predicate 'sits_with' not allowed"],
                           "review_reasons": ["relationship person_1 conducts person_2 dropped: predicate 'conducts' not allowed",
                                              "shot scale: metadata medium, detector mostly close_up"],
                           "evidence_provenance": {}})
    d, _ = run(doc([m]))
    m = d["moments"][0]
    assert m["relationships"] == [{"subject": "person_1", "predicate": "performs_with", "object": "person_2",
                                   "evidence": None, "restored_from": "conducts"}]
    assert [a["type"] for a in m["actions"]] == ["speak", "conduct_interview"]
    assert m["tags"] == ["board meeting", "music"]
    assert m["attributes"]["shot_scale"] == "close_up" and m["attributes"]["shot_scale_source"] == "detector"
    assert m["annotation"]["review_reasons"] == [] and len(m["annotation"]["review_reasons_suppressed"]) == 2
    assert d["video"]["location"]["region"] == "BC" and d["video"]["location"]["place"] == "Coquitlam"


def test_merge_only_silent_near_duplicates():
    desc = "A large orchestra performs on a stage with a conductor."
    ms = [moment(k, [], desc=desc, seg=0, speech=0.0, tags=("orchestra", "concert")) for k in range(3)]
    for k, m in enumerate(ms):
        m["moment_id"] = f"v_s000_m{k + 1:02d}"
    talky = copy.deepcopy(ms)
    for m in talky:
        m["observed"]["speech_share"] = 0.9
    d, _ = run(doc(ms))
    assert len(d["moments"]) == 1 and d["moments"][0]["end_s"] == 30.0
    assert d["moments"][0]["merged_from"] == ["v_s000_m01", "v_s000_m02", "v_s000_m03"]
    assert d["segments"][0]["children"] == ["v_s000_m01"] and d["segments"][0]["single_moment"]
    d2, _ = run(doc(talky))
    assert len(d2["moments"]) == 3


def test_speakers_and_entity_types():
    m = moment(0, [("Newington", "location mentioned in conversation"), ("Kristen", "woman mentioned in conversation"),
                   ("Flash Lady Photography", "woman in a sweater")],
               observed={"speech_share": 0.9, "speakers": ["S0", "S1"], "on_screen_text": [], "shot_scale_detector": []})
    d, _ = run(doc([m]), speakers={"runs": [{"speaker": "S0", "start": 0, "end": 9}, {"speaker": "S1", "start": 9, "end": 10}]})
    m = d["moments"][0]
    assert [e["type"] for e in m["entities"]] == ["place", "person", "organization"]
    assert m["entities"][1]["evidence"] == "transcript" and m["entities"][1]["name_binding"] == "not_visible"
    assert m["observed"]["speakers"] == ["S0"]


def test_location_needs_a_state_tag():
    assert geo.location({"subjects": ["Community Media", "PEG"]}) is None
    loc = geo.location({"subjects": ["Connecticut", "Newington", "NCTV"], "creator": "NCTV"})
    assert (loc["country"], loc["region"], loc["place"]) == ("US", "CT", "Newington")
