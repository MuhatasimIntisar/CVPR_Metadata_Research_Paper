from pathlib import Path

from video_metadata.config import load_config
from video_metadata.ingestion import metadata_parser, transcript_parser
from video_metadata.ingestion.manifest import build_manifest
from video_metadata.utils.timestamps import parse_ts

FIX = Path(__file__).parent / "fixtures" / "corpus"


def test_timestamps():
    assert parse_ts("00:01:02,345") == 62.345
    assert parse_ts("01:02.5") == 62.5
    assert parse_ts("1:00:00.000") == 3600


def test_rolling_srt_dedup_and_word_times():
    t = transcript_parser.parse_track(FIX / "grp/item_a/captions.en.asr.srt", duration=60)
    assert [s["text"] for s in t["segments"]] == ["Hello there friends.", ">> Welcome back."]
    assert t["words"][0] == {"start": 1.0, "end": 1.5, "w": "Hello", "approx": False}
    assert t["words"][1]["start"] == 1.5 and t["words"][3]["w"] == ">>"
    assert t["issues"]["parse"]  # the malformed cue is reported, not fatal
    assert t["segments"][1]["speaker_change_marker"]


def test_vtt_whitespace_lines_inside_cue():
    t = transcript_parser.parse_track(FIX / "grp/item_a/captions.es.vtt", duration=60)
    assert t["segments"][0]["text"] == "Hola amigos" and t["segments"][0]["word_timed"]
    assert t["duplicate_cue_ratio"] == 0.5


def test_primary_prefers_item_language():
    cfg = load_config()
    tracks = [transcript_parser.parse_track(FIX / "grp/item_a" / f, 60) for f in ("captions.en.asr.srt", "captions.es.vtt")]
    for t in tracks:
        t["n_words"] = 100  # bypass the adequacy floor for this tiny fixture
    best, why = transcript_parser.select_primary(tracks, "en", None, cfg["ingestion"])
    assert best["language"] == "en", why


def test_metadata_normalisation_and_missing_fields():
    import json
    doc = json.loads((FIX / "grp/item_a/metadata.json").read_text())
    md, issues = metadata_parser.normalise("item_a", doc, None, 60.0)
    assert md["language_code"] == "en" and md["subjects"] == ["alpha", "beta"]
    assert md["original_platform"]["platform"] == "YouTube" and md["raw_metadata"] == doc
    assert "no rights/licence field in source metadata" in issues
    md2, issues2 = metadata_parser.normalise("x", None, None, None)
    assert md2["title"] is None and "metadata.json missing" in issues2


def test_manifest_flags_folder_without_video():
    man = build_manifest(load_config(), FIX)
    by = {e["video_id"]: e for e in man["videos"]}
    assert by["item_a"]["status"] == "ok"
    assert by["item_nomedia"]["status"] == "invalid" and "no video file in folder" in by["item_nomedia"]["issues"]
