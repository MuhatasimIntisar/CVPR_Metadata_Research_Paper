"""Run every segmenter on ONE video and save each one's raw output as JSON (no snapping/merging/fusion).

    python scripts/run_segmenters.py --item-dir data/pilot/interview/<identifier>
    python scripts/run_segmenters.py --video path/to/video.mp4 --out data/segments/my_video
    python scripts/run_segmenters.py --item-dir ... --only ocr topic --force      # re-run some steps

Output folder (default data/segments/<identifier>/):
    video.json            duration, fps, resolution + archive metadata (title, date, description)
    shots.json            cuts + fades, shot list                    (PySceneDetect)
    scenes.json           shots grouped into scenes                  (CLIP)
    asr.json              words + sentences with timestamps          (Whisper large-v3-turbo)
    speakers.json         speaker runs + every raw turn              (ECAPA + clustering)
    silence.json          silent spans                               (ffmpeg silencedetect)
    audio_events.json     per-second AudioSet scores + on/off events (AST)
    ocr.json              per-second text + text changes             (EasyOCR)
    topic.json            similarity curve + topic shifts            (multilingual-e5-small)
    camera_motion.json    per-sample motion + label changes          (Farneback flow)
    all_boundaries.json   every segmenter's boundaries in one time-sorted list (unmerged)
    summary.json          status, runtime and boundary count per segmenter

Each step is skipped if its JSON already exists (use --force). A failing step is logged in
summary.json and the rest still run.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows; just noisy

from src.segmenters import (asr, audio_events, camera_motion, ocr, scenes, shots, silence,  # noqa: E402
                            speakers, topic)
from src.segmenters.common import extract_audio, get_device, probe  # noqa: E402

STEPS = [shots, scenes, asr, speakers, silence, audio_events, ocr, topic, camera_motion]
NAMES = [m.NAME for m in STEPS]


def find_video(item_dir: Path) -> Path:
    vids = sorted(p for p in item_dir.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"))
    if not vids:
        sys.exit(f"No video file in {item_dir}")
    return vids[0]


def archive_meta(item_dir: Path | None) -> dict:
    if not item_dir or not (item_dir / "metadata.json").exists():
        return {}
    with open(item_dir / "metadata.json", encoding="utf-8") as f:
        m = json.load(f).get("metadata", {})
    keep = ["identifier", "title", "date", "publicdate", "description", "subject", "creator",
            "collection", "language", "runtime"]
    return {k: m[k] for k in keep if k in m}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--item-dir", type=Path, help="folder from ia_pilot_download.py (video + metadata.json)")
    src.add_argument("--video", type=Path, help="a video file")
    ap.add_argument("--out", type=Path, help="output folder (default data/segments/<name>)")
    ap.add_argument("--only", nargs="+", choices=NAMES, help="run only these segmenters")
    ap.add_argument("--skip", nargs="+", choices=NAMES, default=[], help="skip these segmenters")
    ap.add_argument("--force", action="store_true", help="re-run even if the JSON exists")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--language", default=None, help="ASR language code (default: auto-detect)")
    ap.add_argument("--ocr-langs", default="en", help="EasyOCR languages, e.g. en or en,es")
    args = ap.parse_args()

    video = find_video(args.item_dir) if args.item_dir else args.video
    name = args.item_dir.name if args.item_dir else video.stem
    out = args.out or Path("data") / "segments" / name
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    info = probe(video)
    meta = {"video_file": str(video), **{k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()},
            "archive": archive_meta(args.item_dir)}
    with open(out / "video.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    wav = extract_audio(video, out / "_cache" / "audio_16k.wav")
    print(f"Video: {video.name}  {info['duration'] / 60:.1f} min  {info['width']}x{info['height']}  "
          f"device={device}\nOutput: {out}\n")

    lang = args.language or (meta["archive"].get("language") if isinstance(meta["archive"].get("language"), str)
                             and len(meta["archive"]["language"]) == 2 else None)
    kw = {"device": device, "wav": wav, "language": lang, "langs": args.ocr_langs}
    todo = [m for m in STEPS if (not args.only or m.NAME in args.only) and m.NAME not in args.skip]

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for m in todo:
        path = out / f"{m.NAME}.json"
        if path.exists() and not args.force:
            print(f"[skip] {m.NAME}: {path.name} exists")
            continue
        print(f"[run ] {m.NAME} ...", flush=True)
        t0 = time.time()
        try:
            m.run(video, out, info, **kw)
            doc = json.loads(path.read_text(encoding="utf-8"))
            summary[m.NAME] = {"status": "ok", "model": doc["model"], "runtime_s": doc["runtime_s"],
                               "n_boundaries": doc.get("n_boundaries"), "stats": doc.get("stats")}
            print(f"       done in {time.time() - t0:.0f}s  boundaries={doc.get('n_boundaries', '-')}  "
                  f"{json.dumps(doc.get('stats'), ensure_ascii=False)[:160]}")
        except Exception as e:  # keep going; report at the end
            summary[m.NAME] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            print(f"       FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # flat, time-sorted list of every raw boundary (no merging)
    allb = []
    for n in NAMES:
        p = out / f"{n}.json"
        if p.exists():
            doc = json.loads(p.read_text(encoding="utf-8"))
            allb += [{"segmenter": n, **b} for b in doc.get("boundaries", [])]
    allb.sort(key=lambda b: b["t"])
    (out / "all_boundaries.json").write_text(json.dumps(allb, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\nSummary")
    for n in NAMES:
        s = summary.get(n)
        if s:
            nb = s.get("n_boundaries")
            print(f"  {n:14} {s['status']:7} {('%5ss' % s.get('runtime_s', '')) if s['status'] == 'ok' else ''}"
                  f"  {'' if nb is None else f'{nb} boundaries'}{'  ' + s.get('error', '') if s['status'] != 'ok' else ''}")
    print(f"\n{len(allb)} raw boundaries in total -> {out / 'all_boundaries.json'}")


if __name__ == "__main__":
    main()
