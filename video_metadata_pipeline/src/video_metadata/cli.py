"""Command-line interface.

    python -m video_metadata.cli ingest --input <corpus_dir> [--config cfg.yaml]
    python -m video_metadata.cli run    --manifest data/corpus_manifest.json [--stages level1 correction] [--video-id ID]
    python -m video_metadata.cli status --manifest data/corpus_manifest.json
    python -m video_metadata.cli resume --manifest data/corpus_manifest.json
    python -m video_metadata.cli check-vlm --config configs/cluster.yaml   (GPU: start the Level-2 engine, 4 test calls)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import AUTOMATIC, STAGES


def _common(p):
    p.add_argument("--config", action="append", default=[], help="YAML overrides (repeatable, applied in order)")
    p.add_argument("--manifest", help="corpus manifest (default: paths.manifest from the config)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="video_metadata", description="Video metadata construction pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="scan a corpus folder, write the manifest, ingest every valid video")
    _common(p)
    p.add_argument("--input", help="folder with Internet Archive item folders (default: paths.corpus_root)")
    p.add_argument("--manifest-only", action="store_true", help="only (re)build the manifest")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run", help="run pipeline stages for all (or selected) videos")
    _common(p)
    p.add_argument("--stages", nargs="+", choices=STAGES, default=AUTOMATIC)
    p.add_argument("--video-id", nargs="+", help="restrict to these video IDs")
    p.add_argument("--force", action="store_true", help="recompute even if cached")

    p = sub.add_parser("resume", help="continue: run every automatic stage that is not done (failed/stale/missing)")
    _common(p)
    p.add_argument("--stages", nargs="+", choices=STAGES, default=AUTOMATIC)

    p = sub.add_parser("status", help="per-video stage status")
    _common(p)

    p = sub.add_parser("download-models", help="pre-download all models into paths.model_cache (needs internet)")
    _common(p)
    p.add_argument("--include-vlm", action="store_true", help="also download the Level-2 VLM (~10 GB)")

    p = sub.add_parser("check-vlm", help="start the Level-2 backend and run a small batch of image prompts (GPU)")
    _common(p)

    a = ap.parse_args(argv)
    cfg = load_config(*a.config)
    manifest = Path(a.manifest or cfg["paths"]["manifest"])

    if a.cmd == "ingest":
        from .ingestion.manifest import build_manifest
        from .pipeline import run
        from .utils.io import write_json
        root = Path(a.input or cfg["paths"]["corpus_root"])
        cfg["paths"]["corpus_root"] = str(root)
        man = build_manifest(cfg, root)
        write_json(manifest, man)
        print(f"manifest: {manifest} ({man['n_valid']} valid of {man['n_items']} item folders)")
        for e in man["videos"]:
            if e["issues"]:
                print(f"  ! {e['video_id']}: {'; '.join(e['issues'])}")
        if not a.manifest_only:
            res = run(cfg, manifest, ["ingest"], force=a.force)
            _print_results(res)
    elif a.cmd in ("run", "resume"):
        from .pipeline import run
        res = run(cfg, manifest, a.stages, getattr(a, "video_id", None), getattr(a, "force", False))
        _print_results(res)
    elif a.cmd == "check-vlm":
        return _check_vlm(cfg)
    elif a.cmd == "download-models":
        from .models import download
        for x in download(cfg, a.include_vlm):
            print("  ok", x)
    elif a.cmd == "status":
        from .pipeline import status
        rows = status(cfg, manifest)
        print(f"{'video_id':50} " + " ".join(f"{s:11}" for s in STAGES))
        for vid, st, stages, errs in rows:
            print(f"{vid[:50]:50} " + (" ".join(f"{stages[s]:11}" for s in STAGES) if st == "ok" else f"[{st}]"))
            for s, e in errs.items():
                print(f"    {s}: {e[:150]}")
    return 0


def _check_vlm(cfg):
    """Smoke test before a long job: engine start, batching, image placeholders, JSON answers, throughput."""
    import time
    from PIL import Image, ImageDraw
    from .level1.pipeline import device_of
    from .level2.output_validator import extract_json
    from .level2.vlm_backend import get_backend
    dev = device_of(cfg)
    t0 = time.time()
    be = get_backend(cfg, dev)
    print(f"backend {be.name} ({cfg['level2']['model_id']}) on {dev}: loaded in {time.time() - t0:.0f} s")
    imgs = []
    for col in ("red", "blue", "green", "white"):
        im = Image.new("RGB", (448, 252), col)
        ImageDraw.Draw(im).rectangle([150, 60, 300, 190], fill="black")
        imgs.append(im)
    reqs = []
    for k in range(4):
        content = [{"type": "text", "text": "Two frames from a video follow."}]
        for j in (k, (k + 1) % 4):
            content += [{"type": "text", "text": f"Frame at {j}.0 s:"}, {"type": "image", "image": imgs[j]}]
        content.append({"type": "text", "text": 'Answer with ONLY JSON on one line: {"first_colour": "...", "second_colour": "..."}'})
        reqs.append(content)
    t0 = time.time()
    outs = be.generate_batch(reqs, [60] * 4) if hasattr(be, "generate_batch") else [be.generate(r, 60) for r in reqs]
    dt = time.time() - t0
    ok = 0
    for k, o in enumerate(outs):
        try:
            extract_json(o)
            ok += 1
        except Exception:  # noqa: BLE001
            pass
        print(f"  request {k} (expect {('red', 'blue', 'green', 'white')[k]}, {('blue', 'green', 'white', 'red')[k]}): {o[:120]}")
    print(f"4 requests in {dt:.1f} s; valid JSON {ok}/4; GPU peak {be.gpu_memory_gb()} GB")
    return 0 if ok == 4 else 1


def _print_results(res):
    for vid, r in res.items():
        print(f"  {vid[:50]:50} " + " ".join(f"{s}={v}" for s, v in r.items()))
    if any(v == "failed" for r in res.values() for v in r.values()):
        print("Some stages failed; see outputs/<video_id>/logs/ and `status`. Completed work is kept.")


if __name__ == "__main__":
    sys.exit(main())
