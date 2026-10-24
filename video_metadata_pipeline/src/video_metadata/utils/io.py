"""Small JSON helpers: atomic writes so an interrupted run never leaves a half-written file."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def write_json(path, obj, indent=1) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, rows) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, p)
    return p


def append_jsonl(path, row) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_hash(path, full_max_mb: float = 50) -> dict:
    """sha256 of the whole file, or of head+tail+size for large media (fast, still change-sensitive)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if size <= full_max_mb * 2**20:
            for chunk in iter(lambda: f.read(2**20), b""):
                h.update(chunk)
            kind = "sha256"
        else:
            h.update(f.read(2**22))
            f.seek(-2**22, 2)
            h.update(f.read(2**22))
            h.update(str(size).encode())
            kind = "sha256_head_tail_4MiB"
    return {"algorithm": kind, "digest": h.hexdigest(), "bytes": size}


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
