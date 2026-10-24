import json
import subprocess
import sys

from video_metadata.config import load_config
from video_metadata.pipeline import run, status
from video_metadata.state import State


def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "video_metadata.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "ingest" in r.stdout


def test_config_env_override():
    cfg = load_config(env={"VMP_RUNTIME__DEVICE": "cpu", "VMP_CORRECTION__POOL_S": "3.5"})
    assert cfg["runtime"]["device"] == "cpu" and cfg["correction"]["pool_s"] == 3.5


def test_state_resume_and_staleness(tmp_path):
    s = State(tmp_path)
    s.mark("ingest", "done", fingerprint="x")
    s.mark("level1", "done", fingerprint="y")
    assert State(tmp_path).done("ingest", "x") and not State(tmp_path).done("ingest", "other")
    s.invalidate_from("ingest", ["ingest", "level1", "correction"])
    assert State(tmp_path).stage("level1")["status"] == "stale"


def test_failure_is_isolated(tmp_path):
    cfg = load_config()
    cfg["paths"]["outputs_root"] = str(tmp_path / "out")
    man = {"corpus_root": str(tmp_path), "videos": [{"video_id": "v1", "item_dir": "missing", "status": "ok"},
                                                   {"video_id": "v2", "item_dir": "missing2", "status": "ok"}]}
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(man))
    res = run(cfg, mp, ["level1"])  # no ingest outputs -> both fail, independently
    assert res == {"v1": {"level1": "failed"}, "v2": {"level1": "failed"}}
    rows = status(cfg, mp)
    assert all(r[2]["level1"] == "failed" for r in rows)
