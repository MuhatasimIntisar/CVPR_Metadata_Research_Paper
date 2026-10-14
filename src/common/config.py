"""Config loading. Every script takes --env {test_env, storage_env, live}."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
ENVS = ("test_env", "storage_env", "live")


@dataclass
class Paths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest" / "videos.parquet"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def shards(self) -> Path:
        return self.root / "shards"

    @property
    def annotations(self) -> Path:
        return self.root / "annotations"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    def shard_dir(self, shard_id: str) -> Path:
        return self.shards / shard_id


def load_config(env: str) -> dict:
    if env not in ENVS:
        raise ValueError(f"Unknown env '{env}'. Choose from {ENVS}.")
    cfg = yaml.safe_load((CONFIG_DIR / f"{env}.yaml").read_text(encoding="utf-8"))
    cfg["taxonomy"] = yaml.safe_load((CONFIG_DIR / "taxonomy.yaml").read_text(encoding="utf-8"))
    pools_file = cfg.get("search_pools", "search_pools.yaml")
    cfg["search_pools"] = yaml.safe_load((CONFIG_DIR / pools_file).read_text(encoding="utf-8"))
    prov = cfg["search_pools"].pop("_provenance", None)
    if prov and not prov.get("reviewed"):
        print(f"WARNING: {pools_file} was model-generated and is not marked reviewed "
              f"(set _provenance.reviewed: true after checking it).")
    cfg["search_pools_file"] = pools_file
    root = Path(cfg["data_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    cfg["paths"] = Paths(root)
    return cfg


def add_env_arg(parser) -> None:
    parser.add_argument("--env", default="test_env", choices=ENVS,
                        help="Which machine config to use (default: test_env)")
