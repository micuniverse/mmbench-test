#!/usr/bin/env python3
"""Verify local model shards and the frozen MMBench validation source."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ROOT = ROOT.parent.parent / "models"
PARQUET = ROOT / "dataset/data/validation-00000-of-00001-93861598ee8837c0.parquet"
PARQUET_SHA256 = "741977ca4e9badf4383a244815c03b0de2b129aac4c97b0ff71784a0c63fa14f"
TSV_SHA256 = "aeac8ac1c905eec876d824ff19b9d6279903272a0b41d918c34072a437387df4"
MODELS = {
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "Qwen/Qwen3.6-27B": "Qwen3.6-27B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tsv", type=Path, help="Optional original benchmark TSV")
    args = parser.parse_args()
    result = {"dataset": {"path": str(PARQUET), "sha256": sha256(PARQUET)}, "models": {}}
    if result["dataset"]["sha256"] != PARQUET_SHA256:
        raise ValueError("MMBench validation parquet SHA256 mismatch")
    if args.baseline_tsv:
        found = sha256(args.baseline_tsv)
        result["baseline_tsv"] = {"path": str(args.baseline_tsv), "sha256": found}
        if found != TSV_SHA256:
            raise ValueError("Original benchmark TSV SHA256 mismatch")
    else:
        result["baseline_tsv"] = {"status": "not_present"}
    for model_id, leaf in MODELS.items():
        directory = MODEL_ROOT / leaf
        config = json.loads((directory / "config.json").read_text())
        shard_index = json.loads((directory / "model.safetensors.index.json").read_text())
        shard_names = sorted(set(shard_index["weight_map"].values()))
        missing = [name for name in shard_names if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing shards for {model_id}: {missing}")
        result["models"][model_id] = {
            "path": str(directory), "model_type": config.get("model_type"),
            "shard_count": len(shard_names),
            "shard_bytes": sum((directory / name).stat().st_size for name in shard_names),
            "config_sha256": sha256(directory / "config.json"),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
