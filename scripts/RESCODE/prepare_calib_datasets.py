#!/usr/bin/env python
"""
Download calibration datasets for the RESCODE calibration ablation
(Wikipedia / SlimPajama / DCLM / Code), normalised to the same JSONL format
as the existing C4 shard at rescode/data/c4-train.00000-of-01024.json.

Output rows: {"text": "..."} per line (other fields are dropped so the
get_calib_dataloder text-column auto-pick is unambiguous).

Usage:
    python scripts/RESCODE/prepare_calib_datasets.py --dataset wikipedia
    python scripts/RESCODE/prepare_calib_datasets.py --dataset slimpajama --n_docs 50000
    python scripts/RESCODE/prepare_calib_datasets.py --dataset all --n_docs 35000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "rescode" / "data"


# Each spec defines: how to stream the dataset and which field is the text.
# Streaming is mandatory for SlimPajama/DCLM/Stack (too big to materialise).
SPECS = {
    "wikipedia": dict(
        path="wikimedia/wikipedia",
        name="20231101.en",
        split="train",
        text_field="text",
        min_chars=200,
    ),
    "slimpajama": dict(
        # cerebras/SlimPajama-627B is gated; DKYoon/SlimPajama-6B is an
        # open 6B-token clean sample that preserves the 7-source mixture.
        path="DKYoon/SlimPajama-6B",
        name=None,
        split="train",
        text_field="text",
        min_chars=200,
    ),
    "dclm": dict(
        path="mlfoundations/dclm-baseline-1.0-parquet",
        name=None,
        split="train",
        text_field="text",
        min_chars=200,
    ),
    "starcoder": dict(
        # bigcode/the-stack-smol requires auth; codeparrot/github-code-clean
        # is the standard open-access GitHub code corpus.
        path="codeparrot/github-code-clean",
        name="all-all",
        split="train",
        text_field="code",
        min_chars=100,
    ),
}


def _stream(spec: dict) -> Iterator[dict]:
    kwargs = dict(split=spec["split"], streaming=True, trust_remote_code=True)
    if spec.get("name"):
        kwargs["name"] = spec["name"]
    return load_dataset(spec["path"], **kwargs)


def download_one(name: str, n_docs: int, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        print(f"[prepare] {name}: {out_path} already exists, skipping (use --overwrite)")
        return

    spec = SPECS[name]
    text_field = spec["text_field"]
    min_chars = spec["min_chars"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    print(f"[prepare] {name}: streaming {spec['path']} "
          f"(name={spec.get('name')}, split={spec['split']}) -> {out_path}")
    t0 = time.time()
    n_written = 0
    n_seen = 0
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for row in _stream(spec):
            n_seen += 1
            text = row.get(text_field)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < min_chars:
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False))
            fh.write("\n")
            n_written += 1
            if n_written % 1000 == 0:
                elapsed = time.time() - t0
                rate = n_written / max(elapsed, 1e-6)
                print(f"[prepare] {name}: wrote {n_written}/{n_docs} "
                      f"(seen={n_seen}, {rate:.1f} docs/s, {elapsed:.0f}s)")
            if n_written >= n_docs:
                break

    tmp_path.replace(out_path)
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[prepare] {name}: done -> {out_path} "
          f"({n_written} docs, {size_mb:.1f} MB, {elapsed:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=list(SPECS.keys()) + ["all"],
        help="Which calibration dataset to download.",
    )
    ap.add_argument(
        "--n_docs",
        type=int,
        default=35000,
        help="Number of documents to keep per dataset. The C4 loader uses "
             "min(n_blocks_for_stat*16, len), so 35000 covers n_sentences up to "
             "~2180. Bump up for larger calibrations.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory to write JSONL files into (default: rescode/data).",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    names = list(SPECS.keys()) if args.dataset == "all" else [args.dataset]
    for name in names:
        out_path = out_dir / f"{name}-calib.jsonl"
        try:
            download_one(name, args.n_docs, out_path, args.overwrite)
        except Exception as e:
            print(f"[prepare] {name}: FAILED ({type(e).__name__}: {e})", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
