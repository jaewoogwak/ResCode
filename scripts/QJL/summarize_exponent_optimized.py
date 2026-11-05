#!/usr/bin/env python3
"""Combine same-GPU v1/v2 benchmarks, output parity, and ARC evaluation."""

import argparse
import csv
import json
from pathlib import Path

import torch


METHODS = ("base_k2", "base_k4", "qjl_k2_d1n_a1", "qjl_k2_d2n_a1")
TASKS = ("arc_easy", "arc_challenge")


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def accuracy(root, method, variant):
    data = read_json(root / "evaluation" / method / variant / "metrics.json")
    if not data:
        return None
    if data.get("eval_limit") != 0 or any(task not in data.get("task_results", {})
                                         for task in TASKS):
        raise ValueError(f"incomplete ARC evaluation: {method}/{variant}")
    return {task: {metric: data["task_results"][task].get(f"{metric},none")
                   for metric in ("acc", "acc_norm")} for task in TASKS}


def parity(old_path, new_path, original_path=None):
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    if set(old) != set(new):
        raise AssertionError("benchmark traces have different keys")
    result = {}
    for key, before in old.items():
        after = new[key]
        if key in ("routing_topk", "prompt_ids"):
            result[key] = {"exact": torch.equal(before, after)}
            continue
        diff = before.float() - after.float()
        denom = torch.linalg.vector_norm(before.float()).item()
        result[key] = {"max_abs": diff.abs().max().item(),
                       "relative_l2": (torch.linalg.vector_norm(diff).item() / denom
                                       if denom else None),
                       "exact": torch.equal(before, after)}
    if original_path and original_path.is_file():
        original = torch.load(original_path, map_location="cpu", weights_only=False)
        result["routing_unchanged_vs_original"] = (
            torch.equal(old["router"], original["router"])
            and torch.equal(new["router"], original["router"])
            and torch.equal(old["routing_topk"], original["routing_topk"])
            and torch.equal(new["routing_topk"], original["routing_topk"])
            and torch.equal(old["routing_weight"], original["routing_weight"])
            and torch.equal(new["routing_weight"], original["routing_weight"]))
        result["shared_unchanged_vs_original"] = (
            torch.equal(old["shared"], original["shared"])
            and torch.equal(new["shared"], original["shared"]))
    return result


def latency(data, key):
    return data["timing"].get(key, {}).get("median_ms")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/qjl_exponent_base/optimized"))
    args = p.parse_args()
    root = args.root
    rows = []
    parity_results = {}
    original_trace = root / "benchmark" / "original" / "trace.pt"
    for method in METHODS:
        old_dir = root / "benchmark" / method / "legacy"
        new_dir = root / "benchmark" / method / "optimized"
        old = read_json(old_dir / "metrics.json")
        new = read_json(new_dir / "metrics.json")
        if not old or not new:
            continue
        if old["gpu_name"] != new["gpu_name"] or old["torch_version"] != new["torch_version"]:
            raise ValueError(f"benchmark environment differs: {method}")
        trace = parity(old_dir / "trace.pt", new_dir / "trace.pt", original_trace)
        parity_results[method] = trace
        old_acc = accuracy(root, method, "legacy")
        new_acc = accuracy(root, method, "optimized")
        row = {"method": method,
               "legacy_checkpoint_bytes": old["checkpoint_bytes"],
               "optimized_checkpoint_bytes": new["checkpoint_bytes"],
               "legacy_gpu_resident_bytes": old["gpu_resident_bytes"],
               "optimized_gpu_resident_bytes": new["gpu_resident_bytes"],
               "gpu_memory_saved_pct": 100 * (1 - new["gpu_resident_bytes"] /
                                             old["gpu_resident_bytes"]),
               "legacy_gpu_peak_bytes": old["gpu_peak_bytes"],
               "optimized_gpu_peak_bytes": new["gpu_peak_bytes"],
               "legacy_exponent_bytes": old["tensor_bytes"]["exponent"],
               "optimized_exponent_bytes": new["tensor_bytes"]["exponent"],
               "logits_relative_l2": trace["logits"]["relative_l2"],
               "router_unchanged": trace.get("routing_unchanged_vs_original"),
               "shared_unchanged": trace.get("shared_unchanged_vs_original")}
        for key in ("base_gate_m1", "base_gate_m32", "base_down_m32",
                    "qjl_gate_m32", "qjl_down_m32", "expert_m1", "expert_m32",
                    "moe_m1", "moe_m32", "prefill_b1_s16", "prefill_b1_s128",
                    "prefill_b4_s32", "decode_s16", "decode_s128"):
            before, after = latency(old, key), latency(new, key)
            row[f"legacy_{key}_ms"] = before
            row[f"optimized_{key}_ms"] = after
            row[f"{key}_speedup"] = (before / after if before and after else None)
        for task in TASKS:
            for metric in ("acc", "acc_norm"):
                row[f"legacy_{task}_{metric}"] = (old_acc or {}).get(task, {}).get(metric)
                row[f"optimized_{task}_{metric}"] = (new_acc or {}).get(task, {}).get(metric)
                a, b = row[f"legacy_{task}_{metric}"], row[f"optimized_{task}_{metric}"]
                row[f"{task}_{metric}_delta"] = b - a if a is not None and b is not None else None
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    (root / "parity.json").write_text(json.dumps(parity_results, indent=2,
                                                   allow_nan=False) + "\n")
    (root / "summary.json").write_text(json.dumps(rows, indent=2,
                                                  allow_nan=False) + "\n")
    if rows:
        with (root / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"methods": [row["method"] for row in rows],
                      "summary": str(root / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
