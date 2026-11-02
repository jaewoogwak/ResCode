#!/usr/bin/env python3
"""Sequential compact-base sweep; each compression/evaluation is a fresh process."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = "winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte"


def experiments(calibrated=False):
    rows = [{"name": "original", "k": None, "d_factor": None, "alpha": None}]
    rows += [{"name": f"base_k{k}", "k": k, "d_factor": 0, "alpha": "0"}
             for k in (1, 2, 3, 4)]
    rows += [{"name": f"qjl_k{k}_d{d}n_a1", "k": k, "d_factor": d, "alpha": "1"}
             for k in (1, 2, 3) for d in (1, 2)]
    if calibrated:
        rows += [{"name": f"qjl_k{k}_d{d}n_acal", "k": k, "d_factor": d,
                  "alpha": "calibrated"}
                 for k in (1, 2, 3) for d in (1, 2)]
    return rows


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="results/qjl_exponent_base/qwen")
    p.add_argument("--include-calibrated", action="store_true")
    p.add_argument("--only", default=None, help="comma-separated experiment names")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--rerun-eval", action="store_true",
                   help="Preserve an existing evaluation in a timestamped backup and rerun")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--layers", default="all")
    p.add_argument("--train-data-path", default="rescode/data/wikipedia-calib.jsonl")
    p.add_argument("--n-sentences", type=int, default=2400)
    p.add_argument("--calib-seqlen", type=int, default=2048)
    p.add_argument("--max-train-per-expert", type=int, default=32)
    p.add_argument("--max-heldout-per-expert", type=int, default=16)
    p.add_argument("--diagnostic-samples", type=int, default=16)
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--backend", choices=("triton", "triton_legacy", "reference"), default="triton")
    p.add_argument("--task", default=TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.94)
    return p.parse_args()


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def atomic_json(path, rows):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _valid_compress(out, exp, args):
    config = read_json(out / "config.json")
    metrics = read_json(out / "metrics.json")
    checkpoint = out / "compact_qjl_state.pt"
    expected = {"k": exp["k"], "d_factor": exp["d_factor"], "alpha": exp["alpha"],
                "backend": args.backend, "projection_seed": args.projection_seed,
                "calib_seed": args.calib_seed, "layers": args.layers,
                "train_data_path": args.train_data_path,
                "n_sentences": args.n_sentences, "calib_seqlen": args.calib_seqlen,
                "max_train_per_expert": args.max_train_per_expert,
                "max_heldout_per_expert": args.max_heldout_per_expert,
                "diagnostic_samples": args.diagnostic_samples}
    return (config is not None and all(config.get(k) == v for k, v in expected.items())
            and metrics is not None and read_json(out / "diagnostics.json") is not None
            and checkpoint.is_file()
            and checkpoint.stat().st_size == metrics.get("checkpoint_bytes"))


def _valid_eval(out, args):
    metrics = read_json(out / "eval" / "metrics.json")
    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    if not metrics or metrics.get("eval_limit") != args.eval_limit:
        return False
    scores = metrics.get("task_summary", {}).get("tasks", [])
    by_task = {r[0]: r[1] for r in scores if len(r) >= 2}
    return set(by_task) == set(tasks) and all(
        isinstance(by_task[t], (int, float)) and math.isfinite(by_task[t]) for t in tasks)


def _valid_original(out, args):
    metrics = read_json(out / "metrics.json")
    if not metrics or metrics.get("eval_limit") != args.eval_limit:
        return False
    tasks = {t.strip() for t in args.task.split(",") if t.strip()}
    rows = metrics.get("task_summary", {}).get("tasks", [])
    scores = {r[0]: r[1] for r in rows if len(r) >= 2}
    return set(scores) == tasks and all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in scores.values())


def _command(args, exp, out, mode, cache, base_checkpoint=None):
    cmd = [sys.executable, "scripts/QJL/run_exponent_base.py", "--mode", mode,
           "--output-dir", str(out), "--backend", args.backend,
           "--projection-seed", str(args.projection_seed),
           "--calib-seed", str(args.calib_seed), "--layers", args.layers,
           "--train-data-path", args.train_data_path,
           "--n-sentences", str(args.n_sentences), "--calib-seqlen", str(args.calib_seqlen),
           "--max-train-per-expert", str(args.max_train_per_expert),
           "--max-heldout-per-expert", str(args.max_heldout_per_expert),
           "--diagnostic-samples", str(args.diagnostic_samples),
           "--sample-cache", str(cache), "--task", args.task,
           "--eval-limit", str(args.eval_limit),
           "--eval-batch-size", str(args.eval_batch_size),
           "--gpu-memory-fraction", str(args.gpu_memory_fraction)]
    if exp["name"] != "original":
        cmd += ["--k", str(exp["k"]), "--d-factor", str(exp["d_factor"]),
                "--alpha", exp["alpha"]]
    if mode == "compress":
        cmd.append("--skip-eval")
        if base_checkpoint is not None and base_checkpoint.is_file():
            cmd += ["--base-checkpoint", str(base_checkpoint)]
    if mode == "eval":
        cmd += ["--checkpoint", str(out.parent / "compact_qjl_state.pt")]
    return cmd


def _run(cmd, dry_run):
    print("[sweep]", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def _summary(exps, root, args):
    rows = []
    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    for exp in exps:
        name = exp["name"]
        out = root / name
        cm = read_json(out / "metrics.json")
        em = read_json((out if name == "original" else out / "eval") / "metrics.json")
        diag = read_json(out / "diagnostics.json") if name != "original" else None
        row = {**exp, "status": "pending", "residual_norm_ratio": None,
               "exponent_entropy_bits": None, "base_stored_bytes": None,
               "base_stored_bits_per_weight": None, "fitted_alpha_mean": None,
               "qjl_tensor_bytes": None, "checkpoint_bytes": None,
               "compression_ratio": None, "gpu_resident_bytes": None,
               "base_gpu_resident_bytes": None, "qjl_gpu_resident_bytes": None,
               "gpu_peak_bytes": None, "latency_ms": None, "latency_cuda_ms": None,
               "calibration_seconds": None, "compression_seconds": None,
               "diagnostics_seconds": None,
               "exponent_restore_seconds": None, "eval_seconds": None, "mean_acc": None,
               "projection_error": None, "expert_error": None, "moe_error": None,
               "condition1_vs_base": None, "condition2_vs_higher_base": None}
        for task in tasks:
            row[task + "_acc"] = None
            row[task + "_acc_norm"] = None
        if cm:
            row["status"] = "compressed" if name != "original" else "original"
            row["checkpoint_bytes"] = cm.get("checkpoint_bytes")
            row["base_stored_bytes"] = cm.get("base_encoded_bytes")
            row["qjl_tensor_bytes"] = cm.get("qjl_tensor_bytes")
            row["compression_ratio"] = cm.get("checkpoint_ratio_vs_original_routed")
            row["calibration_seconds"] = cm.get("calibration_seconds")
            row["diagnostics_seconds"] = cm.get("diagnostics_seconds")
            combined = cm.get("compression_and_diagnostics_seconds")
            row["compression_seconds"] = (
                combined - row["diagnostics_seconds"]
                if combined is not None and row["diagnostics_seconds"] is not None
                else combined)
            row["gpu_resident_bytes"] = cm.get("gpu_resident_allocated_bytes")
            row["base_gpu_resident_bytes"] = cm.get("base_gpu_resident_bytes")
            row["qjl_gpu_resident_bytes"] = cm.get("qjl_gpu_resident_bytes")
            row["gpu_peak_bytes"] = cm.get("gpu_peak_allocated_bytes")
            row["latency_ms"] = cm.get("inference_forward_median_ms",
                                       cm.get("inference_forward_mean_ms"))
            row["latency_cuda_ms"] = cm.get("inference_cuda_event_median_ms")
            row["exponent_restore_seconds"] = cm.get("exponent_restore_seconds")
        if diag:
            ds = diag["summary"]
            row["residual_norm_ratio"] = ds["residual_norm_ratio"]
            row["exponent_entropy_bits"] = ds["exponent_entropy_bits"]
            row["base_stored_bits_per_weight"] = ds.get(
                "base_stored_bits_per_weight",
                16 * ds["base_stored_bytes"] / cm["original_routed_bf16_bytes"]
                if cm and cm.get("original_routed_bf16_bytes") else None)
            row["fitted_alpha_mean"] = ds["mean_alpha"]
            projection_errors = [ds[p]["mean_relative_l2"] for p in ("gate", "up", "down")]
            if all(value is not None for value in projection_errors):
                row["projection_error"] = sum(projection_errors) / 3
            row["expert_error"] = ds["expert_output"]["mean_relative_l2"]
            row["moe_error"] = ds["moe_output"]["mean_relative_l2"]
        if em and (_valid_original(out, args) if name == "original" else _valid_eval(out, args)):
            row["status"] = "evaluated"
            row["mean_acc"] = em["task_summary"]["avg"]
            row["eval_seconds"] = em.get("eval_seconds")
            row["gpu_peak_bytes"] = em.get("gpu_peak_allocated_bytes", row["gpu_peak_bytes"])
            row["gpu_resident_bytes"] = em.get("gpu_resident_allocated_bytes", row["gpu_resident_bytes"])
            row["latency_ms"] = em.get("inference_forward_median_ms",
                                       em.get("inference_forward_mean_ms", row["latency_ms"]))
            row["latency_cuda_ms"] = em.get("inference_cuda_event_median_ms",
                                            row["latency_cuda_ms"])
            for task, acc, norm in em["task_summary"]["tasks"]:
                row[task + "_acc"], row[task + "_acc_norm"] = acc, norm
        rows.append(row)
    by_name = {r["name"]: r for r in rows}
    for row in rows:
        if not row["name"].startswith("qjl_"):
            continue
        base = by_name.get(f"base_k{row['k']}")
        higher = by_name.get(f"base_k{row['k'] + row['d_factor']}")
        if base:
            row["condition1_vs_base"] = {
                metric: (row[metric] < base[metric]
                         if row[metric] is not None and base[metric] is not None else None)
                for metric in ("projection_error", "expert_error", "moe_error")}
        if higher and row["checkpoint_bytes"] is not None and higher["checkpoint_bytes"] is not None:
            row["condition2_vs_higher_base"] = {
                "accuracy_better": (row["mean_acc"] > higher["mean_acc"]
                                    if row["mean_acc"] is not None
                                    and higher["mean_acc"] is not None else None),
                "heldout_moe_error_better": (
                    row["moe_error"] < higher["moe_error"]
                    if row["moe_error"] is not None and higher["moe_error"] is not None
                    else None),
                "checkpoint_byte_difference": row["checkpoint_bytes"] - higher["checkpoint_bytes"],
                "checkpoint_byte_ratio": row["checkpoint_bytes"] / higher["checkpoint_bytes"],
                "within_5_percent_budget": abs(row["checkpoint_bytes"] / higher["checkpoint_bytes"] - 1) <= 0.05,
                "exact_memory_match": row["checkpoint_bytes"] == higher["checkpoint_bytes"]}
    atomic_json(root / "sweep_summary.json", rows)
    temp = root / f"sweep_summary.csv.tmp{os.getpid()}"
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, root / "sweep_summary.csv")


def main():
    args = arguments()
    os.chdir(ROOT)
    all_exps = experiments(args.include_calibrated)
    exps = all_exps
    if args.only:
        selected = set(args.only.split(","))
        unknown = selected - {e["name"] for e in exps}
        if unknown:
            raise ValueError(f"unknown experiments: {sorted(unknown)}")
        exps = [e for e in exps if e["name"] in selected]
    root = Path(args.output_root)
    cache = root / "shared" / "routed_samples.pt"
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
    for exp in exps:
        out = root / exp["name"]
        if exp["name"] != "original" and not args.eval_only:
            if args.resume and _valid_compress(out, exp, args):
                print(f"[sweep] reused compression: {out}", flush=True)
            else:
                if out.exists() and any(out.iterdir()) and not args.dry_run:
                    raise FileExistsError(f"existing incomplete or mismatched compression: {out}")
                base_checkpoint = (root / f"base_k{exp['k']}" / "compact_qjl_state.pt"
                                   if exp["d_factor"] else None)
                _run(_command(args, exp, out, "compress", cache, base_checkpoint),
                     args.dry_run)
        if not args.skip_eval:
            if exp["name"] != "original" and not _valid_compress(out, exp, args) and not args.dry_run:
                raise ValueError(f"no verified checkpoint: {out}")
            if args.resume and not args.rerun_eval and (
                _valid_original(out, args) if exp["name"] == "original"
                else _valid_eval(out, args)):
                print(f"[sweep] reused evaluation: {out}", flush=True)
            else:
                eval_out = out if exp["name"] == "original" else out / "eval"
                if eval_out.exists() and any(eval_out.iterdir()) and not args.dry_run:
                    if not args.rerun_eval:
                        raise FileExistsError(f"existing evaluation: {eval_out}")
                    backup = eval_out.with_name(eval_out.name + f".prior_{int(time.time())}")
                    eval_out.rename(backup)
                    print(f"[sweep] preserved prior evaluation: {backup}", flush=True)
                _run(_command(args, exp, eval_out,
                              "original" if exp["name"] == "original" else "eval", cache),
                     args.dry_run)
                if not args.dry_run and exp["name"] != "original" and not _valid_eval(out, args):
                    raise RuntimeError(f"evaluation incomplete: {out}")
        if not args.dry_run:
            _summary(all_exps, root, args)


if __name__ == "__main__":
    main()
