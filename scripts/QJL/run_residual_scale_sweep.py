#!/usr/bin/env python3
"""Sequential ResCode QJL residual-scale sweep with resumable checkpoints."""

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
DEFAULT_GROUPS = "results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt"
TASKS = "winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphas", default="0,0.25,0.5,1.0")
    p.add_argument("--output-root", default="results/qjl_residual_scale/qwen/d8192")
    p.add_argument("--groups-path", default=DEFAULT_GROUPS)
    p.add_argument("--train-data-path", default="rescode/data/wikipedia-calib.jsonl")
    p.add_argument("--model-name", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    p.add_argument("--qjl-d", type=int, default=8192)
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--n-sentences", type=int, default=2400)
    p.add_argument("--calib-seqlen", type=int, default=2048)
    p.add_argument("--max-train-per-expert", type=int, default=256)
    p.add_argument("--max-heldout-per-expert", type=int, default=64)
    p.add_argument("--cg-iterations", type=int, default=30)
    p.add_argument("--layers", default="all", help="all or comma-separated layer IDs; smoke tests only")
    p.add_argument("--diagnostic-samples", type=int, default=32)
    p.add_argument("--task", default=TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--backend", choices=("triton", "triton_legacy", "reference"), default="triton")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.94)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def alpha_name(alpha):
    return "alpha_" + (f"{alpha:g}".replace(".", "_"))


def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def verified_compression(out, args, alpha):
    config = load_json(out / "config.json")
    metrics = load_json(out / "metrics.json")
    diagnostics = load_json(out / "diagnostics.json")
    checkpoint = out / "qjl_state.pt"
    expected = {"residual_scale_alpha": alpha, "qjl_d": args.qjl_d,
                "model_name": args.model_name, "backend": args.backend,
                "projection_seed": args.projection_seed,
                "calib_seed": args.calib_seed, "n_sentences": args.n_sentences,
                "calib_seqlen": args.calib_seqlen,
                "max_train_per_expert": args.max_train_per_expert,
                "max_heldout_per_expert": args.max_heldout_per_expert,
                "cg_iterations": args.cg_iterations, "layers": args.layers,
                "diagnostic_samples": args.diagnostic_samples,
                "train_data_path": args.train_data_path,
                "groups_path": args.groups_path, "task": args.task}
    if not config or any(config.get(k) != v for k, v in expected.items()):
        return False
    if not metrics or not diagnostics or not checkpoint.is_file():
        return False
    if diagnostics.get("alpha") != alpha or not diagnostics.get("summary"):
        return False
    return (metrics.get("checkpoint_bytes") == checkpoint.stat().st_size
            and metrics.get("checkpoint_bytes", 0) > 0)


def verified_evaluation(out, tasks, eval_limit):
    metrics = load_json(out / "eval" / "metrics.json")
    if not metrics or metrics.get("eval_limit") != eval_limit:
        return False
    rows = metrics.get("task_summary", {}).get("tasks", [])
    by_task = {row[0]: row[1] for row in rows if len(row) >= 2}
    return (set(by_task) == set(tasks)
            and all(isinstance(by_task[t], (int, float))
                    and math.isfinite(by_task[t]) for t in tasks))


def mean_metric(summary, key):
    return summary.get(key, {}).get("mean")


def make_row(args, alpha, out, error=None):
    row = {"alpha": alpha, "status": "failed" if error else "pending",
           "error": error, "checkpoint_bytes": None,
           "residual_norm_ratio": None,
           "decomposition_error": None,
           "gate_qjl_error": None, "up_qjl_error": None, "down_qjl_error": None,
           "gate_full_projection_error": None,
           "up_full_projection_error": None,
           "down_full_projection_error": None,
           "expert_output_error": None, "moe_output_error": None,
           "compression_seconds": None, "diagnostics_seconds": None,
           "calibration_or_cache_seconds": None,
           "checkpoint_save_seconds": None,
           "eval_seconds": None, "average_acc": None}
    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    for task in tasks:
        row[f"{task}_acc"] = None
        row[f"{task}_acc_norm"] = None
    cm = load_json(out / "metrics.json")
    diag = load_json(out / "diagnostics.json")
    if cm and diag:
        row["status"] = "compressed" if not error else "failed"
        row["checkpoint_bytes"] = cm.get("checkpoint_bytes")
        row["compression_seconds"] = cm.get("compression_seconds")
        row["diagnostics_seconds"] = cm.get("diagnostics_seconds")
        row["calibration_or_cache_seconds"] = cm.get("calibration_or_cache_seconds")
        row["checkpoint_save_seconds"] = cm.get("checkpoint_save_seconds")
        summary = diag.get("summary", {})
        row["residual_norm_ratio"] = mean_metric(summary, "residual_frobenius_ratio")
        row["decomposition_error"] = mean_metric(summary, "decomposition_relative_l2_error")
        for projection in ("gate", "up", "down"):
            row[f"{projection}_qjl_error"] = mean_metric(
                summary, f"qjl_residual_relative_l2_error/heldout/{projection}")
            row[f"{projection}_full_projection_error"] = mean_metric(
                summary, f"full_projection_relative_l2_error/heldout/{projection}")
        row["expert_output_error"] = mean_metric(
            summary, "expert_output_relative_l2_error/heldout")
        row["moe_output_error"] = mean_metric(
            summary, "moe_output_relative_l2_error/heldout")
    em = load_json(out / "eval" / "metrics.json")
    if em:
        row["eval_seconds"] = em.get("eval_seconds")
        for task, acc, norm in em.get("task_summary", {}).get("tasks", []):
            if task in tasks:
                row[f"{task}_acc"] = acc
                row[f"{task}_acc_norm"] = norm
        row["average_acc"] = em.get("task_summary", {}).get("avg")
        if verified_evaluation(out, tasks, args.eval_limit) and not error:
            row["status"] = "evaluated"
    return row


def write_summary(root, rows):
    atomic_json(root / "sweep_summary.json", rows)
    if not rows:
        return
    path = root / "sweep_summary.csv"
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run(command, dry_run):
    print("[sweep]", " ".join(str(x) for x in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    args = arguments()
    alphas = [float(x) for x in args.alphas.split(",")]
    if len(set(alphas)) != len(alphas) or not alphas or any(not 0 <= a <= 1 for a in alphas):
        raise ValueError("--alphas must contain distinct values in [0,1]")
    if args.diagnostic_samples < 0 or args.eval_limit < 0:
        raise ValueError("diagnostic-samples and eval-limit must be nonnegative")
    root = ROOT / args.output_root
    cache = root / "shared_scale_cache.pt"
    groups = ROOT / args.groups_path
    data = ROOT / args.train_data_path
    if not groups.is_file() or not data.is_file():
        raise FileNotFoundError(f"groups={groups.is_file()}, calibration data={data.is_file()}")
    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    if not tasks:
        raise ValueError("--task must list at least one task")
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in alphas:
        out = root / alpha_name(alpha)
        common = ["--model", "qwen", "--model-name", args.model_name,
                  "--qjl-d", str(args.qjl_d), "--backend", args.backend,
                  "--projection-seed", str(args.projection_seed),
                  "--calib-seed", str(args.calib_seed),
                  "--train-data-path", args.train_data_path,
                  "--n-sentences", str(args.n_sentences),
                  "--calib-seqlen", str(args.calib_seqlen),
                  "--max-train-per-expert", str(args.max_train_per_expert),
                  "--max-heldout-per-expert", str(args.max_heldout_per_expert),
                  "--cg-iterations", str(args.cg_iterations),
                  "--layers", args.layers, "--task", args.task,
                  "--eval-limit", str(args.eval_limit),
                  "--eval-batch-size", str(args.eval_batch_size),
                  "--gpu-memory-fraction", str(args.gpu_memory_fraction)]
        try:
            if not (args.resume and verified_compression(out, args, alpha)):
                if out.exists() and any(out.iterdir()) and not args.dry_run:
                    old_config = load_json(out / "config.json")
                    if not args.resume or (old_config is not None and (
                            old_config.get("residual_scale_alpha") != alpha
                            or old_config.get("qjl_d") != args.qjl_d
                            or old_config.get("model_name") != args.model_name)):
                        raise FileExistsError(f"partial or mismatched output: {out}")
                    backup = out.with_name(f"{out.name}.partial_{int(time.time())}_{os.getpid()}")
                    out.rename(backup)
                    print(f"[sweep] preserved partial compression at {backup}", flush=True)
                command = [sys.executable, "scripts/QJL/run.py", "--mode", "compress",
                           "--base-method", "qjl_aware", "--residual-mode", "qjl",
                           "--targets", "gate_up_down", "--skip-eval",
                           "--groups-path", args.groups_path,
                           "--scale-cache", str(cache),
                           "--residual-scale-alpha", str(alpha),
                           "--diagnostic-samples", str(args.diagnostic_samples),
                           "--output-dir", str(out)] + common
                run(command, args.dry_run)
            else:
                print(f"[sweep] reused compression: {out}", flush=True)
            if not args.dry_run and not cache.is_file():
                raise FileNotFoundError(f"shared base/sample cache missing: {cache}")
            if not args.skip_eval and not (args.resume and verified_evaluation(out, tasks, args.eval_limit)):
                eval_out = out / "eval"
                partial_eval = eval_out.exists() and any(eval_out.iterdir())
                if partial_eval and not args.resume and not args.dry_run:
                    raise FileExistsError(f"evaluation already exists: {eval_out}")
                if partial_eval and args.resume and not args.dry_run:
                    backup = eval_out.with_name(
                        f"eval.partial_{int(time.time())}_{os.getpid()}")
                    eval_out.rename(backup)
                    print(f"[sweep] preserved partial evaluation at {backup}", flush=True)
                command = [sys.executable, "scripts/QJL/run.py", "--mode", "eval",
                           "--checkpoint", str(out / "qjl_state.pt"),
                           "--output-dir", str(eval_out)] + common
                run(command, args.dry_run)
                if not args.dry_run and not verified_evaluation(out, tasks, args.eval_limit):
                    raise RuntimeError(f"missing or invalid task accuracy: {eval_out}")
            if not args.dry_run:
                rows.append(make_row(args, alpha, out))
                write_summary(root, rows)
        except Exception as exc:
            if not args.dry_run:
                rows.append(make_row(args, alpha, out, f"{type(exc).__name__}: {exc}"))
                write_summary(root, rows)
            raise


if __name__ == "__main__":
    main()
