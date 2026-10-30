#!/usr/bin/env python3
"""Separate direct-residual QJL experiment; existing RESCODE path is untouched."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rescode.evaluation import evaluate_fewshot, get_calib_dataloder
from rescode.merging.calib_stats import iter_moe_blocks, load_or_collect_calib_stats
from rescode.merging.expert_clusters import form_clusters, residual_params_fn_for
from rescode.merging.eval_summary import report_eval_summary
from rescode.qjl.calibration import ProjectionSampleCollector, collect_projection_samples
from rescode.qjl.core import parse_targets
from rescode.qjl.model import convert_model, load_qjl_checkpoint, save_qjl_checkpoint


TASKS = "winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte"
MODELS = {"qwen": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
          "mixtral": "mistralai/Mixtral-8x7B-v0.1"}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("compress", "eval", "original", "rescode"), default="compress")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--base-method", choices=("mean", "saliency", "qjl_aware"), default="qjl_aware")
    p.add_argument("--residual-mode", choices=("qjl", "exact"), default="qjl")
    p.add_argument("--qjl-d", type=int, default=8192)
    p.add_argument("--targets", choices=("gate", "up", "down", "gate_up", "gate_up_down"), default="gate_up_down")
    p.add_argument("--backend", choices=("triton", "reference"), default="triton")
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--num-groups", type=int, default=None)
    p.add_argument("--singleton-ratio", type=float, default=0.15)
    p.add_argument("--rank-for-grouping", type=int, default=128)
    p.add_argument("--budget-aware", choices=("true", "false"), default="true")
    p.add_argument("--linkage", default="average")
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--cg-iterations", type=int, default=30)
    p.add_argument("--train-data-path", default="rescode/data/dclm-calib.jsonl")
    p.add_argument("--n-sentences", type=int, default=32)
    p.add_argument("--calib-batch-size", type=int, default=1)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--max-train-per-expert", type=int, default=256)
    p.add_argument("--max-heldout-per-expert", type=int, default=64)
    p.add_argument("--calib-stats-cache", default=None)
    p.add_argument("--groups-path", default=None)
    p.add_argument("--layers", default="all", help="all or comma-separated original layer indices")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--rescode-tag", default=None)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--task", default=TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.94,
                   help="Fraction of each GPU offered to transformers device_map=auto")
    return p.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, default=str)
    os.replace(tmp, path)


def reset_cuda_peaks():
    for gpu_index in range(torch.cuda.device_count()):
        with torch.cuda.device(gpu_index):
            torch.cuda.reset_peak_memory_stats()


def synchronize_cuda():
    for gpu_index in range(torch.cuda.device_count()):
        with torch.cuda.device(gpu_index):
            torch.cuda.synchronize()


def calibrator(tokenizer, args):
    if not Path(args.train_data_path).is_file():
        raise FileNotFoundError(
            f"Missing {args.train_data_path}; run "
            "python scripts/RESCODE/prepare_calib_datasets.py --dataset dclm"
        )
    return get_calib_dataloder(
        dataset="c4", tokenizer=tokenizer, max_block_size=args.calib_seqlen,
        n_blocks_for_stat=args.n_sentences, batch_size=args.calib_batch_size,
        num_workers=2, seed=args.calib_seed, data_path=args.train_data_path,
    )


def grouping(model, loader, args, shared_root):
    key = {
        "model_name": args.model_name, "num_groups": args.num_groups,
        "singleton_ratio": args.singleton_ratio, "rank": args.rank_for_grouping,
        "targets": args.targets, "budget_aware": args.budget_aware,
        "linkage": args.linkage, "calib_seed": args.calib_seed,
        "n_sentences": args.n_sentences, "train_data_path": str(Path(args.train_data_path).resolve()),
        "train_data_bytes": Path(args.train_data_path).stat().st_size,
        "train_data_mtime_ns": Path(args.train_data_path).stat().st_mtime_ns,
    }
    name = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    groups_path = Path(args.groups_path or shared_root / f"groups-{name}.pt")
    if groups_path.is_file():
        saved = torch.load(groups_path, map_location="cpu", weights_only=False)
        if saved["key"] != key:
            raise ValueError("groups-path configuration mismatch")
        print(f"[QJL] reused groups: {groups_path}", flush=True)
        return saved["groups"], saved["saliency"], saved["summary"], groups_path, None
    stats_cache = args.calib_stats_cache or str(shared_root / f"calib-stats-{name}.pt")
    sample_layers = None if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    sampler = ProjectionSampleCollector(
        args.model, sample_layers, args.max_train_per_expert, args.max_heldout_per_expert
    )
    stats = load_or_collect_calib_stats(
        model, loader, args.model, stats_cache, sample_hook=sampler.capture
    )
    _, block = next(iter_moe_blocks(model, args.model))
    first = block.experts[0]
    gate = first.w1.weight if args.model == "mixtral" else first.gate_proj.weight
    d_ff, d_model = gate.shape
    groups, summary = form_clusters(
        stats, num_groups=args.num_groups, singleton_ratio=args.singleton_ratio,
        expert_params=3 * d_ff * d_model,
        residual_params_fn=residual_params_fn_for(
            d_model, d_ff, args.rank_for_grouping, len(parse_targets(args.targets))
        ), linkage=args.linkage, budget_aware=args.budget_aware == "true",
    )
    groups_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"key": key, "groups": groups,
                "saliency": {k: v["saliency"] for k, v in stats.items()},
                "summary": summary}, groups_path)
    samples = sampler.result() if sampler.captured else None
    return groups, {k: v["saliency"] for k, v in stats.items()}, summary, groups_path, samples


def evaluate(model, tokenizer, args, out):
    path = out / "lm_eval.txt"
    results = evaluate_fewshot(model, tokenizer, args.task, 0,
                               eval_batch_size=args.eval_batch_size,
                               output_path=str(path),
                               limit=args.eval_limit or None)
    summary = report_eval_summary(results, args.task, str(path), header="Direct QJL evaluation")
    return {"task_summary": summary,
            "task_results": results.get("results", {}),
            "eval_limit": args.eval_limit}


def launch_rescode(args):
    tag = args.rescode_tag or f"qjl-control-seed{args.calib_seed}"
    baseline_out = ROOT / "results" / args.model / f"rescode-{args.model}-{args.num_groups}e-{tag}"
    if (baseline_out / "result.txt").exists() and not args.overwrite:
        raise FileExistsError(f"ResCode baseline exists: {baseline_out}")
    env = os.environ.copy()
    dataset_name = Path(args.train_data_path).name.split("-")[0]
    if dataset_name not in ("c4", "wikipedia", "slimpajama", "dclm", "starcoder"):
        dataset_name = "dclm"
    env.update({
        "MODEL": args.model_name, "NUM_EXPERTS": str(args.num_groups),
        "TRAIN_DATA": args.train_data_path, "N_SENTENCES": str(args.n_sentences),
        "CALIB_SEED": str(args.calib_seed), "TASK": args.task,
        "CALIB_DATASET": dataset_name,
        "EVAL_LIMIT": str(args.eval_limit),
        "TAG": tag,
        "ACCELERATE": str(Path(sys.executable).with_name("accelerate")),
    })
    subprocess.run(["bash", "scripts/RESCODE/run.sh"], cwd=ROOT, env=env, check=True)


def main():
    args = arguments()
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.model_name = args.model_name or MODELS[args.model]
    args.num_groups = args.num_groups or (30 if args.model == "qwen" else 4)
    if args.qjl_d < 1 or args.n_sentences < 1:
        raise ValueError("qjl-d and n-sentences must be positive")
    if not 0.1 <= args.gpu_memory_fraction <= 0.98:
        raise ValueError("gpu-memory-fraction must be between 0.1 and 0.98")
    default_name = (f"{args.base_method}_d{args.qjl_d}" if args.mode == "compress"
                    else f"{args.mode}_d{args.qjl_d}")
    out = Path(args.output_dir or f"results/qjl_direct/{args.model}/{default_name}")
    if (out / "metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"{out} already has results; choose another --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", vars(args))
    if args.mode == "rescode":
        launch_rescode(args)
        return
    torch.manual_seed(args.projection_seed)
    if torch.cuda.is_available():
        reset_cuda_peaks()
    start = time.perf_counter()
    print(f"[QJL] loading {args.model_name}", flush=True)
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory
                             * args.gpu_memory_fraction)
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "128GiB"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True, max_memory=max_memory,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    metrics = {"model_name": args.model_name, "mode": args.mode}
    metrics["environment"] = {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "triton": importlib.metadata.version("triton"),
        "lm_eval": importlib.metadata.version("lm-eval"),
        "gpu_names": [torch.cuda.get_device_name(i)
                      for i in range(torch.cuda.device_count())],
        "model_config_commit": getattr(model.config, "_commit_hash", None),
    }
    if hasattr(model, "hf_device_map"):
        metrics["hf_device_map"] = {str(k): str(v) for k, v in model.hf_device_map.items()}
    if args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for --mode eval")
        payload = load_qjl_checkpoint(model, args.checkpoint, backend=args.backend)
        metrics["checkpoint"] = args.checkpoint
        metrics["base_method"] = payload["config"]["base_method"]
        metrics["checkpoint_bytes"] = Path(args.checkpoint).stat().st_size
    elif args.mode == "compress":
        checkpoint = Path(args.checkpoint or out / "qjl_state.pt")
        if checkpoint.exists() and not args.overwrite:
            raise FileExistsError(f"{checkpoint} already exists; pass --overwrite to replace")
        loader = calibrator(tokenizer, args)
        shared_root = Path(f"results/qjl_direct/shared/{args.model}")
        groups, saliency, group_summary, groups_path, samples = grouping(
            model, loader, args, shared_root
        )
        if args.layers != "all":
            chosen = {int(x) for x in args.layers.split(",")}
            unknown = chosen - set(groups)
            if unknown:
                raise ValueError(f"non-MoE layers requested: {sorted(unknown)}")
            groups = {i: v for i, v in groups.items() if i in chosen}
        if torch.cuda.is_available():
            for layer_idx in groups:
                block = (model.model.layers[layer_idx].block_sparse_moe if args.model == "mixtral"
                         else model.model.layers[layer_idx].mlp)
                weight_device = next(block.experts[0].parameters()).device
                if weight_device.type != "cuda":
                    raise RuntimeError(
                        f"layer {layer_idx} expert is on {weight_device}; "
                        "raise --gpu-memory-fraction or use more GPU memory "
                        "before compression"
                    )
        if samples is None:
            train, heldout, probes = collect_projection_samples(
                model, loader, args.model, groups,
                max_train=args.max_train_per_expert,
                max_heldout=args.max_heldout_per_expert,
            )
        else:
            train, heldout, probes = samples
            print("[QJL] reused projection samples from grouping calibration", flush=True)
        config = {"base_method": args.base_method, "residual_mode": args.residual_mode,
                  "qjl_d": args.qjl_d, "targets": args.targets,
                  "backend": args.backend, "projection_seed": args.projection_seed,
                  "delta": args.delta, "cg_iterations": args.cg_iterations,
                  "model_name": args.model_name}
        original_bytes = 0
        for layer_idx in groups:
            block = (model.model.layers[layer_idx].block_sparse_moe if args.model == "mixtral"
                     else model.model.layers[layer_idx].mlp)
            expert = block.experts[0]
            weights = (expert.w1.weight, expert.w3.weight, expert.w2.weight) if args.model == "mixtral" else (
                expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight)
            original_bytes += len(block.experts) * sum(w.numel() * 2 for w in weights)
        diagnostics = convert_model(model, args.model, groups, saliency,
                                    train, heldout, config, probes)
        sizes = save_qjl_checkpoint(model, args.model, groups, config,
                                    str(checkpoint), diagnostics)
        metrics.update(sizes)
        metrics.update({"original_expert_bf16_bytes": original_bytes,
                        "expert_checkpoint_compression_ratio":
                        1 - sizes["checkpoint_bytes"] / original_bytes if original_bytes else None,
                        "groups_path": str(groups_path),
                        "grouping_rescode_parameter_estimate": group_summary,
                        "diagnostics": diagnostics,
                        "checkpoint": str(checkpoint)})
    metrics["pre_eval_seconds"] = time.perf_counter() - start
    # Measure a fixed short forward after compression/load. Resetting peak
    # excludes model loading and calibration allocations from inference peak.
    encoded = tokenizer("The answer is", return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    probe_ids = encoded["input_ids"].to(input_device)
    with torch.no_grad():
        model(input_ids=probe_ids)
        if torch.cuda.is_available():
            synchronize_cuda()
            metrics["compression_or_load_peak_bytes"] = {
                str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
            }
            reset_cuda_peaks()
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            model(input_ids=probe_ids)
            if torch.cuda.is_available():
                synchronize_cuda()
            times.append(time.perf_counter() - t0)
    metrics["inference_forward_median_ms"] = sorted(times)[1] * 1000
    if torch.cuda.is_available():
        metrics["gpu_peak_allocated_bytes"] = {
            str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        }
        metrics["gpu_resident_allocated_bytes"] = {
            str(i): torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
        }
        if hasattr(model, "_qjl_projection_bank"):
            metrics["projection_gpu_resident_bytes"] = model._qjl_projection_bank.resident_bytes("cuda")
            metrics["projection_cpu_resident_bytes"] = model._qjl_projection_bank.resident_bytes("cpu")
    if not args.skip_eval:
        eval_start = time.perf_counter()
        metrics.update(evaluate(model, tokenizer, args, out))
        metrics["eval_seconds"] = time.perf_counter() - eval_start
    write_json(out / "metrics.json", metrics)
    print(f"[QJL] metrics: {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
