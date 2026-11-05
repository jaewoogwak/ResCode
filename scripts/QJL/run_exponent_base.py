#!/usr/bin/env python3
"""Independent Qwen exponent-base/QJL compression, diagnosis and evaluation."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rescode.evaluation import evaluate_fewshot, get_calib_dataloder
from rescode.merging.eval_summary import report_eval_summary
from rescode.qjl.calibration import collect_projection_samples
from rescode.qjl.exponent_model import (CHECKPOINT_V1, CHECKPOINT_V2,
                                        compress_model, load_checkpoint, save_checkpoint)


TASKS = "winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte"
MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("compress", "eval", "original"), required=True)
    p.add_argument("--model-name", default=MODEL)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--k", type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument("--d-factor", type=int, choices=(0, 1, 2), default=1,
                   help="QJL D = factor * projection input dimension; 0 is Base Only")
    p.add_argument("--alpha", choices=("0", "1", "calibrated"), default="1")
    p.add_argument("--backend", choices=("triton", "triton_legacy", "reference"), default="triton")
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--train-data-path", default="rescode/data/wikipedia-calib.jsonl")
    p.add_argument("--n-sentences", type=int, default=2400)
    p.add_argument("--calib-seqlen", type=int, default=2048)
    p.add_argument("--max-train-per-expert", type=int, default=32)
    p.add_argument("--max-heldout-per-expert", type=int, default=16)
    p.add_argument("--diagnostic-samples", type=int, default=16)
    p.add_argument("--sample-cache", default="results/qjl_exponent_base/shared/routed_samples.pt")
    p.add_argument("--base-checkpoint", default=None,
                   help="Reuse the exact encoded base from a matching Base Only checkpoint")
    p.add_argument("--layers", default="all", help="all or comma-separated layer IDs for smoke tests")
    p.add_argument("--task", default=TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.94)
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp{os.getpid()}")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def summarize(records):
    result = {}
    for key in ("gate", "up", "down", "expert_output", "moe_output"):
        subset = [r for r in records if r["projection"] == key]
        errors = [r["output"]["relative_l2"] for r in subset
                  if r["output"]["relative_l2"] is not None]
        mses = [r["output"]["mse"] for r in subset if r["output"]["mse"] is not None]
        result[key] = {"count": len(errors),
                       "mean_relative_l2": statistics.mean(errors) if errors else None,
                       "mean_mse": statistics.mean(mses) if mses else None}
    projections = [r for r in records if r["projection"] in ("gate", "up", "down")]
    ratios = [r["residual_norm_ratio"] for r in projections
              if r["residual_norm_ratio"] is not None]
    result["residual_norm_ratio"] = statistics.mean(ratios) if ratios else None
    result["exponent_entropy_bits"] = statistics.mean(r["exponent_entropy_bits"] for r in projections)
    result["base_stored_bytes"] = sum(r["base_stored_bytes"] for r in projections)
    result["base_stored_bits_per_weight"] = statistics.mean(
        r["base_stored_bits_per_weight"] for r in projections)
    result["base_gpu_resident_bytes"] = sum(r["base_gpu_resident_bytes"] for r in projections)
    result["mean_alpha"] = statistics.mean(r["alpha"] for r in projections)
    result["max_decomposition_relative_l2"] = max(
        r["decomposition"]["relative_l2"] or 0.0 for r in projections)
    return result


def _cache_key(args, layers):
    path = Path(args.train_data_path)
    stat = path.stat()
    return {"model_name": args.model_name, "calib_seed": args.calib_seed,
            "path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "n_sentences": args.n_sentences,
            "calib_seqlen": args.calib_seqlen,
            "max_train": args.max_train_per_expert,
            "max_heldout": args.max_heldout_per_expert, "layers": layers}


def _samples(model, tokenizer, args, layers):
    cache = Path(args.sample_cache)
    key = _cache_key(args, layers)
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload["key"] != key:
            raise ValueError(f"sample cache configuration mismatch: {cache}")
        print(f"[compact] reused routed calibration samples: {cache}", flush=True)
        return payload["train"], payload["heldout"], payload["probes"]
    loader = get_calib_dataloder(
        dataset="c4", tokenizer=tokenizer, max_block_size=args.calib_seqlen,
        n_blocks_for_stat=args.n_sentences, batch_size=1, num_workers=2,
        seed=args.calib_seed, data_path=args.train_data_path)
    train, heldout, probes = collect_projection_samples(
        model, loader, "qwen", layers, args.max_train_per_expert,
        args.max_heldout_per_expert)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + f".tmp{os.getpid()}")
    torch.save({"key": key, "train": train, "heldout": heldout, "probes": probes}, tmp)
    os.replace(tmp, cache)
    return train, heldout, probes


def _evaluate(model, tokenizer, args, out):
    started = time.perf_counter()
    path = out / "lm_eval.txt"
    results = evaluate_fewshot(model, tokenizer, args.task, 0,
                               eval_batch_size=args.eval_batch_size,
                               output_path=str(path), limit=args.eval_limit or None)
    summary = report_eval_summary(results, args.task, str(path),
                                  header="Exponent Base + QJL evaluation")
    return {"task_summary": summary, "task_results": results.get("results", {}),
            "eval_limit": args.eval_limit, "eval_seconds": time.perf_counter() - started}


def _gpu_bytes(model):
    base, qjl = 0, 0
    for module in model.modules():
        if module.__class__.__name__ == "CompactBase":
            base += module.gpu_resident_bytes()
        elif module.__class__.__name__ == "PackedResidual":
            qjl += module.packed.numel() + module.norms.numel() * 4
    return base, qjl


def _probe(model, tokenizer):
    ids = tokenizer("The answer is", return_tensors="pt")["input_ids"].to(
        model.get_input_embeddings().weight.device)
    with torch.no_grad():
        return model(input_ids=ids).logits.float().cpu()


def _measure_forward(model, tokenizer):
    ids = tokenizer("The answer is", return_tensors="pt")["input_ids"].to(
        model.get_input_embeddings().weight.device)
    wall_ms, event_ms = [], []
    with torch.no_grad():
        for _ in range(2):
            model(input_ids=ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            started = time.perf_counter()
            model(input_ids=ids)
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                event_ms.append(start_event.elapsed_time(end_event))
            wall_ms.append((time.perf_counter() - started) * 1000)
    result = {"inference_forward_median_ms": statistics.median(wall_ms),
              "inference_cuda_event_median_ms": statistics.median(event_ms) if event_ms else None}
    if torch.cuda.is_available():
        result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["gpu_resident_allocated_bytes"] = torch.cuda.memory_allocated()
    return result


def main():
    args = arguments()
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.d_factor == 0 and args.alpha != "0" and args.mode == "compress":
        raise ValueError("Base Only requires --alpha 0")
    if args.d_factor and args.alpha == "0" and args.mode == "compress":
        raise ValueError("alpha 0 uses --d-factor 0 to avoid unused QJL storage")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output exists: {out}")
    if args.mode == "eval" and not args.checkpoint:
        raise ValueError("--checkpoint required for eval")
    if args.mode == "compress" and not Path(args.train_data_path).is_file():
        raise FileNotFoundError(args.train_data_path)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "config.json", vars(args))
    torch.manual_seed(args.projection_seed)
    load_start = time.perf_counter()
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory
                             * args.gpu_memory_fraction)
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "128GiB"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True, max_memory=max_memory)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    metrics = {"model_load_seconds": time.perf_counter() - load_start,
               "model_name": args.model_name, "mode": args.mode}
    if args.mode == "compress":
        layers = ([i for i, layer in enumerate(model.model.layers)
                   if hasattr(layer.mlp, "experts")]
                  if args.layers == "all" else [int(i) for i in args.layers.split(",")])
        if any(not hasattr(model.model.layers[i].mlp, "experts") for i in layers):
            raise ValueError("selected layer is not an MoE layer")
        started = time.perf_counter()
        train, heldout, probes = _samples(model, tokenizer, args, layers)
        metrics["calibration_seconds"] = time.perf_counter() - started
        config = {"format": CHECKPOINT_V2, "model_name": args.model_name,
                  "k": args.k, "d_factor": args.d_factor, "alpha": args.alpha,
                  "backend": args.backend, "projection_seed": args.projection_seed,
                  "calib_seed": args.calib_seed, "layers": layers,
                  "diagnostic_samples": args.diagnostic_samples,
                  "sample_cache": args.sample_cache,
                  "train_data_path": args.train_data_path}
        preencoded = None
        if args.base_checkpoint:
            source = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
            source_config = source.get("config", {})
            if (source.get("format") not in (CHECKPOINT_V1, CHECKPOINT_V2)
                    or source_config.get("d_factor") != 0
                    or source_config.get("k") != args.k
                    or source_config.get("layers") != layers
                    or source_config.get("model_name") != args.model_name):
                raise ValueError("base checkpoint configuration mismatch")
            preencoded = source["layers"]
            del source
            print(f"[compact] reused Base Only encoding: {args.base_checkpoint}", flush=True)
        original_bytes = sum(sum(p.numel() * p.element_size()
                                 for p in model.model.layers[i].mlp.experts[0].parameters()) * 60
                             for i in layers)
        started = time.perf_counter()
        data, records, diagnostics_seconds = compress_model(
            model, train, heldout, probes, config, preencoded=preencoded)
        metrics["compression_and_diagnostics_seconds"] = time.perf_counter() - started
        metrics["diagnostics_seconds"] = diagnostics_seconds
        diagnostic_summary = summarize(records)
        atomic_json(out / "diagnostics.json", {"records": records,
                                                "summary": diagnostic_summary})
        checkpoint = out / "compact_qjl_state.pt"
        sizes = save_checkpoint(model, data, config, str(checkpoint))
        del data, preencoded
        metrics.update(sizes)
        metrics.update({"original_routed_bf16_bytes": original_bytes,
                        "checkpoint_ratio_vs_original_routed": sizes["checkpoint_bytes"] / original_bytes,
                        "diagnostic_summary": diagnostic_summary,
                        "checkpoint": str(checkpoint)})
        before = _probe(model, tokenizer)
        load_checkpoint(model, str(checkpoint), backend=args.backend)
        after = _probe(model, tokenizer)
        metrics["reload_probe_max_abs_error"] = (before - after).abs().max().item()
        if metrics["reload_probe_max_abs_error"] != 0:
            raise AssertionError("checkpoint reload changed probe logits")
    elif args.mode == "eval":
        checkpoint_load_start = time.perf_counter()
        payload = load_checkpoint(model, args.checkpoint, backend=args.backend)
        metrics["checkpoint_load_seconds"] = time.perf_counter() - checkpoint_load_start
        metrics["checkpoint"] = args.checkpoint
        metrics["checkpoint_bytes"] = Path(args.checkpoint).stat().st_size
        metrics["checkpoint_config"] = payload["config"]
        del payload
    if args.mode != "original":
        layers = (config["layers"] if args.mode == "compress"
                  else metrics["checkpoint_config"]["layers"])
        if any(any(not isinstance(expert, torch.nn.Identity) for expert in
                   model.model.layers[i].mlp.experts) for i in layers):
            raise AssertionError("original BF16 routed expert remains resident")
        metrics["base_gpu_resident_bytes"], metrics["qjl_gpu_resident_bytes"] = _gpu_bytes(model)
        metrics["exponent_restore_seconds"] = sum(
            module.exponent_restore_seconds for module in model.modules()
            if module.__class__.__name__ == "CompactBase")
        metrics["projection_gpu_resident_bytes"] = sum(
            b.resident_bytes("cuda") for b in model._compact_projection_banks.values())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics["compression_or_load_peak_bytes"] = torch.cuda.max_memory_allocated()
    metrics.update(_measure_forward(model, tokenizer))
    if not args.skip_eval:
        metrics.update(_evaluate(model, tokenizer, args, out))
    atomic_json(out / "metrics.json", metrics)
    print(f"[compact] metrics: {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
