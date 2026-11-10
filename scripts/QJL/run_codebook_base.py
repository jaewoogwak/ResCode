#!/usr/bin/env python3
"""Qwen 2-bit Codebook Base + packed QJL residual compression/evaluation."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diagnose_codebook_base import aggregate, atomic_json, select
from rescode.evaluation import evaluate_fewshot
from rescode.merging.eval_summary import report_eval_summary
from rescode.qjl.codebook_model import compress_model, load_checkpoint, save_checkpoint

MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_PATH = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
              / "snapshots/ec052fda178e241c7c443468d2fa1db6618996be")
TASKS = "winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("compress", "eval"), required=True)
    p.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/qjl_codebook_base/run")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--backend", choices=("triton", "reference", "triton_legacy"), default="triton")
    p.add_argument("--layers", default="all")
    p.add_argument("--qjl-d-factor", type=float, default=1.0)
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--row-chunk", type=int, default=128)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.94)
    p.add_argument("--task", default=TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()
    if args.qjl_d_factor <= 0 or args.iterations < 1 or args.row_chunk < 1:
        p.error("QJL factor, iterations and row chunk must be positive")
    if args.mode == "eval" and not args.checkpoint:
        p.error("--checkpoint is required for eval")
    return args


def probe(model, tokenizer):
    ids = tokenizer("The answer is", return_tensors="pt")["input_ids"]
    ids = ids.to(model.get_input_embeddings().weight.device)
    with torch.no_grad():
        return model(input_ids=ids).logits.float().cpu()


def main():
    args = parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source = args.model_path.expanduser().resolve()
    if not (source / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"local model snapshot missing: {source}")
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory
                             * args.gpu_memory_fraction)
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "128GiB"
    model = AutoModelForCausalLM.from_pretrained(
        str(source), torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True, max_memory=max_memory)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
    metrics = {"mode": args.mode, "model": MODEL, "source": str(source),
               "backend": args.backend}
    if args.mode == "compress":
        layers = select(args.layers, len(model.model.layers))
        if any(not hasattr(model.model.layers[i].mlp, "experts") for i in layers):
            raise ValueError("selected layer lacks routed experts")
        config = {"format": "rescode-codebook-base-qjl-v1", "model": MODEL,
                  "layers": layers, "qjl_d_factor": args.qjl_d_factor,
                  "projection_seed": args.projection_seed, "backend": args.backend,
                  "iterations": args.iterations, "row_chunk": args.row_chunk}
        data, records = compress_model(model, config)
        path = args.checkpoint or (out / "compact_qjl_state.pt")
        sizes = save_checkpoint(data, config, path)
        metrics.update(sizes)
        metrics["checkpoint"] = str(path)
        summary = aggregate(records)
        summary.update(sizes)
        summary["layers"] = layers
        summary["qjl_d_factor"] = args.qjl_d_factor
        summary["checkpoint_bits_per_weight"] = (
            8 * sizes["checkpoint_bytes"] / summary["overall"]["weight_count"])
        atomic_json(out / "diagnostics.json", {"records": records})
        with (out / "diagnostics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        atomic_json(out / "summary.json", summary)
        del data
        before = probe(model, tokenizer)
        load_checkpoint(model, path, backend=args.backend)
        after = probe(model, tokenizer)
        parity = (before - after).abs().max().item()
        metrics["reload_probe_max_abs_error"] = parity
        if parity != 0:
            raise AssertionError(f"checkpoint reload changed logits: {parity}")
    else:
        payload = load_checkpoint(model, args.checkpoint, backend=args.backend)
        metrics["checkpoint"] = str(args.checkpoint)
        metrics["checkpoint_bytes"] = args.checkpoint.stat().st_size
        metrics["checkpoint_config"] = payload["config"]
        del payload
        if not args.skip_eval:
            path = out / "lm_eval.txt"
            results = evaluate_fewshot(model, tokenizer, args.task, 0,
                eval_batch_size=args.eval_batch_size, output_path=str(path),
                limit=args.eval_limit or None)
            metrics["task_summary"] = report_eval_summary(
                results, args.task, str(path), header="Codebook Base + QJL evaluation")
            metrics["task_results"] = results.get("results", {})
    atomic_json(out / "metrics.json", metrics)
    print(f"[codebook] saved metrics to {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
