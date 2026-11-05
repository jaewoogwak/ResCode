#!/usr/bin/env python3
"""Measure v1/v2 exponent checkpoints on the same GPU and deterministic inputs.

Run once per checkpoint in a fresh process. Each result includes small output
traces for numerical parity checks; no calibration or evaluation data is read.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from rescode.qjl.exponent_model import load_checkpoint


def timing(fn, warmup=5, repeats=20):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    return {"median_ms": statistics.median(ordered),
            "p95_ms": ordered[min(len(ordered)-1, int(0.95 * len(ordered)))],
            "samples_ms": samples}


def tensor_error(actual, expected):
    a, b = actual.float(), expected.float()
    difference = a - b
    denominator = torch.linalg.vector_norm(b).item()
    return {"max_abs": difference.abs().max().item(),
            "relative_l2": (torch.linalg.vector_norm(difference).item() / denominator
                            if denominator else None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-name", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    p.add_argument("--backend", default="triton")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmark")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model_load_s = time.perf_counter() - load_start
    checkpoint_load_s = None
    checkpoint_format = "original_bf16"
    config = {}
    if args.checkpoint:
        start = time.perf_counter()
        payload = load_checkpoint(model, str(args.checkpoint), backend=args.backend)
        torch.cuda.synchronize()
        checkpoint_load_s = time.perf_counter() - start
        checkpoint_format = payload["format"]
        config = payload["config"]
        del payload
        if any(not isinstance(expert, torch.nn.Identity)
               for layer in model.model.layers if hasattr(layer.mlp, "experts")
               for expert in layer.mlp.experts):
            raise AssertionError("original routed BF16 expert remains resident")
    print(f"[bench] loaded {checkpoint_format} in {checkpoint_load_s or 0:.1f}s", flush=True)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    load_peak = torch.cuda.max_memory_allocated()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    model_device = model.get_input_embeddings().weight.device
    result = {"checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "checkpoint_format": checkpoint_format, "config": config,
              "checkpoint_bytes": args.checkpoint.stat().st_size if args.checkpoint else None,
              "model_load_s": model_load_s, "checkpoint_load_s": checkpoint_load_s,
              "gpu_loaded_bytes": resident, "gpu_load_peak_bytes": load_peak,
              "gpu_name": torch.cuda.get_device_name(),
              "torch_version": torch.__version__, "backend": args.backend,
              "timing": {}, "tensor_bytes": {}}
    if args.checkpoint:
        bases = [m for m in model.modules() if m.__class__.__name__ == "CompactBase"]
        residuals = [m for m in model.modules() if m.__class__.__name__ == "PackedResidual"]
        result["tensor_bytes"]["base"] = sum(m.gpu_resident_bytes() for m in bases)
        result["tensor_bytes"]["qjl"] = sum(m.packed.numel() +
                                                m.norms.numel() * m.norms.element_size()
                                                for m in residuals)
        result["tensor_bytes"]["exponent"] = sum(
            m.exponents.numel() if hasattr(m, "exponents") else
            m.exp_data.numel() + m.exp_offsets.numel() * m.exp_offsets.element_size()
            + m.exp_modes.numel() for m in bases)
    block = model.model.layers[0].mlp
    expert = (block._compact_experts[0] if args.checkpoint else block.experts[0])
    x = {m: torch.randn(m, 2048, device=model_device, dtype=torch.bfloat16)
         for m in (1, 4, 32, 128)}
    down_x = {m: torch.randn(m, 1408, device=model_device, dtype=torch.bfloat16)
              for m in (1, 4, 32, 128)}

    def measure(key, fn):
        result["timing"][key] = timing(fn, args.warmup, args.repeats)

    with torch.inference_mode():
        for m in (1, 4, 32, 128):
            if args.checkpoint:
                for proj in ("gate", "up", "down"):
                    inp = down_x[m] if proj == "down" else x[m]
                    measure(f"base_{proj}_m{m}", lambda proj=proj, inp=inp:
                            expert.bases[proj](inp))
                    if proj in expert.residuals:
                        residual = expert.residuals[proj]
                        measure(f"qjl_{proj}_m{m}", lambda residual=residual, inp=inp:
                                residual(inp))
                        bank_matrix = residual.bank.get(inp.shape[-1], inp.device)
                        z = inp.to(bank_matrix.dtype) @ bank_matrix
                        measure(f"qjl_project_{proj}_m{m}",
                                lambda inp=inp, bank_matrix=bank_matrix:
                                inp.to(bank_matrix.dtype) @ bank_matrix)
                        measure(f"qjl_packed_{proj}_m{m}",
                                lambda residual=residual, z=z, inp=inp:
                                residual.forward_projected(z, inp.dtype))
            measure(f"expert_m{m}", lambda m=m: expert(x[m]))
            if m in (1, 32):
                measure(f"moe_m{m}", lambda m=m: block(x[m].unsqueeze(0)))
            print(f"[bench] projection/expert/MoE tokens={m} complete", flush=True)
        trace = {}
        for m in (1, 4, 32, 128):
            trace[f"expert_m{m}"] = expert(x[m]).float().cpu()
            for proj in ("gate", "up", "down"):
                inp = down_x[m] if proj == "down" else x[m]
                if args.checkpoint:
                    trace[f"base_{proj}_m{m}"] = expert.bases[proj](inp).float().cpu()
                    trace[f"projection_{proj}_m{m}"] = expert._projection(proj, inp).float().cpu()
                else:
                    trace[f"projection_{proj}_m{m}"] = getattr(expert, f"{proj}_proj")(inp).float().cpu()
        moe, router = block(x[32].unsqueeze(0))
        probs = torch.softmax(router.float(), dim=-1)
        routing_weight, topk = torch.topk(probs, block.top_k, dim=-1)
        trace.update({"moe": moe.float().cpu(), "router": router.float().cpu(),
                      "routing_topk": topk.cpu(), "routing_weight": routing_weight.cpu(),
                      "shared": block.shared_expert(x[32]).float().cpu()})
        ids = torch.arange(100, 229, device=model_device)[None, :] % model.config.vocab_size
        for batch, length in ((1, 16), (1, 128), (4, 32)):
            input_ids = ids[:, :length].expand(batch, length).contiguous()
            measure(f"prefill_b{batch}_s{length}",
                    lambda input_ids=input_ids: model(input_ids=input_ids, use_cache=False))
            print(f"[bench] prefill batch={batch} length={length} complete", flush=True)
        for length in (16, 128):
            prompt = ids[:, :length]
            cached = model(input_ids=prompt, use_cache=True)
            next_id = ids[:, length:length+1]
            cache_pool = [DynamicCache.from_legacy_cache(cached.past_key_values)
                          for _ in range(args.warmup + args.repeats)]
            cache_iter = iter(cache_pool)
            measure(f"decode_s{length}", lambda next_id=next_id, cache_iter=cache_iter:
                    model(input_ids=next_id, past_key_values=next(cache_iter),
                          use_cache=False))
            del cached, cache_pool, cache_iter
            print(f"[bench] decode context={length} complete", flush=True)
        fixed = ids[:, :16]
        trace["logits"] = model(input_ids=fixed, use_cache=False).logits.float().cpu()
        trace["prompt_ids"] = fixed.cpu()
    torch.cuda.synchronize()
    result["gpu_resident_bytes"] = torch.cuda.memory_allocated()
    result["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
    result["gpu_reserved_bytes"] = torch.cuda.memory_reserved()
    result["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    torch.save(trace, args.output / "trace.pt")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "checkpoint_format": checkpoint_format,
                      "gpu_resident_bytes": result["gpu_resident_bytes"],
                      "gpu_peak_bytes": result["gpu_peak_bytes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
