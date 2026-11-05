#!/usr/bin/env python3
"""Locate v1/v2 MoE and routing divergence with the same fixed model input."""

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM

from rescode.qjl.exponent_model import load_checkpoint


def error(actual, expected):
    diff = actual.float() - expected.float()
    norm = expected.float().norm().item()
    return {"max_abs": diff.abs().max().item(),
            "relative_l2": diff.norm().item() / norm if norm else None}


def run(checkpoint, ids):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat", torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True).eval()
    payload = load_checkpoint(model, checkpoint, backend="triton")
    del payload
    layers = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        block = layer.mlp
        if not hasattr(block, "experts"):
            continue
        record = {}
        layers[index] = record
        def before(module, inputs, record=record):
            record["input"] = inputs[0].detach().cpu()
        def after(module, inputs, output, record=record):
            record["output"] = output[0].detach().cpu()
            record["router"] = output[1].detach().cpu()
            record["topk"] = torch.topk(torch.softmax(output[1].float(), dim=-1),
                                        module.top_k, dim=-1).indices.detach().cpu()
        handles.append(block.register_forward_pre_hook(before))
        handles.append(block.register_forward_hook(after))
    with torch.inference_mode():
        logits = model(input_ids=ids.to(model.device), use_cache=False).logits.float().cpu()
    for handle in handles:
        handle.remove()
    return model, layers, logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy", required=True)
    p.add_argument("--optimized", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ids = torch.arange(100, 116)[None, :]
    old_model, old_layers, old_logits = run(args.legacy, ids)
    del old_model
    gc.collect()
    torch.cuda.empty_cache()
    new_model, new_layers, new_logits = run(args.optimized, ids)
    rows = []
    with torch.inference_mode():
        for index in old_layers:
            old, new = old_layers[index], new_layers[index]
            shared_input = old["input"].to(new_model.device)
            same_input_output, same_input_router = new_model.model.layers[index].mlp(shared_input)
            same_input_topk = torch.topk(torch.softmax(same_input_router.float(), dim=-1),
                                         new_model.model.layers[index].mlp.top_k,
                                         dim=-1).indices.cpu()
            rows.append({"layer": index,
                         "input": error(new["input"], old["input"]),
                         "router": error(new["router"], old["router"]),
                         "topk_exact": torch.equal(new["topk"], old["topk"]),
                         "topk_match_fraction": (new["topk"] == old["topk"]).float().mean().item(),
                         "moe_output": error(new["output"], old["output"]),
                         "same_input_router_exact": torch.equal(same_input_router.cpu(), old["router"]),
                         "same_input_topk_exact": torch.equal(same_input_topk, old["topk"]),
                         "same_input_moe_output": error(same_input_output.cpu(), old["output"])})
    report = {"legacy": args.legacy, "optimized": args.optimized,
              "logits": error(new_logits, old_logits), "layers": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
