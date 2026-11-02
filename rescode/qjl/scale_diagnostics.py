"""Bounded, compression-time diagnostics for the residual-scale experiment."""

import math
import statistics

import torch
import torch.nn.functional as F

from .core import PackedResidual


def relative_l2(candidate, reference):
    denominator = float(torch.linalg.vector_norm(reference.float()))
    if denominator == 0:
        return None, "zero_reference_norm"
    value = float(torch.linalg.vector_norm(candidate.float() - reference.float()) / denominator)
    if not math.isfinite(value):
        return None, "nonfinite_error"
    return value, None


def add_record(records, metric, layer, expert, projection, split, candidate,
               reference, sample_count=None):
    value, reason = relative_l2(candidate, reference)
    records.append({"metric": metric, "layer": layer, "expert": expert,
                    "projection": projection, "split": split, "value": value,
                    "null_reason": reason, "samples": sample_count})


def add_missing(records, metric, layer, expert, projection, split, reason):
    records.append({"metric": metric, "layer": layer, "expert": expert,
                    "projection": projection, "split": split, "value": None,
                    "null_reason": reason, "samples": 0})


def add_norm_ratio(records, metric, layer, expert, projection, numerator, denominator):
    norm = float(torch.linalg.vector_norm(denominator.float()))
    value = float(torch.linalg.vector_norm(numerator.float())) / norm if norm else None
    records.append({"metric": metric, "layer": layer, "expert": expert,
                    "projection": projection, "split": "weight", "value": value,
                    "null_reason": None if norm else "zero_reference_norm", "samples": None})


@torch.no_grad()
def record_projection_errors(records, layer, expert, projection, weight, base,
                             residual, module, train, heldout, max_samples):
    # The residual is computed after the BF16 base has been finalized.
    add_norm_ratio(records, "residual_frobenius_ratio", layer, expert, projection,
                   residual, weight)
    add_record(records, "decomposition_relative_l2_error", layer, expert,
               projection, "weight", base.float() + residual, weight.float())
    for split, sample in (("train", train), ("heldout", heldout)):
        for metric in ("qjl_residual_relative_l2_error",
                       "full_projection_relative_l2_error"):
            if not sample or expert not in sample or max_samples == 0:
                add_missing(records, metric, layer, expert, projection, split,
                            "no_routed_samples" if max_samples else "diagnostics_disabled")
        if not sample or expert not in sample or max_samples == 0:
            continue
        x = sample[expert][0][:max_samples].to(weight.device)
        if not x.numel():
            for metric in ("qjl_residual_relative_l2_error",
                           "full_projection_relative_l2_error"):
                add_missing(records, metric, layer, expert, projection, split,
                            "empty_routed_samples")
            continue
        true_residual = F.linear(x.float(), residual)
        if isinstance(module, PackedResidual):
            p = module.bank.get(module.input_dim, x.device)
            z = x.to(p.dtype) @ p
            estimated_residual = module.forward_projected(z, torch.float32)
        else:
            estimated_residual = module(x).float()
        add_record(records, "qjl_residual_relative_l2_error", layer, expert,
                   projection, split, estimated_residual, true_residual, x.shape[0])
        # Follow the actual BF16 forward arithmetic for the full projection.
        estimated_full = F.linear(x, base) + estimated_residual.to(x.dtype)
        original_full = F.linear(x, weight)
        add_record(records, "full_projection_relative_l2_error", layer, expert,
                   projection, split, estimated_full, original_full, x.shape[0])


@torch.no_grad()
def record_expert_error(records, layer, expert_id, original, cluster, train,
                        heldout, max_samples):
    for split, source in (("train", train), ("heldout", heldout)):
        sample = source.get("gate", {}).get(expert_id)
        if sample is None or max_samples == 0:
            add_missing(records, "expert_output_relative_l2_error", layer,
                        expert_id, "all", split,
                        "no_routed_samples" if max_samples else "diagnostics_disabled")
            continue
        x = sample[0][:max_samples].to(next(original.parameters()).device)
        if not x.numel():
            add_missing(records, "expert_output_relative_l2_error", layer,
                        expert_id, "all", split, "empty_routed_samples")
            continue
        add_record(records, "expert_output_relative_l2_error", layer, expert_id,
                   "all", split, cluster.forward_expert(expert_id, x),
                   original(x), x.shape[0])


def moe_probe_inputs(layer_samples, max_samples):
    if max_samples <= 0:
        return None
    gates = layer_samples.get("gate", {})
    inputs = [gates[i][0][:1] for i in sorted(gates) if gates[i][0].numel()]
    if not inputs:
        return None
    return torch.cat(inputs[:max_samples], dim=0).unsqueeze(0)


def record_moe_error(records, layer, split, estimated, original):
    add_record(records, "moe_output_relative_l2_error", layer, None, "all",
               split, estimated, original.to(estimated.device), estimated.shape[-2])


def summarize_records(records):
    """Aggregate all valid values; retain null counts and reasons."""
    buckets = {}
    for row in records:
        metric = row["metric"]
        for key in (metric, f"{metric}/{row['split']}",
                    f"{metric}/{row['split']}/{row['projection']}",
                    f"{metric}/layer_{row['layer']}",
                    f"{metric}/layer_{row['layer']}/{row['split']}/{row['projection']}"):
            bucket = buckets.setdefault(key, {"values": [], "null_reasons": {}})
            if row["value"] is None:
                reason = row["null_reason"] or "unknown"
                bucket["null_reasons"][reason] = bucket["null_reasons"].get(reason, 0) + 1
            else:
                bucket["values"].append(row["value"])
    summary = {}
    for key, bucket in buckets.items():
        values = sorted(bucket["values"])
        n = len(values)
        summary[key] = {
            "count": n, "null_count": sum(bucket["null_reasons"].values()),
            "null_reasons": bucket["null_reasons"],
            "mean": statistics.fmean(values) if n else None,
            "median": statistics.median(values) if n else None,
            "p90": values[math.ceil(0.9 * n) - 1] if n else None,
            "max": values[-1] if n else None,
        }
    return summary
