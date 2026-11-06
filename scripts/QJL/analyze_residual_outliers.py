#!/usr/bin/env python3
"""Measure original Qwen MoE expert residuals from ResCode saliency bases.

This standalone experiment uses cached ResCode groups and saliency, original
BF16 expert weights, and BF16-rounded shared bases. It never quantizes weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUPS = ROOT / "results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt"
DEFAULT_OUTPUT = ROOT / "results/qjl_outliers/qwen/saliency"
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_REVISION = "ec052fda178e241c7c443468d2fa1db6618996be"
DEFAULT_MODEL_PATH = (Path.home() / ".cache/huggingface/hub"
                      / "models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
                      / "snapshots" / MODEL_REVISION)
PROJECTIONS = ("gate", "up", "down")
KS = (1, 4, 8, 16, 32)
sys.path.insert(0, str(ROOT))

from rescode.merging.expert_clusters import groups_to_labels
from rescode.qjl.base import _initial_base  # Pure ResCode saliency initializer; no QJL projection.


def parse_layers(text: str, count: int) -> list[int]:
    if text == "all":
        return list(range(count))
    try:
        layers = sorted({int(part.strip()) for part in text.split(",")})
    except ValueError as exc:
        raise ValueError("--layers must be 'all' or comma-separated layer indices") from exc
    if not layers or layers[0] < 0 or layers[-1] >= count:
        raise ValueError(f"--layers must be within 0..{count - 1}")
    return layers


def stable_rank(scores: np.ndarray) -> np.ndarray:
    """Descending score order, with original channel index breaking ties."""
    return np.argsort(-scores, kind="stable")


def magnitude_and_channels(matrix: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("expected a finite FP32 [out_features, in_features] matrix")
    absolute = np.abs(matrix)
    quantiles = np.quantile(absolute, [0.5, 0.95, 0.99, 0.999])
    squared = np.square(matrix, dtype=np.float64)
    energy = float(squared.sum(dtype=np.float64))
    mean_abs = absolute.mean(axis=0, dtype=np.float64)
    column_energy = squared.sum(axis=0, dtype=np.float64)
    row_energy = squared.sum(axis=1, dtype=np.float64)
    l2 = np.sqrt(column_energy)
    median_mean = float(np.median(mean_abs))
    median_l2 = float(np.median(l2))
    p99 = float(quantiles[2])
    rms = math.sqrt(energy / matrix.size)
    maximum = float(absolute.max())
    stats = {
        "abs_median": float(quantiles[0]), "abs_p95": float(quantiles[1]),
        "abs_p99": p99, "abs_p999": float(quantiles[3]), "abs_max": maximum,
        "rms": rms, "max_over_p99": maximum / p99 if p99 else math.nan,
        "max_over_rms": maximum / rms if rms else math.nan,
        "top_mean_abs_over_median_channel": float(mean_abs.max() / median_mean) if median_mean else math.nan,
        "top_l2_over_median_channel": float(l2.max() / median_l2) if median_l2 else math.nan,
        "frobenius_norm": math.sqrt(energy), "squared_frobenius_norm": energy,
    }
    return stats, mean_abs, l2, row_energy


def norm_reduction(row_energy: np.ndarray, column_energy: np.ndarray,
                   matrix: np.ndarray, selected: np.ndarray) -> dict:
    """Compute exact full-matrix removal without modifying the source matrix."""
    selected = np.asarray(selected, dtype=np.int64)
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
        raise ValueError("selected channels must be unique")
    if len(selected) and (selected.min() < 0 or selected.max() >= matrix.shape[1]):
        raise ValueError("selected channel outside matrix")
    original_energy = float(row_energy.sum(dtype=np.float64))
    if len(selected) == matrix.shape[1]:
        remaining_row = np.zeros_like(row_energy)
    elif len(selected) == 0:
        remaining_row = row_energy.copy()
    else:
        removed_row = np.square(matrix[:, selected], dtype=np.float64).sum(axis=1, dtype=np.float64)
        remaining_row = np.maximum(row_energy - removed_row, 0.0)
    remaining_energy = float(remaining_row.sum(dtype=np.float64))
    removed_energy = original_energy - remaining_energy
    if not np.isclose(original_energy, removed_energy + remaining_energy, rtol=1e-12, atol=1e-12):
        raise AssertionError("energy identity failed")
    if len(selected) == 0 and remaining_energy != original_energy:
        raise AssertionError("zeroing no channels changed the norm")
    if len(selected) == matrix.shape[1] and remaining_energy != 0:
        raise AssertionError("zeroing all channels did not produce zero norm")
    original_row_norm = np.sqrt(row_energy)
    remaining_row_norm = np.sqrt(remaining_row)
    relative_row_reduction = np.divide(
        original_row_norm - remaining_row_norm, original_row_norm,
        out=np.zeros_like(original_row_norm), where=original_row_norm > 0,
    )
    row_q = np.quantile(relative_row_reduction, [0.5, 0.95, 0.99])
    # Selected column scores independently check the energy calculation.
    score_removed = float(column_energy[selected].sum(dtype=np.float64))
    if not np.isclose(removed_energy, score_removed, rtol=2e-7, atol=1e-9):
        raise AssertionError("selected-column energy disagrees with row accounting")
    return {
        "remaining_frobenius_norm": math.sqrt(remaining_energy),
        "remaining_over_original": math.sqrt(remaining_energy / original_energy) if original_energy else (1.0 if not len(selected) else 0.0),
        "removed_squared_norm_pct": 100 * removed_energy / original_energy if original_energy else 0.0,
        "row_l2_reduction_mean": float(relative_row_reduction.mean()),
        "row_l2_reduction_median": float(row_q[0]),
        "row_l2_reduction_p95": float(row_q[1]),
        "row_l2_reduction_p99": float(row_q[2]),
        "row_l2_reduction_max": float(relative_row_reduction.max()),
    }


def analyze_matrix(matrix: np.ndarray, seed_parts: tuple[int, ...]) -> tuple[dict, list[dict], np.ndarray, np.ndarray, dict]:
    stats, mean_abs, l2, row_energy = magnitude_and_channels(matrix)
    column_energy = np.square(matrix, dtype=np.float64).sum(axis=0, dtype=np.float64)
    order_l2 = stable_rank(l2)
    order_abs = stable_rank(mean_abs)
    rank_l2 = np.empty(matrix.shape[1], dtype=np.int32)
    rank_abs = np.empty_like(rank_l2)
    rank_l2[order_l2] = np.arange(1, matrix.shape[1] + 1)
    rank_abs[order_abs] = np.arange(1, matrix.shape[1] + 1)
    channel = {"mean_abs": mean_abs, "l2": l2, "rank_mean_abs": rank_abs,
               "rank_l2": rank_l2}
    checks = norm_reduction(row_energy, column_energy, matrix, np.array([], dtype=np.int64))
    assert checks["remaining_over_original"] == 1.0
    checks = norm_reduction(row_energy, column_energy, matrix, np.arange(matrix.shape[1]))
    assert checks["remaining_frobenius_norm"] == 0.0
    rows = []
    for label, k in [(str(k), k) for k in KS] + [("top_1pct", math.ceil(matrix.shape[1] * 0.01))]:
        k = min(k, matrix.shape[1])
        for control, chosen in (
            ("top_l2", order_l2[:k]),
            ("random", np.random.default_rng(np.random.SeedSequence([*seed_parts, k]))
             .choice(matrix.shape[1], size=k, replace=False)),
        ):
            rows.append({"k_label": label, "k": k, "selection": control,
                         **stats, **norm_reduction(row_energy, column_energy, matrix, chosen)})
    return stats, rows, order_l2, order_abs, channel


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    left, right = set(a.tolist()), set(b.tolist())
    return len(left & right) / len(left | right) if left or right else 1.0


def validate_group_cache(saved: dict, model_config: dict, index: dict,
                         calibration_path: Path) -> tuple[dict[int, list[list[int]]], dict[int, torch.Tensor], int]:
    """Validate a cache produced by rescode.merging.expert_clusters.form_clusters."""
    key = saved.get("key", {})
    expected = {"model_name": MODEL_ID, "num_groups": 30, "singleton_ratio": 0.15,
                "linkage": "average", "budget_aware": "true", "calib_seed": 42,
                "n_sentences": 2400}
    for name, value in expected.items():
        if key.get(name) != value:
            raise ValueError(f"group-cache {name}={key.get(name)!r}; expected {value!r}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Wikipedia calibration data missing: {calibration_path}")
    if key.get("train_data_bytes") != calibration_path.stat().st_size:
        raise ValueError("cached grouping was built from different calibration-data size")
    if calibration_path.name != "wikipedia-calib.jsonl":
        raise ValueError("expected Wikipedia calibration JSONL")
    groups = {int(k): [[int(i) for i in group] for group in v]
              for k, v in saved["groups"].items()}
    saliency = {int(k): v.float() for k, v in saved["saliency"].items()}
    layer_count = int(model_config["num_hidden_layers"])
    expert_count = int(model_config["num_experts"])
    if layer_count != 24 or len(groups) != layer_count or len(saliency) != layer_count:
        raise ValueError("expected cached groups and saliency for all 24 Qwen layers")
    weight_map = index["weight_map"]
    for layer, partition in sorted(groups.items()):
        if sorted(i for group in partition for i in group) != list(range(expert_count)):
            raise ValueError(f"layer {layer} groups do not partition {expert_count} experts")
        if not 1 <= len(partition) <= 30:
            raise ValueError(f"layer {layer} has an invalid group count: {len(partition)}")
        if saliency[layer].numel() != expert_count or not torch.isfinite(saliency[layer]).all() or (saliency[layer] < 0).any():
            raise ValueError(f"invalid saliency vector for layer {layer}")
        labels = groups_to_labels(partition, expert_count)
        for cid, members in enumerate(partition):
            if not all(int(labels[i]) == cid for i in members):
                raise AssertionError("ResCode group-to-label conversion disagrees with cache")
        for expert in range(expert_count):
            for projection in PROJECTIONS:
                if weight_name(layer, expert, projection) not in weight_map:
                    raise ValueError(f"original model lacks {weight_name(layer, expert, projection)}")
    return groups, saliency, layer_count


def saliency_base(weights: dict[int, torch.Tensor], saliency: torch.Tensor,
                  delta: float = 0.001) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the repository's pure saliency initializer, then round like ResCode BF16 merge."""
    members = list(weights)
    raw = torch.tensor([float(saliency[i]) + delta for i in members], dtype=torch.float32)
    coefficients = raw / raw.sum()
    if not torch.isclose(coefficients.sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-6):
        raise AssertionError("saliency averaging coefficients do not sum to one")
    base = _initial_base(weights, "saliency", saliency, delta).to(torch.bfloat16)
    if base.dtype != torch.bfloat16:
        raise AssertionError("shared base must be BF16")
    return base, coefficients


def weight_name(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"


def plot_representative(matrix: np.ndarray, w: np.ndarray, order_abs: np.ndarray,
                        output: Path, label: str, seed_parts: tuple[int, ...], max_rows: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    rows = np.sort(rng.choice(matrix.shape[0], min(max_rows, matrix.shape[0]), replace=False))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].imshow(np.log1p(np.abs(matrix[rows][:, order_abs])), aspect="auto", interpolation="nearest")
    axes[0, 0].set(title="|Residual|, channels sorted by mean |R|", xlabel="Sorted input channel", ylabel="Sampled output row (not token)")
    axes[0, 1].hist(np.abs(matrix).ravel(), bins=100, histtype="step", density=True, label="Residual")
    axes[0, 1].hist(np.abs(w).ravel(), bins=100, histtype="step", density=True, label="Original W")
    axes[0, 1].set(xscale="symlog", yscale="log", title="Absolute magnitude distribution", xlabel="Magnitude")
    axes[0, 1].legend()
    mean_abs = np.abs(matrix).mean(axis=0)
    l2 = np.sqrt(np.square(matrix, dtype=np.float64).sum(axis=0))
    axes[1, 0].plot(mean_abs[order_abs], linewidth=0.7)
    axes[1, 0].set(title="Per-input-channel mean absolute score", xlabel="Sorted input channel", ylabel="Mean |R|")
    axes[1, 1].plot(l2[order_abs], linewidth=0.7)
    axes[1, 1].set(title="Per-input-channel L2 score", xlabel="Sorted input channel", ylabel="Column L2")
    fig.suptitle(label + " | output rows replace QJL Figure 2 tokens")
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_aggregates(per_expert: pd.DataFrame, overlap: pd.DataFrame,
                    cross_layer: pd.DataFrame, figure_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for projection in PROJECTIONS:
        data = per_expert[(per_expert.projection == projection) & (per_expert.matrix_kind == "residual")]
        fig, ax = plt.subplots(figsize=(9, 5))
        for selection in ("top_l2", "random"):
            subset = data[data.selection == selection]
            curve = subset.groupby("k")["removed_squared_norm_pct"].median().sort_index()
            ax.plot(curve.index, curve.values, marker="o", label=selection)
        ax.set(title=f"{projection}: median squared Frobenius energy removed", xlabel="Input channels removed", ylabel="Energy removed (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"norm_reduction_{projection}.png", dpi=140)
        plt.close(fig)
        sub = overlap[overlap.projection == projection]
        fig, ax = plt.subplots(figsize=(9, 5))
        for k_label in [*(str(k) for k in KS), "top_1pct"]:
            series = sub[sub.k_label == k_label].groupby("layer")["jaccard"].mean()
            ax.plot(series.index, series.values, marker="o", label=k_label)
        ax.set(title=f"{projection}: top-L2 channel overlap within groups", xlabel="Layer", ylabel="Mean pairwise Jaccard", ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"cross_expert_overlap_{projection}.png", dpi=140)
        plt.close(fig)
    if not cross_layer.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for projection in PROJECTIONS:
            data = cross_layer[cross_layer.projection == projection]
            ax.plot(data.layer_b, data.jaccard, marker="o", label=projection)
        ax.set(title="Top-32 pooled normalized-L2 input channels across layers", xlabel="Later analyzed layer",
               ylabel="Jaccard with preceding analyzed layer", ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "cross_layer_overlap.png", dpi=140)
        plt.close(fig)


def write_report(path: Path, per_expert: pd.DataFrame, reductions: pd.DataFrame,
                 overlap: pd.DataFrame, cross_layer: pd.DataFrame,
                 selected_layers: list[int], max_reconstruction_error: float,
                 groups_path: Path, model_path: Path, seed: int):
    top = reductions[(reductions.k_label == "top_1pct") & (reductions.selection == "top_l2")]
    scope = "full 24-layer analysis" if len(selected_layers) == 24 else "selected-layer smoke analysis"
    lines = ["# Saliency residual outliers before QJL", "",
             f"Scope: **{scope}**; layers {', '.join(map(str, selected_layers))}.",
             f"Groups and saliency: `{groups_path}`. Original BF16 model: `{model_path}`.",
             f"R = FP32(W) − FP32(BF16 ResCode saliency-weighted base), delta=0.001. Seed: {seed}.",
             "No QJL projection, quantization, packing, or compressed checkpoint was used.",
             "Heatmap rows are output rows, replacing tokens in QJL Figure 2.", "",
             "## Outlier concentration", ""]
    for projection in PROJECTIONS:
        r = top[(top.projection == projection) & (top.matrix_kind == "residual")]
        w = top[(top.projection == projection) & (top.matrix_kind == "original_weight")]
        random = reductions[(reductions.projection == projection) &
                            (reductions.matrix_kind == "residual") &
                            (reductions.k_label == "top_1pct") &
                            (reductions.selection == "random")]
        channels = per_expert[(per_expert.projection == projection) &
                              (per_expert.matrix_kind == "residual")]
        r_pct, random_pct, w_pct = (float(x.removed_squared_norm_pct.median()) for x in (r, random, w))
        concentrated = "concentrated" if r_pct > 2 * random_pct else "not strongly concentrated relative to random"
        lines.append(
            f"- **{projection}:** Top 1% of input channels remove median {r_pct:.2f}% of residual squared norm "
            f"versus {random_pct:.2f}% for random channels and {w_pct:.2f}% for original W. "
            f"Top/median channel L2 is {channels.top_l2_over_median_channel.median():.2f}×; "
            f"by the 2× random-control criterion, the residual is {concentrated}."
        )
    lines += ["", "## Layer and channel patterns", "",
              "Median top-1% squared energy removed from residuals (%):", "",
              "| Layer | Gate | Up | Down |", "|---:|---:|---:|---:|"]
    layer_table = top[top.matrix_kind == "residual"].groupby(["layer", "projection"])["removed_squared_norm_pct"].median().unstack()
    for layer, row in layer_table.iterrows():
        lines.append(f"| {layer} | {row['gate']:.2f} | {row['up']:.2f} | {row['down']:.2f} |")
    lines += ["", f"Within-group mean top-32 Jaccard: {overlap[overlap.k_label == '32'].jaccard.mean():.3f}."]
    if not cross_layer.empty:
        persistence = []
        for projection in PROJECTIONS:
            observed = float(cross_layer[cross_layer.projection == projection].jaccard.mean())
            dimension = 1408 if projection == "down" else 2048
            random_expectation = 32 / (2 * dimension - 32)
            persistence.append(f"{projection} {observed:.3f} (random-set reference {random_expectation:.3f})")
        lines.append("Cross-layer top-32 Jaccard from pooled median-normalized expert channel L2 (successive analyzed layers): " +
                     ", ".join(persistence) + ".")
    lines += ["", "## Validation and limitations", "",
              "- Reused the ResCode `form_clusters` cache and its per-layer saliency; checked configuration, partitions, labels, and tensor shapes.",
              "- The shared base uses the repository's saliency initializer; every coefficient vector sums to one before BF16 rounding.",
              f"- Maximum relative reconstruction error for W = S + R: {max_reconstruction_error:.3g}.",
              "- Verified squared norm decomposition, zero-all and zero-none masking for every matrix; fixed seeds govern random controls and plotted row samples.",
              "- Full-matrix statistics are unsampled. The experiment measures expert weight matrices, whereas QJL Figure 2 measures KV-cache key embeddings.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-path", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--calibration-path", type=Path, default=ROOT / "rescode/data/wikipedia-calib.jsonl")
    parser.add_argument("--source-config", type=Path, default=ROOT / "results/qjl_direct/qwen/test/config.json",
                        help="Optional experiment metadata for checking calibration length")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Existing local HF snapshot (default: original /root/.cache cache); never downloads")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--figure-rows", type=int, default=256)
    args = parser.parse_args()
    if not args.groups_path.is_file():
        parser.error(f"ResCode grouping/saliency cache missing: {args.groups_path}. Generate compatible calibration and grouping first; no compressed checkpoint is required.")
    if args.figure_rows < 1 or args.seed < 0:
        parser.error("--figure-rows must be positive and --seed nonnegative")
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "config.json").is_file() or not (model_path / "model.safetensors.index.json").is_file():
        parser.error(f"complete local Qwen snapshot not found at {model_path}; pass --model-path. This script does not download model files.")
    model_config = json.loads((model_path / "config.json").read_text())
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    missing_shards = sorted(shard for shard in set(index["weight_map"].values())
                            if not (model_path / shard).is_file())
    if missing_shards:
        parser.error(f"local model cache is incomplete at {model_path}; missing shards: {', '.join(missing_shards)}")
    print(f"[outliers] using existing local model: {model_path}", flush=True)
    saved = torch.load(args.groups_path, map_location="cpu", weights_only=False)
    groups, saliency, layer_count = validate_group_cache(saved, model_config, index, args.calibration_path)
    if args.source_config.is_file():
        source = json.loads(args.source_config.read_text())
        for key, value in {"model_name": MODEL_ID, "num_groups": 30, "singleton_ratio": 0.15,
                           "linkage": "average", "budget_aware": "true", "calib_seed": 42,
                           "n_sentences": 2400, "calib_seqlen": 2048}.items():
            if source.get(key) != value:
                raise ValueError(f"source config {key}={source.get(key)!r}; expected {value!r}")
    layers = parse_layers(args.layers, layer_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    before_stat = args.groups_path.stat()
    expert_rows: list[dict] = []
    reduction_rows: list[dict] = []
    overlap_rows: list[dict] = []
    layer_scores: dict[tuple[int, str], np.ndarray] = {}
    max_reconstruction_error = 0.0
    channel_path = args.output_dir / "channel_scores.csv"
    with channel_path.open("w", newline="") as channel_file:
        writer = csv.DictWriter(channel_file, fieldnames=["layer", "group", "expert", "projection", "channel",
                                                          "mean_abs", "l2", "rank_mean_abs", "rank_l2"])
        writer.writeheader()
        for layer in layers:
            print(f"[outliers] layer {layer}", flush=True)
            plotted = set()
            with ExitStack() as stack:
                shard_handles = {}
                def original_tensor(expert: int, projection: str) -> torch.Tensor:
                    name = weight_name(layer, expert, projection)
                    shard = index["weight_map"][name]
                    if shard not in shard_handles:
                        shard_handles[shard] = stack.enter_context(safe_open(str(model_path / shard), framework="pt", device="cpu"))
                    return shard_handles[shard].get_tensor(name)

                for cid, members in enumerate(groups[layer]):
                    if len(members) == 1:
                        continue
                    for pi, projection in enumerate(PROJECTIONS):
                        weights = {expert: original_tensor(expert, projection) for expert in members}
                        shape = (model_config["moe_intermediate_size"], model_config["hidden_size"]) if projection != "down" else (model_config["hidden_size"], model_config["moe_intermediate_size"])
                        if any(t.dtype != torch.bfloat16 or tuple(t.shape) != shape for t in weights.values()):
                            raise ValueError(f"original BF16 tensor shape/dtype mismatch: layer {layer}, group {cid}, {projection}")
                        base, coefficients = saliency_base(weights, saliency[layer])
                        if not torch.isclose(coefficients.sum(), torch.tensor(1.0), atol=1e-6):
                            raise AssertionError("averaging coefficients do not sum to one")
                        rank_sets: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
                        for expert, weight in weights.items():
                            w = weight.float().numpy()
                            residual = (weight.float() - base.float()).numpy()
                            restored = base.float().numpy() + residual
                            error = float(np.max(np.abs(restored - w)) / max(np.max(np.abs(w)), 1e-12))
                            max_reconstruction_error = max(max_reconstruction_error, error)
                            if not np.allclose(restored, w, rtol=1e-6, atol=1e-6):
                                raise AssertionError(f"residual reconstruction failed: {layer}/{cid}/{expert}/{projection}")
                            for kind, matrix in (("residual", residual), ("original_weight", w)):
                                stats, reductions, order_l2, order_abs, channel = analyze_matrix(
                                    matrix, (args.seed, layer, expert, pi, 0 if kind == "residual" else 1))
                                common = {"layer": layer, "group": cid, "expert": expert,
                                          "projection": projection, "matrix_kind": kind,
                                          "out_features": matrix.shape[0], "in_features": matrix.shape[1]}
                                expert_rows.append({**common, "coefficient_sum": float(coefficients.sum()),
                                                    "reconstruction_relative_error": error if kind == "residual" else "", **stats})
                                reduction_rows.extend({**common, **row} for row in reductions)
                                if kind == "residual":
                                    for k in KS:
                                        rank_sets[str(k)].append((expert, order_l2[:min(k, matrix.shape[1])].copy()))
                                    rank_sets["top_1pct"].append((expert, order_l2[:math.ceil(0.01 * matrix.shape[1])].copy()))
                                    vote_key = (layer, projection)
                                    if vote_key not in layer_scores:
                                        layer_scores[vote_key] = np.zeros(matrix.shape[1], dtype=np.float64)
                                    # Normalize each expert's column L2 scores before pooling,
                                    # so a high-norm expert cannot determine the layer ranking.
                                    layer_scores[vote_key] += channel["l2"] / max(float(np.median(channel["l2"])), 1e-12)
                                    for channel_idx in range(matrix.shape[1]):
                                        writer.writerow({"layer": layer, "group": cid, "expert": expert,
                                                         "projection": projection, "channel": channel_idx,
                                                         "mean_abs": channel["mean_abs"][channel_idx],
                                                         "l2": channel["l2"][channel_idx],
                                                         "rank_mean_abs": channel["rank_mean_abs"][channel_idx],
                                                         "rank_l2": channel["rank_l2"][channel_idx]})
                                    if projection not in plotted:
                                        plot_representative(matrix, w, order_abs,
                                            figure_dir / f"heatmap_layer{layer:02d}_{projection}.png",
                                            f"Layer {layer}, group {cid}, expert {expert}, {projection}",
                                            (args.seed, layer, expert, pi), args.figure_rows)
                                        plotted.add(projection)
                        for k_label, entries in rank_sets.items():
                            for ai, (expert_a, indices_a) in enumerate(entries):
                                for expert_b, indices_b in entries[ai + 1:]:
                                    overlap_rows.append({"layer": layer, "group": cid, "projection": projection,
                                                         "k_label": k_label, "expert_a": expert_a, "expert_b": expert_b,
                                                         "jaccard": jaccard(indices_a, indices_b)})
            channel_file.flush()
    after_stat = args.groups_path.stat()
    if (after_stat.st_size, after_stat.st_mtime_ns) != (before_stat.st_size, before_stat.st_mtime_ns):
        raise AssertionError("group cache changed during read-only analysis")
    per_expert = pd.DataFrame(expert_rows)
    reductions = pd.DataFrame(reduction_rows)
    overlap = pd.DataFrame(overlap_rows)
    per_expert.to_csv(args.output_dir / "per_expert.csv", index=False, float_format="%.10g")
    reductions.to_csv(args.output_dir / "norm_reduction.csv", index=False, float_format="%.10g")
    overlap.to_csv(args.output_dir / "overlap.csv", index=False, float_format="%.10g")
    cross_layer_rows = []
    for projection in PROJECTIONS:
        for first, second in zip(layers, layers[1:]):
            a = stable_rank(layer_scores[(first, projection)])[:32]
            b = stable_rank(layer_scores[(second, projection)])[:32]
            cross_layer_rows.append({"layer_a": first, "layer_b": second, "projection": projection,
                                     "k": 32, "ranking": "sum_expert_median_normalized_l2",
                                     "jaccard": jaccard(a, b)})
    cross_layer = pd.DataFrame(cross_layer_rows)
    cross_layer.to_csv(args.output_dir / "cross_layer_overlap.csv", index=False)
    magnitude_columns = ["abs_median", "abs_p95", "abs_p99", "abs_p999", "abs_max", "rms",
                         "max_over_p99", "max_over_rms", "top_mean_abs_over_median_channel",
                         "top_l2_over_median_channel", "frobenius_norm"]
    reduction_columns = ["remaining_over_original", "removed_squared_norm_pct", "row_l2_reduction_mean",
                         "row_l2_reduction_median", "row_l2_reduction_p95", "row_l2_reduction_p99",
                         "row_l2_reduction_max"]
    summary = reductions.groupby(["layer", "projection", "matrix_kind", "k_label", "k", "selection"], sort=True)[magnitude_columns + reduction_columns].median().reset_index()
    all_layers = reductions.copy()
    all_layers["layer"] = "all"
    summary = pd.concat([summary, all_layers.groupby(["layer", "projection", "matrix_kind", "k_label", "k", "selection"], sort=True)[magnitude_columns + reduction_columns].median().reset_index()], ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.10g")
    plot_aggregates(reductions, overlap, cross_layer, figure_dir)
    write_report(args.output_dir / "REPORT.md", per_expert, reductions, overlap, cross_layer,
                 layers, max_reconstruction_error, args.groups_path, model_path, args.seed)
    print(f"[outliers] analyzed {len(layers)} layers; {len(per_expert)} expert/projection/matrix rows; maximum reconstruction error {max_reconstruction_error:g}")
    print(f"[outliers] outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
