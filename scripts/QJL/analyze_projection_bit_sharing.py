#!/usr/bin/env python3
"""Diagnose Gate/Up/Down directions and Gate-Up QJL sign-bit sharing.

Original Qwen BF16 weights are read from safetensors. Non-singleton residuals
use the existing ResCode group cache and BF16 saliency-weighted shared bases.
The experiment never changes weights or runs model inference/evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rescode.qjl.core import ProjectionBank, pack_residual

import analyze_residual_outliers as previous
from bit_sharing_analysis import (aligned_neurons, bit_statistics, cosine_statistics,
                                  decode_sparse_correction, derangement,
                                  encode_sparse_correction, storage_statistics)

DEFAULT_OUTPUT = ROOT / "results/qjl_bit_sharing/qwen"
PROJECTIONS = ("gate", "up", "down")
PAIRS = ("gate_up", "gate_down", "up_down")


def parse_dimensions(text: str) -> list[int]:
    try:
        dimensions = sorted({int(part.strip()) for part in text.split(",")})
    except ValueError as exc:
        raise ValueError("--qjl-d must be comma-separated positive integers") from exc
    if not dimensions or dimensions[0] < 1 or dimensions[-1] > 65535:
        raise ValueError("--qjl-d must be in 1..65535 for uint16 correction positions")
    return dimensions


def local_model(args) -> Path:
    if args.model_path is not None:
        if args.download_missing:
            raise ValueError("--download-missing uses the standard Hugging Face cache; omit --model-path")
        path = args.model_path.expanduser().resolve()
    elif args.download_missing:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(previous.MODEL_ID, "model.safetensors.index.json",
                                    revision=previous.MODEL_REVISION)).parent
        hf_hub_download(previous.MODEL_ID, "config.json", revision=previous.MODEL_REVISION)
    else:
        path = previous.DEFAULT_MODEL_PATH
    if not (path / "config.json").is_file() or not (path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Qwen config/index missing at {path}; pass --download-missing or --model-path")
    return path


def validate_saved_groups(saved: dict, config: dict, index: dict) -> tuple[dict, dict]:
    """Apply the existing experiment's cache checks without needing raw calib text."""
    expected = {"model_name": previous.MODEL_ID, "num_groups": 30,
                "singleton_ratio": 0.15, "linkage": "average",
                "budget_aware": "true", "calib_seed": 42, "n_sentences": 2400}
    for name, value in expected.items():
        if saved.get("key", {}).get(name) != value:
            raise ValueError(f"group cache {name} does not match ResCode experiment")
    if not isinstance(saved["key"].get("train_data_bytes"), int):
        raise ValueError("group cache lacks calibration-data size metadata")
    layer_count, expert_count = int(config["num_hidden_layers"]), int(config["num_experts"])
    if layer_count != 24 or expert_count != 60:
        raise ValueError("expected 24 Qwen MoE layers with 60 experts each")
    groups = {int(k): [[int(i) for i in group] for group in v]
              for k, v in saved["groups"].items()}
    saliency = {int(k): v.float() for k, v in saved["saliency"].items()}
    if sorted(groups) != list(range(layer_count)) or sorted(saliency) != list(range(layer_count)):
        raise ValueError("cached groups/saliency must cover all 24 layers")
    weight_map = index["weight_map"]
    for layer in range(layer_count):
        partition = groups[layer]
        if len(partition) > 30 or sorted(i for group in partition for i in group) != list(range(expert_count)):
            raise ValueError(f"layer {layer}: groups do not partition 60 experts")
        labels = previous.groups_to_labels(partition, expert_count)
        if any(int(labels[i]) != cid for cid, members in enumerate(partition) for i in members):
            raise AssertionError("ResCode group labels disagree with saved groups")
        scores = saliency[layer]
        if scores.numel() != expert_count or not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError(f"layer {layer}: invalid saliency")
        for expert in range(expert_count):
            for projection in PROJECTIONS:
                if previous.weight_name(layer, expert, projection) not in weight_map:
                    raise ValueError(f"missing original weight: {layer}/{expert}/{projection}")
    return groups, saliency


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def analyze_expert(layer: int, group: int, expert: int, singleton: bool,
                   weights: dict[str, torch.Tensor], bases: dict[str, torch.Tensor] | None,
                   dimensions: list[int], banks: dict[int, ProjectionBank],
                   device: torch.device, seed: int, row_chunk: int) -> tuple[list[dict], list[dict], list[dict], float]:
    original = {p: weights[p].float().numpy() for p in PROJECTIONS}
    matrices = [("original_weight", original)]
    error = 0.0
    if not singleton:
        assert bases is not None
        residual = {p: (weights[p].float() - bases[p].float()).numpy() for p in PROJECTIONS}
        for p in PROJECTIONS:
            restored = bases[p].float().numpy() + residual[p]
            error = max(error, float(np.max(np.abs(restored - original[p])) /
                                     max(np.max(np.abs(original[p])), 1e-12)))
            if not np.allclose(restored, original[p], rtol=1e-6, atol=1e-6):
                raise AssertionError(f"W=S+R failed: {layer}/{group}/{expert}/{p}")
        matrices.append(("residual", residual))
    common = {"layer": layer, "group": group, "expert": expert,
              "singleton": singleton}
    cosine_rows, bit_rows, storage_rows = [], [], []
    for ki, (kind, matrix) in enumerate(matrices):
        pairs = aligned_neurons(matrix["gate"], matrix["up"], matrix["down"])
        permutation = derangement(matrix["gate"].shape[0], (seed, layer, expert, ki))
        gate_up_cosine = None
        for pair, (a, b) in pairs.items():
            for baseline, right in (("aligned", b), ("shuffled_neurons", b[permutation])):
                stats, values = cosine_statistics(a, right)
                cosine_rows.append({**common, "weight_kind": kind, "projection_pair": pair,
                                    "baseline": baseline, **stats})
                if pair == "gate_up" and baseline == "aligned":
                    gate_up_cosine = values
        assert gate_up_cosine is not None
        for d in dimensions:
            bank = banks[d]
            packed = {}
            for p in ("gate", "up"):
                tensor = torch.from_numpy(matrix[p]).to(device)
                bits, norms = pack_residual(tensor, bank, row_chunk=row_chunk)
                packed[p] = bits.cpu().numpy()
                # QJL stores one FP32 norm per row, even if bits are shared.
                if norms.dtype != torch.float32 or len(norms) != matrix[p].shape[0]:
                    raise AssertionError("QJL row-norm format changed")
                del tensor, bits, norms
            stats, mismatch, xor = bit_statistics(
                packed["gate"], packed["up"], d, gate_up_cosine, permutation,
                (seed, layer, expert, ki, d, 97))
            bit_rows.append({**common, "weight_kind": kind,
                             "projection_pair": "gate_up", **stats})
            # Exercise the real sparse encoder and exact reconstruction for
            # deterministic rows; all-row byte counts follow that same layout.
            for row in sorted({0, len(mismatch) // 2, len(mismatch) - 1}):
                invert = bool(mismatch[row] > d - mismatch[row])
                payload = encode_sparse_correction(packed["gate"][row], packed["up"][row], d, invert)
                restored = decode_sparse_correction(packed["gate"][row], payload, d, invert)
                if not np.array_equal(restored, packed["up"][row]):
                    raise AssertionError("lossless XOR correction did not reconstruct Up bits")
                if len(payload) != 2 + 2 * min(int(mismatch[row]), d - int(mismatch[row])):
                    raise AssertionError("sparse payload byte count mismatch")
            storage_rows.append({**common, "weight_kind": kind,
                                 "projection_pair": "gate_up", **storage_statistics(mismatch, d)})
            del packed, xor
    return cosine_rows, bit_rows, storage_rows, error


def make_summary(cosine: pd.DataFrame, bits: pd.DataFrame, storage: pd.DataFrame) -> pd.DataFrame:
    out = []
    specs = (("cosine", cosine, ["signed_cosine_median", "absolute_cosine_median",
                                  "absolute_cosine_mean", "norm_pearson_r",
                                  "abs_cosine_gt_0p8", "abs_cosine_gt_0p9",
                                  "abs_cosine_gt_0p95", "abs_cosine_gt_0p99"]),
             ("qjl_bits", bits, ["direct_agreement_mean", "best_agreement_mean",
                                   "shuffled_agreement_mean", "random_bit_agreement_mean",
                                   "predicted_agreement_mean", "prediction_mae",
                                   "normalized_hamming_median"]),
             ("storage", storage, ["separate_total_bytes", "shared_no_correction_ratio",
                                      "dense_lossless_ratio", "sparse_lossless_ratio",
                                      "selective_lossless_ratio", "maximum_lossless_savings_pct",
                                      "selective_shared_neurons_pct"]))
    for analysis, source, metrics in specs:
        for cohort, data in (("all_experts", source[source.weight_kind == "original_weight"]),
                             ("paired_non_singleton", source[~source.singleton])):
            if cohort == "all_experts":
                data = data.copy()
            for scope in ("layer", "selected_layers"):
                work = data.copy()
                if scope == "selected_layers":
                    work["layer"] = "all_selected"
                keys = ["layer", "weight_kind", "projection_pair"]
                if analysis == "cosine":
                    keys.append("baseline")
                else:
                    keys.append("qjl_d")
                grouped = work.groupby(keys, dropna=False, sort=True)[metrics].median().reset_index()
                grouped.insert(0, "analysis", analysis)
                grouped.insert(1, "cohort", cohort)
                out.append(grouped)
    return pd.concat(out, ignore_index=True)


def make_figures(cosine: pd.DataFrame, bits: pd.DataFrame,
                 storage: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    aligned = cosine[cosine.baseline == "aligned"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, pair in zip(axes, PAIRS):
        for kind in ("original_weight", "residual"):
            values = aligned[(aligned.projection_pair == pair) & (aligned.weight_kind == kind)].absolute_cosine_median
            if len(values):
                ax.hist(values, bins=30, alpha=0.6, label=kind)
        ax.set(title=pair, xlabel="Expert median |cosine|", ylabel="Experts")
        ax.legend()
    fig.savefig(figure_dir / "cosine_distributions.png", dpi=140)
    plt.close(fig)
    for d in sorted(bits.qjl_d.unique()):
        sub = bits[bits.qjl_d == d]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for kind in ("original_weight", "residual"):
            data = sub[sub.weight_kind == kind]
            axes[0].hist(data.direct_agreement_mean, bins=30, alpha=0.6, label=kind)
            curve = data.groupby("layer").direct_agreement_mean.median()
            axes[1].plot(curve.index, curve.values, marker="o", label=kind)
        axes[0].set(title=f"D={d}: Gate-Up direct bit agreement", xlabel="Expert mean agreement", ylabel="Experts")
        axes[1].set(title=f"D={d}: layer trend", xlabel="Layer", ylabel="Median expert agreement")
        for ax in axes:
            ax.legend()
        fig.savefig(figure_dir / f"bit_agreement_d{d}.png", dpi=140)
        plt.close(fig)
        sub = storage[storage.qjl_d == d]
        fig, ax = plt.subplots(figsize=(8, 5))
        for kind in ("original_weight", "residual"):
            data = sub[sub.weight_kind == kind]
            ax.hist(data.maximum_lossless_savings_pct, bins=30, alpha=0.6, label=kind)
        ax.set(title=f"D={d}: maximum selective lossless savings", xlabel="Savings vs separate sketches (%)", ylabel="Experts")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"memory_savings_d{d}.png", dpi=140)
        plt.close(fig)


def write_report(path: Path, cosine: pd.DataFrame, bits: pd.DataFrame,
                 storage: pd.DataFrame, layers: list[int], experts: list[int],
                 dimensions: list[int], max_reconstruction_error: float,
                 model_path: Path, groups_path: Path, seed: int,
                 calibration_present: bool) -> None:
    full = len(layers) == 24 and len(experts) == 60
    scope = "full model" if full else "limited smoke/selected subset"
    lines = ["# QJL Gate–Up bit-sharing diagnostic", "",
             f"Scope: **{scope}**; layers {layers}; expert IDs {experts if len(experts) < 60 else 'all 60'}; D={dimensions}.",
             f"Original BF16 model: `{model_path}`. Cached ResCode groups/saliency: `{groups_path}`.",
             f"Projection seed: {seed}. Residual = FP32(W) − FP32(BF16 saliency base).", "",
             "## Structural direction similarity", "",
             "Down comparisons align Gate/Up rows with Down columns. They describe neuron-vector direction only; Down uses a different QJL input space."]
    for pair in PAIRS:
        data = cosine[(cosine.projection_pair == pair) & (cosine.baseline == "aligned")]
        w = data[(data.weight_kind == "original_weight") & (~data.singleton)]
        r = data[data.weight_kind == "residual"]
        all_w = data[data.weight_kind == "original_weight"]
        lines.append(f"- {pair}: median expert median |cosine| = {all_w.absolute_cosine_median.median():.3f} "
                     f"(all W); paired W {w.absolute_cosine_median.median():.3f}, "
                     f"R {r.absolute_cosine_median.median():.3f}.")
    lines += ["", "## Actual Gate–Up QJL sign bits", "",
              "Signs use the repository's FP16 Gaussian ProjectionBank and pack_residual. Direct and inverted agreement are measured from packed XOR bits."]
    for d in dimensions:
        data = bits[bits.qjl_d == d]
        for kind in ("original_weight", "residual"):
            subset = data[data.weight_kind == kind]
            lines.append(f"- D={d}, {kind}: median expert mean direct agreement "
                         f"{subset.direct_agreement_mean.median():.3f}; best with inversion "
                         f"{subset.best_agreement_mean.median():.3f}; shuffled-neuron "
                         f"{subset.shuffled_agreement_mean.median():.3f}; random-bit "
                         f"{subset.random_bit_agreement_mean.median():.3f}; Gaussian prediction "
                         f"{subset.predicted_agreement_mean.median():.3f}.")
    lines += ["", "## Storage feasibility", "",
              "Bytes include two FP32 row norms, packed sketches, a uint16 count plus uint16 positions per sparse-corrected row, packed inversion/selection flags, and stated 16-byte format metadata. Entropy is reported separately from stored bytes.",
              "Fully shared Gate without correction is lossy; dense XOR is lossless but has no bit-storage savings."]
    for d in dimensions:
        data = storage[storage.qjl_d == d]
        for kind in ("original_weight", "residual"):
            subset = data[data.weight_kind == kind]
            lines.append(f"- D={d}, {kind}: median sparse lossless ratio "
                         f"{subset.sparse_lossless_ratio.median():.3f}; maximum selective lossless savings "
                         f"{subset.maximum_lossless_savings_pct.median():.2f}%; selectively shared neurons "
                         f"{subset.selective_shared_neurons_pct.median():.1f}%.")
    lines += ["", "## Validation and limits", "",
              f"- Maximum relative W=S+R reconstruction error: {max_reconstruction_error:.3g}.",
              "- Group partition, tensor dtype/shape, saliency coefficient sums, exact sparse XOR decoding, and byte accounting are checked.",
              "- Saved cache metadata and membership were validated; grouping was not recomputed." +
              (" Original Wikipedia calibration file size was also checked against the cache."
               if calibration_present else " The original Wikipedia calibration text was unavailable, so its file size could not be checked."),
              "- No model evaluation, perplexity, accuracy, or downstream benchmark was run. Accuracy effects of sharing remain unverified.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="all", help="all or comma-separated layer IDs")
    parser.add_argument("--experts", default="all", help="all or comma-separated expert IDs in each selected layer")
    parser.add_argument("--qjl-d", default="8192,16384", help="comma-separated QJL dimensions")
    parser.add_argument("--groups-path", type=Path, default=previous.DEFAULT_GROUPS)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--download-missing", action="store_true", help="download only needed original model shards into HF cache")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None, help="QJL projection seed; defaults to saved experiment config")
    args = parser.parse_args()
    if not args.groups_path.is_file():
        parser.error(f"saved ResCode grouping/saliency cache missing: {args.groups_path}")
    if args.row_chunk < 1:
        parser.error("--row-chunk must be positive")
    source_config = ROOT / "results/qjl_direct/qwen/test/config.json"
    source = json.loads(source_config.read_text()) if source_config.is_file() else {}
    seed = int(source.get("projection_seed", 0)) if args.seed is None else args.seed
    if seed < 0:
        parser.error("projection seed must be nonnegative")
    if source and source.get("model_name") != previous.MODEL_ID:
        parser.error("saved experiment model does not match Qwen target")
    try:
        dimensions = parse_dimensions(args.qjl_d)
        model_path = local_model(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    config = json.loads((model_path / "config.json").read_text())
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    saved = torch.load(args.groups_path, map_location="cpu", weights_only=False)
    groups, saliency = validate_saved_groups(saved, config, index)
    calibration_path = ROOT / "rescode/data/wikipedia-calib.jsonl"
    if calibration_path.is_file():
        previous.validate_group_cache(saved, config, index, calibration_path)
    layers = previous.parse_layers(args.layers, int(config["num_hidden_layers"]))
    experts = previous.parse_layers(args.experts, int(config["num_experts"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = {"cosine": output / "cosine_per_expert.csv",
             "bits": output / "bit_agreement_per_expert.csv",
             "storage": output / "storage_analysis.csv"}
    for path in paths.values():
        if path.exists():
            parser.error(f"output already exists: {path}; choose a new --output-dir to avoid mixing runs")
    banks = {d: ProjectionBank(d, seed) for d in dimensions}
    expected_shape = {"gate": (config["moe_intermediate_size"], config["hidden_size"]),
                      "up": (config["moe_intermediate_size"], config["hidden_size"]),
                      "down": (config["hidden_size"], config["moe_intermediate_size"])}
    before_stat = args.groups_path.stat()
    max_error = 0.0
    for layer in layers:
        print(f"[bit-sharing] layer {layer}: {len(experts)} selected experts", flush=True)
        layer_rows = {key: [] for key in paths}
        with ExitStack() as stack:
            shard_handles = {}

            def weight(expert: int, projection: str) -> torch.Tensor:
                name = previous.weight_name(layer, expert, projection)
                shard = index["weight_map"][name]
                path = model_path / shard
                if not path.is_file():
                    if not args.download_missing:
                        raise FileNotFoundError(f"model shard missing: {path}; pass --download-missing")
                    from huggingface_hub import hf_hub_download
                    downloaded = Path(hf_hub_download(previous.MODEL_ID, shard,
                                                       revision=previous.MODEL_REVISION))
                    if downloaded != path:
                        raise AssertionError("downloaded shard is outside pinned model snapshot")
                if shard not in shard_handles:
                    shard_handles[shard] = stack.enter_context(
                        safe_open(str(path), framework="pt", device="cpu"))
                tensor = shard_handles[shard].get_tensor(name)
                if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected_shape[projection]:
                    raise ValueError(f"BF16 tensor shape mismatch: {name}")
                return tensor

            for cid, members in enumerate(groups[layer]):
                selected = [expert for expert in members if expert in experts]
                if not selected:
                    continue
                singleton = len(members) == 1
                bases = None
                if not singleton:
                    bases = {}
                    for projection in PROJECTIONS:
                        group_weights = {expert: weight(expert, projection) for expert in members}
                        base, coefficients = previous.saliency_base(group_weights, saliency[layer])
                        if not torch.isclose(coefficients.sum(), torch.tensor(1.0), atol=1e-6):
                            raise AssertionError("saliency coefficients do not sum to one")
                        if tuple(base.shape) != expected_shape[projection]:
                            raise AssertionError("shared-base shape mismatch")
                        bases[projection] = base
                        del group_weights
                for expert in selected:
                    original = {projection: weight(expert, projection) for projection in PROJECTIONS}
                    cos, bit, storage, error = analyze_expert(
                        layer, cid, expert, singleton, original, bases,
                        dimensions, banks, device, seed, args.row_chunk)
                    layer_rows["cosine"].extend(cos)
                    layer_rows["bits"].extend(bit)
                    layer_rows["storage"].extend(storage)
                    max_error = max(max_error, error)
        for name, path in paths.items():
            append_csv(path, layer_rows[name])
        print(f"[bit-sharing] layer {layer} complete", flush=True)
    after_stat = args.groups_path.stat()
    if (before_stat.st_size, before_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
        raise AssertionError("group cache changed during read-only analysis")
    cosine = pd.read_csv(paths["cosine"])
    bits = pd.read_csv(paths["bits"])
    storage = pd.read_csv(paths["storage"])
    summary = make_summary(cosine, bits, storage)
    summary.to_csv(output / "summary.csv", index=False, float_format="%.10g")
    make_figures(cosine, bits, storage, output / "figures")
    write_report(output / "REPORT.md", cosine, bits, storage, layers, experts,
                 dimensions, max_error, model_path, args.groups_path, seed,
                 calibration_path.is_file())
    print(f"[bit-sharing] outputs: {output}; max W=S+R relative error {max_error:.3g}")


if __name__ == "__main__":
    main()
