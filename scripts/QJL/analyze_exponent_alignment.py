#!/usr/bin/env python3
"""Compare raw BF16 exponent agreement before and after MoE neuron alignment.

Only local safetensors are read. One layer and one target expert are resident at a
time; the model is never instantiated. Matching uses tiled cosine products and a
CPU Hungarian assignment. Figures are self-contained SVG files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment

MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
PROJECTIONS = ("gate", "up", "down")
DEFAULT_OUTPUT = Path("results/qjl_exponent_alignment/qwen")
PAIR_FIELDS = ("exact_pct", "within_one_pct", "mean_abs_exponent_diff",
               "chance_pct", "chance_corrected_pct", "signed_diff_entropy_bits")
COLORS = {"before": "#566573", "aligned": "#087e8b", "random": "#d9822b"}


def parse_indices(value: str, size: int, label: str) -> list[int]:
    if value == "all":
        return list(range(size))
    try:
        result = sorted(set(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise ValueError(f"{label} must be 'all' or comma-separated integers") from exc
    if not result or result[0] < 0 or result[-1] >= size:
        raise ValueError(f"{label} must be within 0..{size - 1}")
    return result


def parse_projections(value: str) -> list[str]:
    result = list(dict.fromkeys(part.strip().lower() for part in value.split(",")))
    if not result or any(item not in PROJECTIONS for item in result):
        raise ValueError("--projections must list gate,up,down")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--reference-expert", type=int, default=0)
    parser.add_argument("--projections", default="gate,up,down")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-experts", type=int, default=None)
    parser.add_argument("--max-neurons", type=int, default=None)
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Local HF snapshot containing config.json and safetensors index")
    parser.add_argument("--alignment-device", choices=("auto", "cpu", "cuda"),
                        default="auto", help="Only cosine tiles use CUDA; weights stay on CPU")
    args = parser.parse_args()
    if not 0 <= args.reference_expert < 60:
        parser.error("--reference-expert must be within 0..59")
    if args.max_experts is not None and not 2 <= args.max_experts <= 60:
        parser.error("--max-experts must be within 2..60")
    if args.max_neurons is not None and args.max_neurons < 2:
        parser.error("--max-neurons must be at least 2")
    return args


def resolve_model_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit]
    else:
        homes = [Path(os.environ[name]) for name in ("HF_HOME",) if os.environ.get(name)]
        homes += [Path("/workspace/.hf_home"), Path.home() / ".cache/huggingface"]
        candidates = []
        for home in homes:
            snapshots = (home / "hub" / "models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
                         / "snapshots")
            if snapshots.is_dir():
                candidates.extend(sorted(snapshots.iterdir()))
    for path in candidates:
        if (path / "config.json").is_file() and (path / "model.safetensors.index.json").is_file():
            return path.resolve()
    raise FileNotFoundError("Local Qwen BF16 safetensors snapshot not found; pass --model-dir")


def tensor_key(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"


def exponent_mantissa(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if bits.dtype != np.uint16:
        raise TypeError("raw BF16 bits must have uint16 dtype")
    return ((bits >> 7) & 255).astype(np.uint8), (bits & 127).astype(np.uint8)


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    """BF16 is exactly the upper 16 bits of IEEE FP32, with no rounding here."""
    return np.left_shift(bits.astype(np.uint32), 16).view(np.float32)


def load_expert_bits(model_dir: Path, weight_map: dict, layer: int,
                     expert: int, hidden: int, full_neurons: int,
                     neurons: int) -> dict[str, np.ndarray]:
    result = {}
    for projection in PROJECTIONS:
        key = tensor_key(layer, expert, projection)
        shard = model_dir / weight_map[key]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
            expected = (hidden, full_neurons) if projection == "down" else (full_neurons, hidden)
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected:
                raise ValueError(f"expected BF16 {expected} for {key}, got {tensor.dtype} {tuple(tensor.shape)}")
            if projection == "down":
                tensor = tensor[:, :neurons]
            else:
                tensor = tensor[:neurons, :]
            result[projection] = tensor.contiguous().view(torch.uint16).numpy().copy()
    return result


def entropy(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    total = values.sum()
    if total <= 0:
        raise ValueError("empty symbol histogram")
    p = values[values > 0] / total
    return float(-(p * np.log2(p)).sum())


def histogram_metrics(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    exponent, mantissa = exponent_mantissa(bits)
    exp_counts = np.bincount(exponent.ravel(), minlength=256).astype(np.int64)
    man_counts = np.bincount(mantissa.ravel(), minlength=128).astype(np.int64)
    metrics = {"num_weights": int(bits.size), "exponent_entropy_bits": entropy(exp_counts),
               "mantissa_entropy_bits": entropy(man_counts),
               "min_exponent": int(exponent.min()), "max_exponent": int(exponent.max())}
    for k in (4, 8, 16):
        metrics[f"exponent_top{k}_coverage_pct"] = float(
            100 * np.sort(exp_counts)[-k:].sum() / bits.size)
        metrics[f"mantissa_top{k}_coverage_pct"] = float(
            100 * np.sort(man_counts)[-k:].sum() / bits.size)
    return exp_counts, man_counts, metrics


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    data = bf16_to_float32(matrix)
    lengths = np.linalg.norm(data.astype(np.float64), axis=1)
    np.divide(data, lengths[:, None], out=data, where=lengths[:, None] > 0)
    return np.ascontiguousarray(data)


def neuron_features(bits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"gate": unit_rows(bits["gate"]), "up": unit_rows(bits["up"]),
            "down": unit_rows(bits["down"].T)}


def align_neurons(reference: dict[str, np.ndarray], target: dict[str, np.ndarray],
                  device: str = "cpu", tile_rows: int = 128) -> tuple[np.ndarray, float]:
    """Target index for each reference neuron, using mean Gate/Up/Down cosine."""
    n = reference["gate"].shape[0]
    if any(reference[p].shape != target[p].shape or reference[p].shape[0] != n
           for p in PROJECTIONS):
        raise ValueError("matching features have inconsistent shapes")
    scores = np.zeros((n, n), dtype=np.float32)
    for projection in PROJECTIONS:
        a = torch.as_tensor(reference[projection], device=device)
        b = torch.as_tensor(target[projection], device=device).T.contiguous()
        for start in range(0, n, tile_rows):
            stop = min(start + tile_rows, n)
            tile = torch.mm(a[start:stop], b).cpu().numpy()
            scores[start:stop] += tile / 3.0
        del a, b
    rows, columns = linear_sum_assignment(-scores)
    permutation = np.empty(n, dtype=np.int32)
    permutation[rows] = columns
    assert np.unique(permutation).size == n
    quality = float(scores[rows, columns].mean())
    return permutation, quality


def permute_neurons(bits: dict[str, np.ndarray], permutation: np.ndarray) -> dict[str, np.ndarray]:
    if len(permutation) != bits["gate"].shape[0] or set(permutation.tolist()) != set(range(len(permutation))):
        raise ValueError("invalid one-to-one neuron permutation")
    return {"gate": bits["gate"][permutation, :],
            "up": bits["up"][permutation, :],
            "down": bits["down"][:, permutation]}


def pair_metrics(reference: np.ndarray, target: np.ndarray,
                 ref_hist: np.ndarray, target_hist: np.ndarray) -> dict:
    if reference.shape != target.shape:
        raise ValueError("position-wise comparison requires equal shapes")
    ref_exp = (reference >> 7) & 255
    target_exp = (target >> 7) & 255
    difference = ref_exp.astype(np.int16) - target_exp.astype(np.int16)
    counts = np.bincount((difference.ravel() + 255), minlength=511)
    chance = float(np.dot(ref_hist.astype(np.float64), target_hist.astype(np.float64)) /
                   (reference.size * target.size))
    exact = float(np.count_nonzero(difference == 0) / reference.size)
    return {"exact_pct": 100 * exact,
            "within_one_pct": 100 * float(np.count_nonzero(np.abs(difference) <= 1) / reference.size),
            "mean_abs_exponent_diff": float(np.abs(difference).mean()),
            "chance_pct": 100 * chance,
            "chance_corrected_pct": 100 * (exact - chance),
            "signed_diff_entropy_bits": entropy(counts),
            "num_weights": int(reference.size)}


def jsd_matrix(counts: np.ndarray) -> np.ndarray:
    """Pairwise Jensen-Shannon divergence in bits (range 0..1)."""
    p = counts.astype(np.float64) / counts.sum(axis=1, keepdims=True)

    def h(values):
        logs = np.zeros_like(values)
        np.log2(values, out=logs, where=values > 0)
        return -(values * logs).sum(axis=-1)

    mean = (p[:, None, :] + p[None, :, :]) / 2
    result = h(mean) - (h(p)[:, None] + h(p)[None, :]) / 2
    result = np.maximum(result, 0)
    np.fill_diagonal(result, 0)
    return result.astype(np.float32)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _svg_text(x, y, value, size=12, anchor="start", color="#23313f") -> str:
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" font-family="sans-serif">{html.escape(str(value))}</text>')


def line_figure(path: Path, title: str, layers: list[int], series: list[tuple[str, np.ndarray]],
                ylabel: str, ymin: float, ymax: float) -> None:
    width, height, left, right, top, bottom = 880, 420, 85, 25, 65, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = min(layers), max(layers)
    x = lambda value: left + plot_w * (value - xmin) / max(xmax - xmin, 1)
    y = lambda value: top + plot_h * (ymax - value) / max(ymax - ymin, 1e-12)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>', _svg_text(width / 2, 28, title, 18, "middle"),
             f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#5c6770"/>']
    for t in range(5):
        value = ymin + (ymax - ymin) * t / 4
        parts += [f'<line x1="{left}" y1="{y(value):.1f}" x2="{left + plot_w}" y2="{y(value):.1f}" stroke="#e1e8ed"/>',
                  _svg_text(left - 8, y(value) + 4, f"{value:.1f}", 11, "end")]
    for layer in sorted(set([layers[0], layers[-1], *layers[::max(1, len(layers) // 6)]])):
        parts.append(_svg_text(x(layer), top + plot_h + 20, layer, 11, "middle"))
    for index, (label, values) in enumerate(series):
        color = COLORS[label]
        points = " ".join(f"{x(layer):.1f},{y(float(value)):.1f}" for layer, value in zip(layers, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.extend(f'<circle cx="{x(layer):.1f}" cy="{y(float(value)):.1f}" r="3" fill="{color}"/>'
                     for layer, value in zip(layers, values))
        parts.append(f'<line x1="{left + index * 150}" y1="{height - 17}" x2="{left + index * 150 + 24}" y2="{height - 17}" stroke="{color}" stroke-width="3"/>')
        parts.append(_svg_text(left + index * 150 + 30, height - 13, label))
    parts += [_svg_text(width / 2, height - 43, "MoE layer", 12, "middle"),
              _svg_text(12, 48, ylabel, 12), "</svg>"]
    path.write_text("\n".join(parts))


def heatmap_figure(path: Path, title: str, values: np.ndarray, xlabel: str,
                   ylabel: str, vmax: float, legend: str,
                   row_labels: list[int] | None = None,
                   column_labels: list[int] | None = None) -> None:
    values = np.asarray(values)
    rows, columns = values.shape
    cell_w = max(2.5, min(8, 780 / columns))
    cell_h = max(5, min(13, 560 / rows))
    left, top, bottom, right = 95, 65, 80, 65
    width, height = left + columns * cell_w + right, top + rows * cell_h + bottom
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
             '<rect width="100%" height="100%" fill="white"/>',
             _svg_text(width / 2, 29, title, 17, "middle")]
    scale = max(vmax, 1e-12)
    for i in range(rows):
        for j in range(columns):
            t = min(max(float(values[i, j]) / scale, 0), 1)
            color = f"#{int(245 - 220 * t):02x}{int(249 - 194 * t):02x}{int(252 - 116 * t):02x}"
            parts.append(f'<rect x="{left + j * cell_w:.1f}" y="{top + i * cell_h:.1f}" width="{cell_w + 0.1:.1f}" height="{cell_h + 0.1:.1f}" fill="{color}"/>')
    parts += [f'<rect x="{left}" y="{top}" width="{columns * cell_w:.1f}" height="{rows * cell_h:.1f}" fill="none" stroke="#566"/>',
              _svg_text(left, top + rows * cell_h + 22,
                        (column_labels or list(range(columns)))[0], 11),
              _svg_text(left + columns * cell_w, top + rows * cell_h + 22,
                        (column_labels or list(range(columns)))[-1], 11, "end"),
              _svg_text(width / 2, height - 25, xlabel, 12, "middle"),
              _svg_text(10, 48, ylabel, 12),
              _svg_text(left + columns * cell_w + 8, top + 14, f"max {vmax:.3g}", 11),
              _svg_text(left + columns * cell_w + 8, top + 29, legend, 11)]
    labels = row_labels or list(range(rows))
    for i in sorted(set([0, rows - 1, *range(0, rows, max(1, math.ceil(rows / 8)))])):
        parts.append(_svg_text(left - 7, top + (i + 0.7) * cell_h, labels[i], 10, "end"))
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def summarize_pairs(rows: list[dict], layers: list[int], projections: list[str]) -> list[dict]:
    summary = []
    for layer in layers:
        for projection in projections:
            subset = [r for r in rows if r["layer"] == layer and r["projection"] == projection]
            item = {"layer": layer, "projection": projection, "pairs": len(subset) // 3}
            for field in PAIR_FIELDS:
                for condition in ("before", "aligned", "random"):
                    values = np.array([r[field] for r in subset if r["condition"] == condition])
                    item[f"{field}_{condition}_mean"] = float(values.mean())
                    item[f"{field}_{condition}_std"] = float(values.std(ddof=0))
                paired = {condition: {r["target_expert"]: r[field] for r in subset
                                      if r["condition"] == condition}
                          for condition in ("before", "aligned")}
                delta = np.array([paired["aligned"][expert] - paired["before"][expert]
                                  for expert in sorted(paired["before"])])
                item[f"{field}_aligned_minus_before_mean"] = float(delta.mean())
                item[f"{field}_aligned_minus_before_std"] = float(delta.std(ddof=0))
            summary.append(item)
    return summary


def main() -> None:
    args = arguments()
    model_dir = resolve_model_dir(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("num_hidden_layers") != 24 or config.get("num_experts") != 60 or config.get("torch_dtype") != "bfloat16":
        raise ValueError("expected the original 24-layer, 60-expert BF16 Qwen1.5-MoE checkpoint")
    hidden, all_neurons = int(config["hidden_size"]), int(config["moe_intermediate_size"])
    neurons = min(args.max_neurons or all_neurons, all_neurons)
    layers = parse_indices(args.layers, 24, "--layers")
    projections = parse_projections(args.projections)
    expert_ids = [args.reference_expert] + [i for i in range(60) if i != args.reference_expert]
    expert_ids = sorted(expert_ids[:args.max_experts or 60])
    target_ids = [i for i in expert_ids if i != args.reference_expert]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.alignment_device == "auto" else args.alignment_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    for layer in layers:
        for expert in expert_ids:
            for projection in PROJECTIONS:
                key = tensor_key(layer, expert, projection)
                if key not in weight_map or not (model_dir / weight_map[key]).is_file():
                    raise FileNotFoundError(f"missing local BF16 expert weight: {key}")
    out = args.output_dir
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    shape = (len(layers), len(expert_ids), len(projections))
    exponent_counts = np.zeros((*shape, 256), dtype=np.int64)
    mantissa_counts = np.zeros((*shape, 128), dtype=np.int64)
    exponent_jsd = np.zeros((len(layers), len(projections), len(expert_ids), len(expert_ids)), dtype=np.float32)
    mantissa_jsd = np.zeros_like(exponent_jsd)
    permutations = np.full((len(layers), len(expert_ids), neurons), -1, dtype=np.int32)
    pair_rows, expert_rows, align_rows, jsd_rows = [], [], [], []

    for li, layer in enumerate(layers):
        reference = load_expert_bits(model_dir, weight_map, layer, args.reference_expert,
                                     hidden, all_neurons, neurons)
        ref_features = neuron_features(reference)
        expert_index = expert_ids.index(args.reference_expert)
        permutations[li, expert_index] = np.arange(neurons, dtype=np.int32)

        def record_hist(expert: int, bits: dict[str, np.ndarray]):
            ei = expert_ids.index(expert)
            result = {}
            for projection in projections:
                pi = projections.index(projection)
                ex, ma, metrics = histogram_metrics(bits[projection])
                exponent_counts[li, ei, pi] = ex
                mantissa_counts[li, ei, pi] = ma
                expert_rows.append({"layer": layer, "expert": expert, "projection": projection,
                                    **metrics})
                result[projection] = ex
            return result

        ref_hist = record_hist(args.reference_expert, reference)
        for target_id in target_ids:
            target = load_expert_bits(model_dir, weight_map, layer, target_id,
                                      hidden, all_neurons, neurons)
            target_hist = record_hist(target_id, target)
            target_features = neuron_features(target)
            aligned_perm, quality = align_neurons(ref_features, target_features, device)
            random_perm = np.random.default_rng(np.random.SeedSequence(
                [args.seed, layer, target_id])).permutation(neurons).astype(np.int32)
            permutations[li, expert_ids.index(target_id)] = aligned_perm
            aligned = permute_neurons(target, aligned_perm)
            random = permute_neurons(target, random_perm)
            align_rows.append({"layer": layer, "reference_expert": args.reference_expert,
                               "target_expert": target_id, "neurons": neurons,
                               "matched_mean_cosine": quality,
                               "fixed_position_mean_cosine": float(np.mean([
                                   np.einsum("ij,ij->i", ref_features[p], target_features[p]).mean()
                                   for p in PROJECTIONS]))})
            for projection in projections:
                for condition, candidate in (("before", target), ("aligned", aligned),
                                             ("random", random)):
                    pair_rows.append({"layer": layer, "reference_expert": args.reference_expert,
                                      "target_expert": target_id, "projection": projection,
                                      "condition": condition,
                                      **pair_metrics(reference[projection], candidate[projection],
                                                     ref_hist[projection], target_hist[projection])})
            del target, aligned, random, target_features
        for pi, projection in enumerate(projections):
            exponent_jsd[li, pi] = jsd_matrix(exponent_counts[li, :, pi])
            mantissa_jsd[li, pi] = jsd_matrix(mantissa_counts[li, :, pi])
            mask = np.triu_indices(len(expert_ids), k=1)
            jsd_rows.append({"layer": layer, "projection": projection,
                             "expert_pairs": len(mask[0]),
                             "exponent_jsd_mean_bits": float(exponent_jsd[li, pi][mask].mean()),
                             "exponent_jsd_std_bits": float(exponent_jsd[li, pi][mask].std()),
                             "mantissa_jsd_mean_bits": float(mantissa_jsd[li, pi][mask].mean()),
                             "mantissa_jsd_std_bits": float(mantissa_jsd[li, pi][mask].std())})
        print(f"[alignment] layer {layer}: {len(expert_ids)} experts complete", flush=True)
        del reference, ref_features

    summary_rows = summarize_pairs(pair_rows, layers, projections)
    write_csv(out / "pair_metrics.csv", pair_rows)
    write_csv(out / "expert_distributions.csv", expert_rows)
    write_csv(out / "layer_summary.csv", summary_rows)
    write_csv(out / "alignment_quality.csv", align_rows)
    write_csv(out / "jsd_summary.csv", jsd_rows)
    np.savez_compressed(out / "histograms.npz", layer_ids=layers, expert_ids=expert_ids,
                        projections=projections, exponent_counts=exponent_counts,
                        mantissa_counts=mantissa_counts)
    np.savez_compressed(out / "pairwise_jsd.npz", layer_ids=layers, expert_ids=expert_ids,
                        projections=projections, exponent_jsd=exponent_jsd,
                        mantissa_jsd=mantissa_jsd)
    np.savez_compressed(out / "permutations.npz", layer_ids=layers, expert_ids=expert_ids,
                        permutations=permutations)
    total_exp = exponent_counts.sum(axis=-1, keepdims=True)
    total_man = mantissa_counts.sum(axis=-1, keepdims=True)
    exp_prob = exponent_counts / total_exp
    man_prob = mantissa_counts / total_man
    exp_vmax = float(exp_prob.max())
    man_vmax = float(man_prob.max())
    representatives = sorted(set([0, len(layers) // 2, len(layers) - 1]))
    for pi, projection in enumerate(projections):
        for field, label, bounds in (("exact_pct", "Exact exponent agreement (%)", (0, 100)),
                                     ("chance_corrected_pct", "Agreement above chance (percentage points)", None),
                                     ("signed_diff_entropy_bits", "Signed exponent difference entropy (bits)", None)):
            series = [(condition, np.array([row[f"{field}_{condition}_mean"]
                                            for row in summary_rows if row["projection"] == projection]))
                      for condition in ("before", "aligned", "random")]
            values = np.concatenate([series_item[1] for series_item in series])
            ymin, ymax = bounds if bounds else (float(values.min()) - 0.05, float(values.max()) + 0.05)
            line_figure(figures / f"{projection}_{field}_by_layer.svg",
                        f"{projection.capitalize()}: {label}", layers, series, label, ymin, ymax)
        for counts, name, vmax in ((exp_prob, "exponent", exp_vmax),
                                    (man_prob, "mantissa", man_vmax)):
            heatmap_figure(figures / f"{projection}_{name}_layers.svg",
                           f"{projection.capitalize()} {name} distribution by layer (mean over experts)",
                           counts[:, :, pi, :].mean(axis=1), f"Raw BF16 {name} symbol",
                           "MoE layer", vmax, "probability", row_labels=layers)
            for li in representatives:
                layer = layers[li]
                heatmap_figure(figures / f"{projection}_{name}_experts_layer{layer}.svg",
                               f"{projection.capitalize()} {name}, layer {layer} (experts)",
                               counts[li, :, pi, :], f"Raw BF16 {name} symbol",
                               "Expert ID", vmax, "probability", row_labels=expert_ids)
                matrix = exponent_jsd if name == "exponent" else mantissa_jsd
                heatmap_figure(figures / f"{projection}_{name}_jsd_layer{layer}.svg",
                               f"{projection.capitalize()} {name} JSD, layer {layer}",
                               matrix[li, pi], "Expert ID (column)",
                               "Expert ID (row)", 1.0, "JSD bits",
                               row_labels=expert_ids, column_labels=expert_ids)

    aggregate = {}
    for projection in projections:
        selected = [r for r in pair_rows if r["projection"] == projection]
        aggregate[projection] = {condition: {
            field: float(np.mean([r[field] for r in selected if r["condition"] == condition]))
            for field in PAIR_FIELDS} for condition in ("before", "aligned", "random")}
        aggregate[projection]["aligned_minus_before_exact_percentage_points"] = (
            aggregate[projection]["aligned"]["exact_pct"] - aggregate[projection]["before"]["exact_pct"])
    manifest = {"model": MODEL, "model_dir": str(model_dir), "layers": layers,
                "experts": expert_ids, "reference_expert": args.reference_expert,
                "projections": projections, "hidden_size": hidden, "neurons_analyzed": neurons,
                "full_neurons": all_neurons, "seed": args.seed,
                "alignment_device": device, "partial_smoke": (len(layers) != 24 or len(expert_ids) != 60
                                                        or neurons != all_neurons or len(projections) != 3),
                "percentage_units": "0..100", "jsd_units": "bits", "aggregate": aggregate}
    (out / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[alignment] outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
