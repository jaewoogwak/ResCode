"""Pure row-wise outlier measurements for original expert weights and residuals.

The input is an FP32 [output row, input dimension] matrix. Selection is made
independently in every row. No input matrix is modified or quantized.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROW_KS = (1, 2, 4, 8, 16)
QUANTILES = (0.5, 0.9, 0.95, 0.99, 1.0)
QUANTILE_NAMES = ("median", "p90", "p95", "p99", "max")


def distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{name}": float(value) for name, value in
            zip(QUANTILE_NAMES, np.quantile(values, QUANTILES))}


def top_indices(absolute: np.ndarray, maximum_k: int) -> np.ndarray:
    """Return each row's largest absolute positions in deterministic order."""
    if absolute.ndim != 2 or not 1 <= maximum_k <= absolute.shape[1]:
        raise ValueError("invalid matrix or maximum_k")
    # Argpartition limits the sort to at most 16 candidates per output row.
    candidates = np.argpartition(-absolute, maximum_k - 1, axis=1)[:, :maximum_k]
    values = np.take_along_axis(absolute, candidates, axis=1)
    order = np.lexsort((candidates, -values), axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def jaccard_indices(a: np.ndarray, b: np.ndarray) -> float:
    left, right = set(map(int, a)), set(map(int, b))
    return len(left & right) / len(left | right) if left or right else 1.0


def pair_overlap(indices: np.ndarray, row_pairs: np.ndarray) -> float:
    """Mean Jaccard for fixed distinct row pairs; only overlap is sampled."""
    if len(row_pairs) == 0:
        return math.nan
    a, b = indices[row_pairs[:, 0]], indices[row_pairs[:, 1]]
    intersection = (a[:, :, None] == b[:, None, :]).any(axis=2).sum(axis=1)
    return float(np.mean(intersection / (2 * indices.shape[1] - intersection)))


def analyze_rowwise_matrix(matrix: np.ndarray, seed_parts: tuple[int, ...],
                           overlap_pairs: int = 1024) -> tuple[dict, list[dict], dict]:
    """Full-row magnitude/norm statistics and sampled within-expert overlap.

    The returned position sets are small (at most 16 integers per k); per-row
    arrays are deliberately not retained in output files.
    """
    if matrix.ndim != 2 or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("expected a finite FP32 [out_features, in_features] matrix")
    rows, dims = matrix.shape
    if rows < 1 or dims < 1:
        raise ValueError("matrix dimensions must be positive")
    max_k = min(max(ROW_KS), dims)
    absolute = np.abs(matrix)
    square = np.square(matrix, dtype=np.float64)
    row_energy = square.sum(axis=1, dtype=np.float64)
    row_norm = np.sqrt(row_energy)
    p99 = np.quantile(absolute, 0.99, axis=1)
    maximum = absolute.max(axis=1)
    rms = np.sqrt(row_energy / dims)
    max_over_p99 = np.divide(maximum, p99, out=np.zeros_like(maximum), where=p99 > 0)
    max_over_rms = np.divide(maximum, rms, out=np.zeros_like(rms), where=rms > 0)
    stats = {**distribution(max_over_p99, "max_over_p99"),
             **distribution(max_over_rms, "max_over_rms"),
             **distribution(row_norm, "original_row_l2"),
             "zero_rows": int(np.count_nonzero(row_energy == 0)),
             "row_count": rows, "input_dimensions": dims}

    top_order = top_indices(absolute, max_k)
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    # Independent random keys per row produce nested, distinct random positions.
    random_order = top_indices(rng.random((rows, dims), dtype=np.float32), max_k)
    if rows >= 2:
        first = rng.integers(0, rows, size=overlap_pairs)
        second = rng.integers(0, rows - 1, size=overlap_pairs)
        second += second >= first
        row_pairs = np.column_stack((first, second))
    else:
        row_pairs = np.empty((0, 2), dtype=np.int64)

    total_energy = float(row_energy.sum(dtype=np.float64))
    reductions = []
    dominant_sets = {}
    for k in ROW_KS:
        if k > dims:
            continue
        for selection, order in (("top_abs", top_order), ("random", random_order)):
            selected = order[:, :k]
            if np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0):
                raise AssertionError("selected positions are repeated within a row")
            removed_row = np.take_along_axis(square, selected, axis=1).sum(axis=1, dtype=np.float64)
            remaining_row = row_energy - removed_row
            if np.any(remaining_row < -1e-8 * np.maximum(row_energy, 1)):
                raise AssertionError("selected energy exceeds original row energy")
            remaining_row = np.maximum(remaining_row, 0.0)
            if not np.allclose(row_energy, removed_row + remaining_row, rtol=1e-12, atol=1e-10):
                raise AssertionError("per-row squared norm decomposition failed")
            # Independently check the row-wise masking operation on a subset.
            check_rows = min(8, rows)
            masked = square[:check_rows].copy()
            masked[np.arange(check_rows)[:, None], selected[:check_rows]] = 0
            if not np.allclose(masked.sum(axis=1), remaining_row[:check_rows], rtol=1e-12, atol=1e-10):
                raise AssertionError("row-wise mask does not match energy subtraction")
            remaining_norm = np.sqrt(remaining_row)
            l2_reduction = 100 * np.divide(row_norm - remaining_norm, row_norm,
                                           out=np.zeros_like(row_norm), where=row_norm > 0)
            squared_reduction = 100 * np.divide(removed_row, row_energy,
                                                out=np.zeros_like(row_energy), where=row_energy > 0)
            removed_energy = float(removed_row.sum(dtype=np.float64))
            remaining_energy = float(remaining_row.sum(dtype=np.float64))
            if not math.isclose(total_energy, removed_energy + remaining_energy, rel_tol=1e-12, abs_tol=1e-8):
                raise AssertionError("matrix squared norm decomposition failed")
            frequency = np.bincount(selected.ravel(), minlength=dims)
            # Break frequency ties independently per expert. Index-order tie
            # breaking would create spurious cross-expert overlap, especially
            # for the random control where many dimensions have equal counts.
            dominant = np.lexsort((rng.random(dims), -frequency))[:k]
            dominant_sets[(k, selection)] = dominant
            top_dimension_count = max(1, math.ceil(0.01 * dims))
            concentration = float(np.sort(frequency)[-top_dimension_count:].sum() / (rows * k))
            reductions.append({"k": k, "selection": selection,
                               **distribution(row_norm, "original_row_l2"),
                               **distribution(remaining_norm, "remaining_row_l2"),
                               **distribution(l2_reduction, "row_l2_reduction_pct"),
                               **distribution(squared_reduction, "row_squared_l2_reduction_pct"),
                               "original_frobenius_norm": math.sqrt(total_energy),
                               "remaining_frobenius_norm": math.sqrt(remaining_energy),
                               "matrix_l2_reduction_pct": 100 * (1 - math.sqrt(remaining_energy / total_energy)) if total_energy else 0.0,
                               "matrix_squared_l2_reduction_pct": 100 * removed_energy / total_energy if total_energy else 0.0,
                               "within_expert_row_jaccard": pair_overlap(selected, row_pairs),
                               "top_1pct_dimension_position_share": concentration,
                               "most_frequent_dimension": int(np.argmax(frequency)),
                               "most_frequent_dimension_row_share": float(frequency.max() / rows),
                               "overlap_sample_pairs": len(row_pairs)})
    return stats, reductions, dominant_sets


def plot_representative_rows(residual: np.ndarray, weight: np.ndarray, output: Path,
                             label: str, seed: int, maximum_rows: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(residual.shape[0], size=min(maximum_rows, residual.shape[0]), replace=False))
    top = top_indices(np.abs(residual[rows]), 1)[:, 0]
    frequency = np.bincount(top, minlength=residual.shape[1])
    order = np.argsort(-frequency, kind="stable")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].imshow(np.log1p(np.abs(residual[rows][:, order])), aspect="auto", interpolation="nearest")
    axes[0, 0].set(title="Residual |R|, dimensions sorted by sampled top-1 frequency",
                   xlabel="Input dimension", ylabel="Sampled output row (not token)")
    axes[0, 1].hist(np.abs(residual[rows]).ravel(), bins=100, histtype="step", density=True, label="Residual")
    axes[0, 1].hist(np.abs(weight[rows]).ravel(), bins=100, histtype="step", density=True, label="Original W")
    axes[0, 1].set(xscale="symlog", yscale="log", title="Absolute element magnitudes", xlabel="Magnitude")
    axes[0, 1].legend()
    axes[1, 0].plot(np.sort(np.abs(residual[rows]).max(axis=1)))
    axes[1, 0].set(title="Maximum absolute element by sampled row", xlabel="Sorted output row", ylabel="Maximum |R|")
    axes[1, 1].plot(np.sort(frequency)[::-1])
    axes[1, 1].set(title="Top-1 position frequency", xlabel="Ranked input dimension", ylabel="Sampled row count")
    fig.suptitle(label)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_rowwise_aggregates(per_expert: pd.DataFrame, reductions: pd.DataFrame,
                            overlaps: pd.DataFrame, figure_dir: Path, projections: tuple[str, ...]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for projection in projections:
        data = per_expert[per_expert.projection == projection]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, column, title in zip(axes,
                                     ("max_over_p99_median", "max_over_rms_median"),
                                     ("Row max/P99", "Row max/RMS")):
            for kind in ("residual", "original_weight"):
                values = data[data.matrix_kind == kind][column]
                ax.hist(values, bins=40, alpha=0.6, label=kind)
            ax.set(title=title, xlabel="Expert median across rows", ylabel="Expert count")
            ax.legend()
        fig.savefig(figure_dir / f"row_magnitude_{projection}.png", dpi=130)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        for kind, selection in (("residual", "top_abs"), ("residual", "random"),
                                ("original_weight", "top_abs")):
            sub = reductions[(reductions.projection == projection) &
                             (reductions.matrix_kind == kind) & (reductions.selection == selection)]
            curve = sub.groupby("k")["matrix_squared_l2_reduction_pct"].median().sort_index()
            ax.plot(curve.index, curve.values, marker="o", label=f"{kind} {selection}")
        ax.set(title=f"{projection}: row-wise element removal", xlabel="Elements removed per row",
               ylabel="Median matrix squared Frobenius energy removed (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"row_norm_reduction_{projection}.png", dpi=130)
        plt.close(fig)
        if not overlaps.empty:
            sub = overlaps[overlaps.projection == projection]
            fig, ax = plt.subplots(figsize=(8, 5))
            for selection in ("top_abs", "random"):
                curve = sub[sub.selection == selection].groupby("k")["jaccard"].mean().sort_index()
                ax.plot(curve.index, curve.values, marker="o", label=selection)
            ax.set(title=f"{projection}: dominant position overlap across experts",
                   xlabel="Top-k elements per row", ylabel="Mean group Jaccard", ylim=(0, 1))
            ax.legend()
            fig.tight_layout()
            fig.savefig(figure_dir / f"row_cross_expert_overlap_{projection}.png", dpi=130)
            plt.close(fig)


def write_rowwise_report(path: Path, per_expert: pd.DataFrame, reductions: pd.DataFrame,
                         overlaps: pd.DataFrame, layers: list[int], reconstruction_error: float,
                         groups_path: Path, model_path: Path, seed: int) -> None:
    scope = "full 24-layer analysis" if len(layers) == 24 else "selected-layer analysis"
    lines = ["# Row-wise residual element outliers before QJL", "",
             f"Scope: **{scope}**; layers {', '.join(map(str, layers))}.",
             f"Group/saliency cache: `{groups_path}`. Original BF16 model: `{model_path}`.",
             "R = FP32(W) − FP32(BF16 ResCode saliency base), delta=0.001.",
             f"Random seed: {seed}. No QJL projection or quantization was applied.",
             "Each matrix row is an output row, not a token. Norm and magnitude statistics cover every row; only Jaccard row pairs and plotted rows are sampled.", "",
             "## Element concentration", "",
             "| Projection | Residual row max/P99 median | W row max/P99 median | Residual top-1 squared norm removed | Random top-1 | Residual top-16 | Random top-16 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for projection in ("gate", "up", "down"):
        m = per_expert[per_expert.projection == projection]
        r = reductions[(reductions.projection == projection) & (reductions.matrix_kind == "residual")]
        def med(k: int, selection: str) -> float:
            return float(r[(r.k == k) & (r.selection == selection)].matrix_squared_l2_reduction_pct.median())
        lines.append(f"| {projection} | {m[m.matrix_kind == 'residual'].max_over_p99_median.median():.2f} | "
                     f"{m[m.matrix_kind == 'original_weight'].max_over_p99_median.median():.2f} | "
                     f"{med(1, 'top_abs'):.2f}% | {med(1, 'random'):.2f}% | "
                     f"{med(16, 'top_abs'):.2f}% | {med(16, 'random'):.2f}% |")
    lines += ["", "## Position patterns", ""]
    for projection in ("gate", "up", "down"):
        r = reductions[(reductions.projection == projection) & (reductions.matrix_kind == "residual") & (reductions.k == 16)]
        top = r[r.selection == "top_abs"]
        random = r[r.selection == "random"]
        o = overlaps[(overlaps.projection == projection) & (overlaps.k == 16)]
        lines.append(f"- {projection}: within-expert sampled row Jaccard at k=16: "
                     f"{top.within_expert_row_jaccard.median():.3f} versus random {random.within_expert_row_jaccard.median():.3f}; "
                     f"top 1% of input dimensions account for {100 * top.top_1pct_dimension_position_share.median():.1f}% of top-16 positions; "
                     f"within-group cross-expert Jaccard {o[o.selection == 'top_abs'].jaccard.mean():.3f} versus random {o[o.selection == 'random'].jaccard.mean():.3f}.")
    lines += ["", "## Validation and interpretation", "",
              "- Reused existing ResCode groups, saliency, and shared-base initializer. Singleton experts were excluded.",
              f"- Maximum relative reconstruction error for W = S + R: {reconstruction_error:.3g}.",
              "- Checked BF16 tensor shapes, coefficient sums, independent per-row selection, and row/matrix squared energy identities.",
              "- Large reductions from a few elements would support testing QJL outlier separation; this experiment does not measure quantization quality.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
