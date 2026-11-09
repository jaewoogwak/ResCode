#!/usr/bin/env python3
"""Test position-wise sharing of QJL sketches of BF16 exponent scales.

Reads original Qwen expert safetensors. The only persistent outputs are analysis
tables and plots; neither weights nor existing QJL checkpoints are changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rescode.qjl.core import ProjectionBank, pack_residual, pack_signs, unpack_signs

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
VARIANTS = ("original", "sign_mantissa", "exponent_scale", "centered_exponent_scale")
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


def decompose_bf16(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact FP32 W=A*B, including signed zeros and subnormals."""
    if bits.dtype != np.uint16:
        raise TypeError("BF16 bit pattern must be uint16")
    exponent = (bits >> 7) & 255
    mantissa = bits & 127
    if np.any(exponent == 255):
        raise ValueError("Inf/NaN BF16 weights have no finite A*B decomposition")
    sign = np.where((bits & 0x8000) != 0, -1.0, 1.0).astype(np.float32)
    subnormal = exponent == 0
    a = sign * np.where(subnormal, mantissa / 128.0, 1.0 + mantissa / 128.0).astype(np.float32)
    b = np.ldexp(np.ones(bits.shape, dtype=np.float32),
                 np.where(subnormal, -126, exponent.astype(np.int16) - 127))
    w = a * b
    expected = (bits.astype(np.uint32) << 16).view(np.float32)
    if not np.array_equal(w.view(np.uint32), expected.view(np.uint32)):
        raise AssertionError("BF16 decomposition did not reconstruct raw bit patterns")
    return w, a, b


def matrices(bits: np.ndarray) -> dict[str, np.ndarray]:
    w, a, b = decompose_bf16(bits)
    return {"original": w, "sign_mantissa": a, "exponent_scale": b,
            "centered_exponent_scale": b - b.mean(axis=1, keepdims=True)}


def packed_sketch(matrix: np.ndarray, bank: ProjectionBank, device: torch.device,
                  row_chunk: int) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.ascontiguousarray(matrix)).to(device)
    packed, norms = pack_residual(tensor, bank, row_chunk=row_chunk)
    return packed.cpu().numpy(), norms.cpu().numpy()


def bit_agreement(a: np.ndarray, b: np.ndarray, d: int) -> float:
    if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("packed sketches must have matching uint8 shapes")
    return 1.0 - int(POPCOUNT[np.bitwise_xor(a, b)].sum(dtype=np.int64)) / (a.shape[0] * d)


def bit_balance(packed: np.ndarray, d: int) -> float:
    return int(POPCOUNT[packed].sum(dtype=np.int64)) / (packed.shape[0] * d)


def cosine_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    na = np.linalg.norm(a.astype(np.float64), axis=1)
    nb = np.linalg.norm(b.astype(np.float64), axis=1)
    valid = (na > 0) & (nb > 0)
    if not valid.any():
        return math.nan, math.nan, 0
    dot = np.einsum("ij,ij->i", a[valid].astype(np.float64), b[valid].astype(np.float64))
    cosine = np.clip(dot / (na[valid] * nb[valid]), -1.0, 1.0)
    return float(cosine.mean()), float((1 - np.arccos(cosine) / math.pi).mean()), int(valid.sum())


def cuda_cosine_table(raw_bits: list[np.ndarray], variant: str, seed: int,
                      layer: int, block_rows: int = 32) -> dict[tuple[int, int], tuple]:
    """Compute all pair cosine metrics with bounded GPU row tiles.

    One FP32 variant is staged at a time. The same deterministic random row
    permutations used for packed-bit agreement are applied here.
    """
    experts = len(raw_bits)
    rows, input_dim = raw_bits[0].shape
    features = np.empty((experts, rows, input_dim), dtype=np.float32)
    for expert, bits in enumerate(raw_bits):
        features[expert] = matrices(bits)[variant]
    data = torch.from_numpy(features).to("cuda")
    del features
    norms = torch.linalg.vector_norm(data, dim=-1)
    result = {}

    def accumulate(a: torch.Tensor, b: torch.Tensor, na: torch.Tensor,
                   nb: torch.Tensor, totals: torch.Tensor) -> None:
        valid = (na > 0) & (nb > 0)
        cosine = torch.sum(b * a[None], dim=-1) / (na * nb).clamp_min(1e-30)
        cosine = cosine.clamp(-1, 1)
        totals[:, 0] += torch.where(valid, cosine, 0).sum(dim=1)
        totals[:, 1] += torch.where(valid, 1 - torch.acos(cosine) / math.pi, 0).sum(dim=1)
        totals[:, 2] += valid.sum(dim=1)

    for i in range(experts - 1):
        targets = torch.arange(i + 1, experts, device="cuda")
        permutations = torch.as_tensor(np.stack([
            seeded_permutation(rows, seed, layer, i, j, 17)
            for j in range(i + 1, experts)]), device="cuda", dtype=torch.long)
        same = torch.zeros((len(targets), 3), device="cuda", dtype=torch.float64)
        random = torch.zeros_like(same)
        for start in range(0, rows, block_rows):
            stop = min(start + block_rows, rows)
            a = data[i, start:stop]
            na = norms[i, start:stop][None]
            b = data[i + 1:, start:stop]
            nb = norms[i + 1:, start:stop]
            accumulate(a, b, na, nb, same)
            indices = permutations[:, start:stop]
            random_b = data[targets[:, None], indices]
            random_nb = norms[targets[:, None], indices]
            accumulate(a, random_b, na, random_nb, random)
        same_cpu, random_cpu = same.cpu().numpy(), random.cpu().numpy()
        for offset, j in enumerate(range(i + 1, experts)):
            s, r = same_cpu[offset], random_cpu[offset]
            result[i, j] = (float(s[0] / s[2]) if s[2] else math.nan,
                            float(s[1] / s[2]) if s[2] else math.nan, int(s[2]),
                            float(r[0] / r[2]) if r[2] else math.nan,
                            float(r[1] / r[2]) if r[2] else math.nan, int(r[2]))
        if (i + 1) % 10 == 0 or i == experts - 2:
            print(f"cosine {variant}: expert {i + 1}/{experts - 1}", flush=True)
    del data, norms
    return result


def seeded_permutation(n: int, *parts: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence(parts)).permutation(n)


def payload(header: dict, arrays: tuple[np.ndarray, ...]) -> bytes:
    metadata = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(metadata)) + metadata + b"".join(a.tobytes(order="C") for a in arrays)


def binary_entropy(p: float) -> float:
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def majority_and_corrections(sketches: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vote at each row/bit; ties use expert 0's bit. XOR is exactly invertible."""
    experts, rows, bytes_per_row = sketches.shape
    shared = np.empty((rows, bytes_per_row), dtype=np.uint8)
    corrections = np.empty_like(sketches)
    flips = np.zeros(experts, dtype=np.int64)
    for start in range(0, rows, 16):
        stop = min(start + 16, rows)
        block = sketches[:, start:stop]
        unpacked = np.unpackbits(block, axis=-1, count=d, bitorder="little")
        votes = unpacked.sum(axis=0, dtype=np.uint16)
        chosen = (votes > experts / 2) | ((votes == experts / 2) & (unpacked[0] != 0))
        common = np.packbits(chosen, axis=-1, bitorder="little")
        shared[start:stop] = common
        correction = np.bitwise_xor(block, common[None, :, :])
        if not np.array_equal(np.bitwise_xor(correction, common[None, :, :]), block):
            raise AssertionError("XOR roundtrip failed")
        corrections[:, start:stop] = correction
        flips += POPCOUNT[correction].sum(axis=(1, 2), dtype=np.int64)
    return shared, corrections, flips


def locate_model_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit.expanduser()]
    else:
        candidates = []
        prior = ROOT / "results/qjl_exponent_alignment/qwen/summary.json"
        if prior.exists():
            candidates.append(Path(json.loads(prior.read_text())["model_dir"]))
        candidates.extend([
            Path("/workspace/.hf_home/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat"),
            Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat",
        ])
        candidates += [s for base in candidates.copy() for s in (base / "snapshots").glob("*")]
    for path in candidates:
        if (path / "model.safetensors.index.json").is_file():
            return path.resolve()
    raise FileNotFoundError("Local Qwen BF16 safetensors not found; pass --model-dir")


def load_bits(model_dir: Path, weight_map: dict, layer: int, expert: int,
              projection: str, max_rows: int | None) -> np.ndarray:
    key = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
    shard = model_dir / weight_map[key]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key)
    expected = (2048, 1408) if projection == "down" else (1408, 2048)
    if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected:
        raise ValueError(f"{key}: expected BF16 {expected}, got {tensor.dtype} {tuple(tensor.shape)}")
    if max_rows is not None:
        tensor = tensor[:max_rows]
    return tensor.contiguous().view(torch.uint16).numpy().copy()


def self_test() -> None:
    test = np.array([[0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x0080]], dtype=np.uint16)
    w, a, b = decompose_bf16(test)
    assert w[0, 2] == 2.0 ** -133 and w[0, 3] == -(2.0 ** -133)
    assert w[0, 0] == 0 and w[0, 1] == 0 and np.signbit(w[0, 1])
    np.testing.assert_array_equal((a * b).view(np.uint32), w.view(np.uint32))
    bank = ProjectionBank(13, 0)
    matrix = np.array([[1, -2, 0], [0, 0, 0], [-3, 1, 2]], dtype=np.float32)
    p = bank.get(3, torch.device("cpu"))
    got, norms = packed_sketch(matrix, bank, torch.device("cpu"), 2)
    np.testing.assert_allclose(norms, np.linalg.norm(matrix, axis=1))
    manual = (torch.from_numpy(matrix).to(torch.float16) @ p) >= 0
    manual[1] = False
    np.testing.assert_array_equal(got, pack_signs(manual).numpy())
    np.testing.assert_array_equal(unpack_signs(torch.from_numpy(got), 13).numpy(), manual.numpy())
    np.testing.assert_array_equal(p, ProjectionBank(13, 0).get(3, torch.device("cpu")))
    assert bit_agreement(got, got, 13) == 1.0
    examples = np.stack([got, np.bitwise_xor(got, np.array([[1, 0], [0, 0], [4, 0]], dtype=np.uint8)), got])
    assert abs(bit_agreement(examples[0], examples[1], 13) - (1 - 2 / 39)) < 1e-12
    cosine, predicted, valid = cosine_metrics(
        np.array([[1, 0], [1, 0]], dtype=np.float32),
        np.array([[1, 0], [-1, 0]], dtype=np.float32))
    assert valid == 2 and abs(cosine) < 1e-12 and abs(predicted - 0.5) < 1e-12
    shared, correction, flips = majority_and_corrections(examples, 13)
    np.testing.assert_array_equal(shared, got)
    np.testing.assert_array_equal(np.bitwise_xor(correction, shared[None]), examples)
    np.testing.assert_array_equal(flips, [0, 2, 0])


def plot_summary(out: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(VARIANTS)
    labels = ["W", "A", "B", "B centered"]
    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, title, ylabel in [
        (axes[0, 0], "pair", "Same-row expert-pair agreement", "Agreement (%)"),
        (axes[0, 1], "flip", "Majority-vote correction", "Bit flips (%)"),
        (axes[1, 0], "storage", "Individual / exact shared + XOR", "Compression ratio (>1 saves)"),
        (axes[1, 1], "balance", "Sketch one-bit balance", "One bits (%)"),
    ]:
        actual, random = [], []
        for name in names:
            s = summary["variants"][name]
            if key == "pair":
                actual.append(100 * s["pair_agreement_mean"])
                random.append(100 * s["random_pair_agreement_mean"])
            elif key == "flip":
                actual.append(100 * s["majority"]["aligned_rows"]["flip_rate"])
                random.append(100 * s["majority"]["random_rows"]["flip_rate"])
            elif key == "storage":
                actual.append(s["majority"]["aligned_rows"]["zlib_ratio"])
                random.append(s["majority"]["random_rows"]["zlib_ratio"])
            else:
                actual.append(100 * s["bit_balance"])
                random.append(50.0)
        ax.bar(x - .18, actual, width=.36, label="same row" if key != "storage" else "aligned rows")
        ax.bar(x + .18, random, width=.36, label="random rows" if key != "balance" else "50%")
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Layer {summary['layer']} {summary['projection']}, D={summary['d']}, {summary['experts']} experts")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    self_test()
    if not (0 <= args.layer < 24 and 1 <= args.d and 2 <= args.max_experts <= 60):
        raise ValueError("require layer 0..23, D >= 1, max-experts 2..60")
    model_dir = locate_model_dir(args.model_dir)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    bank = ProjectionBank(args.d, args.seed)
    raw_bits: list[np.ndarray] = []
    packed: dict[str, list[np.ndarray]] = {v: [] for v in VARIANTS}
    row_norms: dict[str, list[np.ndarray]] = {v: [] for v in VARIANTS}
    balance: dict[str, list[float]] = {v: [] for v in VARIANTS}
    for expert in range(args.max_experts):
        bits = load_bits(model_dir, weight_map, args.layer, expert, args.projection, args.max_rows)
        raw_bits.append(bits)
        for name, value in matrices(bits).items():
            sketch, norms = packed_sketch(value, bank, device, args.row_chunk)
            packed[name].append(sketch)
            row_norms[name].append(norms)
            balance[name].append(bit_balance(sketch, args.d))
        print(f"projected expert {expert + 1}/{args.max_experts}", flush=True)
    nrows, input_dim = raw_bits[0].shape
    all_packed = {name: np.stack(value) for name, value in packed.items()}
    all_norms = {name: np.stack(value) for name, value in row_norms.items()}
    del packed
    cosine_tables = ({name: cuda_cosine_table(raw_bits, name, args.seed, args.layer)
                      for name in VARIANTS} if device.type == "cuda" else None)
    pair_rows: list[dict] = []
    by_variant: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    for i in range(args.max_experts):
        left = matrices(raw_bits[i]) if cosine_tables is None else None
        for j in range(i + 1, args.max_experts):
            right = matrices(raw_bits[j]) if cosine_tables is None else None
            random_rows = seeded_permutation(nrows, args.seed, args.layer, i, j, 17)
            for name in VARIANTS:
                p = all_packed[name]
                actual = bit_agreement(p[i], p[j], args.d)
                random = bit_agreement(p[i], p[j, random_rows], args.d)
                if cosine_tables is None:
                    cos, theory, valid = cosine_metrics(left[name], right[name])
                    random_cos, random_theory, random_valid = cosine_metrics(left[name], right[name][random_rows])
                else:
                    cos, theory, valid, random_cos, random_theory, random_valid = cosine_tables[name][i, j]
                row = {"layer": args.layer, "projection": args.projection, "variant": name,
                       "expert_a": i, "expert_b": j, "rows": nrows, "d": args.d,
                       "agreement": actual, "random_row_agreement": random,
                       "row_cosine_mean": cos, "gaussian_theory_agreement": theory,
                       "valid_cosine_rows": valid, "random_row_cosine_mean": random_cos,
                       "random_gaussian_theory_agreement": random_theory,
                       "random_valid_cosine_rows": random_valid}
                pair_rows.append(row)
                by_variant[name].append(row)
            del right
        del left
        print(f"compared expert {i + 1}/{args.max_experts}", flush=True)
    with (out / "pairwise.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(pair_rows)

    bundling_rows: list[dict] = []
    summaries = {}
    random_perms = [seeded_permutation(nrows, args.seed, args.layer, expert, 911)
                    for expert in range(args.max_experts)]
    for name in VARIANTS:
        data = all_packed[name]
        within = []
        for expert in range(args.max_experts):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.layer, expert, 31415]))
            first = rng.integers(nrows, size=args.within_pairs)
            second = rng.integers(nrows - 1, size=args.within_pairs)
            second += second >= first
            within.append(bit_agreement(data[expert, first], data[expert, second], args.d))
        pair = by_variant[name]
        variant_summary = {
            "bit_balance": float(np.mean(balance[name])),
            "pair_agreement_mean": float(np.mean([r["agreement"] for r in pair])),
            "random_pair_agreement_mean": float(np.mean([r["random_row_agreement"] for r in pair])),
            "within_expert_agreement_mean": float(np.mean(within)),
            "within_pairs_sampled_per_expert": args.within_pairs,
            "row_cosine_mean": float(np.mean([r["row_cosine_mean"] for r in pair])),
            "gaussian_theory_agreement_mean": float(np.mean([r["gaussian_theory_agreement"] for r in pair])),
            "majority": {},
        }
        for condition, ordered in (("aligned_rows", data),
                                   ("random_rows", np.stack([data[e, random_perms[e]]
                                                             for e in range(args.max_experts)]))):
            ordered_norms = (all_norms[name] if condition == "aligned_rows" else
                             np.stack([all_norms[name][e, random_perms[e]]
                                       for e in range(args.max_experts)]))
            shared, corrections, flips = majority_and_corrections(ordered, args.d)
            baseline_header = {"format": "individual_bitpacked_v1", "model": MODEL_ID,
                               "layer": args.layer, "projection": args.projection, "variant": name,
                               "shape": list(data.shape), "expert_ids": list(range(args.max_experts)),
                               "d": args.d, "seed": args.seed, "bitorder": "little",
                               "row_norms": "float32", "row_norm_shape": list(ordered_norms.shape),
                               "row_order": condition}
            if condition == "random_rows":
                baseline_header["permutation"] = "numpy.SeedSequence([seed,layer,expert,911])"
            shared_header = {**baseline_header, "format": "shared_xor_bitpacked_v1",
                             "tie": "expert_0"}
            individual_blob = payload(baseline_header, (ordered, ordered_norms))
            shared_blob = payload(shared_header, (shared, corrections, ordered_norms))
            individual_zlib = len(zlib.compress(individual_blob, level=1))
            shared_zlib = len(zlib.compress(shared_blob, level=1))
            total_bits = args.max_experts * nrows * args.d
            rate = float(flips.sum() / total_bits)
            entry = {"flip_rate": rate, "lossy_majority_agreement": 1 - rate,
                     "xor_binary_entropy_bits_per_bit": binary_entropy(rate),
                     "shared_sketch_bytes": shared.nbytes,
                     "correction_bytes": corrections.nbytes,
                     "row_norm_bytes": ordered_norms.nbytes,
                     "metadata_bytes": len(shared_blob) - shared.nbytes - corrections.nbytes - ordered_norms.nbytes,
                     "individual_metadata_bytes": len(individual_blob) - ordered.nbytes - ordered_norms.nbytes,
                     "individual_bitpacked_bytes": len(individual_blob),
                     "shared_xor_bitpacked_bytes": len(shared_blob),
                     "bitpacked_ratio": len(individual_blob) / len(shared_blob),
                     "individual_zlib_bytes": individual_zlib,
                     "shared_xor_zlib_bytes": shared_zlib,
                     "zlib_ratio": individual_zlib / shared_zlib}
            variant_summary["majority"][condition] = entry
            for expert in range(args.max_experts):
                erate = float(flips[expert] / (nrows * args.d))
                bundling_rows.append({**entry, "layer": args.layer, "projection": args.projection,
                                      "variant": name, "condition": condition, "expert": expert,
                                      "bit_balance": balance[name][expert],
                                      "within_expert_agreement": within[expert],
                                      "majority_flip_rate": erate,
                                      "lossy_majority_agreement": 1 - erate,
                                      "xor_binary_entropy_bits_per_bit": binary_entropy(erate)})
            print(f"bundled {name} {condition}", flush=True)
        summaries[name] = variant_summary
    with (out / "bundling.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bundling_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(bundling_rows)
    summary = {"model": MODEL_ID, "model_dir": str(model_dir), "layer": args.layer,
               "projection": args.projection, "d": args.d, "seed": args.seed,
               "experts": args.max_experts, "rows": nrows, "input_dim": input_dim,
               "pairs": args.max_experts * (args.max_experts - 1) // 2,
               "projection_dtype": "float16", "bitorder": "little",
               "cosine_theory": "1 - acos(cosine) / pi; zero-norm rows excluded",
               "within_pairs": "deterministic distinct random row pairs sampled within each expert",
               "storage": "4-byte LE JSON-header length + JSON metadata + bitpacked bytes + FP32 row norms; zlib level 1 separately reported",
               "random_rows": "seeded independent permutation per expert for bundling; per-pair permutation for agreement",
               "self_test": "passed", "variants": summaries}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_summary(out, summary)
    print(f"saved {out.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--projection", choices=("gate", "up", "down"), default="up")
    parser.add_argument("--d", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/qjl_exponent_qjl_agreement")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-experts", type=int, default=60)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--within-pairs", type=int, default=1024)
    parser.add_argument("--row-chunk", type=int, default=128)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 2:
        parser.error("--max-rows must be >= 2")
    if args.within_pairs < 1 or args.row_chunk < 1 or args.seed < 0:
        parser.error("--within-pairs, --row-chunk must be positive and --seed nonnegative")
    run(args)


if __name__ == "__main__":
    main()
