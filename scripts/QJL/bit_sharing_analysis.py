"""Numerical and byte-accounting helpers for the QJL bit-sharing diagnostic."""

from __future__ import annotations

import math

import numpy as np

THRESHOLDS = (0.8, 0.9, 0.95, 0.99)
QUANTILES = (0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0)
QUANTILE_NAMES = ("min", "p10", "median", "p90", "p95", "p99", "max")
POPCOUNT = np.array([int(x).bit_count() for x in range(256)], dtype=np.uint8)
SKETCH_METADATA_BYTES = 16  # four uint32: rows, input_dim, D, projection seed
CORRECTION_METADATA_BYTES = 16  # four uint32: rows, D, encoding, flag layout


def distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("distribution requires finite nonempty vector")
    return {f"{prefix}_{name}": float(value) for name, value in
            zip(QUANTILE_NAMES, np.quantile(values, QUANTILES))}


def aligned_neurons(gate: np.ndarray, up: np.ndarray,
                    down: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Match each intermediate neuron to its Gate/Up row and Down column."""
    if gate.ndim != 2 or gate.shape != up.shape or down.shape != gate.T.shape:
        raise ValueError("Gate/Up must be [intermediate, hidden]; Down [hidden, intermediate]")
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (gate, up, down)):
        raise ValueError("expected finite FP32 weights")
    return {"gate_up": (gate, up), "gate_down": (gate, down.T),
            "up_down": (up, down.T)}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("neuron arrays must have matching [rows, hidden] shape")
    aa = np.einsum("ij,ij->i", a, a, dtype=np.float64)
    bb = np.einsum("ij,ij->i", b, b, dtype=np.float64)
    dot = np.einsum("ij,ij->i", a, b, dtype=np.float64)
    na, nb = np.sqrt(aa), np.sqrt(bb)
    cosine = np.divide(dot, na * nb, out=np.zeros_like(dot), where=(na * nb) > 0)
    return np.clip(cosine, -1, 1), na, nb


def derangement(rows: int, seed_parts: tuple[int, ...]) -> np.ndarray:
    if rows < 2:
        raise ValueError("at least two neurons are needed for mismatched control")
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    order = rng.permutation(rows)
    permutation = np.empty(rows, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    assert np.all(permutation != np.arange(rows))
    return permutation


def norm_correlation(na: np.ndarray, nb: np.ndarray) -> float:
    if len(na) < 2 or np.std(na) == 0 or np.std(nb) == 0:
        return math.nan
    return float(np.corrcoef(na, nb)[0, 1])


def cosine_statistics(a: np.ndarray, b: np.ndarray) -> tuple[dict, np.ndarray]:
    cosine, na, nb = cosine_rows(a, b)
    stats = {**distribution(cosine, "signed_cosine"),
             **distribution(np.abs(cosine), "absolute_cosine"),
             "signed_cosine_mean": float(cosine.mean()),
             "absolute_cosine_mean": float(np.abs(cosine).mean()),
             "norm_pearson_r": norm_correlation(na, nb),
             "zero_norm_pairs": int(np.count_nonzero((na == 0) | (nb == 0))),
             "neurons": len(cosine)}
    stats.update({f"abs_cosine_gt_{str(t).replace('.', 'p')}":
                  float(np.mean(np.abs(cosine) > t)) for t in THRESHOLDS})
    return stats, cosine


def _last_mask(d: int) -> np.uint8:
    return np.uint8((1 << (d % 8)) - 1 if d % 8 else 255)


def bit_mismatches(gate: np.ndarray, up: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Return exact row Hamming counts and packed XOR; ignore padding bits."""
    if gate.shape != up.shape or gate.ndim != 2 or gate.dtype != up.dtype or gate.dtype != np.uint8:
        raise ValueError("packed Gate/Up arrays must have matching uint8 shape")
    if d < 1 or gate.shape[1] != math.ceil(d / 8):
        raise ValueError("packed width does not match D")
    xor = np.bitwise_xor(gate, up)
    if d % 8:
        xor[:, -1] &= _last_mask(d)
    return POPCOUNT[xor].sum(axis=1, dtype=np.int32), xor


def bit_statistics(gate: np.ndarray, up: np.ndarray, d: int,
                   cosine: np.ndarray, permutation: np.ndarray,
                   seed_parts: tuple[int, ...]) -> tuple[dict, np.ndarray, np.ndarray]:
    if len(cosine) != len(gate) or len(permutation) != len(gate):
        raise ValueError("row count mismatch")
    mismatch, xor = bit_mismatches(gate, up, d)
    shuffled, _ = bit_mismatches(gate, up[permutation], d)
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    random_bits = rng.integers(0, 256, size=up.shape, dtype=np.uint8)
    random_mismatch, _ = bit_mismatches(gate, random_bits, d)
    direct = 1 - mismatch / d
    inverted = mismatch / d
    best = np.maximum(direct, inverted)
    predicted = 1 - np.arccos(np.clip(cosine, -1, 1)) / np.pi
    stats = {**distribution(direct, "direct_agreement"),
             **distribution(inverted, "inverted_agreement"),
             **distribution(best, "best_agreement"),
             **distribution(mismatch / d, "normalized_hamming"),
             **distribution(1 - shuffled / d, "shuffled_agreement"),
             **distribution(1 - random_mismatch / d, "random_bit_agreement"),
             "direct_agreement_mean": float(direct.mean()),
             "best_agreement_mean": float(best.mean()),
             "shuffled_agreement_mean": float((1 - shuffled / d).mean()),
             "random_bit_agreement_mean": float((1 - random_mismatch / d).mean()),
             "predicted_agreement_mean": float(predicted.mean()),
             "observed_minus_predicted_mean": float(np.mean(direct - predicted)),
             "prediction_mae": float(np.mean(np.abs(direct - predicted))),
             "inversion_beneficial_fraction": float(np.mean(inverted > direct)),
             "neurons": len(gate), "qjl_d": d}
    return stats, mismatch, xor


def _valid_invert(packed: np.ndarray, d: int) -> np.ndarray:
    result = np.bitwise_not(packed).copy()
    if d % 8:
        result[-1] &= _last_mask(d)
    return result


def encode_sparse_correction(gate_row: np.ndarray, up_row: np.ndarray,
                             d: int, invert: bool) -> bytes:
    """Lossless row encoding: uint16 count followed by uint16 bit indices (LE)."""
    if d > 65535 or d < 1:
        raise ValueError("uint16 positions require 1 <= D <= 65535")
    target = _valid_invert(up_row, d) if invert else up_row
    xor = np.bitwise_xor(gate_row, target).copy()
    if d % 8:
        xor[-1] &= _last_mask(d)
    positions = np.flatnonzero(np.unpackbits(xor, bitorder="little")[:d]).astype("<u2")
    return np.uint16(len(positions)).astype("<u2").tobytes() + positions.tobytes()


def decode_sparse_correction(gate_row: np.ndarray, payload: bytes,
                             d: int, invert: bool) -> np.ndarray:
    if len(payload) < 2 or (len(payload) - 2) % 2:
        raise ValueError("invalid sparse payload length")
    count = int(np.frombuffer(payload[:2], dtype="<u2")[0])
    if len(payload) != 2 + 2 * count:
        raise ValueError("sparse correction count mismatch")
    positions = np.frombuffer(payload, dtype="<u2", offset=2)
    if np.any(positions >= d) or len(np.unique(positions)) != len(positions):
        raise ValueError("invalid sparse bit positions")
    result = gate_row.copy()
    for bit in positions:
        result[int(bit) // 8] ^= np.uint8(1 << (int(bit) % 8))
    if invert:
        result = _valid_invert(result, d)
    return result


def storage_statistics(mismatch: np.ndarray, d: int) -> dict:
    """Exact byte counts for stated raw layouts; no entropy-as-bytes shortcut."""
    if mismatch.ndim != 1 or len(mismatch) == 0 or np.any(mismatch < 0) or np.any(mismatch > d):
        raise ValueError("invalid Hamming counts")
    if not 1 <= d <= 65535:
        raise ValueError("sparse uint16 layout requires D <= 65535")
    rows, packed_bytes = len(mismatch), math.ceil(d / 8)
    gate_bytes = rows * packed_bytes
    norm_bytes = 2 * rows * 4  # Gate and Up retain separate FP32 norms in every mode.
    separate = 2 * gate_bytes + norm_bytes + 2 * SKETCH_METADATA_BYTES
    no_correction = gate_bytes + norm_bytes + SKETCH_METADATA_BYTES
    dense = gate_bytes + gate_bytes + norm_bytes + SKETCH_METADATA_BYTES + CORRECTION_METADATA_BYTES
    corrected = np.minimum(mismatch, d - mismatch)
    inverted = mismatch > d - mismatch
    sparse_payload = int(np.sum(2 + 2 * corrected, dtype=np.int64))
    flags = math.ceil(rows / 8)
    sparse = gate_bytes + sparse_payload + flags + norm_bytes + SKETCH_METADATA_BYTES + CORRECTION_METADATA_BYTES
    # Select rows with the largest positive savings. The selection bitmap and
    # inversion flags are packed, and their byte rounding is charged exactly.
    gains = np.sort(packed_bytes - (2 + 2 * corrected))[::-1]
    prefix = np.concatenate(([0], np.cumsum(gains, dtype=np.int64)))
    best_total, best_count = separate, 0
    for count in range(1, rows + 1):
        payload_bytes = count * packed_bytes - int(prefix[count])
        selection_bytes = 0 if count == rows else math.ceil(rows / 8)
        inversion_bytes = math.ceil(count / 8)
        total = (gate_bytes + norm_bytes + 2 * SKETCH_METADATA_BYTES +
                 (rows - count) * packed_bytes + payload_bytes +
                 selection_bytes + inversion_bytes)
        if total < best_total:
            best_total, best_count = total, count
    return {"neurons": rows, "qjl_d": d, "gate_sketch_bytes": gate_bytes,
            "up_sketch_bytes": gate_bytes, "fp32_norm_bytes": norm_bytes,
            "separate_metadata_bytes": 2 * SKETCH_METADATA_BYTES,
            "correction_metadata_bytes": SKETCH_METADATA_BYTES + CORRECTION_METADATA_BYTES,
            "separate_total_bytes": separate,
            "shared_no_correction_total_bytes": no_correction,
            "shared_no_correction_ratio": no_correction / separate,
            "dense_xor_correction_bytes": gate_bytes,
            "dense_lossless_total_bytes": dense,
            "dense_lossless_ratio": dense / separate,
            "sparse_position_payload_bytes": sparse_payload,
            "sparse_inversion_flag_bytes": flags,
            "sparse_lossless_total_bytes": sparse,
            "sparse_lossless_ratio": sparse / separate,
            "selective_lossless_total_bytes": best_total,
            "selective_lossless_ratio": best_total / separate,
            "selective_shared_neurons": best_count,
            "selective_shared_neurons_pct": 100 * best_count / rows,
            "individually_beneficial_neurons_pct": 100 * float(np.mean((2 + 2 * corrected + 1 / 8) < packed_bytes)),
            "maximum_lossless_savings_pct": 100 * (separate - best_total) / separate,
            "mean_hamming_distance": float(np.mean(mismatch / d)),
            "mean_binary_entropy_bits_per_bit": float(np.mean(
                -(lambda p: np.where((p > 0) & (p < 1),
                    p * np.log2(np.clip(p, 1e-300, 1)) +
                    (1 - p) * np.log2(np.clip(1 - p, 1e-300, 1)), 0))(mismatch / d))) }
