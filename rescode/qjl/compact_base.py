"""Lossless storage for a BF16 base with a rounded, short mantissa.

The checkpoint stores zlib-coded exponents and packed sign/mantissa codes.
At inference the exponent bytes (not BF16 weights) reside on the GPU; a BF16
weight is reconstructed only for the duration of one linear projection.
"""

import math
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rounded_bf16_bits(weight: torch.Tensor, k: int) -> torch.Tensor:
    """Round a BF16 tensor to k fraction bits, ties to even, preserving NaNs."""
    if weight.dtype != torch.bfloat16 or k not in (1, 2, 3, 4):
        raise ValueError("expected BF16 weight and k in {1,2,3,4}")
    bits = weight.contiguous().view(torch.uint16).to(torch.int32)
    exponent = (bits >> 7) & 255
    shift = 7 - k
    retained = (bits & 127) >> shift
    remainder = (bits & 127) & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = ((remainder > half) | ((remainder == half) & ((retained & 1) != 0))).int()
    # Adding the rounded mantissa directly carries into the exponent, including
    # subnormal -> normal. Finite overflow becomes infinity, as in BF16 rounding.
    finite = exponent != 255
    rounded = (bits & ~127) + ((retained + increment) << shift)
    return torch.where(finite, rounded, bits).to(torch.uint16)


def _codes(bits: torch.Tensor, k: int) -> np.ndarray:
    raw = bits.cpu().numpy().astype(np.uint16, copy=False).reshape(-1)
    return (((raw >> 15) << k) | ((raw & 127) >> (7 - k))).astype(np.uint16)


def _pack_codes(codes: np.ndarray, width: int) -> bytes:
    offsets = np.arange(width, dtype=np.uint16)
    bits = ((codes[:, None] >> offsets) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits, bitorder="little").tobytes()


def encode_base(weight: torch.Tensor, k: int) -> dict:
    """Encode a BF16 base; special values retain their complete original bits."""
    rounded = rounded_bf16_bits(weight, k).cpu().contiguous()
    raw = rounded.numpy().astype(np.uint16, copy=False).reshape(-1)
    exponent = ((raw >> 7) & 255).astype(np.uint8)
    counts = np.bincount(exponent, minlength=256)
    probability = counts[counts > 0] / len(exponent)
    entropy = float(-np.sum(probability * np.log2(probability)))
    special = np.flatnonzero(exponent == 255).astype(np.int32)
    return {
        "format": "bf16-rounded-sign-mantissa-zlib-exponent-v1",
        "shape": tuple(weight.shape), "k": k,
        "codes": _pack_codes(_codes(rounded, k), k + 1),
        "exponents": zlib.compress(exponent.tobytes(), level=1),
        "special_indices": special.tobytes(),
        "special_bits": raw[special].tobytes(),
        "exponent_entropy_bits": entropy,
    }


def encoded_bytes(payload: dict) -> int:
    return sum(len(payload[key]) for key in ("codes", "exponents", "special_indices", "special_bits"))


def storage_payload(payload: dict) -> dict:
    """Use raw uint8 tensor storages in torch.save instead of pickle byte strings."""
    result = dict(payload)
    for key in ("codes", "exponents", "special_indices", "special_bits"):
        if not isinstance(result[key], torch.Tensor):
            result[key] = torch.from_numpy(
                np.frombuffer(result[key], dtype=np.uint8).copy())
    return result


def byte_payload(payload: dict) -> dict:
    result = dict(payload)
    for key in ("codes", "exponents", "special_indices", "special_bits"):
        if isinstance(result[key], torch.Tensor):
            result[key] = result[key].cpu().numpy().tobytes()
    return result


class CompactBase(nn.Module):
    """Compact GPU-resident representation; BF16 reconstruction is transient."""

    def __init__(self, payload: dict, device):
        super().__init__()
        payload = byte_payload(payload)
        if payload.get("format") != "bf16-rounded-sign-mantissa-zlib-exponent-v1":
            raise ValueError("unknown compact-base format")
        self.shape = tuple(payload["shape"])
        self.k = int(payload["k"])
        n = math.prod(self.shape)
        started = time.perf_counter()
        exponents = np.frombuffer(zlib.decompress(payload["exponents"]), dtype=np.uint8).copy()
        self.exponent_restore_seconds = time.perf_counter() - started
        if len(exponents) != n or len(payload["codes"]) != (n * (self.k + 1) + 7) // 8:
            raise ValueError("corrupt compact base length")
        self.register_buffer("codes", torch.from_numpy(np.frombuffer(payload["codes"], dtype=np.uint8).copy()).to(device))
        self.register_buffer("exponents", torch.from_numpy(exponents).to(device))
        indices = np.frombuffer(payload["special_indices"], dtype=np.int32).copy()
        bits = np.frombuffer(payload["special_bits"], dtype=np.uint16).copy()
        if len(indices) != len(bits):
            raise ValueError("corrupt special value table")
        self.register_buffer("special_indices", torch.from_numpy(indices.astype(np.int64)).to(device))
        self.register_buffer("special_bits", torch.from_numpy(bits.astype(np.int32)).to(device))

    def decode(self) -> torch.Tensor:
        n = math.prod(self.shape)
        width = self.k + 1
        idx = torch.arange(n, device=self.codes.device, dtype=torch.int64)
        bit_offset = idx * width
        byte_idx = bit_offset >> 3
        low = self.codes[byte_idx].to(torch.int32)
        high = self.codes[torch.clamp(byte_idx + 1, max=self.codes.numel() - 1)].to(torch.int32)
        code = ((low | (high << 8)) >> (bit_offset & 7).to(torch.int32)) & ((1 << width) - 1)
        sign = code >> self.k
        mantissa = code & ((1 << self.k) - 1)
        bits = (sign << 15) | (self.exponents.to(torch.int32) << 7) | (mantissa << (7 - self.k))
        if self.special_indices.numel():
            bits.index_copy_(0, self.special_indices, self.special_bits)
        return bits.to(torch.uint16).view(torch.bfloat16).reshape(self.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.bfloat16 and not self.special_indices.numel():
            from .compact_base_kernel import compact_linear
            return compact_linear(x, self)
        # Exact portable path for special values and reference comparisons.
        return F.linear(x, self.decode())

    def gpu_resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.codes, self.exponents, self.special_indices, self.special_bits))
