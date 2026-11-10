"""Lossless BF16 sign/exponent and FP16 row-mantissa base storage.

The packed exponent stream and GPU matmul are shared with CompactBase.  Its
``k=0`` representation describes sign * 2**e; a separate FP16 beta replaces
the original mantissa for each output row.  Exponent-zero subnormals use the
BF16 minimum normal as their scale and are handled by the compact special
table.  Signed zeros and exponent-255 values retain their original bits.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .compact_base import CompactBase, encoded_bytes, storage_payload
from .packed_exponent import BLOCK_FORMAT, encode_exponents


def _scale_from_bits(bits: torch.Tensor) -> torch.Tensor:
    exponent = ((bits >> 7) & 255).to(torch.int32)
    mantissa = bits & 127
    scale = torch.exp2(torch.where(exponent == 0, -126, exponent - 127).float())
    return torch.where(((exponent == 0) & (mantissa == 0)) | (exponent == 255),
                       torch.zeros_like(scale), scale)


@torch.no_grad()
def fit_row_beta(weight: torch.Tensor) -> torch.Tensor:
    """FP32 least-squares beta, scaled per row to avoid B**2 underflow."""
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("expected a BF16 matrix")
    bits = weight.contiguous().view(torch.uint16).to(torch.int32)
    scale = _scale_from_bits(bits)
    row_scale = scale.amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    relative = scale / row_scale[:, None]
    magnitude = torch.where(torch.isfinite(weight.float()), weight.float().abs(), 0)
    numerator = ((magnitude / row_scale[:, None]) * relative).sum(dim=1)
    denominator = relative.square().sum(dim=1)
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30),
                       torch.ones_like(numerator)).float()


@torch.no_grad()
def encode_exponent_mantissa_base(weight: torch.Tensor, block_size: int = 4096) -> dict:
    """Pack original sign/exponent and store the fitted row beta in FP16."""
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("expected a BF16 matrix")
    beta = fit_row_beta(weight).to(torch.float16).cpu()
    if not torch.isfinite(beta).all():
        raise OverflowError("row beta cannot be stored in FP16")
    raw = weight.contiguous().view(torch.uint16).cpu().numpy().astype(np.uint16, copy=False)
    flat = raw.reshape(-1)
    exponent = ((flat >> 7) & 255).astype(np.uint8)
    mantissa = flat & 127
    signs = np.packbits((flat >> 15).astype(np.uint8), bitorder="little").tobytes()
    # CompactBase k=0 reconstructs signed powers of two.  For subnormals,
    # exponent field 0 has scale 2**-126, represented as BF16 exponent 1.
    special = np.flatnonzero(((exponent == 0) & (mantissa != 0)) |
                             (exponent == 255)).astype(np.int32)
    special_bits = flat[special].copy()
    subnormal = exponent[special] == 0
    special_bits[subnormal] = (special_bits[subnormal] & 0x8000) | 0x0080
    payload = {"format": BLOCK_FORMAT, "shape": tuple(weight.shape), "k": 0,
               "codes": signs, "special_indices": special.tobytes(),
               "special_bits": special_bits.tobytes(), "beta": beta}
    payload.update(encode_exponents(exponent, block_size=block_size))
    return storage_payload(payload)


def exponent_mantissa_base_bytes(state: dict) -> int:
    return encoded_bytes(state) + state["beta"].numel() * state["beta"].element_size()


class PackedExponentMantissaBase(nn.Module):
    """Packed GPU base; dense reconstruction is only made by decode()."""

    def __init__(self, state: dict, device="cpu"):
        super().__init__()
        if state.get("format") != BLOCK_FORMAT or int(state.get("k", -1)) != 0:
            raise ValueError("wrong exponent-mantissa base format")
        self.compact = CompactBase(state, device)
        self.shape = self.compact.shape
        beta = state["beta"]
        if beta.dtype != torch.float16 or beta.numel() != self.shape[0]:
            raise ValueError("expected one FP16 beta per output row")
        self.register_buffer("beta", beta.to(device).contiguous())

    def decode(self) -> torch.Tensor:
        return self.compact.decode().float() * self.beta.float()[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.shape[1]:
            raise ValueError("base input dimension mismatch")
        if x.is_cuda and x.dtype == torch.bfloat16:
            from .compact_base_kernel import compact_linear
            unscaled = compact_linear(x, self.compact, out_dtype=torch.float32)
        else:
            unscaled = F.linear(x.float(), self.compact.decode().float())
        return (unscaled * self.beta.float()).to(x.dtype)

    def resident_bytes(self) -> int:
        return self.compact.gpu_resident_bytes() + self.beta.numel() * 2
