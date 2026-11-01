"""One-bit, row-scaled Gaussian QJL. Projection matrices are shared at runtime."""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGETS = {
    "gate": ("gate",), "up": ("up",), "down": ("down",),
    "gate_up": ("gate", "up"),
    "gate_up_down": ("gate", "up", "down"),
}


def parse_targets(value: str) -> Tuple[str, ...]:
    if value not in TARGETS:
        raise ValueError(f"targets must be one of {tuple(TARGETS)}")
    return TARGETS[value]


class ProjectionBank:
    """Deterministic Gaussian P per (input dimension, D, seed, device).

    Kept outside nn.Module state so a single P is resident for all layers and
    is regenerated from the checkpoint seed on load. FP16 is the intentional
    projection precision; signs and runtime xP use this same matrix.
    """

    def __init__(self, d: int, seed: int):
        if d < 1:
            raise ValueError("QJL dimension must be positive")
        self.d, self.seed = int(d), int(seed)
        self._cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def get(self, input_dim: int, device: torch.device) -> torch.Tensor:
        device = torch.device(device)
        key = (int(input_dim), str(device))
        if key not in self._cache:
            generator = torch.Generator(device="cpu").manual_seed(
                (self.seed + input_dim * 1000003 + self.d * 9176) % (2 ** 63)
            )
            # CPU generation gives the same matrix regardless of GPU model.
            p = torch.randn(input_dim, self.d, generator=generator, dtype=torch.float32)
            self._cache[key] = p.to(device=device, dtype=torch.float16)
        return self._cache[key]

    def resident_bytes(self, device_type=None) -> int:
        return sum(t.numel() * t.element_size() for t in self._cache.values()
                   if device_type is None or t.device.type == device_type)


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Little-endian bit order; pad final byte with zeros."""
    if signs.ndim != 2:
        raise ValueError("expected [rows, D] signs")
    rows, d = signs.shape
    pad = (-d) % 8
    if pad:
        signs = F.pad(signs, (0, pad))
    bits = signs.reshape(rows, -1, 8).to(torch.uint8)
    shifts = torch.arange(8, device=bits.device, dtype=torch.uint8)
    return torch.sum(bits << shifts, dim=-1).to(torch.uint8).contiguous()


def unpack_signs(packed: torch.Tensor, d: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> shifts) & 1).reshape(packed.shape[0], -1)[:, :d].bool()


@torch.no_grad()
def pack_residual(weight: torch.Tensor, bank: ProjectionBank, row_chunk: int = 128):
    """Return packed signs and row norms; never keep the dense residual."""
    p = bank.get(weight.shape[1], weight.device)
    packed, norms = [], []
    for start in range(0, weight.shape[0], row_chunk):
        w = weight[start:start + row_chunk].float()
        n = torch.linalg.vector_norm(w, dim=1)
        # GPU GEMM accumulates FP16 products in FP32; no full FP32 P copy.
        signs = (w.to(p.dtype) @ p) >= 0
        signs[n == 0] = False
        packed.append(pack_signs(signs))
        norms.append(n.to(torch.float32))
    return torch.cat(packed), torch.cat(norms)


def qjl_reference(z: torch.Tensor, packed: torch.Tensor, norms: torch.Tensor, d: int,
                  row_chunk: int = 128) -> torch.Tensor:
    """Reference dot from packed bytes, without materializing a weight matrix."""
    out = []
    z = z.float()
    for start in range(0, packed.shape[0], row_chunk):
        bits = unpack_signs(packed[start:start + row_chunk], d)
        signs = bits.float().mul_(2).sub_(1)
        out.append((z @ signs.T) * norms[start:start + row_chunk].float())
    return torch.cat(out, dim=-1) * (math.sqrt(math.pi / 2) / d)


def qjl_triton(z: torch.Tensor, packed: torch.Tensor, norms: torch.Tensor, d: int) -> torch.Tensor:
    if not z.is_cuda:
        raise RuntimeError("Triton backend requires CUDA")
    from .triton_kernel import packed_qjl
    return packed_qjl(z.contiguous(), packed.contiguous(), norms.contiguous(), d)


def qjl_triton_legacy(z: torch.Tensor, packed: torch.Tensor,
                      norms: torch.Tensor, d: int) -> torch.Tensor:
    if not z.is_cuda:
        raise RuntimeError("Triton backend requires CUDA")
    from .triton_kernel import packed_qjl_legacy
    return packed_qjl_legacy(z.contiguous(), packed.contiguous(), norms.contiguous(), d)


def qjl_triton_fused_gate_up(z: torch.Tensor, gate: "PackedResidual",
                             up: "PackedResidual", output_dtype: torch.dtype):
    from .triton_kernel import packed_qjl_fused_gate_up
    gate_y, up_y = packed_qjl_fused_gate_up(
        z.contiguous(), gate.packed, gate.norms, up.packed, up.norms, gate.d)
    return gate_y.to(output_dtype), up_y.to(output_dtype)


class PackedResidual(nn.Module):
    def __init__(self, packed: torch.Tensor, norms: torch.Tensor, input_dim: int,
                 d: int, bank: ProjectionBank, backend: str = "triton"):
        super().__init__()
        if backend not in ("triton", "triton_legacy", "reference"):
            raise ValueError(f"unknown QJL backend: {backend}")
        self.register_buffer("packed", packed.contiguous())
        self.register_buffer("norms", norms.contiguous())
        self.input_dim, self.d, self.backend = input_dim, d, backend
        self.bank = bank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.bank.get(self.input_dim, x.device)
        z = x.reshape(-1, self.input_dim).to(p.dtype) @ p
        return self.forward_projected(z, x.dtype, x.shape[:-1])

    def forward_projected(self, z: torch.Tensor, output_dtype: torch.dtype,
                          leading_shape=None) -> torch.Tensor:
        if self.backend == "triton":
            y = qjl_triton(z, self.packed, self.norms, self.d)
        elif self.backend == "triton_legacy":
            y = qjl_triton_legacy(z, self.packed, self.norms, self.d)
        else:
            y = qjl_reference(z, self.packed, self.norms, self.d)
        if leading_shape is None:
            leading_shape = z.shape[:-1]
        return y.reshape(*leading_shape, self.packed.shape[0]).to(output_dtype)


class ExactResidual(nn.Module):
    def __init__(self, residual: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", residual.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight.float()).to(x.dtype)
