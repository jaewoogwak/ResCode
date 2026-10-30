"""Packed-bit QJL kernel: reads uint8 signs directly, no dense unpack buffer."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_dot(Z, B, NORM, OUT, D: tl.constexpr, M: tl.constexpr,
                BYTE_STRIDE: tl.constexpr, BLOCK_D: tl.constexpr,
                BLOCK_M: tl.constexpr):
    token = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    ds = tl.arange(0, BLOCK_D)
    acc = tl.full((BLOCK_M, BLOCK_D), 0, tl.float32)
    for start in range(tl.cdiv(D, BLOCK_D)):
        idx = start * BLOCK_D + ds
        valid = idx < D
        zb = tl.load(Z + token * D + idx, valid, other=0)
        byte = tl.load(B + rows[:, None] * BYTE_STRIDE + idx[None, :] // 8,
                       (rows[:, None] < M) & valid[None, :], other=0)
        bit = (byte >> (idx[None, :] % 8)) & 1
        sign = bit.to(tl.float32) * 2.0 - 1.0
        acc += zb[None, :] * sign
    norm = tl.load(NORM + rows, rows < M, other=0)
    y = tl.sum(acc, axis=1) * norm * (1.2533141373155001 / D)
    tl.store(OUT + token * M + rows, y, rows < M)


def packed_qjl(z: torch.Tensor, packed: torch.Tensor, norms: torch.Tensor, d: int):
    if z.ndim != 2 or z.shape[1] != d or packed.ndim != 2:
        raise ValueError("invalid packed QJL shapes")
    if packed.shape[1] != math.ceil(d / 8) or packed.shape[0] != norms.numel():
        raise ValueError("packed signs/norms shape mismatch")
    out = torch.empty((z.shape[0], packed.shape[0]), device=z.device, dtype=torch.float32)
    block_m = 8
    _packed_dot[(z.shape[0], triton.cdiv(packed.shape[0], block_m))](
        z, packed, norms, out, d, packed.shape[0], packed.shape[1], 256, block_m
    )
    return out
