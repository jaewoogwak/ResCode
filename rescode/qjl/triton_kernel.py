"""Packed-sign QJL inference kernels; checkpoint tensors stay unchanged."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_dot_legacy(Z, B, NORM, OUT, D: tl.constexpr, M: tl.constexpr,
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


@triton.jit
def _packed_tiled(Z, PG, PU, NG, NU, OUT,
                  M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  FUSED: tl.constexpr):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    token = tl.program_id(1) * BM + tl.arange(0, BM)
    gate_acc = tl.full((BM, BN), 0, tl.float32)
    if FUSED:
        up_acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(D, BK)):
        k = block * BK + tl.arange(0, BK)
        z = tl.load(Z + token[:, None] * D + k[None, :],
                    (token[:, None] < M) & (k[None, :] < D), other=0).to(tl.float32)
        offset = row[:, None] * tl.cdiv(D, 8) + k[None, :] // 8
        mask = (row[:, None] < N) & (k[None, :] < D)
        gate_byte = tl.load(PG + offset, mask, other=0).to(tl.int32)
        gate_sign = ((gate_byte >> (k[None, :] % 8)) & 1) * 2 - 1
        gate_acc = tl.dot(z, tl.trans(gate_sign.to(tl.float32)), gate_acc,
                          input_precision="tf32x3")
        if FUSED:
            up_byte = tl.load(PU + offset, mask, other=0).to(tl.int32)
            up_sign = ((up_byte >> (k[None, :] % 8)) & 1) * 2 - 1
            up_acc = tl.dot(z, tl.trans(up_sign.to(tl.float32)), up_acc,
                            input_precision="tf32x3")
    gate_norm = tl.load(NG + row, row < N, other=0).to(tl.float32)
    gate = gate_acc * gate_norm[None, :] * (1.2533141373155001 / D)
    tl.store(OUT + token[:, None] * N + row[None, :], gate,
             (token[:, None] < M) & (row[None, :] < N))
    if FUSED:
        up_norm = tl.load(NU + row, row < N, other=0).to(tl.float32)
        up = up_acc * up_norm[None, :] * (1.2533141373155001 / D)
        tl.store(OUT + M * N + token[:, None] * N + row[None, :], up,
                 (token[:, None] < M) & (row[None, :] < N))


def _check(z, packed, norms, d):
    if not z.is_cuda or not packed.is_cuda or not norms.is_cuda:
        raise RuntimeError("Triton QJL backend requires CUDA")
    if z.device != packed.device or z.device != norms.device:
        raise ValueError("QJL tensors must use the same CUDA device")
    if z.ndim != 2 or z.shape[1] != d or packed.ndim != 2:
        raise ValueError("invalid packed QJL shapes")
    if packed.shape[1] != math.ceil(d / 8) or packed.shape[0] != norms.numel():
        raise ValueError("packed signs/norms shape mismatch")
    if packed.dtype != torch.uint8 or norms.dtype != torch.float32:
        raise ValueError("ResCode QJL requires uint8 signs and FP32 row norms")


def packed_qjl_legacy(z: torch.Tensor, packed: torch.Tensor,
                      norms: torch.Tensor, d: int) -> torch.Tensor:
    """The original ResCode kernel, retained for direct comparison."""
    _check(z, packed, norms, d)
    out = torch.empty((z.shape[0], packed.shape[0]), device=z.device, dtype=torch.float32)
    _packed_dot_legacy[(z.shape[0], triton.cdiv(packed.shape[0], 8))](
        z, packed, norms, out, d, packed.shape[0], packed.shape[1], 256, 8)
    return out


def _run_tiled(z, gate, gate_norms, up, up_norms, d, fused):
    _check(z, gate, gate_norms, d)
    if fused:
        _check(z, up, up_norms, d)
        if gate.shape != up.shape:
            raise ValueError("fused Gate/Up QJL requires matching output shapes")
    if not z.is_contiguous():
        z = z.contiguous()
    m, n = z.shape[0], gate.shape[0]
    out = torch.empty((2, m, n) if fused else (m, n),
                      device=z.device, dtype=torch.float32)
    if m == 0 or n == 0:
        return out
    _packed_tiled[(triton.cdiv(n, 32), triton.cdiv(m, 16))](
        z, gate, up, gate_norms, up_norms, out,
        m, n, d, 16, 32, 128, fused, num_warps=4)
    return out


def packed_qjl(z: torch.Tensor, packed: torch.Tensor,
               norms: torch.Tensor, d: int) -> torch.Tensor:
    """FP32 scaled QJL output with direct packed decode."""
    # The original FP32 reduction is faster for short expert batches and
    # small projection dimensions on the measured Blackwell GPU.
    if z.shape[0] <= 20 or d <= 256:
        return packed_qjl_legacy(z, packed, norms, d)
    return _run_tiled(z, packed, norms, packed, norms, d, False)


def packed_qjl_fused_gate_up(z: torch.Tensor, gate: torch.Tensor,
                             gate_norms: torch.Tensor, up: torch.Tensor,
                             up_norms: torch.Tensor, d: int):
    """Fuse Gate/Up for large batches; retain the faster short-batch path."""
    if z.shape[0] <= 20 or d <= 256:
        return (packed_qjl_legacy(z, gate, gate_norms, d),
                packed_qjl_legacy(z, up, up_norms, d))
    outputs = _run_tiled(z, gate, gate_norms, up, up_norms, d, True)
    return outputs[0], outputs[1]
