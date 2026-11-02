"""Direct packed compact-base inference; no dense BF16 weight allocation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(X, CODES, EXP, OUT, M: tl.constexpr, N: tl.constexpr,
            K: tl.constexpr, BITS: tl.constexpr, KEEP: tl.constexpr,
            BO: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(0)
    row = tl.program_id(1) * BO + tl.arange(0, BO)
    acc = tl.full((BO,), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        col = block * BK + tl.arange(0, BK)
        valid = (row[:, None] < N) & (col[None, :] < K)
        offset = row[:, None] * K + col[None, :]
        bit_offset = offset * BITS
        byte_idx = bit_offset // 8
        low = tl.load(CODES + byte_idx, valid, other=0).to(tl.int32)
        # The final byte can be read only where it exists. A code may straddle
        # two bytes; for the last code an absent high byte contributes zeros.
        high = tl.load(CODES + byte_idx + 1,
                       valid & (byte_idx + 1 < tl.cdiv(N * K * BITS, 8)),
                       other=0).to(tl.int32)
        code = ((low | (high << 8)) >> (bit_offset % 8)) & ((1 << BITS) - 1)
        exponent = tl.load(EXP + offset, valid, other=0).to(tl.int32)
        bf16_bits = (((code >> KEEP) << 15) | (exponent << 7)
                     | ((code & ((1 << KEEP) - 1)) << (7 - KEEP)))
        weight = bf16_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        x = tl.load(X + token * K + col, col < K, other=0).to(tl.float32)
        acc += tl.sum(weight * x[None, :], axis=1)
    tl.store(OUT + token * N + row, acc, row < N)


def compact_linear(x: torch.Tensor, base) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.device != base.codes.device:
        raise ValueError("compact Triton linear requires CUDA BF16 inputs on the base device")
    leading = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1]).contiguous()
    n, k = base.shape
    if x.shape[1] != k:
        raise ValueError("compact linear input dimension mismatch")
    out = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
    _linear[(x.shape[0], triton.cdiv(n, 16))](
        x, base.codes, base.exponents, out, x.shape[0], n, k,
        base.k + 1, base.k, 16, 128, num_warps=4)
    return out.reshape(*leading, n)
