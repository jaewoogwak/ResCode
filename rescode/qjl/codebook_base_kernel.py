"""Triton decode and matrix multiplication for the 2-bit codebook base."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(X, CODES, SCALES, Y, M: tl.constexpr, N: tl.constexpr,
            K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    out = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(K, BK)):
        col = step * BK + kk
        offset = out[:, None] * K + col[None, :]
        valid = (out[:, None] < N) & (col[None, :] < K)
        packed = tl.load(CODES + (offset >> 2), valid, other=0).to(tl.int32)
        code = (packed >> ((offset & 3) * 2)) & 3
        scale = tl.load(SCALES + out, out < N, other=0).to(tl.float32)
        sign = tl.where((code & 1) == 0, 1.0, -1.0)
        level = tl.where((code & 2) == 0, 0.5, 2.0)
        weight = (sign * level * scale[:, None]).to(X.dtype.element_ty)
        activation = tl.load(X + row[:, None] * K + col[None, :],
                             (row[:, None] < M) & (col[None, :] < K), other=0)
        acc = tl.dot(activation, tl.trans(weight), acc)
    tl.store(Y + row[:, None] * N + out[None, :], acc,
             (row[:, None] < M) & (out[None, :] < N))


def packed_codebook_linear(x: torch.Tensor, codes: torch.Tensor,
                           scales: torch.Tensor, shape: tuple[int, int],
                           fp32_output: bool = False):
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("GPU codebook inference requires FP16 or BF16 CUDA input")
    n, k = shape
    flat = x.reshape(-1, k).contiguous()
    y = torch.empty((flat.shape[0], n), dtype=torch.float32 if fp32_output else x.dtype,
                    device=x.device)
    _linear[(triton.cdiv(flat.shape[0], 16), triton.cdiv(n, 32))](
        flat, codes, scales, y, flat.shape[0], n, k, 16, 32, 32,
        num_warps=4)
    return y.reshape(*x.shape[:-1], n)
