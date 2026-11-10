"""Direct packed compact-base inference; no dense BF16 weight allocation."""

import torch
import triton
import triton.language as tl

from .packed_exponent import BLOCK_FORMAT


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


@triton.jit(do_not_specialize=["M"])
def _linear_palette(X, CODES, DATA, OFFSETS, MODES, SPECIAL_IDX,
                    SPECIAL_BITS, OUT, M, N: tl.constexpr,
                    K: tl.constexpr, CODE_BITS: tl.constexpr,
                    KEEP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                    DATA_SIZE: tl.constexpr, SPECIAL_COUNT: tl.constexpr,
                    SEARCH_STEPS: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k_range = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(K, BK)):
        kk = step * BK + k_range
        valid = (cols[:, None] < N) & (kk[None, :] < K)
        index = cols[:, None] * K + kk[None, :]

        # Sign and retained mantissa are already packed in the old checkpoint.
        sign_bit = index * CODE_BITS
        code_byte = sign_bit >> 3
        low = tl.load(CODES + code_byte, valid, other=0).to(tl.int32)
        high = tl.load(CODES + code_byte + 1,
                       valid & (code_byte + 1 < tl.cdiv(N * K * CODE_BITS, 8)),
                       other=0).to(tl.int32)
        code = ((low | (high << 8)) >> (sign_bit & 7)) & ((1 << CODE_BITS) - 1)

        block = index // BLOCK_SIZE
        local = index % BLOCK_SIZE
        start = tl.load(OFFSETS + block, valid, other=0)
        mode = tl.load(MODES + block, valid, other=0).to(tl.int32)
        bit = local * mode
        palette_bytes = 1 << mode
        palette_index = start + palette_bytes + (bit >> 3)
        compact = valid & (mode != 0)
        c0 = tl.load(DATA + palette_index,
                     compact & (palette_index < DATA_SIZE), other=0).to(tl.int32)
        c1 = tl.load(DATA + palette_index + 1,
                     compact & (palette_index + 1 < DATA_SIZE), other=0).to(tl.int32)
        palette_code = ((c0 | (c1 << 8)) >> (bit & 7)) & ((1 << mode) - 1)
        palette_exp = tl.load(DATA + start + palette_code, compact, other=0).to(tl.int32)
        literal_exp = tl.load(DATA + start + local, valid & (mode == 0),
                              other=0).to(tl.int32)
        exponent = tl.where(mode == 0, literal_exp, palette_exp)
        bf16_bits = (((code >> KEEP) << 15) | (exponent << 7)
                     | ((code & ((1 << KEEP) - 1)) << (7 - KEEP)))

        if SPECIAL_COUNT > 0:
            lo = tl.full((BN, BK), 0, tl.int32)
            hi = tl.full((BN, BK), SPECIAL_COUNT, tl.int32)
            for _ in tl.static_range(SEARCH_STEPS):
                mid = (lo + hi) >> 1
                at_mid = tl.load(SPECIAL_IDX + mid, valid & (mid < SPECIAL_COUNT),
                                 other=2147483647).to(tl.int32)
                earlier = at_mid < index
                lo = tl.where(earlier, mid + 1, lo)
                hi = tl.where(earlier, hi, mid)
            special_index = tl.load(SPECIAL_IDX + lo,
                                    valid & (lo < SPECIAL_COUNT),
                                    other=-1).to(tl.int32)
            special_bits = tl.load(SPECIAL_BITS + lo,
                                   valid & (lo < SPECIAL_COUNT), other=0)
            bf16_bits = tl.where(valid & (special_index == index),
                                 special_bits, bf16_bits)

        weight = bf16_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        activation = tl.load(X + rows[:, None] * K + kk[None, :],
                             (rows[:, None] < M) & (kk[None, :] < K), other=0)
        acc = tl.dot(activation, tl.trans(weight), acc)
    tl.store(OUT + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit(do_not_specialize=["M"])
def _linear_palette_reduce(X, CODES, DATA, OFFSETS, MODES, SPECIAL_IDX,
                           SPECIAL_BITS, OUT, M, N: tl.constexpr,
                           K: tl.constexpr, CODE_BITS: tl.constexpr,
                           KEEP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                           DATA_SIZE: tl.constexpr, SPECIAL_COUNT: tl.constexpr,
                           SEARCH_STEPS: tl.constexpr, BO: tl.constexpr,
                           BK: tl.constexpr):
    token = tl.program_id(0)
    row = tl.program_id(1) * BO + tl.arange(0, BO)
    acc = tl.full((BO,), 0, tl.float32)
    for step in range(tl.cdiv(K, BK)):
        col = step * BK + tl.arange(0, BK)
        valid = (row[:, None] < N) & (col[None, :] < K)
        index = row[:, None] * K + col[None, :]
        sign_bit = index * CODE_BITS
        code_byte = sign_bit >> 3
        low = tl.load(CODES + code_byte, valid, other=0).to(tl.int32)
        high = tl.load(CODES + code_byte + 1,
                       valid & (code_byte + 1 < tl.cdiv(N * K * CODE_BITS, 8)),
                       other=0).to(tl.int32)
        code = ((low | (high << 8)) >> (sign_bit & 7)) & ((1 << CODE_BITS) - 1)
        block = index // BLOCK_SIZE
        local = index % BLOCK_SIZE
        start = tl.load(OFFSETS + block, valid, other=0)
        mode = tl.load(MODES + block, valid, other=0).to(tl.int32)
        bit = local * mode
        palette_bytes = 1 << mode
        palette_index = start + palette_bytes + (bit >> 3)
        compact = valid & (mode != 0)
        c0 = tl.load(DATA + palette_index,
                     compact & (palette_index < DATA_SIZE), other=0).to(tl.int32)
        c1 = tl.load(DATA + palette_index + 1,
                     compact & (palette_index + 1 < DATA_SIZE), other=0).to(tl.int32)
        palette_code = ((c0 | (c1 << 8)) >> (bit & 7)) & ((1 << mode) - 1)
        palette_exp = tl.load(DATA + start + palette_code, compact, other=0).to(tl.int32)
        literal_exp = tl.load(DATA + start + local, valid & (mode == 0),
                              other=0).to(tl.int32)
        exponent = tl.where(mode == 0, literal_exp, palette_exp)
        bf16_bits = (((code >> KEEP) << 15) | (exponent << 7)
                     | ((code & ((1 << KEEP) - 1)) << (7 - KEEP)))
        if SPECIAL_COUNT > 0:
            lo = tl.full((BO, BK), 0, tl.int32)
            hi = tl.full((BO, BK), SPECIAL_COUNT, tl.int32)
            for _ in tl.static_range(SEARCH_STEPS):
                mid = (lo + hi) >> 1
                at_mid = tl.load(SPECIAL_IDX + mid, valid & (mid < SPECIAL_COUNT),
                                 other=2147483647).to(tl.int32)
                earlier = at_mid < index
                lo = tl.where(earlier, mid + 1, lo)
                hi = tl.where(earlier, hi, mid)
            special_index = tl.load(SPECIAL_IDX + lo,
                                    valid & (lo < SPECIAL_COUNT), other=-1).to(tl.int32)
            special_bits = tl.load(SPECIAL_BITS + lo,
                                   valid & (lo < SPECIAL_COUNT), other=0)
            bf16_bits = tl.where(valid & (special_index == index),
                                 special_bits, bf16_bits)
        weight = bf16_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        activation = tl.load(X + token * K + col, col < K, other=0).to(tl.float32)
        acc += tl.sum(weight * activation[None, :], axis=1)
    tl.store(OUT + token * N + row, acc, row < N)


def compact_linear(x: torch.Tensor, base, *, out_dtype=None) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.device != base.codes.device:
        raise ValueError("compact Triton linear requires CUDA BF16 inputs on the base device")
    leading = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1]).contiguous()
    n, k = base.shape
    if x.shape[1] != k:
        raise ValueError("compact linear input dimension mismatch")
    out = torch.empty((x.shape[0], n), device=x.device,
                      dtype=out_dtype if out_dtype is not None else x.dtype)
    if base.format == BLOCK_FORMAT:
        m = x.shape[0]
        special_count = base.special_indices.numel()
        common = (x, base.codes, base.exp_data, base.exp_offsets, base.exp_modes,
                  base.special_indices, base.special_bits, out,
                  m, n, k, base.k + 1, base.k, base.exp_block_size,
                  base.exp_data.numel(), special_count,
                  (special_count + 1).bit_length())
        if m <= 20:
            # Match the legacy FP32 reduction order for sparse expert calls.
            bo, bk = 16, 128
            _linear_palette_reduce[(m, triton.cdiv(n, bo))](
                *common, bo, bk, num_warps=4)
        else:
            bm, bn, bk = ((32, 64, 64) if m <= 32 else (64, 64, 64))
            _linear_palette[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                *common, bm, bn, bk, num_warps=4)
    else:
        _linear[(x.shape[0], triton.cdiv(n, 16))](
            x, base.codes, base.exponents, out, x.shape[0], n, k,
            base.k + 1, base.k, 16, 128, num_warps=4)
    return out.reshape(*leading, n)
