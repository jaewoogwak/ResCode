"""Lossless, independently addressable blocks of BF16 exponent bytes.

Each block is either literal uint8 bytes or a small palette followed by packed
3/4/5-bit palette indices. Offsets and modes make every weight tile accessible
without decoding preceding blocks. The palette is local to a block, so rare
exponents never require an unbounded escape scan at inference time.
"""

from __future__ import annotations

import math

import numpy as np


BLOCK_FORMAT = "bf16-rounded-sign-mantissa-block-palette-v2"
WIDTHS = (3, 4, 5)


def _pack_codes(codes: np.ndarray, width: int) -> bytes:
    shifts = np.arange(width, dtype=np.uint8)
    bits = ((codes[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits, bitorder="little").tobytes()


def _unpack_codes(packed: bytes, count: int, width: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    bits = bits[:count * width].reshape(count, width)
    return (bits.astype(np.uint16) * (1 << np.arange(width, dtype=np.uint16))).sum(1)


def encode_exponents(exponents: np.ndarray, block_size: int = 512,
                     widths: tuple[int, ...] = WIDTHS) -> dict:
    """Choose the smallest exact representation for each independent block."""
    if block_size < 32 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two >= 32")
    if not widths or any(w not in WIDTHS for w in widths):
        raise ValueError("widths must be a nonempty subset of (3, 4, 5)")
    exponents = np.asarray(exponents, dtype=np.uint8).reshape(-1)
    offsets = [0]
    modes = bytearray()
    pieces = []
    for start in range(0, len(exponents), block_size):
        block = exponents[start:start + block_size]
        # BF16 exponents have only 256 possible symbols.  A histogram gives
        # the same sorted palette as np.unique without sorting every block.
        palette = np.flatnonzero(np.bincount(block, minlength=256)).astype(np.uint8)
        options = [(len(block), 0)]
        options += [((1 << width) + math.ceil(len(block) * width / 8), width)
                    for width in widths if len(palette) <= 1 << width]
        _, mode = min(options)
        if mode:
            lookup = np.empty(256, dtype=np.uint8)
            lookup[palette] = np.arange(len(palette), dtype=np.uint8)
            inverse = lookup[block]
            pad = np.zeros(1 << mode, dtype=np.uint8)
            pad[:len(palette)] = palette
            piece = pad.tobytes() + _pack_codes(inverse, mode)
        else:
            piece = block.tobytes()
        pieces.append(piece)
        offsets.append(offsets[-1] + len(piece))
        modes.append(mode)
    if offsets[-1] >= 2 ** 32:
        raise ValueError("encoded exponent stream exceeds uint32 offset range")
    return {"exp_data": b"".join(pieces),
            "exp_offsets": np.asarray(offsets, dtype=np.uint32).tobytes(),
            "exp_modes": bytes(modes), "exp_block_size": block_size}


def decode_exponents(encoded: dict, count: int) -> np.ndarray:
    data = np.frombuffer(encoded["exp_data"], dtype=np.uint8)
    offsets = np.frombuffer(encoded["exp_offsets"], dtype=np.uint32)
    modes = np.frombuffer(encoded["exp_modes"], dtype=np.uint8)
    block_size = int(encoded["exp_block_size"])
    blocks = math.ceil(count / block_size)
    if (len(offsets) != blocks + 1 or len(modes) != blocks
            or offsets[0] != 0 or offsets[-1] != len(data)
            or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("corrupt exponent block index")
    out = np.empty(count, dtype=np.uint8)
    for i, mode in enumerate(modes):
        n = min(block_size, count - i * block_size)
        part = data[offsets[i]:offsets[i + 1]]
        if mode == 0:
            if len(part) != n:
                raise ValueError("corrupt literal exponent block")
            out[i * block_size:i * block_size + n] = part
        elif mode in WIDTHS:
            width = int(mode)
            palette_size = 1 << width
            if len(part) != palette_size + math.ceil(n * width / 8):
                raise ValueError("corrupt palette exponent block")
            codes = _unpack_codes(part[palette_size:].tobytes(), n, width)
            out[i * block_size:i * block_size + n] = part[:palette_size][codes]
        else:
            raise ValueError(f"unknown exponent block mode: {mode}")
    return out


def resident_bytes(encoded: dict) -> int:
    return sum(len(encoded[key]) for key in ("exp_data", "exp_offsets", "exp_modes"))
