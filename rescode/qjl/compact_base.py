"""Lossless compact BF16 base storage with GPU-resident packed exponents."""

import math
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .packed_exponent import BLOCK_FORMAT, decode_exponents, encode_exponents


LEGACY_FORMAT = "bf16-rounded-sign-mantissa-zlib-exponent-v1"


def _byte_keys(payload: dict) -> tuple[str, ...]:
    if payload.get("format") == LEGACY_FORMAT:
        return ("codes", "exponents", "special_indices", "special_bits")
    if payload.get("format") == BLOCK_FORMAT:
        return ("codes", "exp_data", "exp_offsets", "exp_modes",
                "special_indices", "special_bits")
    raise ValueError("unknown compact-base format")


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


def encode_base(weight: torch.Tensor, k: int, *, legacy: bool = False,
                block_size: int = 4096) -> dict:
    """Encode a BF16 base; special values retain their complete original bits."""
    rounded = rounded_bf16_bits(weight, k).cpu().contiguous()
    raw = rounded.numpy().astype(np.uint16, copy=False).reshape(-1)
    exponent = ((raw >> 7) & 255).astype(np.uint8)
    counts = np.bincount(exponent, minlength=256)
    probability = counts[counts > 0] / len(exponent)
    entropy = float(-np.sum(probability * np.log2(probability)))
    special = np.flatnonzero(exponent == 255).astype(np.int32)
    result = {
        "format": LEGACY_FORMAT if legacy else BLOCK_FORMAT,
        "shape": tuple(weight.shape), "k": k,
        "codes": _pack_codes(_codes(rounded, k), k + 1),
        "special_indices": special.tobytes(),
        "special_bits": raw[special].tobytes(),
        "exponent_entropy_bits": entropy,
    }
    if legacy:
        result["exponents"] = zlib.compress(exponent.tobytes(), level=1)
    else:
        result.update(encode_exponents(exponent, block_size))
    return result


def encoded_bytes(payload: dict) -> int:
    return sum(payload[key].numel() if isinstance(payload[key], torch.Tensor)
               else len(payload[key]) for key in _byte_keys(payload))


def convert_base_payload(payload: dict, block_size: int = 4096,
                         widths: tuple[int, ...] = (3, 4, 5)) -> dict:
    """Recode only the exponent stream; keep BF16 sign/mantissa and specials."""
    payload = byte_payload(payload)
    if payload["format"] == BLOCK_FORMAT:
        return payload
    if payload["format"] != LEGACY_FORMAT:
        raise ValueError("unknown compact-base format")
    count = math.prod(payload["shape"])
    exponents = np.frombuffer(zlib.decompress(payload["exponents"]), dtype=np.uint8)
    if len(exponents) != count:
        raise ValueError("corrupt legacy exponent length")
    result = {key: value for key, value in payload.items() if key != "exponents"}
    result.update(encode_exponents(exponents, block_size, widths))
    result["format"] = BLOCK_FORMAT
    return result


def storage_payload(payload: dict) -> dict:
    """Use raw uint8 tensor storages in torch.save instead of pickle byte strings."""
    result = dict(payload)
    for key in _byte_keys(payload):
        if not isinstance(result[key], torch.Tensor):
            result[key] = torch.from_numpy(
                np.frombuffer(result[key], dtype=np.uint8).copy())
    return result


def byte_payload(payload: dict) -> dict:
    result = dict(payload)
    for key in _byte_keys(payload):
        if isinstance(result[key], torch.Tensor):
            result[key] = result[key].cpu().numpy().tobytes()
    return result


class CompactBase(nn.Module):
    """Compact GPU-resident representation; BF16 reconstruction is transient."""

    def __init__(self, payload: dict, device):
        super().__init__()
        payload = byte_payload(payload)
        self.format = payload.get("format")
        _byte_keys(payload)
        self.shape = tuple(payload["shape"])
        self.k = int(payload["k"])
        n = math.prod(self.shape)
        started = time.perf_counter()
        if len(payload["codes"]) != (n * (self.k + 1) + 7) // 8:
            raise ValueError("corrupt compact base length")
        self.register_buffer("codes", torch.from_numpy(np.frombuffer(payload["codes"], dtype=np.uint8).copy()).to(device))
        if self.format == LEGACY_FORMAT:
            exponents = np.frombuffer(zlib.decompress(payload["exponents"]), dtype=np.uint8).copy()
            if len(exponents) != n:
                raise ValueError("corrupt legacy exponent length")
            self.register_buffer("exponents", torch.from_numpy(exponents).to(device))
        else:
            # Index and dictionary bytes remain packed on GPU. No dense
            # exponent or BF16 weight buffer is retained by this module.
            offsets = np.frombuffer(payload["exp_offsets"], dtype=np.uint32)
            modes = np.frombuffer(payload["exp_modes"], dtype=np.uint8)
            if (len(offsets) != math.ceil(n / payload["exp_block_size"]) + 1
                    or len(modes) != len(offsets) - 1
                    or offsets[-1] != len(payload["exp_data"])):
                raise ValueError("corrupt exponent block index")
            self.exp_block_size = int(payload["exp_block_size"])
            for name in ("exp_data", "exp_offsets", "exp_modes"):
                self.register_buffer(name, torch.from_numpy(
                    np.frombuffer(payload[name], dtype=np.uint8).copy()).to(device))
            self.exp_offsets = self.exp_offsets.view(torch.int32)
        self.exponent_restore_seconds = time.perf_counter() - started
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
        if self.format == LEGACY_FORMAT:
            exponents = self.exponents
        else:
            encoded = {"exp_data": self.exp_data.cpu().numpy().tobytes(),
                       "exp_offsets": self.exp_offsets.cpu().numpy().tobytes(),
                       "exp_modes": self.exp_modes.cpu().numpy().tobytes(),
                       "exp_block_size": self.exp_block_size}
            exponents = torch.from_numpy(decode_exponents(encoded, n)).to(self.codes.device)
        bits = (sign << 15) | (exponents.to(torch.int32) << 7) | (mantissa << (7 - self.k))
        if self.special_indices.numel():
            bits.index_copy_(0, self.special_indices, self.special_bits)
        return bits.to(torch.uint16).view(torch.bfloat16).reshape(self.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.bfloat16 and (
                self.format == BLOCK_FORMAT or not self.special_indices.numel()):
            from .compact_base_kernel import compact_linear
            return compact_linear(x, self)
        # Exact portable path for special values and reference comparisons.
        return F.linear(x, self.decode())

    def gpu_resident_bytes(self) -> int:
        names = ("codes", "special_indices", "special_bits")
        names += (("exponents",) if self.format == LEGACY_FORMAT
                  else ("exp_data", "exp_offsets", "exp_modes"))
        return sum(getattr(self, name).numel() * getattr(self, name).element_size()
                   for name in names)
