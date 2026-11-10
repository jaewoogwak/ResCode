"""Correctness checks for sign/exponent plus row-mantissa Base and QJL."""

import tempfile
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.qjl.codebook_model import (CodebookExpert, encode_projection,
    gpu_resident_breakdown, load_checkpoint, save_checkpoint)
from rescode.qjl.compact_base import CompactBase, byte_payload, encode_base as encode_compact_base
from rescode.qjl.core import ProjectionBank, qjl_reference, qjl_triton
from rescode.qjl.exponent_mantissa_base import (PackedExponentMantissaBase,
    encode_exponent_mantissa_base, exponent_mantissa_base_bytes, fit_row_beta,
    reconstruct_base_from_weight)
from rescode.qjl.packed_exponent import decode_exponents, encode_exponents, _pack_codes


def test_exponent_encoder_legacy_byte_parity():
    """The faster encoder must preserve every field of the checkpoint format."""
    rng = np.random.default_rng(4)
    arrays = [np.array([], dtype=np.uint8),
              rng.integers(0, 256, 8193, dtype=np.uint8),
              rng.integers(115, 124, 8193, dtype=np.uint8),
              rng.integers(0, 20, 8193, dtype=np.uint8),
              np.full(8193, 126, dtype=np.uint8)]
    for values in arrays:
        for block_size in (32, 512, 4096):
            offsets, modes, pieces = [0], bytearray(), []
            for start in range(0, len(values), block_size):
                block = values[start:start + block_size]
                palette, inverse = np.unique(block, return_inverse=True)
                options = [(len(block), 0)]
                options += [((1 << width) + math.ceil(len(block) * width / 8), width)
                            for width in (3, 4, 5) if len(palette) <= 1 << width]
                _, mode = min(options)
                if mode:
                    pad = np.zeros(1 << mode, dtype=np.uint8)
                    pad[:len(palette)] = palette
                    piece = pad.tobytes() + _pack_codes(inverse.astype(np.uint8), mode)
                else:
                    piece = block.tobytes()
                pieces.append(piece)
                offsets.append(offsets[-1] + len(piece))
                modes.append(mode)
            old = {"exp_data": b"".join(pieces),
                   "exp_offsets": np.asarray(offsets, dtype=np.uint32).tobytes(),
                   "exp_modes": bytes(modes), "exp_block_size": block_size}
            new = encode_exponents(values, block_size)
            assert old == new
            assert np.array_equal(decode_exponents(new, len(values)), values)


def test_fast_reconstruction_parity():
    raw = torch.tensor([[0x3f00, 0xbf40, 0x0000, 0x8000, 0x0001, 0x8002,
                         0x7f80, 0xff80, 0x7fc1, 0xffc1]], dtype=torch.uint16)
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        weight = raw.view(torch.bfloat16).to(device)
        state = encode_exponent_mantissa_base(weight, block_size=32)
        packed = PackedExponentMantissaBase(state, device)
        fast = reconstruct_base_from_weight(weight, packed.beta)
        slow = packed.decode()
        assert torch.equal(fast.view(torch.int32), slow.view(torch.int32))
        torch.manual_seed(15)
        regular = torch.randn(19, 65, device=device).to(torch.bfloat16)
        state = encode_exponent_mantissa_base(regular)
        packed = PackedExponentMantissaBase(state, device)
        fast = reconstruct_base_from_weight(regular, packed.beta)
        slow = packed.decode()
        assert torch.equal(fast, slow)


def test_bits_beta_and_special_values():
    raw = torch.tensor([[0x3f00, 0xbf40, 0x4000, 0x0000, 0x8000],
                        [0x0001, 0x8002, 0x3f80, 0x7f80, 0xffc1]], dtype=torch.uint16)
    weight = raw.view(torch.bfloat16)
    state = encode_exponent_mantissa_base(weight, block_size=32)
    repeated = encode_exponent_mantissa_base(weight, block_size=32)
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, repeated[key]), key
        else:
            assert value == repeated[key], key
    exponent = decode_exponents(byte_payload(state), weight.numel())
    assert np.array_equal(exponent, ((raw.numpy().reshape(-1) >> 7) & 255).astype(np.uint8))
    signs = np.unpackbits(state["codes"].numpy(), bitorder="little")[:weight.numel()]
    assert np.array_equal(signs, (raw.numpy().reshape(-1) >> 15).astype(np.uint8))
    base = PackedExponentMantissaBase(state)
    assert torch.isfinite(state["beta"]).all()
    assert base.decode()[0, 3] == 0 and torch.signbit(base.decode()[0, 4])
    assert torch.isinf(base.decode()[1, 3]) and torch.isnan(base.decode()[1, 4])
    assert base.compact.decode()[1, 0] == torch.tensor(2.0 ** -126)
    assert exponent_mantissa_base_bytes(state) <= base.resident_bytes()
    assert all(tuple(buffer.shape) != tuple(weight.shape) for buffer in base.buffers())

    finite = torch.tensor([[0.5, -0.75, 1.0, -1.5],
                           [0.25, -0.375, 0.5, -0.75]], dtype=torch.bfloat16)
    fitted = fit_row_beta(finite)
    bits = finite.view(torch.uint16).int()
    scale = torch.exp2((((bits >> 7) & 255) - 127).float())
    expected = (finite.float().abs() * scale).sum(1) / scale.square().sum(1)
    assert torch.allclose(fitted, expected, atol=1e-6)
    for candidate in (fitted * 0.95, fitted * 1.05):
        assert ((finite.float().abs() - fitted[:, None] * scale).square().sum(1)
                <= (finite.float().abs() - candidate[:, None] * scale).square().sum(1)).all()
    decoded = PackedExponentMantissaBase(encode_exponent_mantissa_base(finite)).decode()
    residual = finite.float() - decoded
    assert torch.allclose(decoded + residual, finite.float(), atol=1e-6)


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        block = nn.Module()
        block.gate = nn.Linear(64, 60, bias=False, dtype=torch.bfloat16)
        block.experts = nn.ModuleList(TinyExpert() for _ in range(60))
        block.top_k = 4
        block.norm_topk_prob = False
        self.model.layers[0].mlp = block


def test_cuda_packed_inference_and_checkpoint():
    if not torch.cuda.is_available():
        print("CUDA unavailable: packed inference test skipped")
        return
    device = "cuda"
    torch.manual_seed(102)
    old_weight = (torch.randn(32, 64, device=device) * 0.01).to(torch.bfloat16)
    old_base = CompactBase(encode_compact_base(old_weight, 2), device)
    old_input = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    assert torch.allclose(old_base(old_input), F.linear(old_input, old_base.decode()),
                          atol=0.02, rtol=0.02)
    expert = TinyExpert().to(device)
    expert.gate_proj.weight.data[0, 0] = 0
    bank64, bank32 = ProjectionBank(64, 11), ProjectionBank(32, 11)
    bases, residuals, states = {}, {}, {}
    for key, dim in (("gate", 64), ("up", 64), ("down", 32)):
        weight = getattr(expert, key + "_proj").weight.detach()
        base, residual, state, record = encode_projection(
            weight, bank64 if dim == 64 else bank32, "triton",
            qjl_row_chunk=16, base_method="exponent-mantissa")
        assert record["base_bytes"] == exponent_mantissa_base_bytes(state["base"])
        x = torch.randn(5, dim, device=device, dtype=torch.bfloat16)
        reference = F.linear(x.float(), base.decode())
        assert torch.allclose(base(x).float(), reference, atol=0.06, rtol=0.03)
        projection = bank64 if dim == 64 else bank32
        z = x.half() @ projection.get(dim, torch.device(device))
        expected = qjl_reference(z, residual.packed, residual.norms, dim)
        actual = qjl_triton(z, residual.packed, residual.norms, dim)
        assert torch.allclose(actual, expected, atol=0.01, rtol=0.01)
        bases[key], residuals[key], states[key] = base, residual, state
    before = CodebookExpert(bases, residuals)
    x = torch.randn(3, 64, device=device, dtype=torch.bfloat16)
    y = before(x)
    config = {"layers": [0], "qjl_d_factor": 1.0, "projection_seed": 11,
              "backend": "triton", "iterations": 8, "row_chunk": 16,
              "base_method": "exponent-mantissa"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "compact.pt"
        data = {0: {i: states for i in range(60)}}
        sizes = save_checkpoint(data, config, path)
        assert sizes["checkpoint_bytes"] == path.stat().st_size
        model = TinyModel().to(device)
        router = model.model.layers[0].mlp.gate
        load_checkpoint(model, path)
        block = model.model.layers[0].mlp
        assert block.gate is router and block.top_k == 4
        assert torch.equal(block._compact_experts[0](x), y)
        resident = gpu_resident_breakdown(model)
        assert resident["gpu_resident_base_bytes"] > 0
        assert resident["gpu_resident_qjl_bytes"] > 0


if __name__ == "__main__":
    test_exponent_encoder_legacy_byte_parity()
    test_fast_reconstruction_parity()
    print("Exponent encoder bytes and fast reconstruction parity: passed")
    test_bits_beta_and_special_values()
    print("BF16 exponent/sign, beta and reconstruction tests: passed")
    test_cuda_packed_inference_and_checkpoint()
    print("GPU packed Base, QJL and checkpoint tests: passed")
