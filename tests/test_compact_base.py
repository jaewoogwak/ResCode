import math

import pytest
import torch

from rescode.qjl.compact_base import (CompactBase, encode_base, encoded_bytes,
                                       rounded_bf16_bits, storage_payload)
from rescode.qjl.core import PackedResidual, ProjectionBank, pack_residual
from rescode.qjl.exponent_model import fit_alpha


@pytest.mark.parametrize("k", (1, 2, 3, 4))
def test_compact_roundtrip_specials_and_carry(k):
    raw = torch.tensor([0x0000, 0x8000, 0x0001, 0x007f, 0x3f90,
                        0x3fb0, 0x3fff, 0x7f80, 0xff80, 0x7fc1],
                       dtype=torch.uint16)
    weight = raw.view(torch.bfloat16).reshape(2, 5)
    payload = encode_base(weight, k)
    decoded = CompactBase(payload, "cpu").decode()
    assert torch.equal(decoded.view(torch.uint16), rounded_bf16_bits(weight, k))
    assert decoded.view(torch.uint16).flatten()[1] == 0x8000
    assert decoded.view(torch.uint16).flatten()[-1] == 0x7fc1
    assert encoded_bytes(payload) < weight.numel() * 2 + 100
    assert 0 <= payload["exponent_entropy_bits"] <= 8


def test_round_to_even_and_subnormal_carry():
    raw = torch.tensor([0x3f90, 0x3fb0, 0x007f, 0x3fff], dtype=torch.uint16)
    result = rounded_bf16_bits(raw.view(torch.bfloat16), 2)
    assert result.tolist() == [0x3f80, 0x3fc0, 0x0080, 0x4000]


def test_tensor_checkpoint_storage_roundtrip(tmp_path):
    weight = torch.randn(128, 256, dtype=torch.bfloat16)
    payload = encode_base(weight, 2)
    path = tmp_path / "base.pt"
    torch.save(storage_payload(payload), path)
    restored = torch.load(path, weights_only=False)
    assert torch.equal(CompactBase(restored, "cpu").decode().view(torch.uint16),
                       rounded_bf16_bits(weight, 2))
    assert path.stat().st_size < encoded_bytes(payload) + 5000


def test_calibrated_alpha_least_squares_and_zero_denominator():
    qjl = torch.tensor([1.0, -2.0, 3.0])
    assert fit_alpha(0.7 * qjl, qjl) == pytest.approx(0.7)
    assert fit_alpha(5 * qjl, qjl) == 5.0
    assert fit_alpha(-qjl, qjl) == -1.0
    assert fit_alpha(qjl, torch.zeros_like(qjl)) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", ((32, 2048), (32, 1408)))
@pytest.mark.parametrize("k", (1, 2, 3, 4))
def test_direct_packed_linear_matches_bf16(shape, k):
    torch.manual_seed(17)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4, shape[1], device="cuda", dtype=torch.bfloat16)
    base = CompactBase(encode_base(weight, k), "cuda")
    assert torch.equal(base.decode().view(torch.uint16), rounded_bf16_bits(weight, k))
    actual = base(x)
    reference = torch.nn.functional.linear(x, base.decode())
    relative = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("input_dim", (2048, 1408))
@pytest.mark.parametrize("factor", (1, 2))
def test_qjl_reference_and_triton_for_projection_dimensions(input_dim, factor):
    torch.manual_seed(23)
    weight = torch.randn(12, input_dim, device="cuda", dtype=torch.float32) * 0.01
    x = torch.randn(4, input_dim, device="cuda", dtype=torch.bfloat16)
    bank = ProjectionBank(input_dim * factor, seed=0)
    packed, norms = pack_residual(weight, bank)
    reference = PackedResidual(packed, norms, input_dim, bank.d, bank, "reference")(x)
    triton = PackedResidual(packed, norms, input_dim, bank.d, bank, "triton")(x)
    relative = (reference.float() - triton.float()).norm() / reference.float().norm()
    assert math.isfinite(relative.item())
    assert relative < 5e-3
