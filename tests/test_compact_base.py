import math

import pytest
import torch
import numpy as np

from rescode.qjl.compact_base import (CompactBase, convert_base_payload,
                                       encode_base, encoded_bytes,
                                       rounded_bf16_bits, storage_payload)
from rescode.qjl.core import PackedResidual, ProjectionBank, pack_residual
from rescode.qjl.exponent_model import fit_alpha
from rescode.qjl.packed_exponent import (BLOCK_FORMAT, decode_exponents,
                                         encode_exponents, resident_bytes)
from scripts.QJL.convert_exponent_checkpoint import convert


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


@pytest.mark.parametrize("block_size", (32, 128, 512))
@pytest.mark.parametrize("widths", ((3,), (4,), (5,), (3, 4, 5)))
def test_block_palette_exponents_lossless(block_size, widths):
    rng = np.random.default_rng(40)
    common = rng.choice(np.arange(112, 128, dtype=np.uint8), 1103)
    common[::37] = rng.integers(0, 256, len(common[::37]), dtype=np.uint8)
    encoded = encode_exponents(common, block_size, widths)
    assert np.array_equal(decode_exponents(encoded, len(common)), common)
    assert resident_bytes(encoded) == sum(len(encoded[k]) for k in
                                           ("exp_data", "exp_offsets", "exp_modes"))


@pytest.mark.parametrize("k", (1, 2, 3, 4))
def test_v1_to_v2_base_bitwise_roundtrip(k):
    edge = torch.tensor([0, 0x8000, 1, 0x007f, 0x3fff, 0x7f80, 0xff80,
                         0x7fc1], dtype=torch.uint16)
    random = torch.randn(512, dtype=torch.bfloat16).view(torch.uint16)
    weight = torch.cat((edge, random)).view(torch.bfloat16).reshape(65, 8)
    old = encode_base(weight, k, legacy=True)
    new = convert_base_payload(old, block_size=32)
    assert new["format"] == BLOCK_FORMAT
    assert "exponents" not in new
    assert torch.equal(CompactBase(new, "cpu").decode().view(torch.uint16),
                       CompactBase(old, "cpu").decode().view(torch.uint16))
    assert new["codes"] == old["codes"]
    assert new["special_bits"] == old["special_bits"]


def test_checkpoint_conversion_reuses_qjl_tensors(tmp_path):
    weight = torch.randn(16, 64, dtype=torch.bfloat16)
    old = storage_payload(encode_base(weight, 2, legacy=True))
    packed = torch.randint(0, 255, (16, 8), dtype=torch.uint8)
    norms = torch.randn(16, dtype=torch.float32)
    source = tmp_path / "source.pt"
    target = tmp_path / "converted" / "compact_qjl_state.pt"
    state = {"base": old, "packed": packed, "norms": norms,
             "alpha": 1.0}
    torch.save({"format": "rescode-exponent-base-qjl-v1",
                "config": {"projection_seed": 0},
                "layers": {0: {0: {"gate": state}}}}, source)
    report = convert(source, target, 32, (3, 4, 5))
    new = torch.load(target, weights_only=False)
    loaded = new["layers"][0][0]["gate"]
    assert new["format"] == "rescode-exponent-base-qjl-v2"
    assert torch.equal(loaded["packed"], packed)
    assert torch.equal(loaded["norms"], norms)
    assert loaded["alpha"] == 1.0
    assert torch.equal(CompactBase(loaded["base"], "cpu").decode().view(torch.uint16),
                       CompactBase(old, "cpu").decode().view(torch.uint16))
    assert report["bitwise_base_verified"]
    with pytest.raises(FileExistsError):
        convert(source, target, 32, (3, 4, 5))


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
@pytest.mark.parametrize("shape", ((1408, 2048), (2048, 1408)))
@pytest.mark.parametrize("k", (1, 2, 3, 4))
@pytest.mark.parametrize("tokens", (1, 4, 32, 128))
def test_gpu_block_palette_linear_parity(shape, k, tokens):
    torch.manual_seed(31)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.01
    x = torch.randn(tokens, shape[1], device="cuda", dtype=torch.bfloat16)
    old_payload = encode_base(weight, k, legacy=True)
    new = CompactBase(convert_base_payload(old_payload, 128), "cuda")
    old = CompactBase(old_payload, "cuda")
    assert not hasattr(new, "exponents")
    assert torch.equal(new.decode().view(torch.uint16), old.decode().view(torch.uint16))
    reference = old(x)
    actual = new(x)
    relative = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_block_palette_special_values():
    raw = torch.randint(0x3f00, 0x4000, (16, 64), device="cuda", dtype=torch.int32)
    raw[0, 0] = 0x7f80
    raw[1, 1] = 0xff80
    raw[2, 2] = 0x7fc1
    weight = raw.to(torch.uint16).view(torch.bfloat16)
    base = CompactBase(convert_base_payload(encode_base(weight, 2), 32), "cuda")
    assert torch.equal(base.decode().view(torch.uint16),
                       rounded_bf16_bits(weight, 2))
    x = torch.ones(1, 64, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(torch.isnan(base(x)), torch.isnan(torch.nn.functional.linear(x, base.decode())))


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
