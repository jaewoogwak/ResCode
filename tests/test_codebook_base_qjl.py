"""Focused tests for the independent 2-bit Codebook Base + QJL method."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.qjl.codebook_base import PackedCodebookBase, encode_base, pack_codes, unpack_codes
from rescode.qjl.codebook_model import (CodebookExpert, encode_projection,
                                         load_checkpoint, save_checkpoint)
from rescode.qjl.core import ProjectionBank, pack_residual, qjl_reference, qjl_triton


def legacy_fit(weight):
    """Original four-start alternating update, kept as a regression oracle."""
    w = weight.float()
    magnitude = w.abs()
    active = magnitude != 0
    mean = magnitude.mean(1)
    rms = magnitude.square().mean(1).sqrt()
    initials = torch.stack((mean, mean * 2, mean * 0.5, rms), dim=0).clamp_min(1e-12)
    best_error = torch.full_like(mean, float("inf"))
    best_alpha = torch.zeros_like(mean, dtype=torch.float16)
    best_index = torch.zeros_like(magnitude, dtype=torch.bool)
    for alpha in initials:
        for _ in range(8):
            index = magnitude > (1.25 * alpha[:, None])
            level = torch.where(index, 2.0, 0.5) * active
            alpha = (magnitude * level).sum(1) / level.square().sum(1).clamp_min(1e-30)
        stored = alpha.to(torch.float16)
        index = magnitude > (1.25 * stored.float()[:, None])
        level = torch.where(index, 2.0, 0.5) * active
        error = (magnitude - stored.float()[:, None] * level).square().sum(1)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_alpha = torch.where(better, stored, best_alpha)
        best_index = torch.where(better[:, None], index, best_index)
    codes = (w < 0).to(torch.uint8) | (best_index.to(torch.uint8) << 1)
    zeros = torch.nonzero(((~active) & (best_alpha.float()[:, None] != 0)).reshape(-1),
                          as_tuple=False).reshape(-1).to(torch.int32)
    return {"packed": pack_codes(codes), "scales": best_alpha, "zero_indices": zeros}


def test_vectorized_fit_matches_legacy():
    torch.manual_seed(314)
    weight = (torch.randn(19, 64) * 0.1).to(torch.bfloat16)
    weight[0, :3] = 0
    weight[7, 23] = 0
    weight[12] = 0
    old = legacy_fit(weight)
    for chunk in (3, 8, 19):
        new = encode_base(weight, row_chunk=chunk)
        for key in ("packed", "scales", "zero_indices"):
            assert torch.equal(new[key], old[key])


def test_codebook_fit_and_residual_identity():
    torch.manual_seed(17)
    sign = torch.where(torch.rand(3, 8) > 0.5, 1.0, -1.0)
    level = torch.where(torch.rand(3, 8) > 0.5, 2.0, 0.5)
    weight = (sign * level * torch.tensor([1.0, 0.75, 1.5])[:, None]).to(torch.bfloat16)
    state = encode_base(weight, row_chunk=2)
    base = PackedCodebookBase(state)
    decoded = base.decode()
    assert torch.equal(unpack_codes(state["packed"], 3, 8),
                       unpack_codes(pack_codes(unpack_codes(state["packed"], 3, 8)), 3, 8))
    assert torch.allclose(decoded, weight.float(), atol=0.01)
    residual = weight.float() - decoded
    assert torch.allclose(decoded + residual, weight.float(), atol=1e-6)
    ratio = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(weight.float())
    assert ratio < 0.01
    # Every row chooses its own scale and scale-index pattern.
    assert not torch.equal(base.scales[0], base.scales[1])


def test_fit_improves_over_single_level():
    weight = torch.tensor([[0.5, -0.5, 2.0, -2.0], [1.0, 0.0, 4.0, -4.0]],
                          dtype=torch.bfloat16)
    state = encode_base(weight)
    decoded = PackedCodebookBase(state).decode()
    single = weight.float().sign() * weight.float().abs().mean(1, keepdim=True)
    assert (weight.float() - decoded).square().sum() < (weight.float() - single).square().sum()
    assert decoded[1, 1] == 0 and state["zero_indices"].numel() == 1


def test_packed_base_and_qjl_triton():
    device = torch.device("cuda")
    torch.manual_seed(6)
    weight = (torch.randn(48, 64, device=device) * 0.05).to(torch.bfloat16)
    weight[0, 0] = 0
    weight[1, 3] = 0
    bank = ProjectionBank(64, 9)
    base, qjl, state, record = encode_projection(weight, bank, "triton", row_chunk=16)
    old_fit = legacy_fit(weight)
    for key in ("packed", "scales", "zero_indices"):
        assert torch.equal(state["base"][key], old_fit[key].cpu())
    old_residual = weight.float() - PackedCodebookBase(
        {**state["base"], **{key: old_fit[key].cpu() for key in old_fit}}, device).decode()
    old_packed, old_norms = pack_residual(old_residual, bank, row_chunk=16)
    assert torch.equal(old_packed, qjl.packed)
    assert torch.equal(old_norms, qjl.norms)
    _, _, split_state, split_record = encode_projection(
        weight, bank, "triton", base_row_chunk=32, qjl_row_chunk=16)
    assert torch.equal(split_state["base"]["packed"], state["base"]["packed"])
    assert torch.equal(split_state["packed"], state["packed"])
    assert torch.equal(split_state["norms"], state["norms"])
    timing_keys = ("time_encode_base_s", "time_residual_s", "time_pack_residual_s",
                   "time_diagnostics_s", "time_cpu_transfer_s")
    assert all(split_record[key] >= 0 for key in timing_keys)
    assert abs(sum(split_record[key] for key in timing_keys)
               - split_record["time_total_s"]) < 1e-8
    x = torch.randn(5, 64, device=device, dtype=torch.bfloat16)
    actual = base(x).float()
    expected = F.linear(x, base.decode().to(x.dtype)).float()
    assert torch.allclose(actual, expected, atol=0.05, rtol=0.02)
    fp32_reference = F.linear(x.float(), base.decode()).float()
    assert torch.allclose(actual, fp32_reference, atol=0.05, rtol=0.02)
    z = x.to(torch.float16) @ bank.get(64, device)
    reference = qjl_reference(z, qjl.packed, qjl.norms, 64)
    triton = qjl_triton(z, qjl.packed, qjl.norms, 64)
    assert torch.allclose(reference, triton, atol=0.01, rtol=0.01)
    assert record["base_bytes"] == (state["base"]["packed"].numel() + 2 * 48
                                    + 4 * state["base"]["zero_indices"].numel())


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(64, 60, bias=False, dtype=torch.bfloat16)
        self.experts = nn.ModuleList([TinyExpert() for _ in range(60)])
        self.top_k = 4
        self.norm_topk_prob = False


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = TinyBlock()


def test_checkpoint_reload_exact_forward_parity():
    device = torch.device("cuda")
    torch.manual_seed(25)
    bank64 = ProjectionBank(64, 11)
    bank32 = ProjectionBank(32, 11)
    expert = TinyExpert().to(device)
    expert.gate_proj.weight.data[0, 0] = 0
    expert.gate_proj.weight.data[0, 1] = 0
    bases, residuals, states = {}, {}, {}
    for key, dim in (("gate", 64), ("up", 64), ("down", 32)):
        weight = getattr(expert, key + "_proj").weight.detach()
        base, residual, state, _ = encode_projection(
            weight, bank64 if dim == 64 else bank32, "triton", row_chunk=16)
        bases[key], residuals[key], states[key] = base, residual, state
    before = CodebookExpert(bases, residuals)
    x = torch.randn(3, 64, device=device, dtype=torch.bfloat16)
    expected = before(x)
    config = {"layers": [0], "qjl_d_factor": 1.0, "projection_seed": 11,
              "backend": "triton", "iterations": 8, "row_chunk": 16}
    data = {0: {i: states for i in range(60)}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.pt"
        sizes = save_checkpoint(data, config, path)
        assert sizes["checkpoint_bytes"] == path.stat().st_size
        model = TinyModel().to(device)
        router = model.model.layers[0].mlp.gate
        load_checkpoint(model, path)
        block = model.model.layers[0].mlp
        assert block.gate is router and block.top_k == 4
        actual = block._compact_experts[0](x)
        assert torch.equal(actual, expected)
        routed, logits = block(x)
        assert torch.equal(logits, router(x))
        probabilities = F.softmax(logits.float(), dim=-1)
        top_probabilities = probabilities.topk(4, dim=-1).values.to(x.dtype)
        expected_routed = (expected * top_probabilities.sum(dim=-1, keepdim=True)).to(x.dtype)
        assert torch.allclose(routed, expected_routed, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    test_vectorized_fit_matches_legacy()
    test_codebook_fit_and_residual_identity()
    test_fit_improves_over_single_level()
    print("CPU codebook tests: passed")
    if torch.cuda.is_available():
        test_packed_base_and_qjl_triton()
        test_checkpoint_reload_exact_forward_parity()
        print("CUDA packed Base, QJL and checkpoint tests: passed")
