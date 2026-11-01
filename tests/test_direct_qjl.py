import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.qjl.base import fit_shared_base, variance_objective
from rescode.qjl.core import (ProjectionBank, PackedResidual, pack_residual,
                              pack_signs, unpack_signs, qjl_reference)
from rescode.qjl.model import convert_model, load_qjl_checkpoint, save_qjl_checkpoint


class ToyExpert(nn.Module):
    def __init__(self, hidden=7, middle=11):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, middle, bias=False)
        self.up_proj = nn.Linear(hidden, middle, bias=False)
        self.down_proj = nn.Linear(middle, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(7, 4, bias=False)
        self.experts = nn.ModuleList([ToyExpert() for _ in range(4)])
        self.shared_expert = ToyExpert()
        self.shared_expert_gate = nn.Linear(7, 1, bias=False)
        self.top_k = 2
        self.norm_topk_prob = False

    def forward(self, hidden):
        flat = hidden.reshape(-1, 7)
        logits = self.gate(flat)
        rw, ids = torch.topk(F.softmax(logits.float(), dim=-1), 2, dim=-1)
        y = torch.zeros_like(flat)
        for j in range(4):
            token, slot = torch.where(ids == j)
            if token.numel():
                y.index_add_(0, token, self.experts[j](flat[token]) * rw[token, slot, None])
        y += torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return y.reshape(hidden.shape), logits


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        layer.mlp = ToyBlock()
        self.model.layers = nn.ModuleList([layer])


class ToyMixtralExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(7, 11, bias=False)
        self.w3 = nn.Linear(7, 11, bias=False)
        self.w2 = nn.Linear(11, 7, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ToyMixtralBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(7, 4, bias=False)
        self.experts = nn.ModuleList([ToyMixtralExpert() for _ in range(4)])
        self.top_k = 2

    def forward(self, hidden):
        flat = hidden.reshape(-1, 7)
        logits = self.gate(flat)
        rw, ids = torch.topk(F.softmax(logits.float(), dim=-1), 2, dim=-1)
        rw /= rw.sum(-1, keepdim=True)
        y = torch.zeros_like(flat)
        for j in range(4):
            token, slot = torch.where(ids == j)
            if token.numel():
                y.index_add_(0, token, self.experts[j](flat[token]) * rw[token, slot, None])
        return y.reshape(hidden.shape), logits


class ToyMixtralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        layer.block_sparse_moe = ToyMixtralBlock()
        self.model.layers = nn.ModuleList([layer])


def synthetic_samples(model, n=64):
    x = torch.randn(n, 7)
    layer = model.model.layers[0]
    mixtral = hasattr(layer, "block_sparse_moe")
    block = layer.block_sparse_moe if mixtral else layer.mlp
    gate = F.softmax(block.gate(x), -1)
    weights, ids = torch.topk(gate, 2, -1)
    if mixtral:
        weights /= weights.sum(-1, keepdim=True)
    samples = {0: {"gate": {}, "up": {}, "down": {}}}
    for eid, expert in enumerate(block.experts):
        token, slot = torch.where(ids == eid)
        h = x[token]
        q = weights[token, slot].square()
        middle = (F.silu(expert.w1(h)) * expert.w3(h) if mixtral
                  else F.silu(expert.gate_proj(h)) * expert.up_proj(h))
        for key in ("gate", "up"):
            samples[0][key][eid] = (h, q)
        samples[0]["down"][eid] = (middle, q)
    return samples


class DirectQJLTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_pack_and_reference(self):
        bits = torch.rand(5, 19) > 0.5
        self.assertTrue(torch.equal(unpack_signs(pack_signs(bits), 19), bits))
        w = torch.randn(5, 7)
        w[0].zero_()
        bank = ProjectionBank(19, seed=17)
        packed, norms = pack_residual(w, bank)
        self.assertEqual(tuple(packed.shape), (5, 3))
        x = torch.randn(4, 7)
        z = x @ bank.get(7, x.device).float()
        y = qjl_reference(z, packed, norms, 19)
        self.assertEqual(tuple(y.shape), (4, 5))
        self.assertEqual(float(y[:, 0].abs().max()), 0)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.equal(bank.get(7, x.device), ProjectionBank(19, 17).get(7, x.device)))

    def test_variance_statistical(self):
        # Independent Gaussian projections; theoretical identity is exact for
        # the real-valued Gaussian estimator, with Monte Carlo sampling error.
        w = torch.tensor([0.8, -1.2, 0.4])
        x = torch.tensor([-0.5, 0.3, 1.1])
        d = 29
        p = torch.randn(6000, 3, d)
        estimates = ((x[None, :, None] * p).sum(1)
                     * (w[None, :, None] * p).sum(1).sign()).mean(-1)
        estimates *= math.sqrt(math.pi / 2) * w.norm()
        measured = float((estimates - w.dot(x)).square().mean())
        theory = float((math.pi / 2 * x.square().sum() * w.square().sum()
                        - w.dot(x).square()) / d)
        self.assertLess(abs(measured / theory - 1), 0.1)

    def test_packed_variance_across_seeds(self):
        w = torch.tensor([[0.8, -1.2, 0.4]])
        x = torch.tensor([[-0.5, 0.3, 1.1]])
        d = 31
        errors = []
        for seed in range(800):
            bank = ProjectionBank(d, seed)
            packed, norms = pack_residual(w, bank)
            z = (x.half() @ bank.get(3, x.device)).float()
            estimate = qjl_reference(z, packed, norms, d)
            errors.append(float((estimate - x @ w.T).square()))
        measured = sum(errors) / len(errors)
        theory = float((math.pi / 2 * x.square().sum() * w.square().sum()
                        - (x @ w.T).square().sum()) / d)
        self.assertLess(abs(measured / theory - 1), 0.15)

    def test_base_optimization_and_identity(self):
        weights = {i: torch.randn(4, 5) for i in range(3)}
        train = {i: (torch.randn(300, 5), torch.rand(300)) for i in weights}
        held = {i: (torch.randn(300, 5), torch.rand(300)) for i in weights}
        saliency = torch.tensor([0.5, 1.1, 0.9])
        base, report = fit_shared_base(weights, train, "qjl_aware", saliency,
                                       d=127, heldout=held)
        self.assertLessEqual(report["objective_after"], report["objective_before"] * 1.00001)
        self.assertLessEqual(report["heldout_after"], report["heldout_before"] * 1.05)
        for w in weights.values():
            self.assertTrue(torch.allclose(base + (w - base), w, atol=1e-6))
        self.assertAlmostEqual(report["objective_after"],
                               variance_objective(base, weights, train, 127), places=3)

    def test_routing_checkpoint_and_memory(self):
        original = ToyModel()
        restored = copy.deepcopy(original)
        x = torch.randn(2, 8, 7)
        before, logits = original.model.layers[0].mlp(x)
        groups = {0: [[0, 1], [2], [3]]}
        saliency = {0: torch.ones(4)}
        samples = synthetic_samples(original)
        config = {"qjl_d": 127, "projection_seed": 5, "base_method": "mean",
                  "residual_mode": "exact", "targets": "gate_up_down",
                  "backend": "reference", "delta": 1e-3, "cg_iterations": 10}
        convert_model(original, "qwen", groups, saliency, samples, {}, config)
        after, new_logits = original.model.layers[0].mlp(x)
        self.assertTrue(torch.equal(logits, new_logits))
        self.assertTrue(torch.allclose(after, before, atol=2e-5, rtol=2e-4))
        state = original.state_dict()
        self.assertFalse(any(k.startswith("model.layers.0.mlp.experts.0.") for k in state))
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "qjl.pt")
            sizes = save_qjl_checkpoint(original, "qwen", groups, config, path)
            self.assertEqual(sizes["checkpoint_bytes"], Path(path).stat().st_size)
            self.assertGreater(sizes["tensor_bytes"], 0)
            load_qjl_checkpoint(restored, path, backend="reference")
            loaded, loaded_logits = restored.model.layers[0].mlp(x)
            self.assertTrue(torch.equal(loaded_logits, logits))
            self.assertTrue(torch.equal(loaded, after))

    def test_mixtral_qjl_save_reload(self):
        model = ToyMixtralModel()
        restored = copy.deepcopy(model)
        x = torch.randn(2, 6, 7)
        _, old_logits = model.model.layers[0].block_sparse_moe(x)
        groups = {0: [[0, 1], [2], [3]]}
        config = {"qjl_d": 127, "projection_seed": 4, "base_method": "saliency",
                  "residual_mode": "qjl", "targets": "gate_up_down",
                  "backend": "reference", "delta": 1e-3, "cg_iterations": 10}
        convert_model(model, "mixtral", groups, {0: torch.ones(4)},
                      synthetic_samples(restored), {}, config)
        output, logits = model.model.layers[0].block_sparse_moe(x)
        self.assertTrue(torch.equal(logits, old_logits))
        self.assertTrue(torch.isfinite(output).all())
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "mixtral.pt")
            save_qjl_checkpoint(model, "mixtral", groups, config, path)
            load_qjl_checkpoint(restored, path, backend="reference")
            loaded, new_logits = restored.model.layers[0].block_sparse_moe(x)
            self.assertTrue(torch.equal(loaded, output))
            self.assertTrue(torch.equal(new_logits, old_logits))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_triton_packed_parity(self):
        from rescode.qjl.core import qjl_triton, qjl_triton_legacy
        from rescode.qjl.triton_kernel import packed_qjl_fused_gate_up
        w = torch.randn(9, 7, device="cuda")
        w[0].zero_()
        for d in (127, 8192, 8193, 16384):
            bank = ProjectionBank(d, 19)
            packed, norms = pack_residual(w, bank)
            self.assertEqual(bank.get(7, w.device).dtype, torch.float16)
            self.assertEqual(norms.dtype, torch.float32)
            up_packed, up_norms = pack_residual(w.flip(0), bank)
            for tokens in (1, 4, 32, 128):
                with self.subTest(d=d, tokens=tokens):
                    x = torch.randn(tokens, 7, device="cuda")
                    z = x.half() @ bank.get(7, x.device)
                    ref = qjl_reference(z, packed, norms, d)
                    got = qjl_triton(z, packed, norms, d)
                    legacy = qjl_triton_legacy(z, packed, norms, d)
                    fused_gate, fused_up = packed_qjl_fused_gate_up(
                        z, packed, norms, up_packed, up_norms, d)
                    self.assertTrue(torch.allclose(got, ref, atol=1e-4, rtol=1e-4))
                    self.assertTrue(torch.allclose(got, legacy, atol=1e-4, rtol=1e-4))
                    self.assertTrue(torch.allclose(fused_gate, ref, atol=1e-4, rtol=1e-4))
                    self.assertTrue(torch.allclose(
                        fused_up, qjl_reference(z, up_packed, up_norms, d),
                        atol=1e-4, rtol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_three_backends_same_checkpoint_qwen_and_mixtral(self):
        from rescode.qjl.core import qjl_triton_fused_gate_up
        for family, factory in (("qwen", ToyModel), ("mixtral", ToyMixtralModel)):
            for d in (127, 8192):
                with self.subTest(family=family, d=d), tempfile.TemporaryDirectory() as temp:
                    original = factory().to("cuda", dtype=torch.bfloat16)
                    untouched = copy.deepcopy(original)
                    groups = {0: [[0, 1], [2], [3]]}
                    config = {"qjl_d": d, "projection_seed": 5, "base_method": "mean",
                              "residual_mode": "qjl", "targets": "gate_up_down",
                              "backend": "triton", "delta": 1e-3, "cg_iterations": 10}
                    convert_model(original, family, groups, {0: torch.ones(4)},
                                  {}, {}, config)
                    path = str(Path(temp) / "qjl.pt")
                    save_qjl_checkpoint(original, family, groups, config, path)
                    models = {}
                    for backend in ("reference", "triton_legacy", "triton"):
                        model = copy.deepcopy(untouched)
                        load_qjl_checkpoint(model, path, backend=backend)
                        models[backend] = model
                    block_name = "mlp" if family == "qwen" else "block_sparse_moe"
                    blocks = {key: getattr(model.model.layers[0], block_name)
                              for key, model in models.items()}
                    for tokens in (1, 4, 32, 128):
                        x = torch.randn(tokens, 7, device="cuda", dtype=torch.bfloat16)
                        reference_output, reference_logits = blocks["reference"](x)
                        reference_selected = torch.topk(reference_logits, 2, dim=-1).indices
                        for backend in ("triton_legacy", "triton"):
                            output, logits = blocks[backend](x)
                            self.assertTrue(torch.equal(logits, reference_logits))
                            self.assertTrue(torch.equal(
                                torch.topk(logits, 2, dim=-1).indices, reference_selected))
                            self.assertTrue(torch.allclose(output, reference_output,
                                                           atol=0.02, rtol=0.02))
                            for singleton in (2, 3):
                                self.assertTrue(torch.equal(
                                    blocks[backend].experts[singleton](x),
                                    blocks["reference"].experts[singleton](x)))
                        clusters = {key: block._qjl_clusters["0"]
                                    for key, block in blocks.items()}
                        packed = {key: cluster.residuals["0"]
                                  for key, cluster in clusters.items()}
                        p = models["reference"]._qjl_projection_bank.get(7, x.device)
                        z = x.to(p.dtype) @ p
                        gate, up = qjl_triton_fused_gate_up(
                            z, packed["triton"]["gate"], packed["triton"]["up"], x.dtype)
                        self.assertTrue(torch.allclose(
                            gate, packed["reference"]["gate"].forward_projected(z, x.dtype),
                            atol=0.02, rtol=0.02))
                        self.assertTrue(torch.allclose(
                            up, packed["reference"]["up"].forward_projected(z, x.dtype),
                            atol=0.02, rtol=0.02))
                        for backend in ("triton_legacy", "triton"):
                            self.assertTrue(torch.allclose(
                                clusters[backend].forward_expert(0, x),
                                clusters["reference"].forward_expert(0, x),
                                atol=0.02, rtol=0.02))
                            candidate = packed[backend]
                            reference = packed["reference"]
                            for projection in ("gate", "up"):
                                self.assertTrue(torch.equal(candidate[projection].packed,
                                                            reference[projection].packed))
                                self.assertTrue(torch.equal(candidate[projection].norms,
                                                            reference[projection].norms))
                                self.assertTrue(torch.allclose(
                                    candidate[projection].forward_projected(z, torch.float32),
                                    reference[projection].forward_projected(z, torch.float32),
                                    atol=1e-4, rtol=1e-4))
                            gate_value = (clusters[backend].bases["gate"](x)
                                          + candidate["gate"].forward_projected(z, x.dtype))
                            up_value = (clusters[backend].bases["up"](x)
                                        + candidate["up"].forward_projected(z, x.dtype))
                            reference_gate = (clusters["reference"].bases["gate"](x)
                                              + reference["gate"].forward_projected(z, x.dtype))
                            reference_up = (clusters["reference"].bases["up"](x)
                                            + reference["up"].forward_projected(z, x.dtype))
                            self.assertTrue(torch.allclose(gate_value, reference_gate,
                                                           atol=0.02, rtol=0.02))
                            self.assertTrue(torch.allclose(up_value, reference_up,
                                                           atol=0.02, rtol=0.02))
                            middle = F.silu(gate_value) * up_value
                            reference_middle = F.silu(reference_gate) * reference_up
                            self.assertTrue(torch.allclose(middle, reference_middle,
                                                           atol=0.02, rtol=0.02))
                            self.assertTrue(torch.allclose(
                                candidate["down"](reference_middle),
                                reference["down"](reference_middle),
                                atol=0.02, rtol=0.02))


if __name__ == "__main__":
    unittest.main()
