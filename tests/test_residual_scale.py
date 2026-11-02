import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rescode.qjl.model import convert_model, load_qjl_checkpoint, save_qjl_checkpoint
from rescode.qjl.scale_diagnostics import relative_l2, summarize_records
from tests.test_direct_qjl import ToyModel, synthetic_samples


GROUPS = {0: [[0, 1], [2], [3]]}
SALIENCY = {0: torch.ones(4)}


def config(alpha=None, residual_mode="qjl", backend="reference"):
    value = {"qjl_d": 127, "projection_seed": 0,
             "base_method": "qjl_aware", "residual_mode": residual_mode,
             "targets": "gate_up_down", "backend": backend,
             "delta": 1e-3, "cg_iterations": 3}
    if alpha is not None:
        value["residual_scale_alpha"] = alpha
    return value


class ResidualScaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)
        self.original = ToyModel().eval()
        self.samples = synthetic_samples(self.original, 24)
        self.x = torch.randn(1, 8, 7)

    def convert(self, alpha, *, diagnostics=True, shared_cache=None,
                residual_mode="qjl"):
        model = copy.deepcopy(self.original)
        records = [] if diagnostics else None
        timer = [0.0]
        cfg = config(alpha, residual_mode)
        convert_model(model, "qwen", GROUPS, SALIENCY,
                      self.samples, self.samples, cfg,
                      shared_cache=shared_cache,
                      diagnostic_records=records, diagnostic_timer=timer,
                      diagnostic_samples=4)
        return model, records, cfg

    def test_alpha_zero_identity_router_singletons_and_nulls(self):
        original_block = self.original.model.layers[0].mlp
        expected, logits = original_block(self.x)
        scaled, records, cfg = self.convert(0.0)
        block = scaled.model.layers[0].mlp
        actual, routed = block(self.x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(routed, logits))
        self.assertTrue(torch.equal(block.experts[2].gate_proj.weight,
                                    original_block.experts[2].gate_proj.weight))
        cluster = block._qjl_clusters["0"]
        self.assertIn("0", cluster.expert_bases)
        for i in (0, 1):
            for proj in ("gate", "up", "down"):
                self.assertEqual(float(cluster.residuals[str(i)][proj].norms.max()), 0.0)
        qjl = [r for r in records if r["metric"] == "qjl_residual_relative_l2_error"]
        self.assertTrue(qjl)
        self.assertTrue(all(r["value"] is None and r["null_reason"] == "zero_reference_norm"
                            for r in qjl))
        summary = summarize_records(records)
        self.assertIsNone(summary["qjl_residual_relative_l2_error"]["mean"])
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = str(Path(temp) / "alpha0.pt")
            save_qjl_checkpoint(scaled, "qwen", GROUPS, cfg, checkpoint)
            restored = copy.deepcopy(self.original)
            load_qjl_checkpoint(restored, checkpoint, backend="reference")
            self.assertTrue(torch.equal(restored.model.layers[0].mlp(self.x)[0], actual))

    def test_alpha_one_matches_existing_qjl_and_reuses_shared_base(self):
        direct = copy.deepcopy(self.original)
        convert_model(direct, "qwen", GROUPS, SALIENCY,
                      self.samples, self.samples, config())
        shared = {}
        scaled, _, cfg = self.convert(1.0, shared_cache=shared)
        self.assertEqual(len(shared), 3)
        direct_cluster = direct.model.layers[0].mlp._qjl_clusters["0"]
        scaled_cluster = scaled.model.layers[0].mlp._qjl_clusters["0"]
        self.assertEqual(len(scaled_cluster.expert_bases), 0)
        for proj in ("gate", "up", "down"):
            self.assertTrue(torch.equal(direct_cluster.bases[proj].weight,
                                        scaled_cluster.bases[proj].weight))
            for i in (0, 1):
                self.assertTrue(torch.equal(direct_cluster.residuals[str(i)][proj].packed,
                                            scaled_cluster.residuals[str(i)][proj].packed))
        self.assertTrue(torch.equal(direct.model.layers[0].mlp(self.x)[0],
                                    scaled.model.layers[0].mlp(self.x)[0]))
        with patch("rescode.qjl.model.fit_shared_base", side_effect=AssertionError("refit")):
            reused, _, _ = self.convert(0.25, shared_cache=shared)
        self.assertTrue(torch.isfinite(reused.model.layers[0].mlp(self.x)[0]).all())
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = str(Path(temp) / "alpha1.pt")
            save_qjl_checkpoint(scaled, "qwen", GROUPS, cfg, checkpoint)
            restored = copy.deepcopy(self.original)
            load_qjl_checkpoint(restored, checkpoint, backend="reference")
            self.assertTrue(torch.equal(restored.model.layers[0].mlp(self.x)[0],
                                        scaled.model.layers[0].mlp(self.x)[0]))

    def test_exact_residual_reconstructs_and_diagnostics_do_not_change_forward(self):
        expected = self.original.model.layers[0].mlp(self.x)[0]
        norms = []
        absolute_qjl_errors = []
        for alpha in (0.0, 0.25, 0.5, 1.0):
            exact, _, _ = self.convert(alpha, diagnostics=False, residual_mode="exact")
            self.assertTrue(torch.allclose(exact.model.layers[0].mlp(self.x)[0],
                                           expected, atol=2e-5, rtol=2e-4))
            with_diag, records, cfg = self.convert(alpha)
            without_diag, _, _ = self.convert(alpha, diagnostics=False)
            self.assertTrue(torch.equal(with_diag.model.layers[0].mlp(self.x)[0],
                                        without_diag.model.layers[0].mlp(self.x)[0]))
            with tempfile.TemporaryDirectory() as temp:
                path = str(Path(temp) / "scaled.pt")
                save_qjl_checkpoint(with_diag, "qwen", GROUPS, cfg, path)
                reloaded = copy.deepcopy(self.original)
                load_qjl_checkpoint(reloaded, path, backend="reference")
                self.assertTrue(torch.equal(with_diag.model.layers[0].mlp(self.x)[0],
                                            reloaded.model.layers[0].mlp(self.x)[0]))
            decomposition = [r for r in records
                             if r["metric"] == "decomposition_relative_l2_error"]
            self.assertTrue(all(r["value"] is not None and r["value"] < 1e-6
                                for r in decomposition))
            norm = summarize_records(records)["residual_frobenius_ratio"]["mean"]
            norms.append(norm)
            cluster = with_diag.model.layers[0].mlp._qjl_clusters["0"]
            module = cluster.residuals["0"]["gate"]
            base = (cluster.expert_bases["0"]["gate"].weight
                    if "0" in cluster.expert_bases else cluster.bases["gate"].weight)
            residual = self.original.model.layers[0].mlp.experts[0].gate_proj.weight - base
            x = self.x.reshape(-1, 7)
            absolute_qjl_errors.append(float(torch.linalg.vector_norm(
                module(x) - F.linear(x, residual))))
        self.assertEqual(norms[0], 0.0)
        self.assertTrue(all(a <= b + 1e-6 for a, b in zip(norms, norms[1:])))
        self.assertEqual(absolute_qjl_errors[0], 0.0)
        self.assertAlmostEqual(absolute_qjl_errors[2] / absolute_qjl_errors[1],
                               2.0, delta=1e-4)
        self.assertAlmostEqual(absolute_qjl_errors[3] / absolute_qjl_errors[2],
                               2.0, delta=1e-4)
        self.assertEqual(relative_l2(torch.zeros(2), torch.zeros(2)),
                         (None, "zero_reference_norm"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_scaled_triton_matches_reference(self):
        samples = synthetic_samples(self.original, 12)
        original = self.original.to("cuda")
        scaled = copy.deepcopy(original)
        cfg = config(0.5, backend="triton")
        convert_model(scaled, "qwen", GROUPS, SALIENCY, samples, {}, cfg,
                      diagnostic_records=[], diagnostic_timer=[0.0],
                      diagnostic_samples=2)
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "scaled.pt")
            save_qjl_checkpoint(scaled, "qwen", GROUPS, cfg, path)
            reference = copy.deepcopy(original)
            load_qjl_checkpoint(reference, path, backend="reference")
            x = self.x.to("cuda")
            a, al = scaled.model.layers[0].mlp(x)
            b, bl = reference.model.layers[0].mlp(x)
            self.assertTrue(torch.equal(al, bl))
            self.assertTrue(torch.allclose(a, b, atol=0.02, rtol=0.02))


if __name__ == "__main__":
    unittest.main()
