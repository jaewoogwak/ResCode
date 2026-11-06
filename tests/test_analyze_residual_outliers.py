"""Small numerical checks for the standalone outlier experiment."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[1] / "scripts/QJL/analyze_residual_outliers.py"
spec = importlib.util.spec_from_file_location("outlier_analysis", MODULE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class OutlierAnalysisTests(unittest.TestCase):
    def test_saliency_base_matches_rescode_merge(self):
        from rescode.merging.grouping_deepseek import _merge_mlp_experts_by_usage_frequency_weighting

        class Expert(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                for name in ("gate_proj", "up_proj", "down_proj"):
                    linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
                    linear.weight.data.fill_(value)
                    setattr(self, name, linear)

        block = torch.nn.Module()
        block.experts = torch.nn.ModuleList([Expert(1), Expert(3)])
        saliency = torch.tensor([1.0, 3.0])
        original = {i: block.experts[i].gate_proj.weight.detach().clone() for i in range(2)}
        base, coefficients = analysis.saliency_base(original, saliency)
        self.assertAlmostEqual(float(coefficients.sum()), 1.0)
        _merge_mlp_experts_by_usage_frequency_weighting(block, torch.tensor([0, 0]), saliency + 0.001)
        torch.testing.assert_close(base, block.experts[0].gate_proj.weight)

    def test_exact_bf16_subtraction_and_top_channel(self):
        weight = torch.tensor([[10, 1, 1, 1], [10, 1, 1, 1], [10, 1, 1, 1]], dtype=torch.bfloat16)
        base = torch.tensor([[0.25, 1, 1, 1], [0.25, 1, 1, 1], [0.25, 1, 1, 1]], dtype=torch.bfloat16)
        residual = (weight.float() - base.float()).numpy()
        self.assertEqual(residual.dtype, np.float32)
        stats, rows, l2_order, abs_order, channel = analysis.analyze_matrix(residual, (7, 0, 0))
        self.assertEqual(int(l2_order[0]), 0)
        self.assertEqual(int(abs_order[0]), 0)
        self.assertEqual(int(channel["rank_l2"][0]), 1)
        self.assertAlmostEqual(stats["frobenius_norm"], float(np.linalg.norm(residual)), places=5)
        top = next(row for row in rows if row["k_label"] == "1" and row["selection"] == "top_l2")
        self.assertAlmostEqual(top["removed_squared_norm_pct"], 100.0)
        self.assertEqual(top["remaining_frobenius_norm"], 0.0)

    def test_energy_identity_and_no_mutation(self):
        matrix = np.arange(20, dtype=np.float32).reshape(4, 5)
        before = matrix.copy()
        squared = np.square(matrix, dtype=np.float64)
        row_energy = squared.sum(axis=1)
        column_energy = squared.sum(axis=0)
        none = analysis.norm_reduction(row_energy, column_energy, matrix, np.array([], dtype=np.int64))
        all_columns = analysis.norm_reduction(row_energy, column_energy, matrix, np.arange(5))
        partial = analysis.norm_reduction(row_energy, column_energy, matrix, np.array([1, 3]))
        self.assertAlmostEqual(none["remaining_over_original"], 1.0)
        self.assertEqual(all_columns["remaining_frobenius_norm"], 0.0)
        self.assertAlmostEqual(partial["removed_squared_norm_pct"],
                               100 * squared[:, [1, 3]].sum() / squared.sum())
        np.testing.assert_array_equal(matrix, before)

    def test_fixed_random_and_jaccard(self):
        matrix = np.random.default_rng(17).normal(size=(11, 40)).astype(np.float32)
        first = analysis.analyze_matrix(matrix, (9, 2, 3))
        second = analysis.analyze_matrix(matrix, (9, 2, 3))
        self.assertEqual(first[1], second[1])
        self.assertAlmostEqual(analysis.jaccard(np.array([1, 2]), np.array([2, 3])), 1 / 3)
        self.assertEqual(analysis.parse_layers("2,0,2", 4), [0, 2])


if __name__ == "__main__":
    unittest.main()
