"""Small exact BF16 neuron-permutation regression tests."""

import unittest

import numpy as np
import torch

from scripts.QJL.analyze_exponent_alignment import (
    align_neurons, exponent_mantissa, histogram_metrics, jsd_matrix,
    neuron_features, pair_metrics, permute_neurons,
)


def bits(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(torch.bfloat16).view(torch.uint16).numpy().copy()


def expert_output(weights, x):
    def fp32(raw):
        return (raw.astype(np.uint32) << 16).view(np.float32)

    gate = fp32(weights["gate"])
    up = fp32(weights["up"])
    down = fp32(weights["down"])
    gate_value = x @ gate.T
    activated = gate_value / (1 + np.exp(-gate_value))
    return (activated * (x @ up.T)) @ down.T


class ExponentAlignmentTest(unittest.TestCase):
    def test_raw_bf16_fields(self):
        exponent, mantissa = exponent_mantissa(
            np.array([0x3F80, 0xBF81, 0x0001], dtype=np.uint16))
        np.testing.assert_array_equal(exponent, [127, 127, 0])
        np.testing.assert_array_equal(mantissa, [0, 1, 1])

    def test_same_permutation_preserves_swiglu_and_histograms(self):
        weights = {"gate": bits((6, 4), 1), "up": bits((6, 4), 2),
                   "down": bits((4, 6), 3)}
        permutation = np.array([2, 0, 5, 1, 4, 3], dtype=np.int32)
        aligned = permute_neurons(weights, permutation)
        x = np.random.default_rng(9).normal(size=(8, 4)).astype(np.float32)
        np.testing.assert_allclose(expert_output(weights, x), expert_output(aligned, x),
                                   rtol=2e-6, atol=2e-6)
        for projection in ("gate", "up", "down"):
            original_exp, original_man, _ = histogram_metrics(weights[projection])
            aligned_exp, aligned_man, _ = histogram_metrics(aligned[projection])
            np.testing.assert_array_equal(original_exp, aligned_exp)
            np.testing.assert_array_equal(original_man, aligned_man)
            self.assertEqual(jsd_matrix(np.stack([original_exp, aligned_exp]))[0, 1], 0)

    def test_assignment_recovers_known_neuron_reordering(self):
        reference = {"gate": bits((6, 8), 11), "up": bits((6, 8), 12),
                     "down": bits((8, 6), 13)}
        shuffled = np.array([2, 0, 5, 1, 4, 3], dtype=np.int32)
        target = permute_neurons(reference, shuffled)
        recovered, _ = align_neurons(neuron_features(reference),
                                     neuron_features(target), device="cpu", tile_rows=2)
        np.testing.assert_array_equal(recovered, np.argsort(shuffled))
        restored = permute_neurons(target, recovered)
        for projection in ("gate", "up", "down"):
            np.testing.assert_array_equal(restored[projection], reference[projection])
            exp, _, _ = histogram_metrics(reference[projection])
            score = pair_metrics(reference[projection], restored[projection], exp, exp)
            self.assertEqual(score["exact_pct"], 100.0)
            self.assertEqual(score["signed_diff_entropy_bits"], 0.0)


if __name__ == "__main__":
    unittest.main()
