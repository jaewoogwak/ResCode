"""Focused synthetic checks for projection and lossless bit correction."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/QJL"))
from analyze_projection_bit_sharing import analyze_expert, parse_dimensions
from bit_sharing_analysis import (aligned_neurons, bit_mismatches, bit_statistics,
                                  cosine_rows, cosine_statistics, decode_sparse_correction,
                                  derangement, encode_sparse_correction, storage_statistics)
from rescode.qjl.core import ProjectionBank, pack_residual, pack_signs, unpack_signs


class ProjectionBitSharingTests(unittest.TestCase):
    def test_neuron_alignment_and_known_cosines(self):
        gate = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        up = np.array([[1, 0, 0], [0, -1, 0]], dtype=np.float32)
        down = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.float32)
        pairs = aligned_neurons(gate, up, down)
        np.testing.assert_array_equal(pairs["gate_down"][1], down.T)
        np.testing.assert_allclose(cosine_rows(*pairs["gate_up"])[0], [1, -1])
        np.testing.assert_allclose(cosine_rows(*pairs["gate_down"])[0], [0, 0])
        stats, _ = cosine_statistics(*pairs["gate_up"])
        self.assertEqual(stats["abs_cosine_gt_0p99"], 1)
        with self.assertRaises(ValueError):
            aligned_neurons(gate, up, down.T)

    def test_projection_precision_seed_and_packing(self):
        x = torch.tensor([[1, 2, 3], [-2, 1, 0], [0, 0, 0]], dtype=torch.float32)
        bank_a, bank_b = ProjectionBank(13, 7), ProjectionBank(13, 7)
        p = bank_a.get(3, torch.device("cpu"))
        self.assertEqual(p.dtype, torch.float16)
        torch.testing.assert_close(p, bank_b.get(3, torch.device("cpu")))
        packed_a, norms = pack_residual(x, bank_a, row_chunk=2)
        packed_b, _ = pack_residual(x, bank_b, row_chunk=1)
        torch.testing.assert_close(packed_a, packed_b)
        reference = (x.to(p.dtype) @ p) >= 0
        reference[2] = False
        torch.testing.assert_close(unpack_signs(packed_a, 13), reference)
        torch.testing.assert_close(norms, torch.linalg.vector_norm(x, dim=1))

    def test_bit_agreement_and_sparse_reconstruction(self):
        gate_bits = torch.tensor([[1] * 13, [0] * 13], dtype=torch.bool)
        up_bits = torch.tensor([[1] * 13, [1] * 13], dtype=torch.bool)
        gate, up = pack_signs(gate_bits).numpy(), pack_signs(up_bits).numpy()
        mismatch, _ = bit_mismatches(gate, up, 13)
        np.testing.assert_array_equal(mismatch, [0, 13])
        stats, _, _ = bit_statistics(gate, up, 13, np.array([1., -1.]),
                                     np.array([1, 0]), (4,))
        self.assertAlmostEqual(stats["direct_agreement_mean"], 0.5)
        self.assertAlmostEqual(stats["best_agreement_mean"], 1.0)
        self.assertAlmostEqual(stats["predicted_agreement_mean"], 0.5)
        for row, invert in ((0, False), (1, True)):
            payload = encode_sparse_correction(gate[row], up[row], 13, invert)
            reconstructed = decode_sparse_correction(gate[row], payload, 13, invert)
            np.testing.assert_array_equal(reconstructed, up[row])
            self.assertEqual(len(payload), 2)
        empty = pack_signs(torch.zeros((1, 13), dtype=torch.bool)).numpy()[0]
        sparse = torch.zeros((1, 13), dtype=torch.bool)
        sparse[0, [0, 5, 12]] = True
        sparse = pack_signs(sparse).numpy()[0]
        payload = encode_sparse_correction(empty, sparse, 13, False)
        self.assertEqual(len(payload), 2 + 2 * 3)
        np.testing.assert_array_equal(decode_sparse_correction(empty, payload, 13, False), sparse)
        mostly_one = pack_signs(torch.ones((1, 13), dtype=torch.bool)).numpy()[0] ^ sparse
        payload = encode_sparse_correction(empty, mostly_one, 13, True)
        self.assertEqual(len(payload), 2 + 2 * 3)
        np.testing.assert_array_equal(decode_sparse_correction(empty, payload, 13, True), mostly_one)
        with self.assertRaises(ValueError):
            decode_sparse_correction(gate[0], b"\x02\x00", 13, False)

    def test_storage_bytes_and_selective_choice(self):
        small = storage_statistics(np.array([0, 8, 16]), 16)
        self.assertEqual(small["separate_total_bytes"], 68)
        self.assertEqual(small["dense_lossless_total_bytes"], 68)
        self.assertEqual(small["sparse_position_payload_bytes"], 22)
        self.assertEqual(small["sparse_lossless_total_bytes"], 85)
        self.assertEqual(small["selective_lossless_total_bytes"], 68)
        large = storage_statistics(np.zeros(16, dtype=np.int32), 8192)
        self.assertLess(large["sparse_lossless_total_bytes"], large["separate_total_bytes"])
        self.assertEqual(large["selective_shared_neurons"], 16)
        self.assertEqual(parse_dimensions("16384,8192,8192"), [8192, 16384])

    def test_residual_reconstruction_and_singleton_exclusion(self):
        gate = torch.tensor([[2, 0, 0, 1], [0, 3, 1, 0]], dtype=torch.bfloat16)
        up = torch.tensor([[2, 1, 0, 0], [0, 2, 1, 1]], dtype=torch.bfloat16)
        down = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.bfloat16)
        weights = {"gate": gate, "up": up, "down": down}
        bases = {p: torch.zeros_like(w) for p, w in weights.items()}
        bank = {16: ProjectionBank(16, 0)}
        w_cos, w_bits, w_storage, _ = analyze_expert(
            0, 0, 1, True, weights, None, [16], bank, torch.device("cpu"), 0, 2)
        self.assertEqual(len(w_cos), 6)
        self.assertEqual(len(w_bits), 1)
        self.assertTrue(all(row["weight_kind"] == "original_weight" for row in w_cos))
        cos, bits, storage, error = analyze_expert(
            0, 0, 1, False, weights, bases, [16], bank, torch.device("cpu"), 0, 2)
        self.assertEqual(len(cos), 12)
        self.assertEqual(len(bits), 2)
        self.assertEqual(len(storage), 2)
        self.assertEqual(error, 0.0)
        np.testing.assert_array_equal(derangement(5, (4,)), derangement(5, (4,)))


if __name__ == "__main__":
    unittest.main()
