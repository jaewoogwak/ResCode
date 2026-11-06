"""Synthetic checks for independent per-row element selection and energy."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/QJL"))
from rowwise_outliers import (analyze_rowwise_matrix, jaccard_indices, top_indices,
                              plot_representative_rows, plot_rowwise_aggregates,
                              write_rowwise_report)


class RowwiseOutlierTests(unittest.TestCase):
    def test_independent_top_element_and_expected_energy(self):
        matrix = np.ones((3, 32), dtype=np.float32)
        matrix[0, 0] = 10
        matrix[1, 7] = 20
        matrix[2, 13] = 30
        before = matrix.copy()
        np.testing.assert_array_equal(top_indices(np.abs(matrix), 1)[:, 0], [0, 7, 13])
        stats, rows, positions = analyze_rowwise_matrix(matrix, (42, 0, 0), overlap_pairs=128)
        top = next(x for x in rows if x["k"] == 1 and x["selection"] == "top_abs")
        energy = 10**2 + 20**2 + 30**2 + 3 * 31
        self.assertAlmostEqual(top["matrix_squared_l2_reduction_pct"], 100 * (10**2 + 20**2 + 30**2) / energy)
        self.assertAlmostEqual(top["remaining_frobenius_norm"] ** 2, 3 * 31)
        self.assertAlmostEqual(top["row_squared_l2_reduction_pct_median"], 100 * 20**2 / (20**2 + 31))
        self.assertEqual(stats["row_count"], 3)
        self.assertEqual(len(positions[(1, "top_abs")]), 1)
        np.testing.assert_array_equal(matrix, before)

    def test_repeated_positions_and_random_determinism(self):
        matrix = np.ones((20, 32), dtype=np.float32)
        matrix[:, 9] = 100
        first = analyze_rowwise_matrix(matrix, (42, 1, 2), overlap_pairs=200)
        second = analyze_rowwise_matrix(matrix, (42, 1, 2), overlap_pairs=200)
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2][(1, "top_abs")].tolist(), [9])
        top = next(x for x in first[1] if x["k"] == 1 and x["selection"] == "top_abs")
        random = next(x for x in first[1] if x["k"] == 1 and x["selection"] == "random")
        self.assertEqual(top["within_expert_row_jaccard"], 1.0)
        self.assertGreater(top["matrix_squared_l2_reduction_pct"], random["matrix_squared_l2_reduction_pct"])
        self.assertAlmostEqual(jaccard_indices(np.array([1, 2]), np.array([2, 3])), 1 / 3)

    def test_zero_rows_and_invalid_inputs(self):
        matrix = np.zeros((2, 20), dtype=np.float32)
        stats, rows, _ = analyze_rowwise_matrix(matrix, (7,))
        self.assertEqual(stats["zero_rows"], 2)
        self.assertEqual(rows[0]["matrix_squared_l2_reduction_pct"], 0)
        with self.assertRaises(ValueError):
            analyze_rowwise_matrix(matrix.astype(np.float64), (7,))

    def test_plot_and_report_artifacts(self):
        matrix = np.ones((8, 32), dtype=np.float32)
        matrix[:, 4] = 10
        expert_rows, reduction_rows, overlap_rows = [], [], []
        for projection in ("gate", "up", "down"):
            for kind in ("residual", "original_weight"):
                stats, reductions, _ = analyze_rowwise_matrix(matrix, (42, len(expert_rows)))
                expert_rows.append({"projection": projection, "matrix_kind": kind, **stats})
                reduction_rows.extend({"projection": projection, "matrix_kind": kind, **row}
                                      for row in reductions)
            for selection in ("top_abs", "random"):
                overlap_rows.append({"projection": projection, "k": 16,
                                     "selection": selection, "jaccard": 0.1})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plot_representative_rows(matrix, matrix, output / "heatmap.png", "synthetic", 42, 4)
            plot_rowwise_aggregates(pd.DataFrame(expert_rows), pd.DataFrame(reduction_rows),
                                    pd.DataFrame(overlap_rows), output, ("gate", "up", "down"))
            write_rowwise_report(output / "REPORT.md", pd.DataFrame(expert_rows),
                                 pd.DataFrame(reduction_rows), pd.DataFrame(overlap_rows),
                                 [0], 0.0, Path("groups.pt"), Path("model"), 42)
            self.assertTrue((output / "heatmap.png").is_file())
            self.assertTrue((output / "row_norm_reduction_down.png").is_file())
            self.assertIn("selected-layer analysis", (output / "REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
