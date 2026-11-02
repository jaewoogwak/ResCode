import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.QJL.run_residual_scale_sweep import (
    alpha_name, verified_compression, verified_evaluation, write_summary,
)


class SweepResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "alpha_0_5"
        self.out.mkdir()
        self.args = SimpleNamespace(
            qjl_d=8192, model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat",
            backend="triton", projection_seed=0, calib_seed=42,
            n_sentences=2400, calib_seqlen=2048,
            max_train_per_expert=256, max_heldout_per_expert=64,
            cg_iterations=30, layers="all", diagnostic_samples=32,
            train_data_path="rescode/data/wikipedia-calib.jsonl",
            groups_path="groups.pt", task="arc_easy,boolq",
        )

    def write(self, name, payload):
        target = self.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))

    def test_compression_resume_requires_matching_config_complete_checkpoint(self):
        self.assertFalse(verified_compression(self.out, self.args, 0.5))
        config = vars(self.args).copy()
        config["residual_scale_alpha"] = 0.5
        self.write("config.json", config)
        self.write("diagnostics.json", {"alpha": 0.5, "summary": {"ratio": 0.2}})
        (self.out / "qjl_state.pt").write_bytes(b"checkpoint")
        self.write("metrics.json", {"checkpoint_bytes": 10})
        self.assertTrue(verified_compression(self.out, self.args, 0.5))
        self.assertFalse(verified_compression(self.out, self.args, 1.0))
        (self.out / "qjl_state.pt").write_bytes(b"partial")
        self.assertFalse(verified_compression(self.out, self.args, 0.5))

    def test_eval_resume_requires_all_tasks_and_limit(self):
        self.assertFalse(verified_evaluation(self.out, ["arc_easy", "boolq"], 0))
        self.write("eval/metrics.json", {
            "eval_limit": 0, "task_summary": {"tasks": [["arc_easy", 0.5, 0.6]]}
        })
        self.assertFalse(verified_evaluation(self.out, ["arc_easy", "boolq"], 0))
        self.write("eval/metrics.json", {
            "eval_limit": 0, "task_summary": {
                "tasks": [["arc_easy", 0.5, 0.6], ["boolq", 0.7, None]]}
        })
        self.assertTrue(verified_evaluation(self.out, ["arc_easy", "boolq"], 0))
        self.assertFalse(verified_evaluation(self.out, ["arc_easy", "boolq"], 2))

    def test_names_and_summary(self):
        self.assertEqual(alpha_name(0.25), "alpha_0_25")
        rows = [{"alpha": 0, "status": "compressed"},
                {"alpha": 0.25, "status": "evaluated"}]
        write_summary(Path(self.temp.name), rows)
        self.assertEqual(json.loads((Path(self.temp.name) / "sweep_summary.json").read_text()),
                         rows)
        self.assertIn("alpha,status", (Path(self.temp.name) / "sweep_summary.csv").read_text())


if __name__ == "__main__":
    unittest.main()
