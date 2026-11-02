import json
from argparse import Namespace

from scripts.QJL.run_exponent_base_sweep import (
    _summary, _valid_eval, experiments,
)


def test_budget_pairs_and_condition_one(tmp_path):
    configs = experiments()
    assert len(configs) == 11
    names = {c["name"] for c in configs}
    assert {"original", "base_k3", "base_k4", "qjl_k2_d1n_a1",
            "qjl_k2_d2n_a1"} <= names
    selected = [c for c in configs if c["name"] in ("base_k2", "qjl_k2_d1n_a1")]
    for config in selected:
        out = tmp_path / config["name"]
        out.mkdir()
        (out / "metrics.json").write_text(json.dumps({
            "checkpoint_bytes": 1000, "original_routed_bf16_bytes": 2000,
            "base_encoded_bytes": 800, "qjl_tensor_bytes": 100,
        }))
        value = 0.2 if config["name"] == "base_k2" else 0.1
        summary = {p: {"mean_relative_l2": value} for p in
                   ("gate", "up", "down", "expert_output", "moe_output")}
        summary.update({"residual_norm_ratio": 0.1,
                        "exponent_entropy_bits": 2.5,
                        "base_stored_bytes": 800, "mean_alpha": 1.0})
        (out / "diagnostics.json").write_text(json.dumps({"summary": summary}))
    _summary(selected, tmp_path, Namespace(task="arc_easy", eval_limit=0))
    rows = json.loads((tmp_path / "sweep_summary.json").read_text())
    assert rows[1]["condition1_vs_base"] == {
        "projection_error": True, "expert_error": True, "moe_error": True}
    assert rows[1]["base_stored_bits_per_weight"] == 6.4


def test_evaluation_requires_all_tasks(tmp_path):
    out = tmp_path / "setting" / "eval"
    out.mkdir(parents=True)
    args = Namespace(task="arc_easy,arc_challenge", eval_limit=0)
    (out / "metrics.json").write_text(json.dumps({
        "eval_limit": 0, "task_summary": {"tasks": [["arc_easy", 0.5, 0.5]]}}))
    assert not _valid_eval(out.parent, args)
    (out / "metrics.json").write_text(json.dumps({
        "eval_limit": 0, "task_summary": {"tasks": [
            ["arc_easy", 0.5, 0.5], ["arc_challenge", 0.4, 0.4]]}}))
    assert _valid_eval(out.parent, args)
