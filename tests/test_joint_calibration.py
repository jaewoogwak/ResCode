from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.merging.calib_stats import collect_calib_stats, route
from rescode.qjl.calibration import ProjectionSampleCollector, collect_projection_samples


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 3, bias=False)
        self.experts = nn.ModuleList([TinyExpert() for _ in range(3)])

    def forward(self, hidden):
        flat = hidden.reshape(-1, 4)
        selected, weights = route(self, flat, "qwen", 2, False)
        output = torch.zeros_like(flat)
        for eid, expert in enumerate(self.experts):
            token, slot = torch.where(selected == eid)
            if token.numel():
                output.index_add_(0, token, weights[token, slot, None] * expert(flat[token]))
        return output.reshape_as(hidden)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        layer.mlp = TinyBlock()
        self.model.layers = nn.ModuleList([layer])
        self.config = SimpleNamespace(num_experts_per_tok=2, norm_topk_prob=False)

    def forward(self, input_ids):
        return self.model.layers[0].mlp(input_ids)


def test_projection_samples_keep_original_split_and_caps():
    torch.manual_seed(5)
    block = TinyBlock()
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    selected = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 2], [0, 2], [1, 2]])
    weights = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 20
    collector = ProjectionSampleCollector("qwen", {0}, max_train=2, max_heldout=1)

    collector.capture(0, block, hidden, selected, weights)
    assert collector.complete_for({0})
    collector.capture(0, block, hidden + 100, selected, weights)
    train, heldout, probes = collector.result()

    for eid in (0, 1):
        assert torch.equal(train[0]["gate"][eid][0], hidden[0, 1:3])
        assert torch.equal(heldout[0]["gate"][eid][0], hidden[0, :1])
        assert torch.equal(train[0]["gate"][eid][1], weights[1:3, eid].square())
        assert torch.equal(train[0]["down"][eid][0],
                           F.silu(block.experts[eid].gate_proj(hidden[0, 1:3]))
                           * block.experts[eid].up_proj(hidden[0, 1:3]))
    assert torch.equal(probes[0][0], hidden)


def test_joint_pass_matches_two_separate_passes_on_gpu():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(7)
    model = TinyModel().cuda().eval()
    batches = [{"input_ids": torch.randn(2, 9, 4)} for _ in range(3)]

    separate_stats = collect_calib_stats(model, batches, "qwen")
    separate_samples = collect_projection_samples(model, batches, "qwen", {0},
                                                  max_train=3, max_heldout=2)
    collector = ProjectionSampleCollector("qwen", {0}, max_train=3, max_heldout=2)
    joint_stats = collect_calib_stats(model, batches, "qwen",
                                      sample_hook=collector.capture)
    joint_samples = collector.result()

    for field in ("saliency", "freq", "dist", "energy", "tokens"):
        assert torch.equal(torch.as_tensor(separate_stats[0][field]),
                           torch.as_tensor(joint_stats[0][field]))
    for separate, joint in zip(separate_samples[:2], joint_samples[:2]):
        for split in ("gate", "up", "down"):
            for eid, (x, q) in separate[0][split].items():
                joint_x, joint_q = joint[0][split][eid]
                assert torch.equal(x, joint_x)
                assert torch.equal(q, joint_q)
    assert torch.equal(separate_samples[2][0][0], joint_samples[2][0][0])
