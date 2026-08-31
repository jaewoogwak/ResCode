from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class ExpertClusterMap:
    """Maps original logical expert ids to physical (cluster) ids for one MoE layer.

    Cluster ids are dense integers in [0, num_clusters). For singleton clusters
    is_singleton[cid] is True; the meta path is skipped for them and the
    original expert is kept untouched.
    """

    layer_idx: int
    num_original_experts: int
    clusters: List[List[int]]
    expert_to_cluster: Dict[int, int]
    cluster_to_experts: Dict[int, List[int]]
    is_singleton: Dict[int, bool]
    expert_to_physical: Dict[int, int]

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)

    @property
    def num_singletons(self) -> int:
        return sum(1 for v in self.is_singleton.values() if v)

    @property
    def num_meta_clusters(self) -> int:
        return self.num_clusters - self.num_singletons

    @property
    def largest_cluster_size(self) -> int:
        return max(len(c) for c in self.clusters) if self.clusters else 0

    def cluster_sizes(self) -> List[int]:
        return [len(c) for c in self.clusters]

    def to_dict(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "num_original_experts": self.num_original_experts,
            "clusters": self.clusters,
            "expert_to_cluster": self.expert_to_cluster,
            "cluster_to_experts": self.cluster_to_experts,
            "is_singleton": self.is_singleton,
            "expert_to_physical": self.expert_to_physical,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExpertClusterMap":
        return cls(
            layer_idx=int(d["layer_idx"]),
            num_original_experts=int(d["num_original_experts"]),
            clusters=[[int(e) for e in c] for c in d["clusters"]],
            expert_to_cluster={int(k): int(v) for k, v in d["expert_to_cluster"].items()},
            cluster_to_experts={int(k): [int(e) for e in v] for k, v in d["cluster_to_experts"].items()},
            is_singleton={int(k): bool(v) for k, v in d["is_singleton"].items()},
            expert_to_physical={int(k): int(v) for k, v in d["expert_to_physical"].items()},
        )


def _build_one_layer(layer_idx: int, group_labels: torch.Tensor) -> ExpertClusterMap:
    """Build a cluster map for one MoE layer from an RESCODE group_labels tensor.

    `group_labels[j]` is the cluster label of original expert j. The labels do
    not need to be contiguous — we re-index to 0..num_clusters-1.
    """
    if not torch.is_tensor(group_labels):
        group_labels = torch.tensor(group_labels)
    group_labels = group_labels.long().cpu()

    num_experts = int(group_labels.numel())
    raw_to_dense: Dict[int, int] = {}
    clusters: List[List[int]] = []

    for j in range(num_experts):
        raw = int(group_labels[j].item())
        if raw not in raw_to_dense:
            raw_to_dense[raw] = len(clusters)
            clusters.append([])
        clusters[raw_to_dense[raw]].append(j)

    expert_to_cluster: Dict[int, int] = {}
    cluster_to_experts: Dict[int, List[int]] = {}
    is_singleton: Dict[int, bool] = {}
    expert_to_physical: Dict[int, int] = {}

    for cid, members in enumerate(clusters):
        cluster_to_experts[cid] = list(members)
        is_singleton[cid] = (len(members) == 1)
        for j in members:
            expert_to_cluster[j] = cid
            # Physical id space: same dense id as cluster id; for singletons
            # this is just a 1-to-1 mapping, for meta clusters all members
            # share one physical id.
            expert_to_physical[j] = cid

    return ExpertClusterMap(
        layer_idx=layer_idx,
        num_original_experts=num_experts,
        clusters=clusters,
        expert_to_cluster=expert_to_cluster,
        cluster_to_experts=cluster_to_experts,
        is_singleton=is_singleton,
        expert_to_physical=expert_to_physical,
    )


def build_cluster_maps_from_group_state(
    group_state_dict: Dict[str, torch.Tensor],
    layer_indices: List[int],
    ffn_name_template: str = "model.layers.{layer_idx}.mlp",
) -> Dict[int, ExpertClusterMap]:
    """Build per-layer cluster maps from an RESCODE grouper's group_state_dict."""
    out: Dict[int, ExpertClusterMap] = {}
    for layer_idx in layer_indices:
        ffn_name = ffn_name_template.format(layer_idx=layer_idx)
        if ffn_name not in group_state_dict:
            raise KeyError(
                f"[RESCODE] group_state_dict missing key '{ffn_name}'. "
                f"Available keys: {list(group_state_dict.keys())[:4]}..."
            )
        out[layer_idx] = _build_one_layer(layer_idx, group_state_dict[ffn_name])
    return out
