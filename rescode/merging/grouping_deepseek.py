import os
import pickle
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from types import MethodType

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# DeepSeek-V1/V2 ship via trust_remote_code (no stable class import in
# transformers). We hint generically and rely on attribute duck-typing.
from transformers import AutoModelForCausalLM, AutoConfig

# Type-alias hacks: keep all `Qwen2Moe*` annotations valid without a hard
# dependency on a class that doesn't exist for DeepSeek. These aliases are
# only used in type hints and the rare zipit-merge path.
Qwen2MoeForCausalLM = nn.Module  # type: ignore[assignment, misc]
Qwen2MoeConfig = object  # type: ignore[assignment, misc]
Qwen2MoeSparseMoeBlock = nn.Module  # type: ignore[assignment, misc]
Qwen2MoeMLP = nn.Module  # type: ignore[assignment, misc]

from .utils import generate_random_group_labels
from rescode.utils.constants import FP32_EPS
from rescode.utils.helper import print_gpu_memory
# Reuse the Qwen2 zipit utilities as fallbacks: DeepSeek expert layout matches
# Qwen (gate_proj/up_proj/down_proj). Main flow (hc_merge="freq"/"saliency")
# does NOT exercise these symbols.
from rescode.models.qwen.utils import (
    merged_qwen2moe_forward,
    Qwen2MoEWrapper,
    ModifiedQwen2MoeSparseMoeBlock,
)
from rescode.merging.clustering import group_experts_by_clustering
from rescode.merging.overlap import compute_kl_divergence, get_prob_distributions, compute_wasserstein_distance

def _forward_with_optional_router_logits(model, batch):
    """DeepSeek stock modeling does not accept `output_router_logits`. Try
    with the kwarg first; fall back to a plain forward. The fallback path
    means callers that consume `outputs.router_logits` will see `None` (or
    an attribute error if the field is missing) - use similarity bases that
    don't depend on router_logits (e.g. `expert-output`, `rescode`) for
    DeepSeek runs.
    """
    try:
        return model(**batch, output_router_logits=True)
    except TypeError:
        return model(**batch)


SIMILARITY_MAPPING_FUNCTION = {
    "cosine": lambda x, y: (F.cosine_similarity(x, y, dim=-1, eps=FP32_EPS) + 1).item() / 2,
    "mse": lambda x, y: 1 / (1 + 0.1 * torch.log(F.mse_loss(x, y, reduction="sum"))).item(),
}

LEGAL_SIMILARITY_BASES = ["weight", "feature", "feature.abs", "weight-feature", "gradient", "weight-gradient",
                          "router-logits", "router-weight", "router-weight-feature", "mse", "random", "no",
                          "feature-correlation.lsa", "feature-correlation.max", "expert-output", "weight+expert-output",
                          "router-logits+weight", "router-logits+expert-output", "router-logits+weight+expert-output",
                          "expert-output-excl-topk", "reap-weighted-output-routed", "rescode"]

class ExpertsGrouperForDeepSeekMoE(object):
    def __init__(
            self,
            config: Qwen2MoeConfig,
            similarity_fn: str = "cosine",
            similarity_base: str = "router-logits",
            start_layer: int = 0,
            group_limit: int = 4,
            data_limit: int = 1000000,
            random_start_center: bool = False,
            cluster: str = "kmeans",
            linkage: str = "ward",
            hierarchical_stopping_metric: str = "silhouette",
            overlap_metric: str = "cosine",
            dynamic_group: bool = False,
            descriptor_tokens: str = "all",   # "all" | "routed"
    ):
        if similarity_fn not in SIMILARITY_MAPPING_FUNCTION:
            raise ValueError(
                f"[RESCODE]similarity_fn should be one of {SIMILARITY_MAPPING_FUNCTION.keys()}, got {similarity_fn} instead."
            )
        if similarity_base not in LEGAL_SIMILARITY_BASES:
            raise ValueError(
                f"[RESCODE] similarity_base should be one of {LEGAL_SIMILARITY_BASES}, got {similarity_base} instead.")
        if descriptor_tokens not in ("all", "routed"):
            raise ValueError(
                f"[RESCODE] descriptor_tokens should be 'all' or 'routed', got {descriptor_tokens} instead.")

        # DeepSeek calls it `n_routed_experts`; fall back to `num_experts` for
        # safety when wrapped configs expose that alias.
        self.num_experts = int(
            getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
        )
        self.d_model = config.hidden_size
        self.d_ff = config.moe_intermediate_size
        self.top_k = config.num_experts_per_tok
        self.group_limit = group_limit
        self.data_limit = data_limit
        self.random_start_center = random_start_center
        self.cluster = cluster
        self.linkage = linkage
        self.hierarchical_stopping_metric = hierarchical_stopping_metric
        self.overlap_metric = overlap_metric
        self.dynamic_group = dynamic_group
        self.descriptor_tokens = descriptor_tokens

        # Skip the dense layer(s) at the front of the stack. DeepSeek sets
        # `first_k_dense_replace=1` (layer 0 is dense, all others are MoE).
        first_k_dense = int(getattr(config, "first_k_dense_replace", 0))
        effective_start = max(int(start_layer), first_k_dense)
        self.sparse_layer_indices = list(range(effective_start, config.num_hidden_layers))
        self.similarity_fn = SIMILARITY_MAPPING_FUNCTION[similarity_fn]
        self.similarity_base = similarity_base
        self._group_state_dict = None
        self._similarity_state_dict = None
        self._usage_frequency_state_dict = None
        self._init_center_state_dict = None
        self.reset_all()


    def reset_all(self):
        if self.similarity_base == "mse":
            self.similarity_fn = SIMILARITY_MAPPING_FUNCTION["mse"]
            print("[RESCODE]Set similarity_fn to mse for mse similarity_base.")
        self._group_state_dict = dict()
        self._similarity_state_dict = dict()
        self._usage_frequency_state_dict = dict()
        self._init_center_state_dict = dict()
        # RESCODE: saliency a_i and routed-output descriptor o_i per expert.
        self._saliency_state_dict = dict()
        self._descriptor_state_dict = dict()
        # Similarity range: [0, 2]
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            self._group_state_dict[ffn_name] = torch.arange(self.num_experts, device="cpu")
            self._similarity_state_dict[ffn_name] = torch.zeros(
                (self.num_experts, self.num_experts), device="cpu") + torch.eye(self.num_experts, device="cpu")
            self._usage_frequency_state_dict[ffn_name] = torch.zeros(self.num_experts, device="cpu")
            self._saliency_state_dict[ffn_name] = torch.zeros(self.num_experts, device="cpu")
            self._descriptor_state_dict[ffn_name] = None

    def similarity_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._similarity_state_dict)

    def group_state_dict(self) -> Dict[str, torch.LongTensor]:
        return deepcopy(self._group_state_dict)

    def usage_frequency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._usage_frequency_state_dict)

    def saliency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._saliency_state_dict)

    def descriptor_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: (v.clone() if v is not None else None)
                for k, v in self._descriptor_state_dict.items()}

    def save_similarity(self, mlp_name: str, i: int, j: int, similarity: float):
        self._similarity_state_dict[mlp_name][i, j] = similarity
        self._similarity_state_dict[mlp_name][j, i] = similarity

    def get_similarity(self, mlp_name: str, i: int, j: int) -> float:
        return self._similarity_state_dict[mlp_name][i, j].item()

    def get_similarity_matrix(self, mlp_name: str) -> torch.Tensor:
        return deepcopy(self._similarity_state_dict[mlp_name])

    def save_group_state_dict(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._group_state_dict, os.path.join(save_dir, "group_state_dict.pt"))

    def load_group_state_dict(self, load_dir: str):
        self._group_state_dict = torch.load(os.path.join(load_dir, "group_state_dict.pt"))

    def load_init_center_state_dict(self, load_path: str):
        init_centers = pickle.load(open(load_path, "rb"))
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            self._init_center_state_dict[ffn_name] = torch.tensor(init_centers[layer_idx])

    def _assign_num_groups_per_layer(
            self,
            num_average_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, int]:
        num_grouping_layers = len(merging_layers)
        total_num_groups = num_average_groups * num_grouping_layers + self.num_experts * (
                len(self.sparse_layer_indices) - num_grouping_layers
        )
        print("total_num_groups: ", total_num_groups)
        all_usage_frequency = []
        usage_frequency_dict = deepcopy(self._usage_frequency_state_dict)
        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"

            # 1. Experts in the excluded layers are always not merged.
            if layer_idx not in merging_layers:
                usage_frequency_dict[ffn_name] = torch.ones_like(usage_frequency_dict[ffn_name])

            # 2. Each layer must have at least one group, set the most used expert in a layer to frequency 1.
            k = (self.num_experts // self.group_limit) + 1 if (self.num_experts % self.group_limit) != 0 else (self.num_experts // self.group_limit)
            # max_usage_index = torch.argmax(usage_frequency_dict[ffn_name])
            # usage_frequency_dict[ffn_name][max_usage_index] = 1.0

            # 3. Collect all usage frequency.
            all_usage_frequency.append(usage_frequency_dict[ffn_name])

        all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
        sorted_usage_frequency, sorted_indices = torch.sort(all_usage_frequency, descending=True)
        num_groups_per_layer = dict()

        # Note: When threshold is 0.0, the actual number of groups is smaller than total_num_groups.
        if num_average_groups == self.num_experts:
            total_num_groups = total_num_groups - 1
        frequency_threshold = sorted_usage_frequency[total_num_groups]
        print(f"[RESCODE] Frequency threshold: {frequency_threshold}")

        if frequency_threshold == 1.0:
            raise ValueError("[RESCODE] The number of groups is too large, please reduce the number of groups.")

        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups_per_layer[ffn_name] = torch.sum(
                (usage_frequency_dict[ffn_name] > frequency_threshold).long()
            ).item()

        return num_groups_per_layer

    def group_experts_randomly(
        self,
        num_groups: int,
    ):
        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc=f"Randomly merging experts into {num_groups} clusters"):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            group_labels = generate_random_group_labels(self.num_experts, num_groups)
            self._group_state_dict[ffn_name] = group_labels
    
    #NOTE: Sihouette Score
    def compute_sihouette_score(self, model, dataloader):
        if self.similarity_base == "expert-output":
            # collect expert outputs
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            forwarded_hidden_states = {}
            handles = []
            def _get_activation_hook(name):
                def hook(module, input, output):
                    forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1])) # .cpu()
                return hook
            
            for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[RESCODE] Registering forward hook..."):
                ffn_name = f"model.layers.{layer_idx}.mlp"
                forwarded_hidden_states[ffn_name] = []
                moe = model.model.layers[layer_idx].mlp
                handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))

            for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to collect moe inputs"):
                batch = {k: v.cuda() for k, v in batch.items()}
                if "labels" in batch:
                    batch.pop("labels")
                with torch.no_grad():
                    outputs = model(**batch)
                    del outputs
            
            for handle in handles:
                handle.remove()

            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                layer_input = torch.cat(forwarded_hidden_states[ffn_name]) # .cuda()
                expert_outputs = [] # (E, #T, D) -> average -> (E, D)
                with torch.no_grad():
                    for i in range(self.num_experts):
                        expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                    expert_outputs = torch.stack(expert_outputs)
                    score = self.silhouette_score(expert_outputs, self._group_state_dict[ffn_name])
                    print(f"layer {layer_idx}: {score}")
                del layer_input
            del forwarded_hidden_states
        elif self.similarity_base == "weight":
            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                moe = model.model.layers[layer_idx].mlp
                experts = []
                for i in range(self.num_experts):
                    weight_flat = torch.cat(
                        [moe.experts[i].gate_proj.weight.flatten(),
                        moe.experts[i].down_proj.weight.flatten(),
                        moe.experts[i].up_proj.weight.flatten()],
                        dim=0
                    )
                    experts.append(weight_flat.to(torch.float))
                experts = torch.stack(experts)
                score = self.silhouette_score(experts, self._group_state_dict[ffn_name])
                del experts
                print(f"layer {layer_idx}: {score}")
        elif self.similarity_base == "router-logits":
            all_router_logits = []
            for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to get routing logits"):
                batch = {k: v.cuda() for k, v in batch.items()}
                if "labels" in batch:
                    batch.pop("labels")
                with torch.no_grad():
                    outputs = _forward_with_optional_router_logits(model, batch)
                batch_router_logits = outputs.router_logits
                batch_router_logits = torch.stack(batch_router_logits)
                all_router_logits.append(batch_router_logits)
                del outputs
            all_router_logits = torch.cat(all_router_logits, dim=1)
            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T
                score = self.silhouette_score(layer_router_logits, self._group_state_dict[ffn_name])
                print(f"layer {layer_idx}: {score}")
        else:
            pass

    def silhouette_score(self, tensor_list, cluster_labels):
        """
        Compute the silhouette score based on a list of tensors, 
        the cluster assignments, and the dominant experts for each group.
        """
        
        def compute_pairwise_distances(tensor_list):
            """Compute pairwise distances between all tensors in the list."""
            num_tensors = tensor_list.shape[0]
            distances = torch.zeros((num_tensors, num_tensors))

            with torch.no_grad():
                for i in range(num_tensors):
                    for j in range(i, num_tensors):
                        # print(i, j)
                        # print(torch.cuda.memory_summary(device=4))
                        dist = torch.norm(tensor_list[i] - tensor_list[j])
                        # print(dist)
                        distances[i, j] = dist
                        distances[j, i] = dist  # Symmetric matrix

            return distances
        
        # Step 1: Compute pairwise distances
        pairwise_distances = compute_pairwise_distances(tensor_list)

        num_tensors = tensor_list.shape[0]
        unique_labels = torch.unique(cluster_labels)
        num_clusters = len(unique_labels)

        silhouette_scores = torch.zeros(num_tensors)

        # Step 2: For each sample, compute silhouette score
        for i in range(num_tensors):
            # a(i): Mean intra-cluster distance (within the same cluster)
            same_cluster = [j for j in range(num_tensors) if cluster_labels[j] == cluster_labels[i] and j != i]
            if len(same_cluster) > 0:
                a_i = torch.mean(pairwise_distances[i, same_cluster])
            else:
                a_i = 0  # If there are no other points in the cluster

            # b(i): Mean nearest-cluster distance (distance to points in the nearest cluster)
            b_i = float('inf')
            for label in unique_labels:
                if label == cluster_labels[i]:
                    continue
                other_cluster = [j for j in range(num_tensors) if cluster_labels[j] == label]
                if len(other_cluster) > 0:
                    mean_dist_to_other_cluster = torch.mean(pairwise_distances[i, other_cluster])
                    b_i = min(b_i, mean_dist_to_other_cluster)

            # Step 3: Compute silhouette score for the sample
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

        # Step 4: Average silhouette score across all samples
        overall_silhouette_score = torch.mean(silhouette_scores)

        return overall_silhouette_score
    
    
    #NOTE: Clustering
    def cluster_experts(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
            exclude_top_percent: float = 0.0,
            excl_mode: str = "per-layer",
    ):
        if self.similarity_base == "weight":
            dom_experts = self.group_experts_by_clustering_weight(
                model=model,
                num_groups=num_groups
            )
        elif self.similarity_base == "expert-output":
            dom_experts = self.group_experts_by_clustering_output(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "expert-output-excl-topk":
            dom_experts = self.group_experts_by_clustering_output_excl_topk(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups,
                exclude_top_percent=float(exclude_top_percent),
                excl_mode=str(excl_mode),
            )
        elif self.similarity_base == "reap-weighted-output-routed":
            dom_experts = self.group_experts_by_clustering_reap_weighted_output_routed(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups,
            )
        elif self.similarity_base == "rescode":
            dom_experts = self.group_experts_by_clustering_rescode(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups,
                exclude_top_percent=float(exclude_top_percent),
                excl_mode=str(excl_mode),
            )
        elif self.similarity_base == "weight+expert-output":
            dom_experts = self.group_experts_by_clustering_weight_and_output(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "router-logits":
            dom_experts = self.group_experts_by_clustering_router_score(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "router-logits+weight":
            dom_experts = self.group_experts_by_clustering_router_score_and_weight(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "router-logits+expert-output":
            dom_experts = self.group_experts_by_clustering_router_score_and_output(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        elif self.similarity_base == "router-logits+weight+expert-output":
            dom_experts = self.group_experts_by_clustering_router_score_and_weight_and_output(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )
        else:
            raise ValueError(f"Unknown similarity base: {self.similarity_base}")
        return dom_experts
    
    def group_experts_by_clustering_weight_layerwise(
            self,
            moe: Qwen2MoeSparseMoeBlock,
            ffn_name: str,
            num_groups: int,
    ):
        experts = []
        for i in range(self.num_experts):
            weight_flat = torch.cat(
                [moe.experts[i].gate_proj.weight.flatten(),
                moe.experts[i].down_proj.weight.flatten(),
                moe.experts[i].up_proj.weight.flatten()],
                dim=0
            )
            experts.append(weight_flat.to(torch.float))
        experts = torch.stack(experts)
        dom_experts, label = group_experts_by_clustering(
            model="qwen",
            num_groups=num_groups,
            cluster=self.cluster,
            linkage=self.linkage,
            hierarchical_stopping_metric=self.hierarchical_stopping_metric,
            num_experts=self.num_experts,
            experts=experts,
            init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
        self._group_state_dict[ffn_name] = label.cpu()
        return dom_experts
    
    def group_experts_by_clustering_weight(
        self,
        model: Qwen2MoeForCausalLM,
        num_groups: int,
    ):
        model.eval()
        dom_experts = dict()
        if self.dynamic_group:
            num_groups_per_layer = self._assign_num_groups_per_layer(
                num_groups, self.sparse_layer_indices
            )
            for k in num_groups_per_layer.keys():
                print(f"{k}: {num_groups_per_layer[k]}")
        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[RESCODE] Clustering experts by weight"):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            num_groups_in_layer = num_groups_per_layer[ffn_name] if self.dynamic_group else num_groups
            dom_experts[ffn_name] = self.group_experts_by_clustering_weight_layerwise(moe, ffn_name, num_groups_in_layer)
        return dom_experts
    
    def group_experts_by_clustering_output_layerwise(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
            layer_idx: int,
            ffn_name: str,
            num_groups: int,
    ):
        model.eval()
        moe_input = []
        moe = model.model.layers[layer_idx].mlp
        
        def _get_hook(_, input, __): # module, input, output
            moe_input.append(input[0].detach().reshape(-1, input[0].shape[-1])) # .cpu()
        
        with torch.no_grad():
            handle = moe.register_forward_hook(_get_hook)
            for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to collect moe inputs"):
                batch = {k: v.cuda() for k, v in batch.items()}
                if "labels" in batch:
                    batch.pop("labels")
                outputs = model(**batch)
            handle.remove()
            torch.cuda.empty_cache()
            
            layer_input = torch.cat(moe_input)
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            for i in range(self.num_experts):
                expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
            expert_outputs = torch.stack(expert_outputs)

            dom_experts, label = group_experts_by_clustering(
                model="qwen",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=expert_outputs,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
            return dom_experts

            return self.group_experts_by_clustering(ffn_name, num_groups, expert_outputs)

    def group_experts_by_clustering_output(
        self,
        model: Qwen2MoeForCausalLM,
        dataloader: DataLoader,
        num_groups: int,
    ):
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging]Registering forward hook..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))

        for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        if self.dynamic_group:
            num_groups_per_layer = self._assign_num_groups_per_layer(
                num_groups, self.sparse_layer_indices
            )

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] Computing similarities by expert outputs..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            _device = model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.device
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]).to(_device) # .cuda()
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            with torch.no_grad():
                for i in range(self.num_experts):
                    expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                expert_outputs = torch.stack(expert_outputs)
                num_groups_in_layer = num_groups_per_layer[ffn_name] if self.dynamic_group else num_groups
                dom_experts[ffn_name], label = group_experts_by_clustering(
                    model="qwen",
                    num_groups=num_groups_in_layer,
                    cluster=self.cluster,
                    linkage=self.linkage,
                    hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                    num_experts=self.num_experts,
                    experts=expert_outputs,
                    init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
                self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts
    
    def group_experts_by_clustering_weight_and_output(
        self,
        model: Qwen2MoeForCausalLM,
        dataloader: DataLoader,
        num_groups: int,
    ):
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging]Registering forward hook..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))

        for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] Computing similarities by weights and expert outputs..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            _device = model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.device
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]).to(_device) # .cuda()
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            weights = []
            with torch.no_grad():
                for i in range(self.num_experts):
                    weight_flat = torch.cat(
                        [moe.experts[i].gate_proj.weight.flatten(),
                        moe.experts[i].down_proj.weight.flatten(),
                        moe.experts[i].up_proj.weight.flatten()],
                        dim=0
                    )
                    weights.append(weight_flat)
                    expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                expert_outputs = torch.stack(expert_outputs)
                weights = torch.stack(weights)
                dom_experts[ffn_name], label = group_experts_by_clustering(
                    model="qwen",
                    num_groups=num_groups,
                    cluster=self.cluster,
                    linkage=self.linkage,
                    hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                    num_experts=self.num_experts,
                    experts=weights,
                    experts2=expert_outputs,
                    init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
                self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts

    def group_experts_by_clustering_router_score(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        print("group_experts_by_clustering_router_score")
        model.eval()
        dom_experts = dict()
        all_router_logits = []
        for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = _forward_with_optional_router_logits(model, batch)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)
            del outputs 
        
        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, num_tokens, num_experts)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T
            dom_experts[ffn_name], label = group_experts_by_clustering(
                model="qwen",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=layer_router_logits,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
        torch.cuda.empty_cache()
        return dom_experts

    def group_experts_by_clustering_router_score_and_weight(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        print("group_experts_by_clustering_router_score_weight")
        model.eval()
        dom_experts = dict()
        all_router_logits = []
        for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = _forward_with_optional_router_logits(model, batch)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)
            del outputs 
        
        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, num_tokens, num_experts)
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            all_weights = []
            for i in range(self.num_experts):
                weight_flat = torch.cat(
                        [moe.experts[i].gate_proj.weight.flatten(),
                        moe.experts[i].down_proj.weight.flatten(),
                        moe.experts[i].up_proj.weight.flatten()],
                        dim=0
                    )
                all_weights.append(weight_flat)
            weights = torch.stack(all_weights)
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T.to(weights.device)
            all_weights.clear()
            dom_experts[ffn_name], label = group_experts_by_clustering(
                model="qwen",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=layer_router_logits,
                experts2=weights,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
        torch.cuda.empty_cache()
        return dom_experts

    def group_experts_by_clustering_router_score_and_output(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        print("group_experts_by_clustering_router_score_output")
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []
        all_router_logits = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook

        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging]Registering forward hook..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))
        

        for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = _forward_with_optional_router_logits(model, batch)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)
            del outputs 
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, num_tokens, num_experts)
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            _device = model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.device
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]).to(_device)
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            for i in range(self.num_experts):
                output = model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0)
                expert_outputs.append(output)
            expert_outputs = torch.stack(expert_outputs)
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T.to(expert_outputs.device)
            dom_experts[ffn_name], label = group_experts_by_clustering(
                model="qwen",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=layer_router_logits,
                experts2=expert_outputs,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts

    def group_experts_by_clustering_router_score_and_weight_and_output(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        print("group_experts_by_clustering_router_score_output")
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []
        all_router_logits = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook

        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging]Registering forward hook..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))
        

        for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = _forward_with_optional_router_logits(model, batch)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)
            del outputs 
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, num_tokens, num_experts)
        
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            _device = model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.device
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]).to(_device)
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            weights = []
            for i in range(self.num_experts):
                output = model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0)
                weight_flat = torch.cat(
                        [moe.experts[i].gate_proj.weight.flatten(),
                        moe.experts[i].down_proj.weight.flatten(),
                        moe.experts[i].up_proj.weight.flatten()],
                        dim=0
                    )
                weights.append(weight_flat)
                expert_outputs.append(output)
            expert_outputs = torch.stack(expert_outputs)
            weights = torch.stack(weights)
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts).T.to(weights.device)
            dom_experts[ffn_name], label = group_experts_by_clustering(
                model="qwen",
                num_groups=num_groups,
                cluster=self.cluster,
                linkage=self.linkage,
                hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                num_experts=self.num_experts,
                experts=layer_router_logits,
                experts2=weights,
                experts3=expert_outputs,
                init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
            self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts

    
    def group_experts_globally_from_dominant_experts(
            self,
            num_average_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, List[int]]:
        """
        Globally group experts into clusters by routing-guided clustering, each layer will have different number of
         clusters. The total number of clusters is determined by num_average_groups.

        Parameters
        ----------
        num_average_groups: int
            The average number of clusters for all layers.
        merging_layers: List[int]
            The layers of decoder that are excluded from merging.

        Returns
        -------
        core_experts: Dict[str, List[int]]
            The core experts of each cluster
        """

        # 1. Assign num_groups respectively for each layer according to num_average_groups
        num_groups_per_layer = self._assign_num_groups_per_layer(
            num_average_groups, merging_layers
        )
        print(f"[RESCODE] Number of groups per layer: {num_groups_per_layer}")

        # 2. Group experts into clusters for each layer
        dom_experts = dict()
        for layer_idx in tqdm(
                self.sparse_layer_indices,
                desc=f"[RESCODE] Globally routing-guided grouping experts into average {num_average_groups} clusters"
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups = num_groups_per_layer[ffn_name]
            group_member_count = torch.zeros(num_groups)

            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[ffn_name], descending=True)

            # 1 Assign top-K most-used experts with label 0 to K-1 respectively
            group_dict = {} 
            core_expert_indices = indices_sorted_by_usage[:num_groups]
            dom_experts[ffn_name] = core_expert_indices.tolist()
            for i in range(num_groups):
                self._group_state_dict[ffn_name][indices_sorted_by_usage[i]] = i
                group_member_count[i] += 1
                group_dict[i] = [core_expert_indices[i].item()]
            # 2 Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(ffn_name)
            print(similarity_matrix)
            print(core_expert_indices)
            for i in range(0, self.num_experts):
                if i in core_expert_indices:
                    continue
                # Find the most similar core
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[i, core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[ffn_name][most_similar_core]
                self._group_state_dict[ffn_name][i] = most_similar_group_label
                group_member_count[most_similar_group_label] += 1
                group_dict[most_similar_group_label.item()].append(i)
                print(f"--expert {i} is assigned to group {most_similar_group_label}, the core expert is {most_similar_core}")
                if group_member_count[self._group_state_dict[ffn_name][i]] > self.group_limit:
                    if len(core_expert_indices) == 1 and self.group_limit < self.num_experts:
                        raise ValueError(
                            f"[Merging]The number of groups at Encoder layer {layer_idx} is too small!"
                        )
                    
                    while group_member_count[most_similar_group_label] > self.group_limit:
                        print(f"----meet group limit {self.group_limit} with group {most_similar_group_label} (core: {most_similar_core})")
                        # Find the most unsimilar expert in the exceed group
                        sim = similarity_matrix[most_similar_core, group_dict[most_similar_group_label.item()]]
                        print(sim, group_dict[most_similar_group_label.item()])
                        unsimilar_pos = torch.argmin(sim).item()
                        if (unsimilar_pos == 0): # do not let it choose the dominant expert
                            unsimilar_pos = 1
                        unsimilar_idx = group_dict[most_similar_group_label.item()][unsimilar_pos]
                    
                        group_member_count[most_similar_group_label] -= 1
                        group_dict[most_similar_group_label.item()].remove(unsimilar_idx)
                        similarity_matrix[unsimilar_idx, most_similar_core] = -100
                        similarity_matrix[most_similar_core, unsimilar_idx] = -100
                        print(f"----kick out {unsimilar_idx} from group ")
                        # Reassign group label
                        print(similarity_matrix[unsimilar_idx, core_expert_indices])
                        most_similar_core = core_expert_indices[
                            torch.argmax(similarity_matrix[unsimilar_idx, core_expert_indices])
                        ]
                        most_similar_group_label = self._group_state_dict[ffn_name][most_similar_core]
                        self._group_state_dict[ffn_name][unsimilar_idx] = most_similar_group_label
                        group_member_count[most_similar_group_label] += 1
                        group_dict[most_similar_group_label.item()].append(unsimilar_idx)
                        print(f"--expert {unsimilar_idx} is assigned to group {most_similar_group_label}, the core expert is {most_similar_core}")
        return dom_experts

    # ============================================================ excl-topk
    def _compute_global_excluded_set(
        self,
        exclude_top_percent: float,
        excl_mode: str = "per-layer",
    ) -> Dict[str, torch.LongTensor]:
        """Experts in the top k% by top-2 pick count -> per-layer excluded indices.

        excl_mode:
          - "global"   : pick the top k% from the pooled (layer, expert) set.
          - "per-layer": pin exactly round(E*k/100) experts within every layer (default).

        compute_all_usages(mode="frequency") must be called beforehand.
        Returns {ffn_name: LongTensor of excluded expert indices}.
        """
        if exclude_top_percent <= 0.0:
            return {f"model.layers.{i}.mlp": torch.empty(0, dtype=torch.long)
                    for i in self.sparse_layer_indices}

        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            freq = self._usage_frequency_state_dict[ffn_name]
            if freq.sum().item() == 0:
                raise RuntimeError(
                    f"[excl-topk] usage_frequency not populated for {ffn_name}. "
                    f"compute_all_usages must be called before clustering."
                )

        excluded_per_layer: Dict[str, List[int]] = {
            f"model.layers.{i}.mlp": [] for i in self.sparse_layer_indices
        }

        if excl_mode == "per-layer":
            per_layer_k = int(round(self.num_experts * exclude_top_percent / 100.0))
            per_layer_k = max(0, min(per_layer_k, self.num_experts))
            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                freq = self._usage_frequency_state_dict[ffn_name]
                top_idx = (torch.topk(freq, k=per_layer_k, largest=True).indices.tolist()
                           if per_layer_k > 0 else [])
                excluded_per_layer[ffn_name] = sorted(int(x) for x in top_idx)
            print(f"[excl-topk] mode=per-layer  k={per_layer_k}/{self.num_experts} per layer "
                  f"(={exclude_top_percent}%, total={per_layer_k * len(self.sparse_layer_indices)} pairs).")
        else:
            all_scores = []
            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                freq = self._usage_frequency_state_dict[ffn_name]
                for e_idx in range(self.num_experts):
                    all_scores.append((layer_idx, e_idx, freq[e_idx].item()))
            total = len(all_scores)
            k_count = int(round(total * exclude_top_percent / 100.0))
            all_scores.sort(key=lambda x: x[2], reverse=True)
            for layer_idx, e_idx, _ in all_scores[:k_count]:
                excluded_per_layer[f"model.layers.{layer_idx}.mlp"].append(e_idx)
            print(f"[excl-topk] mode=global  Excluding top {exclude_top_percent}% = {k_count}/{total} (layer,expert) pairs.")

        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            ex = sorted(excluded_per_layer[ffn_name])
            print(f"  {ffn_name}: {len(ex)} excluded -> {ex}")

        return {k: torch.tensor(sorted(v), dtype=torch.long) for k, v in excluded_per_layer.items()}

    def group_experts_by_clustering_output_excl_topk(
        self,
        model: Qwen2MoeForCausalLM,
        dataloader: DataLoader,
        num_groups: int,
        exclude_top_percent: float = 15.0,
        excl_mode: str = "per-layer",
    ):
        """Cluster experts by output similarity, but pin top-k% (by usage freq)
        as singletons. Remaining experts are clustered into (num_groups - n_excl).
        """
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []

        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(
                    input[0].detach().cpu().reshape(-1, input[0].shape[-1])
                )
            return hook

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[excl-topk] Registering hooks..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))

        for batch in tqdm(dataloader, desc="[excl-topk] Forward to collect MoE inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs

        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        excluded_per_layer = self._compute_global_excluded_set(exclude_top_percent, excl_mode=excl_mode)

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[excl-topk] Clustering non-excluded..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            _device = model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.device
            with torch.no_grad():
                layer_input = torch.cat(forwarded_hidden_states[ffn_name]).to(_device)
                expert_outputs = []
                for i in range(self.num_experts):
                    expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                expert_outputs = torch.stack(expert_outputs)
                del layer_input

            excluded = excluded_per_layer[ffn_name].tolist()
            excluded_set = set(excluded)
            kept_indices = [i for i in range(self.num_experts) if i not in excluded_set]
            num_kept = len(kept_indices)
            num_excluded = len(excluded)

            target_clusters_for_kept = num_groups - num_excluded
            if target_clusters_for_kept < 1:
                raise ValueError(
                    f"[excl-topk] {ffn_name}: excluded={num_excluded} >= num_groups={num_groups}. "
                    f"Lower exclude_top_percent or raise num_groups."
                )
            if target_clusters_for_kept > num_kept:
                target_clusters_for_kept = num_kept

            if num_kept > 0:
                kept_feats = expert_outputs[kept_indices]
                _, kept_labels = group_experts_by_clustering(
                    model="qwen",
                    num_groups=target_clusters_for_kept,
                    cluster=self.cluster,
                    linkage=self.linkage,
                    hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                    num_experts=num_kept,
                    experts=kept_feats,
                    init_center=None,
                )
                kept_labels = kept_labels.cpu().tolist() if torch.is_tensor(kept_labels) else list(kept_labels)
            else:
                kept_labels = []

            label_tensor = torch.zeros(self.num_experts, dtype=torch.long)
            for rank, ex_idx in enumerate(excluded):
                label_tensor[ex_idx] = rank
            for pos, kept_idx in enumerate(kept_indices):
                label_tensor[kept_idx] = num_excluded + int(kept_labels[pos])

            self._group_state_dict[ffn_name] = label_tensor

            dom_list: List[int] = []
            for rank, ex_idx in enumerate(excluded):
                dom_list.append(ex_idx)
            if num_kept > 0:
                kept_labels_t = torch.tensor(kept_labels, dtype=torch.long)
                expert_outputs_f32 = expert_outputs.float()
                for k in range(target_clusters_for_kept):
                    mask = (kept_labels_t == k)
                    members = [kept_indices[j] for j in range(num_kept) if mask[j].item()]
                    if len(members) == 0:
                        continue
                    member_feats = expert_outputs_f32[members]
                    centroid = member_feats.mean(dim=0, keepdim=True)
                    dists = torch.cdist(member_feats, centroid).squeeze(-1)
                    dom_list.append(members[int(torch.argmin(dists).item())])
            dom_experts[ffn_name] = dom_list

            del expert_outputs
        torch.cuda.empty_cache()
        return dom_experts

    # ============================================================ REAP-weighted
    def group_experts_by_clustering_reap_weighted_output_routed(
        self,
        model: Qwen2MoeForCausalLM,
        dataloader: DataLoader,
        num_groups: int,
    ):
        """Cluster experts minimizing saliency-weighted routed-output reconstruction cost.

        Per layer:
            f_i = mean_{t: i in topk(t)} expert_i(x_t)                (routed-output feature)
            s_i = mean_{t: i in topk(t)} [g_tilde_{t,i} * ||expert_i(x_t)||_2]  (REAP saliency)
            cost(C) = sum_{i in C} s_i * ||f_i - base(C)||_2
            base(C) = freq-weighted mean of f_i in C
        """
        from rescode.merging.saliency_clustering import (
            saliency_weighted_hierarchical_clustering,
            saliency_weighted_kmedoids,
        )

        model.eval()
        dom_experts = dict()
        forwarded_hidden_states: Dict[str, list] = {}
        handles = []
        all_router_logits = []

        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(
                    input[0].detach().cpu().reshape(-1, input[0].shape[-1])
                )
            return hook

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[REAP] Registering hooks..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))

        for batch in tqdm(dataloader, desc="[REAP] Forward to collect inputs+router logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                outputs = _forward_with_optional_router_logits(model, batch)
            batch_router_logits = torch.stack(outputs.router_logits)
            all_router_logits.append(batch_router_logits.cpu())
            del outputs

        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        all_router_logits = torch.cat(all_router_logits, dim=1)

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[REAP] Per-layer clustering..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            _device = model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.device

            layer_input = torch.cat(forwarded_hidden_states[ffn_name])
            layer_logits = all_router_logits[layer_idx]

            routing_weights = F.softmax(layer_logits, dim=1, dtype=torch.float)
            topk_w, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            renorm_topk = topk_w / (topk_w.sum(dim=-1, keepdim=True) + FP32_EPS)
            gate_full = torch.zeros_like(routing_weights)
            gate_full.scatter_(1, selected_experts, renorm_topk)

            expert_features = []
            saliencies = torch.zeros(self.num_experts, dtype=torch.float32)
            freq = torch.zeros(self.num_experts, dtype=torch.float32)

            with torch.no_grad():
                for i in range(self.num_experts):
                    token_mask = (selected_experts == i).any(dim=-1)
                    n_active = int(token_mask.sum().item())
                    if n_active == 0:
                        routed = layer_input[:1].to(_device)
                        expert_out = model.model.layers[layer_idx].mlp.experts[i](routed)
                        expert_features.append(expert_out.mean(dim=0).cpu())
                        saliencies[i] = 0.0
                        freq[i] = 0.0
                        continue
                    routed = layer_input[token_mask].to(_device)
                    expert_out = model.model.layers[layer_idx].mlp.experts[i](routed)
                    expert_features.append(expert_out.mean(dim=0).cpu())

                    norms = torch.linalg.norm(expert_out.float(), dim=-1)
                    gates = gate_full[token_mask, i].to(_device)
                    saliencies[i] = (norms * gates).mean().item()
                    freq[i] = float(n_active)

                expert_features = torch.stack(expert_features)

                if self.cluster == "kmedoids":
                    label, centers = saliency_weighted_kmedoids(
                        features=expert_features,
                        saliencies=saliencies,
                        num_groups=num_groups,
                    )
                else:
                    label, centers = saliency_weighted_hierarchical_clustering(
                        features=expert_features,
                        saliencies=saliencies,
                        num_groups=num_groups,
                        freq=freq,
                        base_weighting="freq",
                        verbose=False,
                    )
                print(
                    f"  Layer {layer_idx}: saliency range "
                    f"[{saliencies.min().item():.3e}, {saliencies.max().item():.3e}] "
                    f"-> {num_groups} clusters ({self.cluster})"
                )
                dom_experts[ffn_name] = centers
                self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts

    # ============================================================ RESCODE
    @torch.no_grad()
    def compute_saliency_and_descriptors(
        self,
        model: Qwen2MoeForCausalLM,
        dataloader: DataLoader,
    ):
        """One forward pass to fill _saliency_state_dict and _descriptor_state_dict.

        Per layer, per expert i:
            a_i = E_x [ 1(i in topk) * pi_i(x) * ||E_i(x)||_2 ]   (REAP saliency)
            o_i = mean of E_i(x) over ALL tokens ("all", default)  (descriptor)
                  or over routed tokens only ("routed"), per
                  `self.descriptor_tokens`
        """
        model.eval()
        forwarded_hidden_states: Dict[str, list] = {
            f"model.layers.{i}.mlp": [] for i in self.sparse_layer_indices
        }
        # DeepSeek's MoEGate.forward returns a tuple (not raw logits) and the
        # model doesn't surface outputs.router_logits. Capture logits ourselves
        # via a forward-pre hook on each MoE block: logits = F.linear(input, gate.weight).
        layer_logits_batches: Dict[int, list] = {
            i: [] for i in self.sparse_layer_indices
        }
        handles = []

        def _hook(name):
            def f(_m, inp, _o):
                forwarded_hidden_states[name].append(
                    inp[0].detach().cpu().reshape(-1, inp[0].shape[-1])
                )
            return f

        def _gate_pre_hook(layer_idx):
            def f(module, inputs):
                hidden = inputs[0]
                flat = hidden.reshape(-1, hidden.shape[-1])
                w = module.gate.weight
                logits = F.linear(flat.to(w.dtype), w, None)
                layer_logits_batches[layer_idx].append(logits.detach().cpu())
            return f

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] hooks"):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            block = model.model.layers[layer_idx].mlp
            handles.append(block.register_forward_hook(_hook(ffn_name)))
            handles.append(block.register_forward_pre_hook(_gate_pre_hook(layer_idx)))

        for batch in tqdm(dataloader, desc="[RESCODE] Forward pass for saliency+descriptors"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            with torch.no_grad():
                out = _forward_with_optional_router_logits(model, batch)
            del out

        for h in handles:
            h.remove()
        torch.cuda.empty_cache()
        # Dict keyed by absolute layer_idx (downstream uses all_router_logits[layer_idx]).
        all_router_logits = {
            li: torch.cat(layer_logits_batches[li], dim=0)
            for li in self.sparse_layer_indices
        }

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] per-layer compute"):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            moe = model.model.layers[layer_idx].mlp
            _device = moe.experts[0].gate_proj.weight.device

            layer_input = torch.cat(forwarded_hidden_states[ffn_name])
            layer_logits = all_router_logits[layer_idx]
            rw = F.softmax(layer_logits, dim=1, dtype=torch.float)
            topk_w, sel = torch.topk(rw, self.top_k, dim=-1)
            renorm = topk_w / (topk_w.sum(dim=-1, keepdim=True) + FP32_EPS)
            gate_full = torch.zeros_like(rw)
            gate_full.scatter_(1, sel, renorm)

            feats = []
            sal = torch.zeros(self.num_experts, dtype=torch.float32)
            # Paper Sec. 4.2 Eq. (saliency):
            #     a_i = E_x[ 1(i in T_k(x)) * pi_i(x) * ||E_i(x)||_2 ]
            # The expectation is over ALL tokens (denominator = total_tokens),
            # so the indicator effectively zeroes contributions from non-routed
            # tokens. Equivalently:
            #     a_i = (1/N) * sum_{x in R_i} pi_i(x) * ||E_i(x)||_2
            # where N = total tokens, R_i = tokens routed to expert i.
            total_tokens = float(sel.shape[0])
            # Descriptor token set (`descriptor_tokens`):
            #   "all"    - o_i = mean of E_i(x) over ALL calibration tokens, so
            #              every expert is evaluated on the same token set,
            #              matching the paper's output-distance definition
            #              (default).
            #   "routed" - o_i = mean of E_i(x) over tokens routed to expert i
            #              (conditional-mean descriptor).
            use_all_tokens = self.descriptor_tokens == "all"
            chunk = 8192
            with torch.no_grad():
                for i in range(self.num_experts):
                    mask = (sel == i).any(dim=-1)
                    if use_all_tokens:
                        feat_sum = torch.zeros(layer_input.shape[1], dtype=torch.float32)
                        sal_sum = 0.0
                        for s in range(0, layer_input.shape[0], chunk):
                            x = layer_input[s:s + chunk].to(_device)
                            eo = moe.experts[i](x).float()
                            feat_sum += eo.sum(dim=0).cpu()
                            m = mask[s:s + chunk]
                            if m.any():
                                norms = torch.linalg.norm(eo, dim=-1).cpu()
                                gates = gate_full[s:s + chunk][:, i]
                                sal_sum += float((norms[m] * gates[m]).sum().item())
                        feats.append(feat_sum / max(1.0, float(layer_input.shape[0])))
                        sal[i] = sal_sum / total_tokens
                        continue
                    if int(mask.sum().item()) == 0:
                        routed = layer_input[:1].to(_device)
                        eo = moe.experts[i](routed)
                        feats.append(eo.mean(dim=0).cpu())
                        sal[i] = 0.0
                        continue
                    routed = layer_input[mask].to(_device)
                    eo = moe.experts[i](routed)
                    feats.append(eo.mean(dim=0).cpu())
                    norms = torch.linalg.norm(eo.float(), dim=-1)
                    gates = gate_full[mask, i].to(_device)
                    sal[i] = (norms * gates).sum().item() / total_tokens
            feats = torch.stack(feats)
            self._saliency_state_dict[ffn_name] = sal
            self._descriptor_state_dict[ffn_name] = feats
            del layer_input, forwarded_hidden_states[ffn_name]
        torch.cuda.empty_cache()
        print(f"[RESCODE] saliency+descriptors computed for {len(self.sparse_layer_indices)} layers")

    def _compute_excluded_by_saliency(
        self,
        exclude_top_percent: float,
        excl_mode: str = "per-layer",
    ) -> Dict[str, torch.LongTensor]:
        """Like _compute_global_excluded_set but ranks by saliency instead of freq."""
        if exclude_top_percent <= 0.0:
            return {f"model.layers.{i}.mlp": torch.empty(0, dtype=torch.long)
                    for i in self.sparse_layer_indices}

        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            sal = self._saliency_state_dict[ffn_name]
            if sal.sum().item() == 0:
                raise RuntimeError(
                    f"[RESCODE-excl] saliency not populated for {ffn_name}. "
                    f"compute_saliency_and_descriptors must be called first."
                )

        excluded_per_layer: Dict[str, List[int]] = {
            f"model.layers.{i}.mlp": [] for i in self.sparse_layer_indices
        }

        if excl_mode == "per-layer":
            per_layer_k = int(round(self.num_experts * exclude_top_percent / 100.0))
            per_layer_k = max(0, min(per_layer_k, self.num_experts))
            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                sal = self._saliency_state_dict[ffn_name]
                top_idx = (torch.topk(sal, k=per_layer_k, largest=True).indices.tolist()
                           if per_layer_k > 0 else [])
                excluded_per_layer[ffn_name] = sorted(int(x) for x in top_idx)
            print(f"[RESCODE-excl] saliency-based per-layer pin: k={per_layer_k}/{self.num_experts}")
        else:
            all_scores = []
            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                sal = self._saliency_state_dict[ffn_name]
                for e_idx in range(self.num_experts):
                    all_scores.append((layer_idx, e_idx, sal[e_idx].item()))
            total = len(all_scores)
            k_count = int(round(total * exclude_top_percent / 100.0))
            all_scores.sort(key=lambda x: x[2], reverse=True)
            for layer_idx, e_idx, _ in all_scores[:k_count]:
                excluded_per_layer[f"model.layers.{layer_idx}.mlp"].append(e_idx)
            print(f"[RESCODE-excl] saliency-based global pin: k={k_count}/{total}")

        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            ex = sorted(excluded_per_layer[ffn_name])
            print(f"  {ffn_name}: {len(ex)} excluded -> {ex}")

        return {k: torch.tensor(sorted(v), dtype=torch.long)
                for k, v in excluded_per_layer.items()}

    def group_experts_by_clustering_rescode(
        self,
        model: Qwen2MoeForCausalLM,
        dataloader: DataLoader,
        num_groups: int,
        exclude_top_percent: float = 15.0,
        excl_mode: str = "per-layer",
    ):
        """RESCODE: saliency-based singleton pin + output-similarity clustering.

        Pre-condition: compute_saliency_and_descriptors must have run.
        Stores saliency in _saliency_state_dict so the saliency-weighted merger
        (merge_by_groups_with_saliency_weighted) can use it later.
        """
        # Lazy compute if not done
        first_ffn = f"model.layers.{self.sparse_layer_indices[0]}.mlp"
        if (self._descriptor_state_dict.get(first_ffn) is None
                or self._saliency_state_dict[first_ffn].sum().item() == 0):
            self.compute_saliency_and_descriptors(model, dataloader)

        excluded_per_layer = self._compute_excluded_by_saliency(exclude_top_percent, excl_mode=excl_mode)
        dom_experts = dict()
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] clustering non-excluded"):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            descriptors = self._descriptor_state_dict[ffn_name]  # (E, D)
            excluded = excluded_per_layer[ffn_name].tolist()
            excluded_set = set(excluded)
            kept_indices = [i for i in range(self.num_experts) if i not in excluded_set]
            num_kept = len(kept_indices)
            num_excluded = len(excluded)
            target_clusters_for_kept = num_groups - num_excluded
            if target_clusters_for_kept < 1:
                raise ValueError(
                    f"[RESCODE] {ffn_name}: excluded={num_excluded} >= num_groups={num_groups}."
                )
            if target_clusters_for_kept > num_kept:
                target_clusters_for_kept = num_kept

            if num_kept > 0:
                kept_feats = descriptors[kept_indices]
                _, kept_labels = group_experts_by_clustering(
                    model="qwen", num_groups=target_clusters_for_kept,
                    cluster=self.cluster, linkage=self.linkage,
                    hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                    num_experts=num_kept, experts=kept_feats, init_center=None,
                )
                kept_labels = kept_labels.cpu().tolist() if torch.is_tensor(kept_labels) else list(kept_labels)
            else:
                kept_labels = []

            label_tensor = torch.zeros(self.num_experts, dtype=torch.long)
            for rank, ex_idx in enumerate(excluded):
                label_tensor[ex_idx] = rank
            for pos, kept_idx in enumerate(kept_indices):
                label_tensor[kept_idx] = num_excluded + int(kept_labels[pos])
            self._group_state_dict[ffn_name] = label_tensor

            dom_list: List[int] = []
            for rank, ex_idx in enumerate(excluded):
                dom_list.append(ex_idx)
            if num_kept > 0:
                kept_labels_t = torch.tensor(kept_labels, dtype=torch.long)
                descs_f32 = descriptors.float()
                for k in range(target_clusters_for_kept):
                    mask = (kept_labels_t == k)
                    members = [kept_indices[j] for j in range(num_kept) if mask[j].item()]
                    if not members:
                        continue
                    member_feats = descs_f32[members]
                    centroid = member_feats.mean(dim=0, keepdim=True)
                    dists = torch.cdist(member_feats, centroid).squeeze(-1)
                    dom_list.append(members[int(torch.argmin(dists).item())])
            dom_experts[ffn_name] = dom_list
        return dom_experts

    def compute_all_usages(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
            mode: str = "frequency", # frequency, routing-score
    ):
        model.eval()
        config = model.config

        # DeepSeek's forward doesn't expose outputs.router_logits and its
        # MoEGate.forward returns a tuple - register forward-pre hooks on each
        # MoE block and recompute logits ourselves via F.linear(input, gate.weight).
        # Also: original code hardcoded top-2 in selected_experts which is wrong
        # for DeepSeek (top_k=6); use self.top_k.
        topk_for_freq = self.top_k

        _captured: Dict[int, torch.Tensor] = {}

        def _make_router_hook(layer_idx: int):
            def hook(module, inputs):
                hidden = inputs[0]
                flat = hidden.reshape(-1, hidden.shape[-1])
                w = module.gate.weight
                logits = F.linear(flat.to(w.dtype), w, None)
                _captured[layer_idx] = logits.detach()
            return hook

        _handles = [
            model.model.layers[li].mlp.register_forward_pre_hook(_make_router_hook(li))
            for li in self.sparse_layer_indices
        ]
        try:
            for batch in tqdm(dataloader, desc=f"[RESCODE] Evaluating routing distribution"):
                batch = {k: v.cuda() for k, v in batch.items()}
                if "labels" in batch:
                    batch.pop("labels")
                with torch.no_grad():
                    outputs = _forward_with_optional_router_logits(model, batch)
                model_router_logits = getattr(outputs, "router_logits", None)
                if mode == "frequency":
                    if model_router_logits is not None:
                        stacked = torch.stack(model_router_logits)
                        selected_experts = torch.topk(stacked, topk_for_freq, dim=-1)[1].reshape(
                            config.num_hidden_layers, -1
                        )
                        for layer_idx in self.sparse_layer_indices:
                            ffn_name = f"model.layers.{layer_idx}.mlp"
                            unique, counts = torch.unique(selected_experts[layer_idx], return_counts=True)
                            self._usage_frequency_state_dict[ffn_name][unique.cpu()] += counts.cpu()
                    else:
                        for layer_idx in self.sparse_layer_indices:
                            rl = _captured.get(layer_idx)
                            if rl is None:
                                continue
                            ffn_name = f"model.layers.{layer_idx}.mlp"
                            sel = torch.topk(rl, topk_for_freq, dim=-1)[1].reshape(-1)
                            unique, counts = torch.unique(sel, return_counts=True)
                            self._usage_frequency_state_dict[ffn_name][unique.cpu()] += counts.cpu()
                else:  # routing-score
                    for layer_idx in self.sparse_layer_indices:
                        rl = (model_router_logits[layer_idx] if model_router_logits is not None
                              else _captured.get(layer_idx))
                        if rl is None:
                            continue
                        ffn_name = f"model.layers.{layer_idx}.mlp"
                        router_score = F.softmax(rl, dim=1)
                        scores = router_score.float().sum(0) / router_score.shape[0]
                        self._usage_frequency_state_dict[ffn_name] += scores.cpu()
                _captured.clear()
        finally:
            for h in _handles:
                h.remove()
        self._usage_frequency_state_dict = {
            k: v / torch.sum(v) for k, v in self._usage_frequency_state_dict.items()
        }

    def compute_all_similarities(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader = None
    ):
        # if os.path.exists("similarity.pkl"):
        #     with open("similarity.pkl", "rb") as f:
        #         self._similarity_state_dict = pickle.load(f)
        #     return
        
        similarity_list = ["weight", "router-weight", "router-logits", "expert-output"]
        if self.similarity_base not in similarity_list and dataloader is None:
            raise ValueError(
                "[RESCODE] `dataloader` should be provided when similarity_base is not 'weight' or 'router-weight'")
        model = model.eval()
        if self.similarity_base == "weight":
            self._compute_all_similarities_by_weight(model.state_dict())
        elif self.similarity_base == 'router-weight':
            self._compute_all_similarities_by_router_weight(model.state_dict())
        elif self.similarity_base == 'router-logits':
            self._compute_all_similarities_by_router_logits(model, dataloader)
        elif self.similarity_base == 'expert-output':
            self._compute_all_similarities_by_expert_outputs(model, dataloader)
        else:
            raise NotImplementedError
        
        # with open("similarity.pkl", "wb") as f:
        #     pickle.dump(self._similarity_state_dict, f)

    def _compute_all_similarities_by_weight(self, state_dict: Dict[str, torch.Tensor]):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE]  Computing similarities by weight..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    i_flat = torch.cat(
                        [state_dict[f"{ffn_name}.experts.{i}.gate_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{i}.down_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{i}.up_proj.weight"].flatten()],
                        dim=0
                    )
                    j_flat = torch.cat(
                        [state_dict[f"{ffn_name}.experts.{j}.gate_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{j}.down_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{j}.up_proj.weight"].flatten()],
                        dim=0
                    )
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(ffn_name, i, j, similarity)

    def _compute_all_similarities_by_router_weight(
            self, state_dict: Dict[str, torch.Tensor]
    ):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] Computing similarities by router rows..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    i_flat = state_dict[f"{ffn_name}.gate.weight"][i]
                    j_flat = state_dict[f"{ffn_name}.gate.weight"][j]
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(ffn_name, i, j, similarity)

    def _compute_all_similarities_by_router_logits(
            self, model: Qwen2MoeForCausalLM, dataloader: DataLoader
    ):
        model.eval()
        all_router_logits = []
        for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = _forward_with_optional_router_logits(model, batch)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)

        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, *, num_experts)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts)
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        i_flat = layer_router_logits[:, i].flatten()
                        j_flat = layer_router_logits[:, j].flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(ffn_name, i, j, similarity)
    
    def _compute_all_similarities_by_expert_outputs(
            self, model: Qwen2MoeForCausalLM, dataloader: DataLoader
    ):
        model.eval()
        forwarded_hidden_states = {} # moe input
        handles = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                # forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1]))
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))
            return hook
        
        for layer_idx in tqdm(
                self.sparse_layer_indices,
                desc=f"[Merging]Registering forward hook..."
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            handles.append(model.model.layers[layer_idx].mlp.register_forward_hook(
                _get_activation_hook(ffn_name))
            )

        for batch in tqdm(dataloader, desc=f"[RESCODE] Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        for handle in handles:
            handle.remove()
        # torch.cuda.empty_cache()

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[RESCODE] Computing similarities by expert outputs..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]).cuda()
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            with torch.no_grad():
                for i in range(self.num_experts):
                    if self.overlap_metric == "cosine":
                        expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                    else:
                        expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input))
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        if i == j:
                            self.save_similarity(ffn_name, i, j, 1.0)
                            continue
                        if self.overlap_metric == "kl-divergence":
                            p = get_prob_distributions(expert_outputs[i])
                            q = get_prob_distributions(expert_outputs[j])
                            similarity = compute_kl_divergence(p, q)
                        elif self.overlap_metric == "wasserstein": # wasserstein
                            similarity = compute_wasserstein_distance(expert_outputs[i], expert_outputs[j])
                        else: # cosine
                            i_flat = expert_outputs[i].flatten()
                            j_flat = expert_outputs[j].flatten()
                            similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(ffn_name, i, j, similarity)
                        self.save_similarity(ffn_name, j, i, similarity)
        

def apply_mask(module, _mask):
    # applying masks to the input to compute gradients
    def masking(_, i):
        return _mask * i[0]

    handle = module.register_forward_pre_hook(masking)
    return handle

def hijack(module, _list, _hijack_input, _stop_forward=False):
    # if _stop_forward=True, then it raise error after forwarding the module
    if _hijack_input:
        def input_hook(_, inputs, __):
            _list.append(inputs[0].detach().cpu()) # .clone().data
            if _stop_forward:
                raise RuntimeError("Stop forward")

        handle = module.register_forward_hook(input_hook)
    else:
        raise NotImplementedError
    return handle

def remove_col(x, idx):
    return torch.cat([x[:, :idx], x[:, idx+1:]], dim=-1)

def remove_row(x, idx):
    return torch.cat([x[:idx], x[idx+1:]], dim=0)

@torch.no_grad()
def collect_act(data, weight1, weight3=None):
    activations = []
    act = torch.nn.SiLU()
    if weight3 is not None:
        cur = act(torch.matmul(data, weight1.T)) * torch.matmul(data, weight3.T)
    else:
        cur = torch.matmul(data, weight1.T)
    activations.append(cur.reshape(-1, cur.shape[-1]))
    return torch.cat(activations, dim=0)

@torch.no_grad()
def collect_feature(ingredient, data, weight1, weight2, weight3):
    if ingredient == "act":
        return collect_act(data, weight1, weight3)
    elif ingredient == "weight":
        # weigh1, weigh3: NxD, weigh2: DxN
        return torch.cat([weight1.T, weight2, weight3.T], dim=0)
    else: # both
        return collect_act(data, weight1, weight3), torch.cat([weight1.T, weight2, weight3.T], dim=0)

@torch.no_grad()
def compute_covariance(act1, act2):
    print(f"compute covariance: {act1.shape}, {act2.shape}")
    mean1 = act1.mean(dim=0, keepdim=True)
    mean2 = act2.mean(dim=0, keepdim=True)
    std1 = act1.std(dim=0, keepdim=True)
    std2 = act2.std(dim=0, keepdim=True)
    corr_matrix = torch.matmul((act1 - mean1).T, act2 - mean2) / (act1.shape[0] - 1)
    corr_matrix = corr_matrix / (std1.T * std2 + FP32_EPS)
    del mean1, mean2, std1, std2
    return corr_matrix

@torch.no_grad()
def compute_feature_covariance(ingredient, data1, data2):
    if ingredient == "act+weight":
        corr1 = compute_covariance(data1[0], data2[0])
        corr2 = compute_covariance(data1[1], data2[1])
        return corr1 + corr2
    else:
        return compute_covariance(data1, data2)

def get_coef(num_ffn, input_weight, average_coefs, d_ff=None):
    if d_ff == None: # fix-dom-same
        if input_weight is not None:
            coef = input_weight
        elif average_coefs is None:
            coef = [1.0] * num_ffn
        elif len(average_coefs) == num_ffn:
            coef = average_coefs
        else:
            coef = [1.0] * num_ffn
    else: # zipit
        if input_weight is not None:
            coef = []
            for w in input_weight:
                coef = [w] * d_ff
                coef.extend(coef)
        elif average_coefs is None:
            coef = [1.0] * num_ffn * d_ff
        elif len(average_coefs) == num_ffn:
            coef = [coef for coef in average_coefs for _ in range(d_ff)]
        elif len(average_coefs) != num_ffn * d_ff:
            raise ValueError(
                f"The length of average_coefs should be either {num_ffn} or {num_ffn * d_ff}, "
                f"but got {len(average_coefs)}."
            )
    return coef

@torch.no_grad()
def _merge_mlp_experts_by_usage_frequency_weighting(
        ffn: Qwen2MoeSparseMoeBlock,
        group_labels: torch.LongTensor,
        usage_frequencies: torch.Tensor,
) -> Qwen2MoeSparseMoeBlock:
    """Weighted average of each group's experts (accumulated in fp32), bound to its first member."""
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        weights = usage_frequencies[expert_indices].float()
        weights = weights / (weights.sum() + FP32_EPS)
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            merged = sum(
                weights[k] * getattr(ffn.experts[e], proj).weight.float()
                for k, e in enumerate(expert_indices.tolist())
            )
            getattr(ffn.experts[expert_indices[0]], proj).weight.copy_(merged)

        for expert_idx in expert_indices[1:]:
            # Binding merged experts to the first of them
            ffn.experts[expert_idx] = ffn.experts[expert_indices[0]]

    return ffn


@torch.no_grad()
def _zipit_merge(temp_dim, target_dim, weight1, weight3, data, _device, _dtype):
    permutation_matrix = torch.eye(temp_dim, temp_dim, dtype=_dtype, device=_device)
    ROUND = 0
    act = torch.nn.SiLU()
    while temp_dim > target_dim:
        ROUND += 1
        odd = temp_dim % 2
        target_dim_this_round = max(target_dim, temp_dim // 2 + odd)
        print(f"ROUND {ROUND}. From {temp_dim} to {target_dim_this_round}")
        
        ### Collect activations
        activations = []
        if weight3 is None:
            cur = torch.matmul(data, weight1.T)
        else:
            cur = act(torch.matmul(data, weight1.T)) * torch.matmul(data, weight3.T)
        activations.append(cur.reshape(-1, cur.shape[-1]))
        activations = torch.cat(activations, dim=0)
        print("Activations: ", activations.shape)
        ### Compute covariance
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, keepdim=True)
        covar = torch.matmul((activations - mean).T, activations - mean) / (activations.shape[0] - 1)
        corr_matrix = covar / (std.T * std + FP32_EPS)
        del mean, std, covar
        torch.cuda.empty_cache()
        corr_matrix[torch.arange(temp_dim), torch.arange(temp_dim)] = -1 # Remove self-correlation
        print(corr_matrix)
        ### Merge temp_dim / 2 times
        for _ in range(temp_dim - target_dim_this_round):
            max_index = torch.argmax(corr_matrix)
            row, col = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]
            permutation_matrix[:, row] += permutation_matrix[:, col]
            permutation_matrix = remove_col(permutation_matrix, col)

            # row_coef, col_coef = average_coefs[row], average_coefs[col]
            row_coef, col_coef = 1.0, 1.0
            weight1[row] = (row_coef * weight1[row] + col_coef * weight1[col]) / (row_coef + col_coef + FP32_EPS)
            if weight3 is not None:
                weight3[row] = (row_coef * weight3[row] + col_coef * weight3[col]) / (row_coef + col_coef + FP32_EPS)
                weight3 = remove_row(weight3, col)
            weight1 = remove_row(weight1, col)
            
            corr_matrix[row] = FP32_EPS # set very small number to avoid repeated merging
            corr_matrix[:, row] = FP32_EPS
            corr_matrix[row, row] = -1
            corr_matrix = remove_col(corr_matrix, col)
            corr_matrix = remove_row(corr_matrix, col)
        temp_dim = weight1.shape[0]
    for i in range(20): # permutation_matrix.shape[1]
        print(permutation_matrix[:, i].nonzero().squeeze())
    return permutation_matrix

@torch.no_grad()
def _merge_qwen_moe_by_zipit(
    ffn_list: List[Qwen2MoeMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = None,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
) -> Qwen2MoeMLP:
    d_ff, d_model = ffn_list[0].gate_proj.out_features, ffn_list[0].gate_proj.in_features
    num_ffn = len(ffn_list)
    temp_dim = d_ff * num_ffn
    average_coefs = [1.0] * temp_dim
    act = torch.nn.SiLU()

    _device = ffn_list[0].gate_proj.weight.device
    _dtype = ffn_list[0].gate_proj.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Data shape: {forwarded_hidden_states.shape}, temp_dim: {temp_dim}, target_dim: {d_ff}")

    ### Merge gate_proj and up_proj
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    first_permutation_matrix = _zipit_merge(d_ff * num_ffn, d_ff, ffn_all_gate_proj, ffn_all_up_proj, forwarded_hidden_states, _device, _dtype)
    first_unmerge_matrix = first_permutation_matrix
    first_merge_matrix = torch.div(first_permutation_matrix, torch.sum(first_permutation_matrix, dim=0, keepdim=True))

    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_gate_proj = torch.matmul(first_merge_matrix.T, ffn_all_gate_proj)
    ffn_up_proj = torch.matmul(first_merge_matrix.T, ffn_all_up_proj)

    ### Merge down_proj
    new_data = act(torch.matmul(forwarded_hidden_states, ffn_gate_proj.T)) * torch.matmul(forwarded_hidden_states, ffn_up_proj.T)
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=0) # (d_model * num_ffn, d_ff)
    second_permutation_matrix = _zipit_merge(d_model * num_ffn, d_model, ffn_all_down_proj, None, new_data, _device, _dtype)
    second_merge_matrix = torch.div(second_permutation_matrix, torch.sum(second_permutation_matrix, dim=0, keepdim=True))
    second_unmerge_matrix = second_permutation_matrix
    ffn_down_proj = torch.zeros(d_model, d_ff).to(_device)
    for i in range(num_ffn):
        ffn_down_proj += torch.matmul(second_merge_matrix.T[:, i*d_model:(i+1)*d_model], torch.matmul(ffn_all_down_proj[i*d_model:(i+1)*d_model], first_unmerge_matrix.T[:, i*d_ff:(i+1)*d_ff]))

    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.gate_proj.weight.data = ffn_gate_proj
    merged_ffn.down_proj.weight.data = ffn_down_proj
    merged_ffn.up_proj.weight.data = ffn_up_proj

    return merged_ffn


@torch.no_grad()
def _merge_qwen_moe_experts_with_dominant(
        ffn_list: List[Qwen2MoeMLP],
        forwarded_hidden_states: torch.Tensor,
        mini_batch_size: Optional[int] = None,
        alpha_for_repeated_merging: Optional[float] = 0.1,
        average_coefs: Optional[List[float]] = None,
        input_weight: Optional[List[float]] = None,
        dominant_index: Optional[int] = 0,
) -> Qwen2MoeMLP:
    print("merge: fix-dom-independent-rule-without-unmerge")
    d_ff, d_model = ffn_list[0].gate_proj.out_features, ffn_list[0].gate_proj.in_features
    num_ffn = len(ffn_list)
    need_pinv = False
    if input_weight is not None:
        coef = input_weight
        need_pinv = True
    elif average_coefs is None:
        coef = [1.0] * num_ffn
    elif len(average_coefs) == num_ffn:
        coef = average_coefs
        need_pinv = True
    else:
        coef = [1.0] * num_ffn
    
    if dominant_index != 0:
        ffn_list[0], ffn_list[dominant_index] = ffn_list[dominant_index], ffn_list[0]
    print("dominant_index: ", dominant_index)
    _device = ffn_list[0].gate_proj.weight.device
    _dtype = ffn_list[0].gate_proj.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Data shape: {forwarded_hidden_states.shape}, temp_dim: {d_ff * num_ffn}, target_dim: {d_ff}, dominant_index: {dominant_index}")
    # Compute Permutation Matrix for gate_proj and up_proj
    permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, device=_device, dtype=_dtype) * coef[0]
    dom_act = collect_act(forwarded_hidden_states, ffn_list[dominant_index].gate_proj.weight.data, ffn_list[dominant_index].up_proj.weight.data)
    group_indexes = []
    for i in range(num_ffn):
        if i == dominant_index:
            continue
        other_act = collect_act(forwarded_hidden_states, ffn_list[i].gate_proj.weight.data, ffn_list[i].up_proj.weight.data)
        corr_matrix = compute_covariance(dom_act, other_act)
        max_index = torch.argmax(corr_matrix, dim=1)
        group_indexes.append(max_index)
    for i in range(d_ff):
        for j in range(num_ffn - 1):
            index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_ff * (j + 1)
            permutation_matrix[i, index_in_this_group] = coef[j]
    if not need_pinv:
        unmerge_1 = permutation_matrix
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
    else:
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
        unmerge_1 = torch.linalg.pinv(permutation_matrix.to(torch.float)).to(_dtype).T
        permutation_matrix = permutation_matrix.to(_dtype)
    
    print(f"first permutation_matrix: {permutation_matrix.shape} {permutation_matrix[0]}")
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_gate_proj = torch.matmul(permutation_matrix, ffn_all_gate_proj)
    ffn_up_proj = torch.matmul(permutation_matrix, ffn_all_up_proj)

    del ffn_all_gate_proj, ffn_all_up_proj

    # Compute Permutation Matrix for down_proj
    permutation_matrix = torch.eye(d_model, d_model * num_ffn, dtype=_dtype, device=_device) * coef[0]
    new_data = collect_act(forwarded_hidden_states, ffn_gate_proj, ffn_up_proj)
    dom_act = collect_act(new_data, ffn_list[dominant_index].down_proj.weight.data, None)
    group_indexes.clear()
    for i in range(num_ffn):
        if i == dominant_index:
            continue
        other_act = collect_act(new_data, ffn_list[i].down_proj.weight.data, None)
        corr_matrix = compute_covariance(dom_act, other_act)
        max_index = torch.argmax(corr_matrix, dim=1)
        group_indexes.append(max_index)
    for i in range(d_model):
        for j in range(num_ffn - 1):
            index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_model * (j + 1)
            permutation_matrix[i, index_in_this_group] = coef[j]
    permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
    print(f"second permutation_matrix: {permutation_matrix.shape} {permutation_matrix[0]}")
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=0) # (d_model * num_ffn, d_ff)
    ffn_down_proj = torch.zeros(d_model, d_ff).to(_device)
    for i in range(num_ffn):
        ffn_down_proj += torch.matmul(permutation_matrix[:, i*d_model:(i+1)*d_model],
            torch.matmul(ffn_all_down_proj[i*d_model:(i+1)*d_model], 
                         unmerge_1[:, i*d_ff:(i+1)*d_ff])
        )

    del ffn_all_down_proj

    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.gate_proj.weight.data = ffn_gate_proj
    merged_ffn.down_proj.weight.data = ffn_down_proj
    merged_ffn.up_proj.weight.data = ffn_up_proj

    return merged_ffn

@torch.no_grad()
def _merge_qwen_moe_experts_with_dominant_same_rule(
        ffn_list: List[Qwen2MoeMLP],
        forwarded_hidden_states: torch.Tensor,
        average_coefs: Optional[List[float]] = None,
        input_weight: Optional[List[float]] = None,
        dominant_index: Optional[int] = 0,
        ingredient: Optional[str] = "act", # "act", "weight", "act+weight"
        mode: Optional[str] = "normal", # normal, cluster
):
    print("merge: fix-dom-same-rule-without-unmerge")
    d_ff, d_model = ffn_list[0].gate_proj.out_features, ffn_list[0].gate_proj.in_features
    num_ffn = len(ffn_list)
    coef = get_coef(num_ffn, input_weight, average_coefs)
    
    if dominant_index != 0:
        ffn_list[0], ffn_list[dominant_index] = ffn_list[dominant_index], ffn_list[0]
        dominant_index = 0
    print("dominant_index: ", dominant_index)
    _device = ffn_list[0].gate_proj.weight.device
    _dtype = ffn_list[0].gate_proj.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Data shape: {forwarded_hidden_states.shape}, temp_dim: {d_ff * num_ffn}, target_dim: {d_ff}, dominant_index: {dominant_index}")
    # Compute Permutation Matrix for gate_proj and up_proj
    
    if mode == "cluster":
        # 1. Concat activations
        activations = []
        for ffn in ffn_list:
            cur = collect_act(forwarded_hidden_states, ffn.gate_proj.weight.data, ffn.up_proj.weight.data)
            activations.append(cur)
        activations = torch.cat(activations, dim=1).T
        print(activations.shape)

        # 2. Kmeans clustering
        centers = activations[:d_ff]
        min_points_per_cluster = 1

        for _ in range(100):
            distance = torch.cdist(activations, centers)
            assignments = torch.argmin(distance, dim=1)
            del distance
            for i in range(d_ff):
                num_points_in_cluster = torch.sum(assignments == i)
                if num_points_in_cluster < min_points_per_cluster:
                    # Find overpopulated clusters
                    for j in range(d_ff):
                        if i != j and torch.sum(assignments == j) > num_ffn:
                            # Move points from overpopulated cluster j to underpopulated cluster i
                            diff = torch.sum(assignments == j) - min_points_per_cluster

                            # Select `num_to_move` points from cluster j and reassign them to cluster i
                            reassign_indices = torch.where(assignments == j)[0][0]
                            assignments[reassign_indices] = i
                            print(f"Group {i} has {num_points_in_cluster} points, move 1 point from group {j}")
                            break
                            
            # Recompute the centers after ensuring the minimum number of points
            group_members = []
            for i in range(d_ff):
                group_member = activations[assignments == i].mean(dim=0)
                if torch.isnan(group_member).sum().item() > 0:
                    print(f"Group {i}: {torch.nonzero(assignments == i).squeeze()} {group_member.shape} {torch.isnan(group_member).sum()}")
                group_members.append(group_member)
            new_centers = torch.stack(group_members)
            max_diff = 0
            for i in range(d_ff):
                diff = torch.max(torch.abs(centers[i] - new_centers[i]))
                max_diff = max(max_diff, diff.item())
            print(f"max_diff: {max_diff}")
            if max_diff < 1e-4:
                print("Converged!")
                break
            centers = new_centers
        
        permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, dtype=torch.float16, device=_device)
        for i in range(d_ff):
            index_in_this_group = (assignments == i).nonzero().squeeze()
            permutation_matrix[i, index_in_this_group] = 1
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True)).to(_dtype)
        for i in range(5): # permutation_matrix.shape[1]
            print(permutation_matrix[:, i].nonzero().squeeze())
    else:
        # dom_act = collect_act(forwarded_hidden_states, ffn_list[0].gate_proj.weight.data, ffn_list[0].up_proj.weight.data)
        dom_act = collect_feature(ingredient, forwarded_hidden_states, ffn_list[0].gate_proj.weight.data, ffn_list[0].down_proj.weight.data, ffn_list[0].up_proj.weight.data)
        group_indexes = [[]]
        for i in range(1, num_ffn):
            # other_act = collect_act(forwarded_hidden_states, ffn_list[i].gate_proj.weight.data, ffn_list[i].up_proj.weight.data)
            other_act = collect_feature(ingredient, forwarded_hidden_states, ffn_list[i].gate_proj.weight.data, ffn_list[i].down_proj.weight.data, ffn_list[i].up_proj.weight.data)
            # corr_matrix = compute_covariance(dom_act, other_act)
            corr_matrix = compute_feature_covariance(ingredient, dom_act, other_act)
            max_index = torch.argmax(corr_matrix, dim=0)
            group_indexes.append(max_index)
        
        permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, dtype=_dtype, device=_device) * coef[0]
        for i in range(d_ff):
            for j in range(1, num_ffn):
                index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_ff * j
                permutation_matrix[i, index_in_this_group] = coef[j]
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
        print(f"first permutation_matrix: {permutation_matrix.shape} {permutation_matrix[0]}")
        del dom_act
    
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=1) # (d_model, d_ff * num_ffn)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_gate_proj = torch.matmul(permutation_matrix, ffn_all_gate_proj)
    ffn_down_proj = torch.matmul(permutation_matrix, ffn_all_down_proj.T)
    ffn_up_proj = torch.matmul(permutation_matrix, ffn_all_up_proj)

    del ffn_all_gate_proj, ffn_all_down_proj, ffn_all_up_proj

    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.gate_proj.weight.data = ffn_gate_proj
    merged_ffn.down_proj.weight.data = ffn_down_proj.T
    merged_ffn.up_proj.weight.data = ffn_up_proj

    return merged_ffn

@torch.no_grad()
def process_coef(num_ffn, d_ff, d_model, average_coefs=None, input_weight=None):
    if input_weight is not None:
        first_coef = []
        second_coef = []
        for w in input_weight:
            coef_1 = [w] * d_ff
            first_coef.extend(coef_1)
            coef_2 = [w] * d_model
            second_coef.extend(coef_2)
    elif average_coefs is None:
        first_coef = [1.0] * num_ffn * d_ff
        second_coef = [1.0] * num_ffn * d_model
    elif len(average_coefs) == num_ffn:
        first_coef = [coef for coef in average_coefs for _ in range(d_ff)]
        second_coef = [coef for coef in average_coefs for _ in range(d_model)]
    else:
        raise ValueError("The argument `avearge_coefs` should be either None or have the same length as `num_ffn`, or you need to provide `input_weight`.")
    return first_coef, second_coef



@torch.no_grad()
def compute_merging(temp_dim, target_dim, corr_matrix, coef, alpha, _device):
    permutation_matrix = torch.eye(temp_dim, temp_dim, dtype=torch.float, device=_device)
    while corr_matrix.shape[0] > target_dim:
        max_index = torch.argmax(corr_matrix)
        max_i, max_j = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]

        # Update permutation matrix
        i_coef, j_coef = coef[max_i], coef[max_j]
        permutation_matrix[:, max_i] = (i_coef * permutation_matrix[:, max_i] + j_coef * permutation_matrix[:, max_j]) / (i_coef + j_coef + FP32_EPS)
        permutation_matrix = remove_col(permutation_matrix, max_j)

        # Update corr_matrix
        updated_corr_vec = alpha * torch.min(torch.stack([corr_matrix[max_i], corr_matrix[max_j]]), dim=0).values
        corr_matrix[max_i] = updated_corr_vec
        corr_matrix[:, max_i] = updated_corr_vec
        corr_matrix[max_i, max_i] = -1
        # Remove second feature from the correlation matrix
        corr_matrix = remove_col(corr_matrix, max_j)
        corr_matrix = remove_row(corr_matrix, max_j)
    return permutation_matrix

@torch.no_grad()
def _merge_qwen_moe_by_zipit_with_unmerge(
    ffn_list: List[Qwen2MoeMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = 5000,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
):
    print("merge: zipit-independe-rule-with-unmerge")
    ffn_list = [ffn.eval() for ffn in ffn_list]
    d_ff, d_model = ffn_list[0].gate_proj.out_features, ffn_list[0].gate_proj.in_features
    num_ffn = len(ffn_list)
    first_coef, second_coef = process_coef(num_ffn, d_ff, d_model, average_coefs, input_weight)
    
    _device = ffn_list[0].gate_proj.weight.device
    _dtype = ffn_list[0].gate_proj.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Collect activations with batch size {mini_batch_size} with original data length {forwarded_hidden_states.shape}")

    # Compute gate_proj and up_proj's permutation matrix
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # 
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0)
    act = torch.nn.SiLU()

    activations = []
    cur = act(torch.matmul(forwarded_hidden_states, ffn_all_gate_proj.T)) * torch.matmul(forwarded_hidden_states, ffn_all_up_proj.T)
    activations.append(cur.reshape(-1, cur.shape[-1]))
    cat_activtaions = torch.cat(activations, dim=0)
    activations.clear()
    corr_matrix = compute_covariance(cat_activtaions, cat_activtaions)
    corr_matrix[torch.arange(d_ff * num_ffn), torch.arange(d_ff * num_ffn)] = -1  # Remove self-correlation
    print(f"corr_matrix: {corr_matrix.shape}")
    first_permutation_matrix = compute_merging(d_ff * num_ffn, d_ff, corr_matrix, first_coef, alpha_for_repeated_merging, _device)
    first_permutation_matrix = first_permutation_matrix / torch.sum(first_permutation_matrix, dim=0, keepdim=True)
    first_unmerge_matrix = torch.linalg.pinv(first_permutation_matrix)
    first_permutation_matrix = first_permutation_matrix.to(_dtype)
    ffn_gate_proj = torch.matmul(first_permutation_matrix.T, ffn_all_gate_proj)
    ffn_up_proj = torch.matmul(first_permutation_matrix.T, ffn_all_up_proj)
    print(f"first_permutation_matrix: {first_permutation_matrix.shape}, first_unmerge_matrix: {first_unmerge_matrix.shape}")
    
    # Compute down_proj's permutation matrix
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=0)
    new_data = act(torch.matmul(forwarded_hidden_states, ffn_gate_proj.T)) * torch.matmul(forwarded_hidden_states, ffn_up_proj.T)
    activations = []
    new_cur = torch.matmul(new_data, ffn_all_down_proj.T)
    activations.append(new_cur.reshape(-1, new_cur.shape[-1]))
    cat_activtaions = torch.cat(activations, dim=0)
    activations.clear()
    corr_matrix = compute_covariance(cat_activtaions, cat_activtaions)
    corr_matrix[torch.arange(d_model * num_ffn), torch.arange(d_model * num_ffn)] = -1  # Remove self-correlation
    print(f"corr_matrix: {corr_matrix.shape}")
    second_permutation_matrix = compute_merging(d_model * num_ffn, d_model, corr_matrix, second_coef, alpha_for_repeated_merging, _device)
    second_permutation_matrix = second_permutation_matrix / torch.sum(second_permutation_matrix, dim=0, keepdim=True)
    second_unmerge_matrix = torch.linalg.pinv(second_permutation_matrix) # DxED
    print(f"second_permutation_matrix: {second_permutation_matrix.shape}, second_unmerge_matrix: {second_unmerge_matrix.shape}")
    second_permutation_matrix = second_permutation_matrix.to(_device).to(_dtype)
    first_unmerge_matrix = first_unmerge_matrix.to(_device).to(_dtype)
    ffn_down_proj = torch.zeros(d_model, d_ff, device=_device)
    for i in range(num_ffn):
        ffn_down_proj += torch.matmul(second_permutation_matrix.T[:, i*d_model:(i+1)*d_model], 
            torch.matmul(ffn_all_down_proj[i*d_model:(i+1)*d_model], first_unmerge_matrix[:, i*d_ff:(i+1)*d_ff]))
    
    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.gate_proj.weight.data = ffn_gate_proj
    merged_ffn.down_proj.weight.data = ffn_down_proj
    merged_ffn.up_proj.weight.data = ffn_up_proj
    
    # TODO: use a warpper to warp moe and assign unmerge matrix to it
    # TODO: consider (gate_proj, up_proj) and (down_proj) has differnt unmerge matrix, use down_proj's unmerge matrix to unmerge down_proj's output
    return merged_ffn, second_unmerge_matrix

@torch.no_grad()
def _merge_qwen_moe_by_activation_matching_within_and_across_models(
    ffn_list: List[Qwen2MoeMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = None,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
    ingredient: Optional[str] = "act", # "act" or "weight" or "act+weight"
    mode: Optional[str] = "normal", # "normal" or "cluster"
) -> Qwen2MoeMLP:
    print("merge: zipit-same-rule-without-unmerge")
    ffn_list = [ffn.eval() for ffn in ffn_list]
    concat_ffn = deepcopy(ffn_list[0])
    d_ff, d_model = concat_ffn.gate_proj.out_features, concat_ffn.gate_proj.in_features
    num_ffn = len(ffn_list)
    average_coefs = get_coef(num_ffn, input_weight, average_coefs, d_ff)
    
    if len(forwarded_hidden_states) == 0 or len(forwarded_hidden_states) == 1:
        return concat_ffn
    if mini_batch_size is None:
        mini_batch_size = forwarded_hidden_states.shape[0]

    _device = forwarded_hidden_states.device
    _dtype = ffn_list[0].gate_proj.weight.dtype
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0)
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=1)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0)
    concat_ffn.gate_proj = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.down_proj = torch.nn.Linear(d_ff * num_ffn, d_model, bias=False)
    concat_ffn.up_proj = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.gate_proj.weight.data = ffn_all_gate_proj
    concat_ffn.down_proj.weight.data = ffn_all_down_proj
    concat_ffn.up_proj.weight.data = ffn_all_up_proj
    # concat_ffn = concat_ffn.eval()
    concat_ffn.gate_proj.weight.to(_device)
    concat_ffn.up_proj.weight.to(_device)
    concat_ffn.down_proj.weight.to(_device)
    
    activations, weights = None, None
    if "act" in ingredient:
        activations = []
        def _activation_hook(module, input, output):
            activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))
            return _activation_hook
        print(f"Collect activations with batch size {mini_batch_size} with original data length {forwarded_hidden_states.shape[0]}")
        handle = concat_ffn.down_proj.register_forward_hook(_activation_hook)
        for i in range(0, forwarded_hidden_states.shape[0], mini_batch_size):
            concat_ffn(forwarded_hidden_states[i:i + mini_batch_size])
        handle.remove()
        del handle, forwarded_hidden_states
        activations = torch.cat(activations, dim=0)  # (batch_size * seq_len, d_ff * num_ffn)
    if "weight" in ingredient:
        if "act" not in ingredient:
            del forwarded_hidden_states
        weights = []
        for ffn in ffn_list:
            concat_weight = torch.cat([ffn.gate_proj.weight.data.T, ffn.down_proj.weight.data, ffn.up_proj.weight.data.T], dim=0) # 3DxN
            weights.append(concat_weight)
        weights = torch.cat(weights, dim=1) # 3Dx(N*num_ffn)

    if ingredient == "act":
        corr_matrix = compute_covariance(activations, activations)
    elif ingredient == "weight":
        corr_matrix = compute_covariance(weights, weights)
    else:
        corr_matrix = compute_covariance(activations, activations) + compute_covariance(weights, weights)
    torch.cuda.empty_cache()

    corr_matrix[torch.arange(d_ff * num_ffn), torch.arange(d_ff * num_ffn)] = -1  # Remove self-correlation


    if mode == "cluster":
        center_indices = [0]
        mask = torch.ones(corr_matrix.size(0), dtype=torch.bool, device=corr_matrix.device)  # Mask to exclude already selected centers
        mask[0] = False
        for _ in range(1, d_ff):
            # EN x num_centers
            dist = torch.min(corr_matrix[:, center_indices], dim=1).values ** 2
            probabilities = dist / dist.sum()
            probabilities = probabilities * mask.float()  # Set probabilities of selected centers to 0
            next_center_idx = torch.multinomial(probabilities, 1).item()
            center_indices.append(next_center_idx)
            mask[next_center_idx] = False
        center_indices.sort()
        print(len(center_indices))

        activations = activations.T
        centers = activations[center_indices]
        min_points_per_cluster = 1

        for _ in range(100):
            distance = torch.cdist(activations, centers)
            assignments = torch.argmin(distance, dim=1)
            del distance
            # Ensure each cluster has at least min_points_per_cluster points
            for i in range(d_ff):
                num_points_in_cluster = torch.sum(assignments == i)
                if num_points_in_cluster < min_points_per_cluster:
                    # Find overpopulated clusters
                    for j in range(d_ff):
                        if i != j and torch.sum(assignments == j) > num_ffn:
                            # Move points from overpopulated cluster j to underpopulated cluster i
                            diff = torch.sum(assignments == j) - min_points_per_cluster

                            # Select `num_to_move` points from cluster j and reassign them to cluster i
                            reassign_indices = torch.where(assignments == j)[0][0]
                            assignments[reassign_indices] = i
                            print(f"Group {i} has {num_points_in_cluster} points, move 1 point from group {j}")
                            break
                            
            # Recompute the centers after ensuring the minimum number of points
            group_members = []
            for i in range(d_ff):
                group_member = activations[assignments == i].mean(dim=0)
                if torch.isnan(group_member).sum().item() > 0:
                    print(f"Group {i}: {torch.nonzero(assignments == i).squeeze()} {group_member.shape} {torch.isnan(group_member).sum()}")
                group_members.append(group_member)
            new_centers = torch.stack(group_members)
            max_diff = 0
            for i in range(d_ff):
                diff = torch.max(torch.abs(centers[i] - new_centers[i]))
                max_diff = max(max_diff, diff.item())
            print(f"max_diff: {max_diff}")
            if max_diff < 1e-4:
                print("Converged!")
                break
            centers = new_centers
        
        # Assign the group index
        permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, dtype=torch.float16, device=_device)
        for i in range(d_ff):
            index_in_this_group = (assignments == i).nonzero().squeeze()
            permutation_matrix[i, index_in_this_group] = 1
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True)).to(_dtype)
        for i in range(5): # permutation_matrix.shape[1]
            print(permutation_matrix[:, i].nonzero().squeeze())
        ffn_gate_proj = torch.matmul(permutation_matrix, ffn_all_gate_proj)
        ffn_down_proj = torch.matmul(permutation_matrix, ffn_all_down_proj.T)
        ffn_up_proj = torch.matmul(permutation_matrix, ffn_all_up_proj)

        del ffn_all_gate_proj, ffn_all_down_proj, ffn_all_up_proj
        merged_ffn = deepcopy(ffn_list[0])
        merged_ffn.gate_proj.weight.data = ffn_gate_proj
        merged_ffn.down_proj.weight.data = ffn_down_proj.T
        merged_ffn.up_proj.weight.data = ffn_up_proj
        return merged_ffn

    # Greedy Merging!
    while ffn_all_gate_proj.shape[0] > d_ff:
        # Select the most correlated pair
        max_index = torch.argmax(corr_matrix)
        max_i, max_j = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]

        # Merge the most correlated pair, replace the first feature with the merged one
        i_coef, j_coef = average_coefs[max_i], average_coefs[max_j]
        ffn_all_gate_proj[max_i] = (i_coef * ffn_all_gate_proj[max_i] + j_coef * ffn_all_gate_proj[max_j]) / (i_coef + j_coef + FP32_EPS)
        ffn_all_up_proj[max_i] = (i_coef * ffn_all_up_proj[max_i] + j_coef * ffn_all_up_proj[max_j]) / (i_coef + j_coef + FP32_EPS)
        ffn_all_down_proj[:, max_i] = (i_coef * ffn_all_down_proj[:, max_i] + j_coef * ffn_all_down_proj[:, max_j]) / (
                i_coef + j_coef + FP32_EPS)
       
        # Remove the second feature
        ffn_all_gate_proj = torch.cat([
            ffn_all_gate_proj[:max_j],
            ffn_all_gate_proj[max_j + 1:]
        ], dim=0)
        ffn_all_up_proj = torch.cat([
            ffn_all_up_proj[:max_j],
            ffn_all_up_proj[max_j + 1:]
        ], dim=0)
        ffn_all_down_proj = torch.cat([
            ffn_all_down_proj[:, :max_j],
            ffn_all_down_proj[:, max_j + 1:]
        ], dim=1)

        # Update the correlation matrix
        updated_corr_vec = alpha_for_repeated_merging * torch.min(
            torch.stack([corr_matrix[max_i], corr_matrix[max_j]]), dim=0
        ).values
        corr_matrix[max_i] = updated_corr_vec
        corr_matrix[:, max_i] = updated_corr_vec
        corr_matrix[max_i, max_i] = -1  # Remove self-correlation

        # Remove the second feature from the correlation matrix
        corr_matrix = torch.cat([
            corr_matrix[:, :max_j],
            corr_matrix[:, max_j + 1:]
        ], dim=1)
        corr_matrix = torch.cat([
            corr_matrix[:max_j],
            corr_matrix[max_j + 1:]
        ], dim=0)

        # Update the average coefs
        average_coefs[max_i] += average_coefs[max_j]
        average_coefs = average_coefs[:max_j] + average_coefs[max_j + 1:]

    # handle.remove()
    del corr_matrix
    merged_ffn = deepcopy(ffn_list[0])
   
    merged_ffn.gate_proj.weight.data = ffn_all_gate_proj
    merged_ffn.down_proj.weight.data = ffn_all_down_proj
    merged_ffn.up_proj.weight.data = ffn_all_up_proj

    return merged_ffn

def prune_experts(
        moe: Qwen2MoeSparseMoeBlock,
        dominant_experts: List[int],
):
    with torch.no_grad():
        r = len(dominant_experts)
        dominant_experts.sort()
        gate_new = torch.nn.Linear(in_features=moe.gate.in_features, out_features=r, bias=False, dtype=torch.bfloat16)
        gate_new.weight.data = moe.gate.weight.data[dominant_experts]
        moe.gate = gate_new

        moe.experts = torch.nn.ModuleList(
            [moe.experts[i] for i in dominant_experts])
        moe.num_experts = r
        moe.top_k = min(r, moe.top_k)
    return moe


@torch.no_grad()
def _merge_moe_experts_within_and_across_models(
        moe: Qwen2MoeSparseMoeBlock,
        group_labels: torch.LongTensor,
        forwarded_hidden_states: Tuple[torch.Tensor],
        dominant_alone: bool,
        merge: str,
        mode: Optional[str] = "normal",
        core_expert_indices: Optional[List[int]] = None,
        usage_frequencies: Optional[torch.Tensor] = None,
        moe_scores: Optional[torch.Tensor] = None,
        data_limit: Optional[int] = 50000,
        ingredient: Optional[str] = "act",
) -> Qwen2MoeSparseMoeBlock:
    

    if merge == "weighted":
        merging_coefficient = [float(coef) for coef in mode]
        if len(merging_coefficient) != 2:
            raise ValueError("The argument `mode` should be in the format of `weight,weight`, but got {mode}.")
        merging_weight = torch.zeros(len(moe.experts))
        for e in range(len(moe.experts)):
            merging_weight[e] = merging_coefficient[0] if e in core_expert_indices else merging_coefficient[1]
        return _merge_mlp_experts_by_usage_frequency_weighting(
            ffn=moe,
            group_labels=group_labels,
            usage_frequencies=merging_weight,
        )
    elif merge == "prune" and mode == "normal":
        # Prune experts and its routing weights
        moe = prune_experts(moe, core_expert_indices)
        return moe

    _device = moe.experts[0].gate_proj.weight.device
    _dtype = moe.experts[0].gate_proj.weight.dtype
    
    # moe.expert_dict = {}
    input_weight = None
    if merge == "unmerge":
        moe = Qwen2MoEWrapper(moe)
    print("core_expert_indices: ", core_expert_indices)

    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        print(f"Group {label}: {expert_indices}")
        if core_expert_indices is not None:
            core_expert_index = [i for i, idx in enumerate(expert_indices) if idx in core_expert_indices]
        if mode == "input-weight" or mode == "all":
            input_weight = []
            for expert_idx in expert_indices:
                input_weight.append(forwarded_hidden_states[expert_idx].shape[0])
            s = sum(input_weight)
            input_weight = [w / s for w in input_weight]
            print(f"Input weight: {input_weight}")
        # not dominant
        group_forwarded_hidden_states = torch.cat([
            forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
        ], dim=0)
        if data_limit != 1000000:
            randperm_indices = torch.randperm(group_forwarded_hidden_states.shape[0])
            group_forwarded_hidden_states = group_forwarded_hidden_states[randperm_indices[:data_limit]]
        if len(expert_indices) == 1:
            if merge == "unmerge":
                merged_expert = moe.model.experts[expert_indices[0]]
                moe.unmerge_matrix[label.item()] = None
            else:
                merged_expert = moe.experts[expert_indices[0]]
        else:
            if merge == "update":
                merged_expert = _merge_qwen_moe_by_zipit(
                    ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    mini_batch_size=5000,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                    input_weight=input_weight,
                )
            elif merge == "fix-dom":
                merged_expert = _merge_qwen_moe_experts_with_dominant(
                    ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    mini_batch_size=5000,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                    input_weight=input_weight,
                    dominant_index=core_expert_index[0],
                )
            elif merge == "fix-dom-same":
                merged_expert = _merge_qwen_moe_experts_with_dominant_same_rule(
                    ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    input_weight=input_weight,
                    dominant_index=core_expert_index[0],
                    ingredient=ingredient,
                    mode=mode,
                )
            elif merge == "unmerge":
                merged_expert, unmerge_matrix = _merge_qwen_moe_by_zipit_with_unmerge(
                    ffn_list=[moe.model.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    mini_batch_size=5000,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                    input_weight=input_weight,
                )
                moe.unmerge_matrix[label.item()] = unmerge_matrix.to(_device).to(_dtype)
            elif merge == "prune":
                pass
            else:
                merged_expert = _merge_qwen_moe_by_activation_matching_within_and_across_models(
                    ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    mini_batch_size=2048,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                    input_weight=input_weight,
                    ingredient=ingredient,
                    mode=mode,
                )
        
        if merge == "unmerge":
            moe.model.experts[expert_indices[0].item()].gate_proj.weight.copy_(merged_expert.gate_proj.weight)
            moe.model.experts[expert_indices[0].item()].down_proj.weight.copy_(merged_expert.down_proj.weight)
            moe.model.experts[expert_indices[0].item()].up_proj.weight.copy_(merged_expert.up_proj.weight)
            moe.expert_to_group[expert_indices[0].item()] = label.item()
            moe.group_to_expert[label.item()] = [expert_indices[0].item()]
            for expert_idx in expert_indices[1:]:
                moe.model.experts[expert_idx.item()] = moe.model.experts[expert_indices[0].item()]
                moe.expert_to_group[expert_idx.item()] = label.item()
                moe.group_to_expert[label.item()].append(expert_idx.item())
            moe.group_to_expert[label.item()] = torch.tensor(moe.group_to_expert[label.item()])
        elif merge == "prune" and mode == "zero-output":
            for expert_idx in expert_indices:
                if expert_idx == core_expert_indices[0]:
                    continue
                moe.experts[expert_idx.item()].down_proj.weight.copy_(torch.zeros_like(moe.experts[expert_idx.item()].down_proj.weight))
        else:
            moe.experts[expert_indices[0]].gate_proj.weight.copy_(merged_expert.gate_proj.weight)
            moe.experts[expert_indices[0]].down_proj.weight.copy_(merged_expert.down_proj.weight)
            moe.experts[expert_indices[0]].up_proj.weight.copy_(merged_expert.up_proj.weight)

            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                moe.experts[expert_idx] = moe.experts[expert_indices[0]]
    return moe

@torch.no_grad()
def merge_by_groups_with_saliency_weighted(
        model: Qwen2MoeForCausalLM,
        grouper: ExpertsGrouperForDeepSeekMoE,
        merging_layers: Optional[List[int]] = None,
        delta: float = 1.0e-3,
) -> Qwen2MoeForCausalLM:
    """RESCODE-style saliency-weighted merge: alpha_i = (a_i + delta) / sum(a_j + delta)."""
    saliency_dict = grouper.saliency_state_dict()
    group_labels_dict = grouper.group_state_dict()
    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[RESCODE] Merging experts with saliency-weighted averaging (delta={delta})..."
    ):
        if merging_layers is not None and layer_idx not in merging_layers:
            continue
        ffn_name = f"model.layers.{layer_idx}.mlp"
        group_labels = group_labels_dict[ffn_name]
        saliencies = saliency_dict[ffn_name].clone()
        if saliencies.sum().item() == 0:
            raise RuntimeError(
                f"[RESCODE-merge] saliency not populated for {ffn_name}. "
                f"compute_saliency_and_descriptors must be called first."
            )
        weights = saliencies + float(delta)  # delta-floored saliency
        # Reuse the existing weighted-average function with saliency as weight.
        model.model.layers[layer_idx].mlp = _merge_mlp_experts_by_usage_frequency_weighting(
            ffn=model.model.layers[layer_idx].mlp,
            group_labels=group_labels,
            usage_frequencies=weights,
        )
    return model


def merge_by_groups_with_usage_weighted(
        model: Qwen2MoeForCausalLM,
        grouper: ExpertsGrouperForDeepSeekMoE,
        merging_layers: Optional[List[int]] = None,
) -> Qwen2MoeForCausalLM:
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    group_labels_dict = grouper.group_state_dict()

    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[RESCODE] Merging experts with usage-frequency-weighted averaging..."
    ):
        if merging_layers is not None and layer_idx not in merging_layers:
            continue
        ffn_name = f"model.layers.{layer_idx}.mlp"
        group_labels = group_labels_dict[ffn_name]
        usage_frequencies = usage_frequency_dict[ffn_name]
        model.model.layers[layer_idx].mlp = _merge_mlp_experts_by_usage_frequency_weighting(
            ffn=model.model.layers[layer_idx].mlp,
            group_labels=group_labels,
            usage_frequencies=usage_frequencies,
        )
    return model


@torch.no_grad()
def merge_by_groups_within_and_across_models(
    qwen_model: Qwen2MoeForCausalLM,
    grouper: ExpertsGrouperForDeepSeekMoE,
    dataloader: DataLoader,
    merge: str,
    mode: Optional[str] = "normal",
    partition: Optional[int] = 1,
    dominant_alone: Optional[bool] = False,
    core_experts: Optional[Dict[str, List[int]]] = None,
    usage_weighted: Optional[bool] = False,
    ingredient: Optional[str] = "act",
) -> Qwen2MoeForCausalLM:
    
    forwarded_hidden_states = dict()

    usage_frequencies = grouper.usage_frequency_state_dict()
    num_experts = grouper.num_experts

    def part_processor(sparse_layer_indices):
        qwen_model.eval() #.cuda()
        handles = []

        def _get_activation_hook(name):
            def hook(module, input, output):
                # forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1]))
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))
            return hook
        
        for layer_idx in tqdm(
                sparse_layer_indices,
                desc=f"[Merging]Registering forward hook..."
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            handles.append(qwen_model.model.layers[layer_idx].mlp.register_forward_hook(
                _get_activation_hook(ffn_name))
            )
        
        router_indices = {name: [] for name in forwarded_hidden_states.keys()}
        if mode == "activation-with-router-logits" or mode == "all":
            router_weights = {name: [] for name in forwarded_hidden_states.keys()}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[Merging]Computing activations..."):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = _forward_with_optional_router_logits(qwen_model, batch)
                for layer_idx in sparse_layer_indices:
                    ffn_name = f"model.layers.{layer_idx}.mlp"
                    routing_weights = F.softmax(outputs.router_logits[layer_idx], dim=1)
                    routing_weights, selected_experts = torch.topk(routing_weights, qwen_model.config.num_experts_per_tok, dim=-1)
                    router_indices[ffn_name].append(selected_experts)
                    if mode == "activation-with-router-logits" or mode == "all":
                        router_weights[ffn_name].append(routing_weights)
                        
        for handle in handles:
            handle.remove()
        
        
        for layer_idx in tqdm(
                sparse_layer_indices,
                desc=f"[Merging]Merging by groups within and across experts..."
        ):
            _st = time.time()
            ffn_name = f"model.layers.{layer_idx}.mlp"
            group_labels = grouper.group_state_dict()[ffn_name]
            layer_forwarded_hidden_states = tuple()
            hidden_states = torch.cat(forwarded_hidden_states[ffn_name], dim=0) # T x D
            concat_router_indices = torch.cat(router_indices[ffn_name], dim=0) # BT x k
            if mode == "activation-with-router-logits" or mode == "all":
                concat_router_weights = torch.cat(router_weights[ffn_name], dim=0) # BT x k
            for expert_idx in range(num_experts): # expert num
                expert_mask = (concat_router_indices == expert_idx)
                batch_tensor = torch.any(expert_mask, dim=-1).to(hidden_states.device)
                choice_input = hidden_states[batch_tensor]
                if mode == "activation-with-router-logits" or mode == "all":
                    router_weight = torch.masked_select(concat_router_weights, expert_mask).view(-1, 1).to(choice_input.device)
                    layer_hidden_states = choice_input * router_weight
                else:
                    layer_hidden_states = choice_input
                layer_forwarded_hidden_states += (layer_hidden_states,)
            qwen_model.model.layers[layer_idx].mlp = _merge_moe_experts_within_and_across_models(
                moe=qwen_model.model.layers[layer_idx].mlp,
                group_labels=group_labels,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                dominant_alone=dominant_alone,
                merge=merge,
                mode=mode,
                core_expert_indices=core_experts[ffn_name] if core_experts is not None else None,
                usage_frequencies=usage_frequencies[ffn_name] if usage_weighted else None,
                data_limit=grouper.data_limit,
                ingredient=ingredient,
            )
            del layer_forwarded_hidden_states
            print(f"------- Layer {layer_idx} took {time.time() - _st:.2f}s -------\n")

    print(grouper.sparse_layer_indices)
    partition_num = len(grouper.sparse_layer_indices) // partition
    for i in range(0, len(grouper.sparse_layer_indices), partition_num):
        cur_indices = grouper.sparse_layer_indices[i:i+partition_num]
        print("cur: ", cur_indices)
        part_processor(cur_indices)
    return qwen_model
