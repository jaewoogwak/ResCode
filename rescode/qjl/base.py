"""Shared-base fitting for the Gaussian one-bit QJL variance objective.

For each expert i, H_i = (pi/2) sum(q||x||²) I - X_i^T diag(q) X_i.
The normal equation is S (sum H_i) = sum W_i H_i. We apply H_i
matrix-free, so down projection never builds a d_ff by d_ff covariance.
The 1/D multiplier is common to every candidate S and omitted in the solve.
"""

import math
from typing import Dict, Tuple

import torch


Samples = Dict[int, Tuple[torch.Tensor, torch.Tensor]]
_A = math.pi / 2


@torch.no_grad()
def variance_objective(base: torch.Tensor, weights: Dict[int, torch.Tensor],
                       samples: Samples, d: int) -> float:
    total = torch.zeros((), device=base.device, dtype=torch.float64)
    for expert_id, weight in weights.items():
        if expert_id not in samples:
            continue
        x, q = samples[expert_id]
        x, q = x.to(base.device).float(), q.to(base.device).float()
        if x.numel() == 0:
            continue
        residual = weight.to(base.device).float() - base.float()
        t = (q * x.square().sum(-1)).sum()
        v = (_A * t * residual.square().sum()
             - ((x @ residual.T).square() * q[:, None]).sum()) / d
        total += v.double()
    return float(total)


def _initial_base(weights: Dict[int, torch.Tensor], method: str,
                  saliency: torch.Tensor, delta: float) -> torch.Tensor:
    ids = list(weights)
    if method == "mean":
        coeff = torch.ones(len(ids), device=weights[ids[0]].device)
    else:
        coeff = torch.tensor([float(saliency[i]) + delta for i in ids],
                             device=weights[ids[0]].device)
    coeff = coeff / coeff.sum()
    base = torch.zeros_like(weights[ids[0]], dtype=torch.float32)
    for c, i in zip(coeff, ids):
        base.add_(weights[i].float(), alpha=float(c))
    return base


@torch.no_grad()
def fit_shared_base(weights: Dict[int, torch.Tensor], samples: Samples,
                    method: str, saliency: torch.Tensor, d: int,
                    heldout: Samples = None, delta: float = 1e-3,
                    max_iter: int = 30, tolerance: float = 1e-5,
                    ridge: float = 1e-7):
    """Fit one projection's shared base; return FP32 base and diagnostics.

    The saliency method uses ResCode's (saliency+delta) weighting. Its weight
    vector is the full-layer statistic, restricted and renormalized to this
    cluster. `qjl_aware` starts from that same saliency base.
    """
    if method not in ("mean", "saliency", "qjl_aware"):
        raise ValueError(method)
    if not weights:
        raise ValueError("empty cluster")
    device = next(iter(weights.values())).device
    base = _initial_base(weights, "mean" if method == "mean" else "saliency",
                         saliency, delta)
    before = variance_objective(base, weights, samples, d)
    held_before = variance_objective(base, weights, heldout, d) if heldout else None
    iterations = 0
    if method == "qjl_aware":
        records = []
        weighted_weight = torch.zeros_like(base)
        diagonal = 0.0
        for expert_id, w in weights.items():
            if expert_id not in samples:
                continue
            x, q = samples[expert_id]
            x, q = x.to(device).float(), q.to(device).float()
            if not x.numel():
                continue
            t = (q * x.square().sum(-1)).sum()
            diagonal += _A * float(t)
            wf = w.float()
            weighted_weight += _A * t * wf - ((wf @ x.T) * q[None, :]) @ x
            records.append((x, q))
        if records and diagonal > 0:
            diagonal *= 1.0 + ridge

            def apply(s):
                out = diagonal * s
                for x, q in records:
                    out = out - ((s @ x.T) * q[None, :]) @ x
                return out

            # Global matrix CG solves all output rows together, without any
            # inverse or dense input covariance. FP32 is sufficient here.
            current = base.clone()
            residual = weighted_weight - apply(current)
            direction = residual.clone()
            rs = torch.sum(residual * residual)
            rs0 = max(float(rs), 1e-30)
            for step in range(max_iter):
                ad = apply(direction)
                denom = torch.sum(direction * ad)
                if float(denom) <= 0:
                    break
                alpha = rs / denom
                current += alpha * direction
                residual -= alpha * ad
                new_rs = torch.sum(residual * residual)
                iterations = step + 1
                if float(new_rs) <= rs0 * tolerance ** 2:
                    break
                direction = residual + (new_rs / rs) * direction
                rs = new_rs
            candidate = variance_objective(current, weights, samples, d)
            # The unregularized objective must not regress due to ridge or
            # finite-precision CG. Retain the saliency initialization if it does.
            if candidate <= before * (1 + 1e-6):
                base = current
    after = variance_objective(base, weights, samples, d)
    held_after = variance_objective(base, weights, heldout, d) if heldout else None
    return base, {
        "method": method, "objective_before": before, "objective_after": after,
        "heldout_before": held_before, "heldout_after": held_after,
        "cg_iterations": iterations,
    }
