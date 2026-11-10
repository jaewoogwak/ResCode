"""Two-bit signed codebook base with FP16 row scales and packed GPU inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack four [sign, scale-index] codes per byte, row-major."""
    if codes.ndim != 2 or (codes > 3).any() or (codes < 0).any():
        raise ValueError("codes must be a [rows, columns] tensor with values 0..3")
    flat = codes.reshape(-1).to(torch.uint8)
    flat = F.pad(flat, (0, (-flat.numel()) % 4))
    c = flat.reshape(-1, 4)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).contiguous()


def unpack_codes(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    shifts = torch.tensor((0, 2, 4, 6), device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(1) >> shifts) & 3).reshape(-1)[:rows * cols].reshape(rows, cols)


def decode_base(packed: torch.Tensor, scales: torch.Tensor, cols: int,
                zero_indices: torch.Tensor | None = None) -> torch.Tensor:
    codes = unpack_codes(packed, scales.numel(), cols)
    sign = 1.0 - 2.0 * (codes & 1).float()
    level = torch.where((codes & 2) != 0, 2.0, 0.5)
    decoded = sign * level * scales.float()[:, None]
    if zero_indices is not None and zero_indices.numel():
        decoded.reshape(-1)[zero_indices.long()] = 0
    return decoded


@torch.no_grad()
def encode_base(weight: torch.Tensor, iterations: int = 8, row_chunk: int = 128):
    """Alternating row-wise assignment/least squares with four initial scales.

    The best candidate is selected using the *stored FP16* scale.  The final
    assignment is refreshed after FP16 rounding so the residual uses exactly
    the inference base. No dense base is retained in the returned payload.
    """
    if weight.ndim != 2 or iterations < 1 or row_chunk < 1:
        raise ValueError("expected 2D weight and positive iterations/chunk")
    rows, cols = weight.shape
    code_parts, scales_parts, zero_parts = [], [], []
    for start in range(0, rows, row_chunk):
        w = weight[start:start + row_chunk].float()
        magnitude = w.abs()
        active = magnitude != 0
        mean = magnitude.mean(1)
        rms = magnitude.square().mean(1).sqrt()
        initials = torch.stack((mean, mean * 2, mean * 0.5, rms), dim=0).clamp_min(1e-12)
        best_error = torch.full_like(mean, float("inf"))
        best_alpha = torch.zeros_like(mean, dtype=torch.float16)
        best_index = torch.zeros_like(magnitude, dtype=torch.bool)
        for alpha in initials:
            for _ in range(iterations):
                index = magnitude > (1.25 * alpha[:, None])
                level = torch.where(index, 2.0, 0.5) * active
                alpha = (magnitude * level).sum(1) / level.square().sum(1).clamp_min(1e-30)
            # Candidate quality is measured after storing alpha in FP16.
            stored = alpha.to(torch.float16)
            if not torch.isfinite(stored).all():
                raise OverflowError("row scale cannot be stored in FP16")
            index = magnitude > (1.25 * stored.float()[:, None])
            level = torch.where(index, 2.0, 0.5) * active
            error = (magnitude - stored.float()[:, None] * level).square().sum(1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_alpha = torch.where(better, stored, best_alpha)
            best_index = torch.where(better[:, None], index, best_index)
        codes = (w < 0).to(torch.uint8) | (best_index.to(torch.uint8) << 1)
        code_parts.append(codes.cpu())
        scales_parts.append(best_alpha.cpu())
        zero_parts.append(torch.nonzero((~active) & (best_alpha.float()[:, None] != 0),
                                        as_tuple=False).cpu())
    zero_indices = torch.cat([((part[:, 0] + chunk * row_chunk) * cols + part[:, 1])
                              for chunk, part in enumerate(zero_parts)]).to(torch.int32)
    return {"packed": pack_codes(torch.cat(code_parts)),
            "scales": torch.cat(scales_parts), "zero_indices": zero_indices,
            "shape": (rows, cols), "codebook": (0.5, 2.0),
            "iterations": iterations}


class PackedCodebookBase(nn.Module):
    def __init__(self, state: dict, device="cpu"):
        super().__init__()
        if tuple(state["codebook"]) != (0.5, 2.0):
            raise ValueError("wrong fixed codebook")
        self.shape = tuple(int(i) for i in state["shape"])
        rows, cols = self.shape
        if state["packed"].numel() != (rows * cols + 3) // 4 or state["scales"].numel() != rows:
            raise ValueError("invalid packed base shape")
        self.register_buffer("packed", state["packed"].to(device).contiguous())
        self.register_buffer("scales", state["scales"].to(device).contiguous())
        zeros = state.get("zero_indices", torch.empty(0, dtype=torch.int32))
        if zeros.numel() and (zeros.min() < 0 or zeros.max() >= rows * cols):
            raise ValueError("invalid zero indices")
        self.register_buffer("zero_indices", zeros.to(device).contiguous())

    def decode(self):
        return decode_base(self.packed, self.scales, self.shape[1], self.zero_indices)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.shape[1]:
            raise ValueError("base input dimension mismatch")
        if x.is_cuda:
            from .codebook_base_kernel import packed_codebook_linear
            y = packed_codebook_linear(x, self.packed, self.scales, self.shape,
                                       fp32_output=bool(self.zero_indices.numel()))
            if self.zero_indices.numel():
                rows = torch.div(self.zero_indices.long(), self.shape[1], rounding_mode="floor")
                cols = self.zero_indices.long() % self.shape[1]
                flat = x.reshape(-1, self.shape[1])
                correction = flat.float()[:, cols] * (self.scales.float()[rows] * 0.5)
                y = y.reshape(-1, self.shape[0])
                y.scatter_add_(1, rows[None, :].expand(flat.shape[0], -1), -correction)
                return y.reshape(*x.shape[:-1], self.shape[0]).to(x.dtype)
            return y
        return F.linear(x.float(), self.decode()).to(x.dtype)

    def resident_bytes(self):
        return (self.packed.numel() + self.scales.numel() * self.scales.element_size()
                + self.zero_indices.numel() * self.zero_indices.element_size())
