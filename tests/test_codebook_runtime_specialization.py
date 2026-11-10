"""Variable token counts must share Triton kernels without changing outputs."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_token_count_reuses_compiled_kernels():
    import torch
    if not torch.cuda.is_available():
        print("CUDA unavailable: specialization test skipped")
        return
    with tempfile.TemporaryDirectory(prefix="codebook_triton_shapes_") as cache:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache
        env["PYTHONPATH"] = str(ROOT)
        subprocess.run([sys.executable, __file__, "--worker"], cwd=ROOT,
                       env=env, check=True)


def _worker():
    import torch
    import torch.nn.functional as F
    from rescode.qjl.codebook_base import PackedCodebookBase, encode_base
    from rescode.qjl.core import pack_signs, qjl_reference
    from rescode.qjl.triton_kernel import packed_qjl, packed_qjl_fused_gate_up

    torch.manual_seed(101)
    weight = (torch.randn(64, 64) * 0.05).to(torch.bfloat16)
    base = PackedCodebookBase(encode_base(weight), "cuda")
    packed = pack_signs(torch.rand(64, 512) > 0.5).cuda()
    up = pack_signs(torch.rand(64, 512) > 0.5).cuda()
    norms = torch.rand(64, device="cuda") + 0.1
    up_norms = torch.rand(64, device="cuda") + 0.1
    for tokens in (24, 37, 48, 63):
        x = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16)
        actual = base(x)
        expected = F.linear(x, base.decode().to(x.dtype))
        assert torch.allclose(actual, expected, atol=0.03, rtol=0.02)
        z = torch.randn(tokens, 512, device="cuda", dtype=torch.float16)
        single = packed_qjl(z, packed, norms, 512)
        reference = qjl_reference(z, packed, norms, 512)
        assert torch.allclose(single, reference, atol=0.002, rtol=0.002)
        gate_y, up_y = packed_qjl_fused_gate_up(z, packed, norms, up, up_norms, 512)
        assert torch.allclose(gate_y, reference, atol=0.002, rtol=0.002)
        assert torch.allclose(up_y, qjl_reference(z, up, up_norms, 512),
                              atol=0.002, rtol=0.002)
        torch.cuda.synchronize()
        print(f"passed token batch {tokens}", flush=True)
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    for name, expected in (("_linear.json", 1), ("_packed_tiled.json", 2)):
        count = len(list(cache.rglob(name)))
        print(f"{name}: {count} compiled variants", flush=True)
        assert count == expected, (name, count)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker()
    else:
        test_dynamic_token_count_reuses_compiled_kernels()
