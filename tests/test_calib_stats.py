import torch

from rescode.merging.calib_stats import accumulate_output_gram


def test_chunked_output_gram_preserves_close_expert_distances():
    generator = torch.Generator().manual_seed(13)
    outputs = torch.randn(8, 97, 256, generator=generator)
    outputs[1] = outputs[0] + 0.01 * torch.randn(97, 256, generator=generator)
    outputs = outputs.bfloat16().float()

    def distances(matrix):
        diagonal = matrix.diag()
        return (diagonal[:, None] + diagonal[None, :] - 2 * matrix) / 97

    devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
    for device in devices:
        device_outputs = outputs.to(device)
        flattened = device_outputs.reshape(8, -1)
        reference = flattened.double() @ flattened.double().T
        gram = torch.zeros(8, 8, dtype=torch.float64, device=device)
        accumulate_output_gram(gram, device_outputs)

        assert torch.allclose(distances(gram), distances(reference), atol=5e-5, rtol=1e-4)
        assert torch.allclose(gram.diag() / 97, reference.diag() / 97, atol=5e-5, rtol=1e-4)
