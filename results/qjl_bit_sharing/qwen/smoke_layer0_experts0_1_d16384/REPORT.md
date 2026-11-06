# QJL Gate–Up bit-sharing diagnostic

Scope: **limited smoke/selected subset**; layers [0]; expert IDs [0, 1]; D=[16384].
Original BF16 model: `/root/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be`. Cached ResCode groups/saliency: `/home/jaewoo/ResCode/results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt`.
Projection seed: 0. Residual = FP32(W) − FP32(BF16 saliency base).

## Structural direction similarity

Down comparisons align Gate/Up rows with Down columns. They describe neuron-vector direction only; Down uses a different QJL input space.
- gate_up: median expert median |cosine| = 0.027 (all W); paired W 0.025, R 0.025.
- gate_down: median expert median |cosine| = 0.019 (all W); paired W 0.018, R 0.019.
- up_down: median expert median |cosine| = 0.028 (all W); paired W 0.033, R 0.032.

## Actual Gate–Up QJL sign bits

Signs use the repository's FP16 Gaussian ProjectionBank and pack_residual. Direct and inverted agreement are measured from packed XOR bits.
- D=16384, original_weight: median expert mean direct agreement 0.500; best with inversion 0.511; shuffled-neuron 0.500; random-bit 0.500; Gaussian prediction 0.500.
- D=16384, residual: median expert mean direct agreement 0.500; best with inversion 0.510; shuffled-neuron 0.500; random-bit 0.500; Gaussian prediction 0.500.

## Storage feasibility

Bytes include two FP32 row norms, packed sketches, a uint16 count plus uint16 positions per sparse-corrected row, packed inversion/selection flags, and stated 16-byte format metadata. Entropy is reported separately from stored bytes.
Fully shared Gate without correction is lossy; dense XOR is lossless but has no bit-storage savings.
- D=16384, original_weight: median sparse lossless ratio 4.404; maximum selective lossless savings 0.00%; selectively shared neurons 0.0%.
- D=16384, residual: median sparse lossless ratio 4.411; maximum selective lossless savings 0.00%; selectively shared neurons 0.0%.

## Validation and limits

- Maximum relative W=S+R reconstruction error: 0.
- Group partition, tensor dtype/shape, saliency coefficient sums, exact sparse XOR decoding, and byte accounting are checked.
- The original Wikipedia calibration text is unavailable on this instance. The saved cache metadata and membership are validated; grouping is not recomputed.
- No model evaluation, perplexity, accuracy, or downstream benchmark was run. Accuracy effects of sharing remain unverified.
