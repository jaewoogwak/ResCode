# QJL Gate–Up bit-sharing diagnostic

Scope: **full model**; layers [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]; expert IDs all 60; D=[8192, 16384].
Original BF16 model: `/workspace/.hf_home/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be`. Cached ResCode groups/saliency: `/home/jaewoo/ResCode/results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt`.
Projection seed: 0. Residual = FP32(W) − FP32(BF16 saliency base).

## Structural direction similarity

Down comparisons align Gate/Up rows with Down columns. They describe neuron-vector direction only; Down uses a different QJL input space.
- gate_up: median expert median |cosine| = 0.046 (all W); paired W 0.042, R 0.040.
- gate_down: median expert median |cosine| = 0.044 (all W); paired W 0.041, R 0.040.
- up_down: median expert median |cosine| = 0.075 (all W); paired W 0.070, R 0.069.

## Actual Gate–Up QJL sign bits

Signs use the repository's FP16 Gaussian ProjectionBank and pack_residual. Direct and inverted agreement are measured from packed XOR bits.
- D=8192, original_weight: median expert mean direct agreement 0.500; best with inversion 0.521; shuffled-neuron 0.500; random-bit 0.500; Gaussian prediction 0.500.
- D=8192, residual: median expert mean direct agreement 0.500; best with inversion 0.518; shuffled-neuron 0.500; random-bit 0.500; Gaussian prediction 0.500.
- D=16384, original_weight: median expert mean direct agreement 0.500; best with inversion 0.520; shuffled-neuron 0.500; random-bit 0.500; Gaussian prediction 0.500.
- D=16384, residual: median expert mean direct agreement 0.500; best with inversion 0.517; shuffled-neuron 0.500; random-bit 0.500; Gaussian prediction 0.500.

## Storage feasibility

Bytes include two FP32 row norms, packed sketches, a uint16 count plus uint16 positions per sparse-corrected row, packed inversion/selection flags, and stated 16-byte format metadata. Entropy is reported separately from stored bytes.
Fully shared Gate without correction is lossy; dense XOR is lossless but has no bit-storage savings.
- D=8192, original_weight: median sparse lossless ratio 4.322; maximum selective lossless savings 0.00%; selectively shared neurons 0.0%.
- D=8192, residual: median sparse lossless ratio 4.346; maximum selective lossless savings 0.00%; selectively shared neurons 0.0%.
- D=16384, original_weight: median sparse lossless ratio 4.330; maximum selective lossless savings 0.00%; selectively shared neurons 0.0%.
- D=16384, residual: median sparse lossless ratio 4.355; maximum selective lossless savings 0.00%; selectively shared neurons 0.0%.

## Validation and limits

- Maximum relative W=S+R reconstruction error: 3.23e-08.
- Group partition, tensor dtype/shape, saliency coefficient sums, exact sparse XOR decoding, and byte accounting are checked.
- Saved cache metadata and membership were validated; grouping was not recomputed. The original Wikipedia calibration text was unavailable, so its file size could not be checked.
- No model evaluation, perplexity, accuracy, or downstream benchmark was run. Accuracy effects of sharing remain unverified.
