# Continuation 01109: DC-only candidate fails rendered acceptance

The user reports that frame-shift, shock and tone changes remain in the completed
01109 render and supplies two adjacent boundary snapshots. The run executes the
current DC-only contract; it is a failed rendered qualification, not evidence
that the older structural bridge or temporary video-prefix release still runs.

PR #93's reviewed source head before this qualification is `a928e31e`. It is
preserved at `checkpoint/01109-before-boundary-review-20261004`. The available log
does not identify the installed Flow source bytes, so this qualification identifies
the executed runtime receipts rather than claiming a verified installation SHA.

## Executed path

- Source/target geometry is `50x38 -> 72x54`, with 12 protected prefix tokens and
  50 generated tokens. This differs from 01097's `32x58 -> 46x82`; these runs are
  not a matched comparison of correction quality.
- The bridge reports `partitioned_exact_overlap_dc_only_v5`, structural support
  zero, DC support one, weights `[1.0]`, and no later-suffix extrapolation. The
  first-token channel-mean correction applies with RMS `0.20956181`.
- Target-high reports `partitioned_video_high_exact_context_v4`, requested video
  width six, applied width zero, exact sampler video mask and exact model video
  context. Final prefix restoration is exact.
- All six continuation guidance calls use the actual learned provider pair and
  `exact_prefix_guidance_reference_dc_v1`. The captured-versus-provider source
  suffix differs by max `4.7683716e-7`, RMS `1.4889282e-8`.
- Provider soft support, spatial registration mutation, high prediction gauge
  repair and VAE-window repair do not apply. The provider soft-support selector
  remains loaded but reports production mutation disabled.

These receipts establish the current path's ownership and correction scope.
They do not establish that the rendered boundary is smooth.

## Latent-stage localization

First boundary-pair translations, in each diagnostic's latent measurement cells:

| Stage | upper45 dx / dy | full dx / dy |
| --- | --- | --- |
| Learned provider native | -0.081142 / +0.014469 | +0.215291 / +0.450016 |
| Exact-restored, DC-corrected pre-high | +0.124212 / +0.079022 | +0.773949 / +0.927286 |
| First actual high prediction before Flow | +0.104574 / +0.294506 | +0.326764 / +0.584829 |
| Same prediction after Flow | +0.104743 / +0.293816 | +0.328151 / +0.585795 |
| Final post-high | +0.115207 / +0.316409 | +0.214253 / +0.424831 |

Exact-prefix restoration leaves a substantial native boundary mismatch in the
full region. The first high prediction increases upper-region vertical motion,
while reducing the full-region pre-high displacement. Flow changes the measured
translation only slightly. Thus this run does not support a guidance-only fix,
nor does it isolate target-high as the sole origin of the decoded artifact.

The final raw exact-seam RMS and lowpass ratios improve relative to pre-high
(`0.9351102` and `0.6988390`) despite the failed render. Those tensor-space
reductions cannot substitute for decoded motion/tone acceptance.

## Decoded evidence

All 22 sampled duplicate-overlap frames match exactly: 17 interior samples and
five right-context tail samples have zero RGB difference. The native decoder
uses five real future latents for right context. Continuum seam assembly reports
`patch_frames=0`, `action='kept native boundary'`.

The decoded trajectory spans several generated frames:

| Region | Pre-boundary median dy, px | First three generated dy values, px |
| --- | --- | --- |
| upper45 | +0.6130 | +0.2532, +2.7965, +5.3073 |
| full | +0.7153 | +0.7633, +3.2358, +5.0350 |

Both regions agree on the larger second/third transition. This is a measured
motion discontinuity; it does not establish a skipped output frame. The affine
receipts do not establish a coherent calibrated zoom event.

Against the last preceding decoded frame, the third generated frame changes
luma mean by `-6.5346%`, fifth percentile by `-50.1483%`, and deviation by
`+2.3999%`. These are scene-content statistics, not exposure calibration or proof
of a single tone actuator. The decoder's `clean_boundary` classification and
zero seam patch do not certify perceptual continuity.

## Next causal evidence

This run already exports eight native boundary-window tensors, totaling
19,035,648 bytes (about 18.2 MiB), without another model/provider/VAE call:

```text
h3_flow_regenerate/residual_geometry/
session-continuum_chunk-2_seed-16337513745387792240_sigma-0.87804878_1791091815413576674/
```

The directory is relative to ComfyUI's configured output directory. Its manifest
SHA256 is `608ac4ef58619be438f212230ab4aab16f4f602fe76512f1e2c9c568f5d03ee2`.
The snapshots contain the authoritative prefix, provider clean output, corrected
pre-high clean window, first actual high sampler input, initial video mask, first
high predictions before/after Flow, and final clean window.

The archive is not among the supplied attachments. Receipts and two cropped
decoded frames cannot reconstruct these tensor operands. Before choosing another
runtime mutation, inspect the saved bytes and compare the native seven-token
windows across provider, exact-prefix splice, first high prediction and final
state. For a returned-suffix decoder comparison, replace the first two window
tokens with the saved authoritative prefix, then use the model's latent-output
conversion and native VAE. Inspect the first retained local frame five and the
following frames, rather than just one latent pair. Any decoded comparison
requires the actual native VAE; a surrogate is not rendered acceptance evidence.

No new sampler arithmetic or actuator is promoted by this qualification. The
one-token DC path is retained as the current numerical contract, not declared a
solution. Previously rejected prefix release, spatial residual transplants,
prediction/VAE warps and fixed successor guards remain retired. A new render is
not required to obtain the already-exported bundle.

## Evidence identity

- Metrics SHA256: `af7463c500aa8a48c2795bd9dc7d1c9661ee04264daa40a1f8cbd160d8029a13`.
- Process log SHA256: `b0fb5e26468f824346ca0f903af83df83b0c94e0d82e0e2d348350189eb9372c`.
- Replay exposed an offline receipt-name drift: the partitioned scheduler emits
  `transferred_prefix_output_discarded`, while the gate still required the older
  `upscaler_prefix_output_discarded`. The validator now accepts either boolean
  proof, rejects conflicting aliases, and rejects missing or non-boolean proof.
  This repairs evidence replay, not sampling or rendered continuity.
- The green source tests and CI recorded for `a928e31e` establish code contracts,
  not a visual fix. This follow-up changes qualification documentation and the
  offline validator only; production sampler arithmetic stays unchanged.
