# 01150: the residual candidate is insufficient

The user reports another visible boundary shift after the deterministic-sampler
guard and overridden-Linear repair. This run executes dense v2 clean transfer
and `source_residual_dense_drift_v2` with zero-churn `sample_euler`. Its provenance
reports `deterministic_initial_noise_flow`. The existing three-frame boundary
dense policy and five-token right decoder context execute. A selector mismatch
does not explain this failure.

The log reports zero **user** DoRA model/CLIP hooks. VDN's built-in default and
turbo adapters remain active at strength 0.5 each. Zero user hooks must not be
described as zero adapters or zero forward hooks.

## Matched operands

All eight exported tensors pass manifest hashes, byte counts, geometry,
finite-value and protected-prefix checks. The seed, geometry and sigma match
01144: seed `16182742365559695871`, source 44x44, target 62x62, 47 tokens,
prefix 12 and sigma `0.8780487775802612`. The exported window is [10,17).

[Verified receipts](CONTINUATION_01150_VERIFIED.json) retain the input identities,
manifest, selected runtime events and direct matched tensor differences.
Authoritative prefix, provider clean, restored pre-high clean and initial mask
are byte-identical between 01144 and 01150. The high sampler input changes only
in generated rows, by 0.01566889 RMS over the exported window. First-high before
Flow changes by 0.07224118 RMS; final changes by 0.07074518 RMS. Exact clean-prefix
ownership is retained. The residual correction changes the actual sampler input
but does not remove the reported failure.

## Native tile replay

Replay first-high before Flow and final with the same official unquantized VAE,
FP32 CPU arithmetic, native SDPA, spatial tile (1,2) and unblended spatial interior
used for 01144. Byte-identical pre-high operands permit reusing that stage's
01144 decode. The [replay receipt](CONTINUATION_01150_NATIVE_TILE.json) records
decoder/source hashes and scope.

| Stage | 01144 dx (px) | 01144 dy (px) | 01150 dx (px) | 01150 dy (px) |
| --- | ---: | ---: | ---: | ---: |
| Pre-high after DC | -0.089682 | +0.027372 | -0.089682 | +0.027372 |
| First high before Flow | -0.599750 | +3.246390 | -0.432725 | +3.333073 |
| Final | -1.590492 | +2.905442 | -1.004600 | +2.539960 |

These are local first-transition phase-correlation estimates. Production's raw
decoded upper45 receipt independently reports dx=-2.379722, dy=+4.107829 px,
before the one-frame photometric assembly correction. Its affine receipt reports
boundary axis scales about 0.990 versus pre-boundary scales near 1.0. Do not
interpret the local estimate as a calibrated uniform whole-frame translation.

[Same-frame comparisons](CONTINUATION_01150_DECODED_SAME_FRAME.json) compare the
same decoded frame before and after high refinement. In this interior the first
retained frame changes by approximately dx=+1.50, dy=+1.99 px in first-high;
later matching frames have much smaller estimates. The first two saved frames
are carried-prefix output. Their geometry estimates stay below 0.08 px, but RGB
RMS changes about 0.012 because the noncausal decoder sees different future
latents. Exact latent-prefix equality alone does not imply exact decoded pixels.

The limited replay omits preceding temporal-window blending and full-frame
spatial blending, and does not reproduce production INT8 ConvRot arithmetic or
rerun the H3 transformer. It establishes a local change by the first actual high
prediction before Flow guidance. It cannot certify rendered acceptance or rule
out every transfer/decoder contribution elsewhere.

## Native execution check

The existing oracle compared partitioned probe with partitioned high; a shared
error could pass that comparison. Extend it to compare generated video and audio
directly with unwrapped Core using identical conditioned inputs. Include short
and 12-token/47-token prefixes, image-reference packing, and FP32/BF16 video
inputs. Keep Core's per-call layout writes in a separate options dictionary and
check stage cleanup and input immutability.

This uses actual Core blocks and Sol's dense CPU arithmetic, with small random
weights. It does not test the trained VDN branches, production sparse CUDA
kernel, full checkpoint, or rendered continuation. Passing this oracle must not
be promoted to a claim that production high refinement is equivalent or fixed.

## Review conclusion and next discriminating run

The continuous clean map fixes an analytical coordinate inconsistency. Separating
initial Gaussian noise from structured drift fixes a second demonstrated
operator inconsistency. **Neither is a sufficient rendered fix for 01150.**
Preserve Claude's sampler restriction and Core forward-identity guard; this
evidence does not justify reverting either or extending the boundary-dense
policy again. Later Spectrum forecasts, Flow guidance and the photometric patch
cannot solely explain a shift already measurable before those operations.

The next discriminating production control is the existing
`spatial_stage_control=same_grid_target_control`, with the same seed, prompt,
references, masks, sampler schedule, adapters and decoder precision. It runs
low/probe on the target grid and bypasses both learned spatial transfer and
cross-grid residual transport through exact identity. Compare its first-high
and raw-decode receipts with 01150. A persistent jump implicates the native
split/high conditioning path; its disappearance implicates the heterogeneous
handoff or its interaction with the trained high model. This is a diagnostic
control, not a visual repair or a production recommendation. Its GPU cost is
higher and has not been measured here.

Flow #93 remains a draft and fails rendered continuity acceptance. No new
production warp, operator, guidance term, adapter-strength default, sampler
lifetime or model evaluation is introduced by this review.
