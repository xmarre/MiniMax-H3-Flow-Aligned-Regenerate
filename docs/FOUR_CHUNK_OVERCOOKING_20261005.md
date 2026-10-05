# Four-chunk overcooking investigation — 2026-10-05

Status: evidence audit complete; visual root cause and runtime fix are **not yet
established**. This record changes no sampling behavior and is not a runtime
overlay. A four-chunk GPU reproduction has not been performed in this workspace.

## Capture identity

The video is 832×832, 24 fps, 672 frames / 28 seconds with stereo audio. Its
attached log and metrics describe three progressive continuation handoffs.
The generating workflow and exact loaded source revision are unavailable;
ffprobe exposes no workflow/prompt tags in the MP4.

| Attachment | SHA-256 |
| --- | --- |
| `metrics_01173_.json` | `b2234cf0e29f5ea5445c52a34c66283e33d8b30945b094d69bdac07171aa71b3` |
| `Pasted text(20261005-054345).txt` | `f10adce440fc0fa0b08d83c1cf6abfdbca8bb70ac43c162544568de80822b298` |
| `manifest(5).json` | `14c3117a546528b0a5a5f312ccd988f63e55ac6516f5876f1496f7b43b905cb8` |
| `MiniMax_H3_00034-audio.mp4` | `ceafa8fee1de0e504ff14324e138511dc92b4bf2567f2253e3c5e5fa782abc3a` |

The native boundary manifest matches chunk 4: 52×52 target, seed
`6898659038325342056`, sigma `0.8780487775802612`, twelve protected tokens and
67 temporal tokens. It captures tokens 10 through 16: two protected tokens
and the first five generated tokens. All eight tensor byte counts and hashes
match this manifest, and all values are finite. The separate 56×74 residual
geometry capture is excluded from this run's stage comparison.

Visual inspection confirms increasingly exaggerated facial detail and shading
in later generated content. Camera framing, expression, and shot changes also
vary; the video alone cannot attribute those changes to one sampler component.

## Confirmed execution facts

All three `partitioned_stage_plan` receipts report:

- `spatial_stage_control=progressive_low_to_high` and
  `same_grid_control_active=false`;
- configured and effective generated-suffix low grid 36×36, target grid 52×52;
- twelve protected video latent tokens on the target grid;
- handoff index 10, sigma `0.8780487775802612`, then six high-stage steps;
- `frame_gauge` result `identity`, reason `already_aligned`, with no spatial warp.

The installation receipt reports `frame_gauge_repair_default=false` and
`production_default_changed=false`. Current Flow v0.3.9's production node has
both values true and defaults to `same_grid_target_control`. A compatibility
node, older loaded code, or an older explicit workflow can explain parts of
this mismatch; the receipts do not identify which one was used. Updating a node
default does not overwrite explicit saved workflow selections.

The released target-grid continuation uses target H/W for low/probe execution
and an identity clean-video handoff. All-generated first chunks still use
learned progressive transfer. Thus this capture is **not a four-chunk test of
the released target-grid continuation default**.

Other receipts narrow the investigation:

- Four distinct trajectory run IDs are present; each continuation selects its
  own guidance trajectory and resets guidance state before high sampling.
- VDN default and turbo strengths are 0.5 for all four chunks. Adapter hook
  ownership in the audited source replaces active registrations rather than
  accumulating post-forward adapters.
- Continuation residual RMS is 1.00167, 1.00156, and 1.00187. Correlation with
  initial effective noise remains approximately 0.9996. These measurements do
  not show growing residual amplitude.
- Decoded copied overlap has sampled RGB difference RMS zero. Assembly reports
  no applied video tone correction. This narrows the issue to generated content
  or its conditioning rather than modification of that measured overlap.
- All three boundaries are detected as scene cuts. RGB mean/std changes across
  those cuts are not causal proof of contrast amplification.
- All four conditioning manifests retain three image-reference slots with
  identical shapes. The source preserves these references through native masked
  continuation. The logged shapes do not establish byte identity.

## Chunk 4 stage comparison

The authoritative two-token prefix is byte-exact in both the pre-high clean
state and the final clean output. The first native prediction's protected
prefix equals its inpainted sampler input, and Flow preserves that prediction
prefix. The initial video mask is exactly zero on the two protected tokens and
one on all five generated tokens.

The DC bridge adds a per-channel spatial constant to the first generated token
(RMS 0.212962), with spatial variation below 8e-8 and exactly zero change to
later generated tokens. It does not increase their spatial variation.

| Stage | Mean channel spatial std over five generated tokens | Horizontal gradient RMS |
| --- | ---: | ---: |
| Learned provider clean | 0.777938 | 0.532381 |
| Exact-restored pre-high clean | 0.777938 | 0.532381 |
| First high native prediction before Flow | 0.801559 | 0.568066 |
| First high prediction after Flow | 0.795424 | 0.567619 |
| Final high clean | 0.796654 | 0.573260 |

These are latent statistics, not decoded contrast or image-quality scores.
At the captured first evaluation, the native high prediction increases both
statistics; Flow guidance reduces them slightly. Guidance changes the generated
suffix by RMS 0.046305; the native prediction differs from provider clean by
RMS 0.336424. This does not establish cumulative causality across chunks or
exclude effects of earlier low-stage forecasts.

## Source audit and limits

| Component | Audited revision |
| --- | --- |
| Flow | `d44174d7873f4da4158f072aa98629d43001eda5` |
| Continuum | `0ba5ab3f9cdaf9a17c34fead979e16b441b27a3c` |
| VDN | `e4232ce584c3449cb66d772e9bc7f6c7b4463c82` |
| Latent Upscaler | `fc58fb80246bc58383929d21e78f54cf87d6507a` |
| ComfyUI Core | `5c460d8172fe30761ff67c0df3d5643bb74e0d70` |
| Spectrum | `5161f0457bc8c52535212d6783eee73f439e1537` |

No decode/re-encode loop between Continuum chunks, repeated adapter injection,
or guidance-history leakage was found in these sources. Exact-prefix integrity
does not rule out recursive drift in newly generated suffixes.
Spectrum creates a fresh runtime on MODEL clone and resets its forecasters and
model-aware controller at each sampler lifetime. The run uses Spectrum forecasts
in low and high stages, plus DiffAid text activation modulation and visual
reference modulation; their numerical influence is not isolated by this capture.

Repeated learned transfer and heterogeneous low/probe execution remain plausible
contributors, not proven causes. GroupNorm depends on complete temporal/spatial
context, so an unqualified temporal-window change is not an equivalent fix.

Core's VAE returns normalized video latents; the upscaler applies channel
statistics again. That additional transform is also present in the checkpoint
author's reference inference implementation. Removing it without the training
input contract would be speculative. See the author's
[reference implementation](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_3d.py)
and [published normalization specification](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/config.json).

Existing targeted checks: stage/native-modulation/guidance-gauge/primary-execution
suite 84 passed, 10 skipped; diagnostic/guidance-gauge/residual-lattice/frame-gauge
suite 225 passed; fresh-clone/single-wrapper/fresh-stage-call checks 3 passed.
Suites overlap and their counts must not be added as unique coverage. Native
modulation tests were skipped because Core was not on the test import path.
These CPU checks do not validate trained-checkpoint video quality.

## Next discriminating reproduction

Recover the generating workflow, including effective Patcher overlays and loaded
revisions. With the same seed, prompt, reference,
model/adapters, sampler, step schedule, target size, and four-chunk duration,
compare the captured progressive selection with `same_grid_target_control` on
the coordinated released stack. Confirm 52×52 low/probe and identity continuation
handoff in receipts before judging the video.

If the target-grid control also deteriorates, investigate the native continuation
conditioning/model response. If it is stable, the comparison localizes the issue
to the removed progressive continuation path as a whole; it does not by itself
separate learned transfer from heterogeneous attention. Keep any production
correction scoped to the demonstrated mechanism and validate all four chunks.
