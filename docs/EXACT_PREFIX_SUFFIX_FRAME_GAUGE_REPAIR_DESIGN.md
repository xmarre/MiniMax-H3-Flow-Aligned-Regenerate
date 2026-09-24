# Exact-prefix / suffix frame-gauge repair design

Status: investigation checkpoint; architecture and validation contract are being completed. No production fix is implemented by this document commit.

## Scope and evidence boundary

Flow constructs a low-resolution continuation, transfers its clean video through a learned 3D upscaler, replaces the upscaler's generated prefix with caller-owned exact target-grid latents, and continues high-resolution sampling. The defect under investigation is a visible spatial jump at that replacement boundary. Auto-strength must be OFF for validation. User-observed improvement of background coherence with auto-strength disabled is hardware evidence supplied with the task; the historical artifacts inspected so far do not contain an authoritative resolved auto-strength-off receipt.

## Verified source anchors

- Production main / v0.3.8: `a6249b8343becc1458a4555d2bb25cd523983c90`.
- Exact physical-prefix runtime used for 00625: `e57f320b3ece93e3289ebcccd4104ea6a1265526`; tree `61bce93066b30c9923d8a99a0eec0ea80d0f54e5`.
- PR #87 is open/draft at that exact historical commit on `mirror/00625-first-chunk-provenance-20260923`. Preserve it as the reproduction control.
- PR #88 is open/draft at `f883adebd30925ef2f1f86bc0bd5c716703cdb59`, diagnostic only.
- PR #89 is open/draft on `candidate/frame-shift-exact-prefix-gauge-20260924` at `ae1a7795254dec213a35bf09d83e4b6324a91c54`, one commit over main. It is the existing frame-shift candidate owner. Its additive residual bridge must be audited, not silently accepted.
- PRs #81/#84/#86 are closed, unmerged historical evidence.
- Learned provider source: `xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus` at `620165a311de9b28a36260219fb5cd370a304e3c`.
- Auto-strength owner: `xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader` at `51c44419bbe3bbdea2d7ea080e32bd61e9f0fa38`, `nodes.py`, `PowerLoraLoader` path and `_auto_strength_analyze_base_targets`.

## Findings that constrain the design

1. The 00625 source prefixes used for low/probe and learned transfer are the same H3 physical 2x2 patch-lattice projection. Preserve this behavior.
2. The provider processes all video frames together with learned 3D convolutions, normalization and `trilinear` interpolation with `align_corners=False`. Temporal length remains unchanged. Its prefix is paired temporal evidence, but source does not guarantee that learned distortion is one rigid translation or that it transfers unchanged to the suffix.
3. Existing `measure_translation_trajectory` compares distinct temporal frames. In 00625, full-frame learned-native `(-0.140466,-0.010790)` is the first anchor pair, not `anchor_final`; the final anchor is `(0.037421,-0.066870)`. Exact-restored final anchor is `(0.699761,0.758296)`. These values localize a discontinuity and must not be used directly as a correction vector.
4. The old rigid implementation already compared same-index learned and exact prefix frames. It used four tail frames, phase correlation, componentwise lower medians, a peak/mean-absolute response and loose MAD checks. It shifted the whole suffix in recovered clean space, retained handoff noise through affine mapping, then shifted every source-grid guidance sample by scaled learned displacement. Its failure cannot be dismissed as adjacent-frame motion estimation.
5. Run 00627 actually applied `(-0.359221,-0.496345)` target latent cells to 50 suffix frames and still failed the reported frame-shift acceptance. X estimates `[0.335381,0.657972,0.359221,0.664755]` yield lower-median MAD `0.023841`, understating the split into two clusters. This is a concrete confidence weakness; it does not establish the sole cause of failure.
6. Full-field residual transport `S_t + (E-L)` preserves algebraic temporal differences, but is not a spatial translation. The historical 00629 candidate was reported to preserve those differences numerically and still fail visible frame-shift acceptance. PR #89 reuses that principle. Do not equate its green CPU tests with spatial correctness.
7. `build_handoff_state` uses `Y=(1-sigma)L+sigma*N`, preserving raw audio. The scheduler regenerates the deterministic noise and inverts Y for diagnostics, then feeds a DC correction derived from that recovered state into production. A new spatial estimator should observe the actual provider output before re-noise instead.
8. High-stage Flow reference `G=resize_video(source_ref, target_hw)` uses generic bicubic coordinates. G and learned output L are different mappings of the source. A displacement measured between L and E cannot automatically be applied to G.
9. Auto-strength is upstream of Flow, changes per-target LoRA strengths, and can be resolved from State Manager rather than visible node kwargs. Its current report exposes resolved enablement and row strengths; complete validation provenance needs resolution origin and effective per-target adjustment identity as well.

## Intended architecture

Implement a bounded, opt-in paired-prefix registration transaction at the clean learned-video handoff. Retain actual learned clean output, estimate same-time target-grid alignment, validate rigid fit on held-out frames/regions, and apply one constant target-grid translation only to generated suffix frames. Reject ambiguous or nonrigid cases to the unchanged physical-prefix baseline without resampling or restarting a sampler.

Keep the existing noise realization unchanged. Apply a clean correction before re-noise, or its algebraically equivalent affine difference to the existing conditional state. Preserve exact prefix, original protected noise, masks, audio, and caller-owned data. Guidance needs an independently validated target-grid reference mapping; do not blindly reuse the learned displacement or edit shared trajectory samples. No extra H3 evaluations or sampler lifetimes are permitted.

The completed specification will provide precise estimator gates, coordinate formulas, guidance-field transport, API changes, tests, hardware acceptance and rollback rules. A universal rigid-translation root cause is not yet established by available runtime evidence.

## Topology

This documentation-only branch is `design/exact-prefix-suffix-frame-gauge-20260924`, based on production main. Implementation belongs on the existing #89 candidate after live re-fetch, developed on a separate mirror and consolidated to one clean implementation commit only after validation. Preserve #87/#88 and every historical evidence checkpoint. Do not modify production in this investigation.
