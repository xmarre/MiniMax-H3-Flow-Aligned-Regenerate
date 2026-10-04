# Continuation 01097: exact high context still fails rendered continuity

Follow-up: [01097's DC-only candidate also fails in 01109](CONTINUATION_01109.md).
The candidate rationale and pre-render limitations below describe the state at
the time of this review, not current rendered qualification.

Run 01097 follows 01093 after target-high video-prefix release was retired.
It is not a matched comparison: source/target spatial geometries changed from
`54x36 -> 76x50` to `32x58 -> 46x82`. The legacy node value remains
`video_guided_overlap_tokens=6`, but
the runtime reports `partitioned_video_high_exact_context_v4`,
`requested_tokens=6`, `applied_tokens=0`,
`high_sampler_video_mask_exact=true`, and
`high_model_video_context_exact=true`. The old overlap closure does not execute.

The actual learned provider pair is also bound successfully. Captured-versus-
provider source suffix disagreement is floating-point roundoff only
(max `4.7683716e-7`, RMS `1.4583e-8`), and continuation high guidance uses
both `handoff_reference_used=true` and
`temporal_handoff_reference_used=true`.

Despite those corrections, the rendered issue survives. One remaining active
pre-high mutation is the v4 exact-overlap successor bridge selected after video
registration/boundary motion accept but Euler guidance registration rejects its
unsupported sampler contract. In this run its complete exact-vs-learned residual
has RMS `0.4698693`; the zero-spatial-mean structural component alone is
`0.4094602` RMS and the DC component is `0.2304769` RMS. The bridge writes
that spatial residual into four generated tokens using
`(1.0, 0.75, 0.5, 0.25)`.

That bridge's unit contract is algebraic: after adding the same residual to the
first suffix token, `corrected_suffix - exact_prefix` equals the provider's
`learned_suffix - learned_prefix`. This does not imply equal H3 or VAE behavior
under a spatially varying additive field. 01097 demonstrates that the algebraic
first-transition guarantee is insufficient
for rendered acceptance. It does not isolate this bridge as the cause. The
learned-provider boundary has a large phase displacement, while the v4
transplant nearly zeros the immediate latent transition; the decoded boundary
nevertheless retains the reported shock and tone/content change.

The production response is therefore to retire the spatially varying successor
transplant rather than add another rigid/prediction/VAE actuator. The v5 candidate
restores the older one-token DC correction only:
per-channel spatial mean is applied to the first generated token, no spatial
structure is copied from the exact prefix, and every later suffix token remains
unchanged. The high guidance reference gauge is likewise forced to
channel-mean-only first-token support; otherwise guidance would silently
reintroduce the retired structural residual even after the state mutation was
removed. Rigid registration, high-prediction gauge repair, VAE-window repair
and video-prefix release remain disabled.

This document records localization and the next candidate. It does not claim
rendered acceptance.

## Localization and limits

All 22 sampled duplicate-overlap decoded frames match exactly, including the five
right-context tail samples; seam replacement is zero. Native decoder alignment
and overlap assembly therefore have no measured discrepancy in these samples.
This does not exclude H3 or VAE behavior at the new generated content boundary.

In upper45, the first latent pair dy changes from `+0.0614874` pre-high to
`+0.2446821` before the first Flow correction and `+0.2431063` after it.
Final post-high is `+0.1637346`. Flow's correction is small relative to that
first high-stage change. This is latent-space localization, not proof of a
literal skipped decoded frame or calibrated zoom.

Against the last pre-boundary decoded frame, the third generated frame's luma
fifth percentile falls about `36.50%` and deviation rises about `6.63%`. At the
fourth generated frame, deviation rises about `13.08%`. These support a tone
change, but are scene-content statistics, not a calibrated exposure measurement.

The leading unresolved alternatives are the learned transfer's native boundary,
target-high conditioning/refinement, and the removed structural/DC transplant.
No captured tensor bundle or matched render of the DC-only candidate is available
in this review. Removing the transplant may expose an existing native structural
mismatch. Source tests establish the new mutation and guidance contracts; a render
with the same 01097 seed, prompt, references, geometry, sampler and schedule must
establish whether the visible artifact improves.

## Source validation

The reviewed mirror is preserved at
`checkpoint/01097-dc-only-targeted-green-20261004` (`5ace0630`). Its validation is:

- targeted bridge, learned/DC guidance, seam and runtime-gate tests: **249 passed**;
- full local Python 3.12 CPU suite: **850 passed, 27 skipped**; the skipped native
  oracles run in source-contract CI;
- Ruff check and format check, compileall, sdist and wheel build: pass;
- pinned Sol/VDN exact-prefix ABI check: pass, including `same_grid_history=True`;
- [mirror CI](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/actions/runs/37178408407):
  all five jobs pass, including Python 3.10-3.13 and native/sibling source contracts.

Historical structural gauge counterexamples are retained through the historical
primitive, independent of the production bridge that now rejects multi-token
support. New tests cover stationary/moving temporal correspondence with the
learned provider pair and DC-only gauge, actual and forecast prediction wrappers,
exact audio/prefix preservation, recursive lifetime cleanup, and runtime-gate
rejection of structural reactivation or mismatched support/ownership receipts.

Consolidation changes only this validation record and a module docstring after
that green source tree. These checks establish code contracts and compatibility;
they do not substitute for a render with the actual H3 and upscaler checkpoints.
