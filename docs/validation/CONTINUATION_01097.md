# Continuation 01097: exact high context isolates the structural successor transplant

Run 01097 is the matched follow-up to 01093 after target-high video-prefix release
was retired. The legacy node value remains `video_guided_overlap_tokens=6`, but
the runtime reports `partitioned_video_high_exact_context_v4`,
`requested_tokens=6`, `applied_tokens=0`,
`high_sampler_video_mask_exact=true`, and
`high_model_video_context_exact=true`. The old overlap closure does not execute.

The actual learned provider pair is also bound successfully. Captured-versus-
provider source suffix disagreement is floating-point roundoff only
(max `4.7683716e-7`, RMS `1.4583e-8`), and continuation high guidance uses
both `handoff_reference_used=true` and
`temporal_handoff_reference_used=true`.

Despite those corrections, the rendered issue survives. The remaining active
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
under a spatially varying additive field. 01097 shows the distinction. The
learned-provider boundary has a large phase displacement, while the v4
transplant nearly zeros the immediate latent transition; the decoded boundary
nevertheless retains the reported shock and tone/content change.

The production response is therefore to retire the spatially varying successor
transplant rather than add another rigid/prediction/VAE actuator. The v5 path
restores the older hardware-qualified one-token DC correction only:
per-channel spatial mean is applied to the first generated token, no spatial
structure is copied from the exact prefix, and every later suffix token remains
unchanged. The high guidance reference gauge is likewise forced to
channel-mean-only first-token support; otherwise guidance would silently
reintroduce the retired structural residual even after the state mutation was
removed. Rigid registration, high-prediction gauge repair, VAE-window repair
and video-prefix release remain disabled.

This document records localization and the next candidate. It does not claim
rendered acceptance.
