# Exact-prefix boundary-motion validation

The optional frame-gauge repair registers the learned provider's clean video
against an authoritative exact prefix. The prefix remains caller-owned. Only an
accepted transaction can translate the generated suffix and its independently
registered Flow guidance reference.

The boundary-motion gate checks whether exact-prefix replacement disturbs the
provider's transition. Let `P` and `S` be the provider's last prefix and first
suffix frames, `E` the authoritative last prefix frame, `W` the proposed spatial
translation, and `M(A, B)` the bounded phase-correlation motion estimate.

Policy `native_boundary_motion_preservation_v2` measures:

- Before repair: `||M(E, S) - M(P, S)||`.
- After repair: `||M(E, W(S)) - M(W(P), W(S))||`.

Both comparisons use the provider transition in the corresponding coordinate
domain. The transformed prefix `W(P)` is a diagnostic witness; it is never
substituted for `E` in the output.

The estimator crops a region, applies a Hann window, and selects a phase peak.
Jointly translating and interpolating both input frames can change the measured
peak, especially when several motions compete. Comparing the corrected splice
against `M(P, S)` would count this estimator change as splice error. In particular,
when `E = W(P)`, the after-repair replacement error must be zero.

The upper-45% and full-frame checks retain the same requirements: unclipped
estimates with response at least 3, non-increasing replacement error, and at least
25% improvement when the before-repair error is at least 0.125 target latent
cells. The transformed-native estimate must also pass the ambiguity checks.
Video registration and independent guidance registration remain mandatory.

When frame-gauge repair is enabled and rigid-v2 rejects solely because an
otherwise unambiguous candidate fails one of the native-boundary improvement
requirements, partitioned exact-prefix continuation has a separate bounded
fallback. It does **not** convert the rigid transaction into an acceptance:
`spatial_warp_applied` remains false and no registered Flow guidance reference
is published.

The fallback uses the exact same-time overlap already available from the learned
3D transfer. For the learned provider's last prefix token `P`, authoritative
token `E`, and first learned suffix token `S`, define `D = E - P`. The
structural bridge adds only `D - mean_spatial(D)` to `S`; the existing
one-token DC bridge supplies `mean_spatial(D)`. Together they preserve the
provider's immediate native clean-domain transition algebraically:

`(S + D) - E = S - P`.

Eligibility is fail-closed. The video registration must have accepted, the
rigid-v2 receipt must contain both `upper45` and `full` checks, and
`native`, `transformed_native`, `exact_restored`, and `candidate` must
all be finite, unclipped, and have response at least 3. Ambiguous registration,
invalid-area, guidance, or other rejection classes keep the existing baseline
fallback. The structural correction touches only the first suffix token; the
authoritative prefix and all later suffix tokens remain unchanged. It adds no
model evaluation, provider call, VAE decode, random draw, or sampler lifetime.

Receipts report this path separately as
`partitioned_exact_overlap_structural_plus_dc_v1`. The offline runtime gate
replays the rigid-v2 boundary arithmetic and requires the recorded fallback
trigger to be exactly the rejection that the receipt reproduces.

The gate reuses the already-transformed two-frame witness and adds two bounded
motion estimates. It adds no model evaluation, provider call, VAE decode, random
draw, or sampler lifetime. Repair disabled preserves the existing path.

Runtime receipts include `native`, `transformed_native`, `exact_restored`, and
`candidate` measurements for each region. The offline validator recomputes v2
errors and improvement from those measurements. Historical v1 receipts retain
their original interpretation; they cannot establish a v2 acceptance result.

This gate validates geometric replacement consistency. It does not establish
semantic continuity or decoded-video quality. A successful structural test or
runtime receipt still requires inspection of matching decoded boundary media.


## Diagnostic-only boundary content continuity

00675 separated the remaining failure from the rigid frame-gauge defect: direct
video inspection found the camera/frame snap substantially gone while multiple
background objects still changed across the exact-prefix boundary. That is a
content/state continuity problem, not evidence that the rigid threshold should
be weakened.

When frame-gauge repair is enabled, the partitioned runtime now records
`partitioned_boundary_content_continuity_v1` at three existing clean-domain
stages:

- `provider_native`: the learned provider's own prefix -> suffix transition.
- `pre_high_exact_restored`: authoritative exact prefix -> corrected learned
  suffix immediately before target-grid high-stage sampling.
- `post_high_internal_clean`: authoritative exact prefix -> final generated
  suffix after high-stage sampling, converted through the existing model latent
  input transform for a common-domain comparison.

The diagnostic is observation-only. It never alters the sampler state, guidance
reference, prefix ownership, suffix tokens, frame-gauge decision, or fallback
selection. It adds no H3 NFE, provider call, VAE call, sampler lifetime, or
history boundary.

Each stage compares the boundary pair against the previous three prefix-internal
pairs. A fixed 4x4 target-latent tile grid publishes raw RMS, spatial low-pass
RMS, spatial-mean-removed low-pass RMS, low-pass gradient RMS, per-channel
spatial-mean RMS, normalized cross-correlation, and conservative local
correspondence statistics. The local matcher uses a bounded 3-cell search,
minimum similarity 0.35, minimum uniqueness margin 0.02, and exact integer
forward/backward cycle consistency. These values are diagnostic parameters, not
production acceptance thresholds.

The receipt also ranks tile identities by boundary centered-structural residual
relative to the local prefix baseline. A separate
`partitioned_boundary_content_stage_delta` compares pre-high and post-high
receipts. This is intended to answer two questions before any new correction is
designed:

1. Is the changed object/region already inconsistent immediately after the
   exact-prefix restoration and learned handoff?
2. Or does target-grid high-stage sampling introduce or amplify the local
   structural change?

The diagnostic deliberately does **not** re-enable temporal correspondence
across the exact-prefix crossing. Registered Flow guidance currently excludes
prefix-internal pairs and the exact-prefix -> first generated-suffix pair, and
00675 reported only about 0.12% valid temporal support after registration. A
future boundary-local correction must therefore be justified by measured local
evidence rather than by broadly enabling ambiguous cross-prefix transport.

The offline runtime validator treats these receipts as optional for historical
evidence, but when they are present it requires all three stages, exactly one
pre/post stage delta, the fixed diagnostic policy/geometry, finite metrics, and
zero extra model/provider/VAE/sampler/history work. It also requires
`diagnostic_only=true` and `production_gate=false`.
