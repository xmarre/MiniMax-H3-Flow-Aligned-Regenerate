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
