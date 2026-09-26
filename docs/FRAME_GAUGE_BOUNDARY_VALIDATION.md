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


## 00676 fallback validation and provider-native localization

00676 reproduces the original 00674 exact-prefix realization byte-for-byte at
the authoritative prefix hash and therefore exercises the rejected rigid-v2 arm
that selected the exact-overlap fallback.

The rigid transaction rejects for the same
`boundary_upper45_insufficient_improvement` reason. The exact-overlap fallback
then applies from the actual provider boundary pair, modifies exactly one suffix
token, leaves the authoritative prefix and later suffix untouched, and adds no
model/provider/VAE/sampler/history work. On the clean transfer receipt the
corrected boundary returns to the provider-native seam exactly for raw RMS and
low-pass RMS, with the spatial-mean ratio equal to numerical precision. This is
the expected algebraic result of preserving `S - P` after replacing `P` with
`E`.

The content-continuity diagnostic localizes the remaining problem one layer
earlier than the exact-prefix replacement. Globally, the provider-native
boundary is not unusually large relative to the previous three provider-prefix
transitions, but the fixed regional grid exposes strong local outliers. In
00676, tile `r2c0` has a centered low-pass boundary residual about 2.16 times
its recent local prefix median; `r1c1`, `r2c2`, `r2c1`, and `r3c1` are
also elevated.

Crucially, the pre-high exact-restored boundary has the same raw, low-pass and
centered-low-pass boundary magnitudes as the provider-native transition to
floating-point precision, and it preserves the same leading anomalous regions.
The exact-overlap bridge is therefore faithfully carrying the learned
provider's immediate transition rather than creating the localized structural
change.

The high stage does not explain the dominant anomaly either. The global
centered-low-pass boundary residual decreases slightly from pre-high to
post-high, while the dominant `r2c0` region also decreases modestly. The high
stage amplifies several other regions by smaller amounts, so it can modulate the
boundary, but 00676 does not support treating it as the origin of the principal
localized discontinuity.

The local correspondence branch is not suitable as a production transport
signal in this evidence. More than 98% of candidate positions are ambiguous in
the boundary-content receipts and cycle-supported unique correspondence is
near zero. These fields remain useful as negative evidence against broad
cross-prefix transport; they are not used to move content.

### Provider-native temporal predictor

The next diagnostic layer is
`partitioned_provider_boundary_temporal_predictor_v1`. It remains
observation-only and runs on the learned provider's native clean trajectory
before exact-prefix replacement.

For each global region and fixed 4x4 tile, it forms the previous three
low-pass provider deltas, removes per-channel spatial DC inside that region,
and takes their elementwise median as a conservative recent-motion predictor.
The actual first-suffix delta is then compared with that predictor. Receipts
report:

- actual and predicted centered low-pass delta RMS;
- prediction-error RMS;
- prediction error relative to recent prefix-delta dispersion and to the actual
  boundary delta;
- cosine similarity between the actual and predicted delta directions;
- projection gain of the actual delta on the predicted direction;
- tile rankings by absolute prediction error and by error relative to recent
  temporal dispersion.

This predictor does not create a correction or candidate output. Its purpose is
to distinguish a legitimate continuation of recent provider dynamics from a
first-suffix state change that is both locally large and inconsistent in
direction with the immediately preceding provider trajectory. Only after that
evidence exists should a bounded provider-native stabilization be designed.

The runtime gate treats predictor receipts as optional for historical evidence.
When present, it requires one receipt, the fixed predictor policy and geometry,
finite global/tile metrics, complete 4x4 rankings, diagnostic-only ownership,
and zero extra H3 NFE/provider/VAE/sampler/history work.


## 00677 provider predictor result and held-out calibration

00677 reproduces the 00676 provider-native handoff state. The authoritative
exact-prefix SHA-256 remains
`36ea0b7b0578642f895a7a2e2fdfab16f23755e509d8f2e84d248d585c425b7a`;
rigid-v2 again rejects on
`boundary_upper45_insufficient_improvement`; the exact-overlap fallback
applies from the actual provider boundary pair; and the provider-native,
pre-high exact-restored, frame-gauge, exact-overlap, and clean seam values are
identical to 00676 apart from elapsed-time fields. The high-stage output remains
allowed to differ numerically because those downstream GPU operations are not
used as an identity requirement for the provider-native diagnostic.

The first provider temporal predictor receipt confirms that the dominant local
boundary regions are not merely large continuations of recent motion. Globally,
the first-suffix structural delta is poorly aligned with the median of the
previous three provider deltas: cosine is about 0.033 and prediction error is
about 1.16 times recent prefix-delta dispersion.

The regional evidence is stronger. `r2c0`, which was already the largest
content-continuity outlier, is also the largest dispersion-normalized predictor
outlier: prediction error is about 2.64 times recent local dispersion while the
actual/predicted cosine is only about 0.13. `r2c2`, `r2c1`, `r1c1`, and
`r3c1` follow, with error/dispersion ratios about 1.74, 1.61, 1.55, and 1.53.
Their cosines are approximately 0.008, -0.212, 0.032, and -0.396 respectively.
The same five regions are the leading provider-native centered-structural
outliers. This is consistent with a first-suffix provider-state discontinuity,
not a rigid gauge failure or an exact-prefix representation seam.

Those ratios are not yet production thresholds. The denominator in the first
predictor receipt measures dispersion of the same three history deltas used to
form the predictor; it does not establish how surprising that prediction error
is relative to ordinary held-out provider transitions. Using a threshold
selected from 00677 would therefore overfit this single boundary.

The next diagnostic layer is
`partitioned_provider_boundary_temporal_calibration_v1`. It keeps the same
three-transition median predictor but evaluates it on up to the previous five
held-out prefix transitions. For each historical target, only earlier provider
deltas are used to predict it. The actual first-suffix prediction error is then
reported relative to the median and maximum held-out historical error, both in
absolute RMS and after normalization by each target's preceding-delta
dispersion. Cosine and projection-gain baselines are reported as well.

This calibration remains observation-only. It does not select tiles for
correction, alter the provider output, modify exact-prefix ownership, change
frame-gauge decisions, or add model/provider/VAE/sampler/history work. Its
purpose is to establish whether the 00677 first-suffix outliers exceed the
provider predictor's own recent held-out error envelope before any stabilization
policy is designed.

The offline runtime gate treats this new receipt as optional for historical
evidence. When present, it requires one calibration receipt, the fixed
three-transition predictor, five requested held-out targets, fixed 4x4
geometry and low-pass kernel, complete finite global/tile metrics, complete
rankings, diagnostic-only ownership, and zero extra work.


## 00678 held-out calibration result

00678 reproduces the same rejected rigid-v2 / exact-overlap state as 00676 and
00677. The exact-prefix hash remains
`36ea0b7b0578642f895a7a2e2fdfab16f23755e509d8f2e84d248d585c425b7a`,
rigid-v2 again rejects on
`boundary_upper45_insufficient_improvement`, and the exact-overlap bridge
again reconstructs the provider-native immediate clean-domain boundary.

The held-out calibration materially narrows the provider-native hypothesis.
Five prefix-internal targets (transition indices 6 through 10) provide the
recent out-of-sample predictor envelope. Globally the first-suffix boundary is
inside that envelope: its absolute predictor error is about 0.678 times the
largest held-out error, while its dispersion-normalized error is about 0.718
times the held-out maximum.

Only one fixed 4x4 region exceeds **both** held-out maxima: `r2c0`.

For `r2c0`:

- boundary prediction-error RMS / held-out maximum is about **1.271**;
- dispersion-normalized boundary error / held-out maximum is about **1.475**;
- the corresponding ratios to the held-out medians are about **1.538** and
  **1.806**;
- actual-vs-predictor cosine is about **+0.133**, versus a historical median of
  about **+0.035**;
- projection gain is about **+0.455**, versus a historical median of about
  **+0.056**.

This changes the interpretation of 00677. The other prominent
provider-content regions are unusual relative to recent medians, but they do
not exceed the recent held-out maxima on both error measures. In particular,
`r2c1`, `r2c2`, `r1c1`, and `r3c1` remain inside at least one held-out
maximum. They are therefore not justified targets for a production clamp from
this evidence.

`r2c0` is different. Its first-suffix prediction residual exceeds both
independent recent envelopes, but its motion direction is not reversed relative
to historical behavior. The evidence is therefore more consistent with a
**localized residual-amplitude overshoot** than with a wrong-direction
transition. A directional replacement or broad temporal smoothing would be
poorly matched to the measured failure.

### Provider-boundary stabilization shadow

Before mutating production state, the next layer is
`partitioned_provider_boundary_stabilization_shadow_v1`.

The shadow candidate is deliberately observation-only. A tile is eligible only
when both measured error signals exceed a numerical floor of `1e-6` and both:

1. `boundary_error_over_historical_max > 1`; and
2. `boundary_dispersion_ratio_over_historical_max > 1`.

The `1e-6` floor is only a fail-closed floating-point stability guard. It
prevents ratios of near-zero predictor residuals from selecting tiles; it is
not a content-quality threshold.

For an eligible tile the shadow policy leaves the recent median predictor
unchanged and scales only the first-suffix prediction residual by the tighter
of the two factors required to return both ratios to the held-out envelope:

`scale = min(1, 1 / absolute_ratio, 1 / dispersion_ratio)`.

For the measured 00678 `r2c0` values this predicts a residual scale of about
`0.678`, i.e. a bounded reduction of roughly 32% in the excess prediction
residual. It does not reverse the residual or replace it with the predictor.

The implementation builds this candidate on a cloned provider-clean tensor and
measures its resulting boundary-content metrics. It never returns the clone to
the sampler. The production provider output, exact-prefix owner, frame-gauge
transaction, exact-overlap bridge, guidance reference, and high-stage state all
remain unchanged. The first implementation intentionally uses hard fixed-tile
support **only in the shadow clone** so that no unmeasured spatial blending
policy is silently introduced. Its edge behavior and local structural effect
must be measured before any production spatial support is designed.

The runtime validator requires the shadow receipt to replay tile eligibility and
the residual scale from the held-out ratios, requires ineligible tiles to have
zero correction, requires both predicted post-clamp ratios to be no greater
than the historical maxima for eligible tiles, and requires
`production_applied=false`, `output_mutated=false`, and zero extra
model/provider/VAE/sampler/history work.
