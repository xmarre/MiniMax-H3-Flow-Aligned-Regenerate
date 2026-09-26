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


## 00679 hard-shadow result

00679 is matched to the 00678 provider-native state. The exact-prefix hash,
rigid-v2 rejection, exact-overlap fallback, held-out calibration and selected
provider region all reproduce. The hard stabilization shadow selects exactly
one tile, `r2c0`, with the predicted residual scale
`0.6781307988`.

The measured hard-shadow result is favorable inside that selected region:

- centered low-pass boundary RMS falls from about `0.302924` to
  `0.238555`, a ratio of about **0.7875**;
- gradient RMS falls from about `0.096559` to `0.085084`, a ratio of about
  **0.8812**;
- NCC rises from about `0.93250` to `0.94877`, a gain of about
  **+0.01627**;
- the nominal dispersion-normalized predictor error is returned exactly to the
  held-out maximum while the absolute error is predicted to fall inside it.

The whole-frame response is small but also favorable: centered low-pass RMS is
about `0.9887x` provider-native, gradient RMS about `0.9929x`, and NCC
improves by about `+0.00136`.

The shadow remains non-mutating and observation-only. The correction RMS over
the full frame is about `0.02447`, correction absolute maximum about
`0.57936`, all extra-work counters remain zero, and only `r2c0` receives a
direct correction.

Several neighboring tile *measurements* move slightly even though their direct
correction is zero. That is expected because the 5x5 low-pass measurement
kernel crosses fixed-tile boundaries. It must not be misread as direct
mutation of those ineligible regions.

00679 therefore supports the residual-amplitude hypothesis, but it still does
not justify shipping the hard fixed-tile support. The current per-tile gradient
metric is computed after low-pass filtering inside each tile and does not
directly measure the raw correction jump across the selected/ineligible tile
frontier. A hard support boundary could therefore hide a spatial edge cost.

### Soft-support shadow

The next diagnostic layer is
`partitioned_provider_boundary_soft_support_shadow_v1`.

It inherits the exact same held-out eligibility and residual scale from the
hard shadow. It changes only spatial support on a discarded clone. Selected
tiles use an inside-only raised-cosine taper across each frontier with an
ineligible neighboring tile. Shared borders between selected tiles are not
tapered, and image-frame borders are not tapered.

The taper width is derived from the measurement support rather than selected
from 00679:

`feather_width = lowpass_kernel // 2 + 1`

which is 3 cells for the fixed 5x5 low-pass kernel.

The soft shadow publishes both the hard and soft direct-correction jump across
selected/ineligible tile frontiers, plus the soft candidate's:

- per-tile and global centered low-pass boundary ratios;
- per-tile and global gradient ratios;
- NCC deltas;
- recalibrated held-out-max predictor-error ratios;
- support statistics and correction magnitude.

The soft candidate remains diagnostic-only. It is constructed on a clone,
measured, and discarded. It cannot alter the provider output, exact-prefix
owner, frame-gauge transaction, exact-overlap fallback, guidance registration,
or high-stage state.

The purpose of the next matched hardware run is to determine whether the
localized 00679 benefit survives a spatially continuous support function and
whether the direct frontier jump is materially reduced. A production
stabilizer remains unimplemented until that comparison is measured.


## 00680 soft-support shadow result

00680 reproduces the same rejected rigid-v2 and exact-overlap state as the
previous matched runs. The authoritative exact-prefix hash is unchanged,
rigid-v2 again rejects on
`boundary_upper45_insufficient_improvement`, no spatial warp or registered
guidance reference is published, and the exact-overlap fallback applies from
the actual provider boundary pair.

The hard stabilization shadow also reproduces 00679: `r2c0` is the only
eligible 4x4 region and its residual scale remains
`0.6781307988`.

The new soft-support shadow provides the missing spatial-support evidence. Its
inside-only raised-cosine taper reduces the measured direct selected/ineligible
frontier jump from:

- RMS `0.1223239` to **0.0**; and
- absolute maximum `0.5793642` to **0.0**.

This is not achieved by discarding the correction. The soft candidate retains
a full-frame correction RMS of about `0.01837` and maximum magnitude about
`0.35294`. Within `r2c0`, support has mean `0.7363`, maximum `1.0`,
minimum `0.0`, and nonzero coverage of about `82.0%`.

The tapered candidate retains most of the hard-shadow local benefit:

- centered low-pass boundary RMS changes from `0.302924` provider-native to
  `0.257208`, a ratio of **0.8491x**;
- gradient RMS changes from `0.096559` to `0.087962`, a ratio of
  **0.9110x**;
- NCC changes from `0.93250` to `0.94441`, a gain of about
  **+0.01191**.

Relative to the hard-shadow improvement, that is roughly 71% of the centered
low-pass reduction, 75% of the gradient reduction, and 73% of the NCC gain,
while the direct measured frontier discontinuity is eliminated.

The taper intentionally does not force the complete tile aggregate back inside
both held-out maxima. After tapering, `r2c0` has absolute prediction error
about `1.077x` the held-out maximum and dispersion-normalized error about
`1.249x` the held-out maximum. This is expected from an inside-only support
function: edge pixels are attenuated to preserve spatial continuity. Those
post-taper ratios are therefore observations, not a second threshold that the
production candidate is allowed to defeat by increasing correction strength.

Globally the soft candidate remains conservative: centered low-pass RMS is
about `0.99244x` provider-native, gradient RMS about `0.99478x`, and NCC
improves by about `+0.00085`. The shadow remains non-mutating and adds no H3
NFE, provider call, VAE call, sampler lifetime, or history boundary.

### 00680 promotion decision (historical; superseded by 00681)

00680 established that the held-out-calibrated soft-support correction improved
the provider-native first-suffix boundary on a discarded shadow clone while
eliminating the measured hard tile frontier. That was sufficient to justify one
explicitly opt-in hardware trial, but not to establish complete-path safety.

The trial used `provider_boundary_stabilization=soft_support_v1` and the policy
`partitioned_provider_boundary_soft_support_production_v1`. It modified exactly
the first generated suffix token in provider-native clean space before the
existing exact-overlap mapping, left the authoritative prefix and later suffix
tokens unchanged, and added no model/provider/VAE/sampler/history work.

00681 subsequently disproved the promotion criterion. The current runtime
therefore does not apply this mutation; the historical details below are kept
only to document how the failed candidate was derived.

## 00680 soft-support result

00680 reproduces the same provider-native boundary state as 00679. The hard
shadow again selects only `r2c0` with residual scale `0.6781307988`.
The raised-cosine soft-support shadow therefore compares the same residual
correction under a different spatial support policy rather than selecting a new
content region or threshold.

The direct selected/ineligible correction frontier is eliminated by the soft
support in this realization:

- hard frontier correction-jump RMS: about `0.122324`;
- hard frontier correction-jump absolute maximum: about `0.579364`;
- soft frontier correction-jump RMS: `0.0`;
- soft frontier correction-jump absolute maximum: `0.0`.

This is the intended effect of the inside-only taper: the correction reaches
zero at every selected/ineligible boundary before entering the neighboring
tile. The soft correction is correspondingly smaller overall, with full-frame
RMS about `0.018365` versus `0.024470` for hard support and absolute maximum
about `0.352940` versus `0.579364`.

The selected `r2c0` region retains most of the hard-shadow benefit:

- centered low-pass RMS ratio: `0.84908` soft versus `0.78751` hard;
- gradient RMS ratio: `0.91096` soft versus `0.88116` hard;
- NCC delta: `+0.01191` soft versus `+0.01627` hard.

Relative to the hard-shadow improvement above provider-native, the soft support
retains about 71% of the centered-lowpass reduction, 75% of the gradient
reduction, and 73% of the NCC gain while eliminating the measured hard frontier
jump.

The taper does deliberately reduce the effective correction amplitude near the
frontier. Consequently the recalibrated `r2c0` prediction error remains
slightly outside the held-out maxima after the soft candidate:

- absolute prediction error / held-out maximum: about `1.0770`;
- dispersion-normalized prediction error / held-out maximum: about `1.2495`.

This is not treated as a reason to amplify the taper. The held-out envelope was
used to establish that the original boundary was anomalous and to derive the
bounded residual scale. Once spatial support is tapered, forcing every supported
cell back to the same scalar envelope would require a new amplification rule
chosen after observing 00680. That would overfit this boundary and could undo
the edge-safety evidence. The 00680 promotion decision therefore used the measured soft candidate exactly as tested.

Globally the soft candidate remains small and favorable: centered low-pass RMS
is about `0.99244x`, gradient RMS about `0.99478x`, and NCC improves by about
`+0.00085`. The candidate remains a one-token, local provider-native
correction.

## 00681 production validation failure and current fail-closed contract

00681 was the first run that actually enabled the 00680 soft-support mutation.
The pre-mutation provider state, rigid-v2 transaction, predictor calibration,
hard shadow, and soft-support shadow reproduced the 00680 evidence. The runtime
applied one correction to `r2c0` with RMS `0.0183653` and maximum absolute
magnitude `0.352940`, then continued through the existing exact-overlap and
target-high stages.

The mutation improved the selected tile before target-high:

- `r2c0` centered low-pass boundary RMS: `0.302924` -> `0.257208`;
- `r2c0` gradient RMS: `0.096559` -> `0.087962`;
- exact-restored `r2c0` NCC: `0.916876` -> `0.930873`.

That local pre-high result did **not** survive the complete continuation. From
the corrected pre-high state to the existing post-high clean state, `r2c0`
changed by:

- centered low-pass RMS ratio: `1.143122`;
- gradient RMS ratio: `1.103335`;
- NCC delta: `-0.021323`.

In the matched 00680 shadow-only run the same stage ratios were `0.973768`,
`1.006299`, and `-0.007935`, respectively. `r2c0` also moved from outside the
largest structural-amplification tiles to the second-ranked tile in 00681.
The final `r2c0` centered low-pass RMS (`0.294021`) was consequently close to
the 00680 shadow-only final value (`0.294978`): target-high largely erased the
local pre-high improvement rather than preserving it.

Decoded-media validation failed as well: the visible frame shift returned and
the background still changed at the boundary. The decoded trajectory classifier
continued to label the boundary as clean, so that classifier is not an
acceptance gate for this defect.

The violated invariant is therefore clear: **a provider-native/pre-high shadow
improvement is not evidence that the corresponding mutation is stable through
target-high denoising and decoding.** The 00680 promotion gate tested the wrong
lifetime boundary.

Current behavior is fail-closed:

- `provider_boundary_stabilization=off` remains unchanged;
- the serialized `soft_support_v1` value remains accepted for workflow
  compatibility, but it no longer mutates provider or sampler state;
- an eligible `soft_support_v1` run records
  `reason=disabled_pending_post_high_validation`;
- the existing exact-overlap path receives the original provider state;
- after target-high, the runtime re-runs the same held-out calibration and
  hard/soft-support hypotheses on the already-existing
  `post_high_internal_clean` tensor and emits
  `partitioned_provider_boundary_post_high_shadow_v1`;
- that post-high path is diagnostic-only, cannot become a production gate, and
  adds zero H3 NFE, provider calls, VAE calls, sampler lifetimes, or history
  boundaries.

No replacement production correction is promoted by this change. A later
candidate must first demonstrate, on the final high-stage state, both that the
visible defect is represented by the measured signal and that correcting the
first boundary does not merely move discontinuity into the next suffix
transition.
