# High-stage boundary ownership and audio context audit

An exact continuation prefix is caller-owned even when spatial registration is
disabled, rejected, or observation-only. In partitioned target-high sampling,
Flow guidance excludes that prefix from corrections, correction-RMS bounds, and
temporal correspondence. The prefix/suffix crossing is excluded too. Temporal
cache identity includes the protected-prefix length. Ownership is scoped to the
high-stage call and released on success or exception.

This preserves the existing generated-to-generated guidance algorithm. It does
not register or replace the low-resolution direction reference. Remaining motion
can originate in the model prediction, forecasting, guidance, or solver update;
prefix ownership alone is not a guarantee of decoded continuity.

## Video evidence

On the partitioned continuation path, `frame_gauge_residual_mode=measure` adds
`partitioned_high_boundary_prediction` receipts before and after Flow guidance.
Each receipt includes sigma, logical call index, actual/forecast classification,
and upper-45%/full-frame motion for the first four generated tokens. A disposable
five-token witness uses the authoritative last prefix token to represent the
subsequent inpaint restore. It never changes model or sampler inputs/outputs.

Evidence is limited to the first 16 high-stage prediction calls. No extra H3
evaluation, provider call, VAE decode, sampler lifetime, or history boundary is
introduced. FFT measurement and scalar synchronization have a runtime cost;
measurement-OFF runs are required for production timing claims.

Compare the existing pre-high receipt with the first `before_flow` receipt, then
compare each paired `before_flow`/`after_flow` receipt. A displacement already
present before Flow cannot be attributed solely to Flow guidance. Changes across
later prediction calls include solver evolution and subsequent model predictions;
these observations alone do not isolate a solver defect. A forecast label is
provenance, not evidence that forecasting caused an error.


## Prediction-domain exact-prefix gauge release

A hard high-stage guard over generated video tokens is not the production
boundary policy. Holding the first generated token exact can make the immediate
splice numerically exact while moving the discontinuity to the first unguarded
token. A fixed clean-reference anchor has the related problem that it can freeze
generated content to an earlier clean state while target-high refinement evolves.

The current candidate instead corrects the representation mismatch at the point
where it is produced: each target-high model prediction, before Flow guidance.
Let `P` be the model-predicted clean value of the last protected prefix token,
`E` the authoritative exact value that native inpaint semantics will restore,
and `S_i` the generated suffix prediction. Define `D = E - P`. The bridge
modifies only the first four generated suffix predictions:

`S'_i = S_i + w_i D`

with fixed temporal weights
`[1.0, 0.8535533906, 0.5, 0.1464466094]`.

Because the first weight is exactly one,

`S'_0 - E = S_0 - P`.

The exact-prefix -> first-generated transition therefore matches the model's own
native predicted transition for that model call instead of introducing a new
jump when `P` is replaced by `E`. The remaining weights release the same
per-call gauge residual smoothly toward zero rather than transferring the full
jump to the next temporal token. The residual is recomputed for every high-stage
model call; no generated clean token is pinned to a fixed pre-high reference.

The bridge is output-only. It does not expand the sampler denoise mask, alter
sampler entry state, change the caller-owned exact prefix, touch audio, modify
video outside the four-token release support, or add an H3 evaluation, provider
call, VAE decode, sampler lifetime, or history boundary. Flow guidance continues
to protect only the authoritative historical prefix.

Runtime evidence is fail-closed. Every selected high-stage model call emits
`partitioned_high_prediction_gauge_bridge`; the receipt proves the caller
prefix, audio, and suffix outside support are unchanged and checks that the
rebased first transition equals the model-native transition within numerical
tolerance. `partitioned_high_prediction_gauge_bridge_verified` requires bridge
coverage for every high-stage model call and proves that no fixed clean reference
or sampler-state mutation was used. Two
`final_post_high_prediction_gauge_release` trajectory receipts measure the
upper-45% and full-frame transition at the end of the four-token release so a
delayed discontinuity cannot be hidden by the first-transition invariant.

These structural and unit-test invariants do not establish decoded-video
quality. The candidate still requires matched hardware validation of both the
exact-prefix boundary and the release frontier before it can be treated as a
production repair.

## Audio context intervention

**MiniMax H3 Audio Boundary Audit** is an optional node after Audio VAE Decode:

1. Connect the same audio VAE and original physical audio-latent list used by the
   decoder to `audio_vae` and `audio_latents`.
2. Connect the decoder's AUDIO list to `audio` and the original Continuum assembly
   plan to `assembly_plan`.
3. Connect the audit's AUDIO output to the existing Assemble audio input. Its
   output is the original AUDIO list, unchanged. The STRING output and console
   contain the report.

For each eligible exact carried overlap, the node performs **two extra bounded
AudioVAE decodes** after sampling. Both contain identical overlap, generated
suffix, and right context. One prepends 32 actual earlier latent ticks. The
comparison measures the same 500 ms suffix in both outputs. It therefore tests
left-context dependence without changing generated content or production audio.

`same_suffix_context_comparison` reports the paired intervention.
`production_crop_comparison` separately compares the baseline bounded decode
with the ordinary full decode; truncation effects must be assessed here before
extrapolating the bounded result to production. Its gain interpretation is valid
only when `production_crop_comparison_valid=true`, which establishes that the
standard Comfy audio normalization was inactive. The paired raw decoder
comparison does not apply this normalization.

`common_decode_boundary` measures pre/post loudness within the extended decode.
If this remains discontinuous while the identical suffix is insensitive to
extended context, missing left context is not supported as the explanation.
Latent RMS and decoded waveform RMS are different quantities: neither a small
latent ratio nor an exact carried overlap proves smooth generated audio.

The audit requires native 32-channel stereo latents at 40 Hz, 32 kHz audio,
20–128 carried ticks, 32 additional earlier ticks, and 52 generated ticks. It
skips unsupported/non-exact boundaries with an explicit report and no decode.
Each eligible decode is at most 212 ticks. Latents, saved continuation state,
assembled duration, masks, and production waveforms are not modified. Synthetic
tests verify isolation and sensitivity; actual model/media validation is still
required before claiming a video or audio repair.
