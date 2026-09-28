# High-stage boundary ownership and audio context audit

An exact continuation prefix is caller-owned even when spatial registration is
disabled, rejected, or observation-only. In partitioned target-high sampling,
Flow guidance excludes that prefix from corrections, correction-RMS bounds, and
temporal correspondence. The prefix/suffix crossing is excluded too. Temporal
cache identity includes the protected-prefix length. Ownership is scoped to the
high-stage call and released on success or exception.

This preserves the existing generated-to-generated guidance algorithm. It does
not register or replace the low-resolution direction reference.

When frame-gauge repair is enabled on the exact target-partitioned path, the
selected pre-high production clean trajectory also owns a bounded generated
boundary reference. Video holds the first generated token exactly and blends
the next three toward that pre-high trajectory with weights 1, 0.75, 0.5 and
0.25. Selection follows the production pre-high trajectory after registration,
fallback and DC handling; it is not conditional on one particular frame-gauge
transaction outcome.

Audio uses a separate clean-domain reference. The low/probe continuation is
translated onto the caller-owned exact-prefix gauge using the difference
between the authoritative and low/probe final protected audio tick. This keeps
the low/probe first generated edge unchanged while making it relative to the
actual prefix that will survive final exact restoration. Target-high model
predictions hold that aligned trajectory exactly across the 20-tick (500 ms)
seam window and the following 32 latent ticks required as future decoder
context. Ownership then tapers to zero over another 32 ticks instead of
creating a second hard boundary when the reference support ends. Final exact-prefix canonicalization preserves the sampled first-edge relation by
distributing its restore delta across the same 52-tick
seam-plus-decoder-context interval. After the high-stage solver returns, the
scheduler projects the remaining bounded solver-integration residual back onto
the same clean reference: video uses the same 1/0.75/0.5/0.25 four-token
weights; audio holds the full 52-tick seam-plus-decoder-context interval exactly
and uses the existing 32-tick tapered release. The caller-owned prefix and every
suffix value outside those supports remain byte-for-byte unchanged. The 32-tick
future-context bound is tied to the pinned MiniMax-H3 BigVGAN decoder architecture
and is source-contract checked. No additional H3 evaluation, sampler lifetime,
provider call, or VAE decode is introduced.

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
