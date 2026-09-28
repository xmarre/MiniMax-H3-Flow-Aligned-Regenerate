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

On the main/main partitioned audio diagnostic path, Flow retains one bounded
in-memory caller-domain audio witness from the completed low/probe stage. The
witness is kept on CPU and excluded from metrics JSON. Connecting the Flow
node's `H3_FLOW_METRICS` output to the audit's optional `metrics` input lets
the audit consume that witness. It is matched to each physical boundary by
physical order, audio geometry, exact-prefix width, and a SHA-256 digest of the
final authoritative carried prefix. A stale or mismatched witness is rejected.

For a matched witness, the audit performs one additional bounded AudioVAE decode.
It uses the same 32 actual earlier latent ticks as the final extended-context
decode, keeps the **final authoritative prefix** unchanged, and replaces only
the first 52 generated audio ticks with the low/probe stage's generated suffix.
`low_probe_counterfactual` then reports:

- `low_probe_common_decode_boundary`: the 500 ms pre/post boundary ratio if
  low/probe generated audio had been retained;
- `final_vs_low_probe_suffix`: the decoded change introduced between low/probe
  and the final generated suffix.

This isolates stage ownership without changing sampling. If the low/probe
counterfactual is continuous while the ordinary common decode is discontinuous,
the discontinuity was introduced after low/probe. If both are discontinuous, the
defect was already present by low/probe and a high-stage-only clamp is not
causally justified. The receipt is evidence only; it does not make either
interpretation automatically.

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
Each eligible decode is at most 212 ticks. Without a metrics witness it performs
two extra VAE calls per eligible boundary; with a matched low/probe witness it
performs three. Capturing the witness also introduces a small diagnostic tensor
copy, so runs using this path are not production timing controls. Latents, saved
continuation state, assembled duration, masks, and production waveforms are not
modified. Synthetic tests verify isolation and sensitivity; actual model/media
validation is still required before claiming a video or audio repair.
