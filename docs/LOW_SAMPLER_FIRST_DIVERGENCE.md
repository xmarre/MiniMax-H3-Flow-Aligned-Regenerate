# Frozen low/probe replay first-divergence trace

The opt-in `low_sampler_aba` diagnostic records production arm A, a protected
source-prefix intervention B, and replay A′. The original A clean state remains
the learned-transfer/high handoff. An identical first **video** prediction is
insufficient to establish replay: audio can differ at that same call and feed
back into later joint AV predictions.

The `partitioned_frozen_low_source_aba` metrics event and exported manifest
metadata contain `first_divergence_trace` (`h3_low_first_divergence_v1`). It
records full tensor SHA256, shape/dtype and, for matched shadow operands,
float32 elementwise RMS and maximum error. Bitwise identity is reported
separately from the existing `1e-5` numerical tolerance. Nonfinite differences,
changed call topology and missing records cannot qualify reproduction.

## Observed boundaries

- Live `x` in the exact selected sampler function's Python frame, before Core
  inpainting. The callable and sampling algorithm are not replaced.
- Flow predictor input after Core's sampler mask reconstruction, and timestep.
- Native diffusion wrapper input video/audio, native timestep, returned
  video/audio velocities and full conditioning/mask/payload hashes within a
  64 MiB hashing budget. Opaque objects and budget omissions are explicit.
- Full packed AV actual or forecast predictor output, distinguished by Flow's
  existing active/completed Spectrum provenance and solver phase.
- Native callback latent and denoised state, **before** the solver update.
- Low outer return, raw-state reconstruction, probe noise and full clean
  probe output in model-internal coordinates.

The native diffusion observer follows existing wrappers. Spectrum forecasts
can bypass that observer; their returned prediction is still recorded at the
logical predictor boundary. Multiple native calls have distinct ordinals.
Alignment requires matching stage, evaluation, native-call order, timestep,
actual/forecast identity, phase and outer-step index. A changed topology stops
alignment instead of shifting subsequent calls to fit.

For zero-churn Euler with matching call order, the next live `x` is the actual
preceding updated state. Other solvers can include intermediate updates or
stochastic operations; `solver_interval_before_mask` identifies that interval
without claiming a generic after-update hook. Core's final pre-inverse-scaling
solver return is unobserved: `raw_reconstruction` is derived from the actual
outer output using Flow's existing conversion. A first difference there
localizes the terminal conversion interval, not a proven solver defect.

`first_observed_divergence` identifies the earliest matched differing boundary.
`alignment_failure` identifies the earliest topology mismatch. Full matching
record exactness and numerical reproduction are separate fields. These fields
never establish physical model/cache isolation or rendered acceptance.

## Ownership and storage

Only the explicit A/B/A′ mode installs the temporary observer. Contexts and
original wrapper dictionaries restore on exceptions; the enclosing source-carry
transaction still owns publication. Baseline operands reside in temporary CPU
disk files. B/A′ comparisons read those files, and only the first differing A′
tensor pair is retained for export. No additional CUDA trajectory is retained.

Bounds are 2 GiB baseline tensor payload, 512 tensor records per arm, 128 MiB
per operand and 128 retained omission descriptions. File headers and one
additional A′ mismatch operand add bounded overhead. Large contiguous Spectrum
history tensors are hashed completely through 8 MiB CPU buffers. Other cache
and conditioning digests use a 64 MiB per-digest budget. Limits mark observation
incomplete and prevent a reproduction pass; they do not silently sample bytes.

The manifest can include `first_divergence_A` and
`first_divergence_A_replay`. They are raw operands at the saved boundary,
possibly packed AV or native velocities; they are **not** VAE-decodable clean
video by default. Their scalar trace row gives the domain, stage and hashes.
A topology or conditioning mismatch can precede the first tensor mismatch.
Interpret both records together.

Spectrum run/step/forecaster lifetime IDs, history generation and visible
history/evidence/controller digests are telemetry. VDN pool identity/cache
occupancy, Flow provider identity and Sol request/kernel identity are recorded
when their modules and owners are present. VDN cache contents, model-owned
runtime buffers and attention-kernel internals remain unobserved. Lifetime IDs
are not numerical equality keys. Hashing can synchronize GPU work and adds
host-transfer/disk overhead; observer overhead is not a production speed result.
Loaded Core/Spectrum/VDN/Sol source paths and SHA256 identify observed source
files without asserting they are the user's complete installed repository.

## Offline control

An explicitly unqualified three-decode Frozen Low A/B Audit also reports
`replay_A` static-background tracks and
`temporal_increment_delta_A_to_A_replay`, using the existing A′ decode. It adds
zero H3 or VAE calls. Absolute same-frame gradients and each arm's own temporal
sharpness retention remain distinct. Independently selected feature landmarks
cannot establish a paired physical camera-scale improvement.

## Review and GPU validation

Review observer ordering, restoration, bounds and exact-head Python/native
contract tests before a GPU replay. Keep selected Sol/Spectrum/VDN controls,
model/LoRA/reference inputs, masks, sigmas and seed fixed. Tracing adds zero
sampling lifetimes beyond the existing two shadow low/probe pairs and zero
high/VAE calls. Its measured host-transfer/storage overhead is not yet known.
Return the metrics, complete ComfyUI log and exported frozen-low manifest with
its first-divergence operands. A′ divergence is a diagnostic result requiring
source-backed analysis; it is not authorization to change brightness, geometry,
prefix policy, sampler mathematics or attention-provider selection.
