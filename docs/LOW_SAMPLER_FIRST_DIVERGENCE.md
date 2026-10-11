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
Every logical prediction input requires one returned output with actual/forecast,
phase and outer-step identity. This includes low/probe calls with configured
Flow guidance and no active guidance run. An interrupted or missing output
makes coverage incomplete, even when all remaining tensor rows match.

## Observed boundaries

- Live `x` in the exact selected sampler function's Python frame, before Core
  inpainting. The callable and sampling algorithm are not replaced.
- Flow predictor input after Core's sampler mask reconstruction, and timestep.
- Native diffusion wrapper input video/audio, native timestep, returned
  video/audio velocities and conditioning/mask/payload hashes within a 64 MiB
  hashing budget. Positional transformer options are included. Source-backed
  data owners expose prefix/noise, geometry, position policy and domain data;
  metrics, executable wrappers, provider caches and run IDs remain separately
  labeled backend state. Unknown option values stay mandatory data. Opaque
  mandatory objects, depth/node/string limits or byte omissions disqualify
  complete replay. Matching omission descriptors never establish equality.
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
additional A′ mismatch operand add bounded overhead. Hash transfers are at most
8 MiB. Non-contiguous tensors are sliced in logical row order before copying
to CPU and reshaping. Copy implementations can allocate scratch for a bounded
slice; the observer never reshapes an entire non-contiguous device tensor.

Spectrum history/evidence/controller inspection shares a 16 MiB receipt limit
and a 256 MiB invocation limit, reserved equally for A/B/A′. History lists are
limited to 128 entries. Budgets exhausted by A cannot consume the shadow arms'
reserved observation. Complete tensor hashes are produced within these bounds;
larger operands are explicitly unobserved, never sampled. Backend receipts
report hashed bytes, remaining budgets and `state_digest_complete`.

Mandatory conditioning digests use a 64 MiB budget, at most 2,048 nodes, depth
12 and 4,096 characters per string. Tensor/mandatory semantic omissions prevent
a reproduction pass. Partial backend cache inspection remains caveated and does
not independently prevent operational tensor matching. Neither result proves
physical model/cache isolation.

The audio model-timestep context fingerprints its mask, overlap ticks, prefix
ticks and mask kind. Its metrics owner and accumulated call/verification counters
are telemetry, not model inputs. Core's executable callback registry is listed
as unobserved backend state alongside wrapper registrations. Excluding these
owners does not establish callback or backend-state equivalence; unknown
conditioning objects still invalidate mandatory coverage.

## Native execution observation

`low_native_trace_step=-1` disables additional native observation. With frozen
replay enabled, a nonnegative value selects one zero-based **logical low
prediction**, including forecasts in its numbering. Only its first actual native
call is observed. A forecast, missing prediction or unsupported block path has
no selected native witness and is explicitly incomplete.

The observer fingerprints every byte of the packed hidden input to block zero
and outputs of blocks zero, `block_count // 2`, and `block_count - 1`, before
Flow maps the final hidden state back to native carrier rows. For outputs of
blocks strictly between the middle and final block, it additionally hashes all
features in every 32nd token row. Those probes are labeled
`coverage=token_rows_stride_32` and are **not** complete block hashes.
For a 50-block model these are blocks 26–48; four full witnesses plus 23 sampled
witnesses consume approximately 1.72 GiB per A/B/A′ arm at 36,445 × 5,376
bf16 hidden shape, under the existing 2 GiB limit.

For the confirmed 01867 sampled discrepancy first observed at block 45,
`low_native_trace_pair_block=45` selects a separate **full-byte** pair mode.
It keeps full packed-hidden fingerprints of input to block 0 and the outputs of
block 25, block 44, block 45, and block 49; **no intermediate sampled rows** are
recorded in pair mode. The pair must be after `block_count // 2 + 1` and before
the final block. At the 01867 geometry, five 36,445 × 5,376 bf16 fingerprints
cost approximately 1.82 GiB per arm, below the existing 2 GiB inspection budget.
The setting is optional, defaults to `-1` (preserving the original sampled mode),
and requires `low_sampler_aba=true` and `low_native_trace_step>=0`.
The selected pair is validated against the actual block count before hashing;
invalid pair identities and missing/full-hash budget failures never pass
observation completeness. If A/A′ full outputs after block 44 are identical
but full outputs after block 45 differ, the mismatch is created within the
block-45 execution interval. Otherwise the prior sampled equality missed some
earlier differing tokens and source-level attribution must remain open.

Hidden states include conditioning and both target AV streams; they are not clean
video latents. Full selected hash identity establishes only equality at those
boundaries. A differing strided probe identifies the earliest *observed sampled*
difference, not proof that the preceding unobserved token rows or earlier blocks
are identical. Follow up on a suspect block with a full-byte observation before
attributing divergence to that specific block, attention primitive, or weight.

`native_execution_trace` in the first-divergence report contains the observations,
topology checks, source receipts, inspection budgets and omissions. Full hidden
tensors are never retained on GPU or exported. Hashing streams at most 8 MiB of
logical tensor bytes per host slice, charges at most 2 GiB per arm (6 GiB across
the three arms), and rejects individual observed tensors larger than 512 MiB. Budget/layout/read
failures, missing boundaries or different topology prevent selected-boundary
equality. Existing native AV divergence operands remain a separate export.

The 01867 manifest `e2c38cb34a789dd66190e1da6aa0b2301020dd15323c7a3b3ca7db38c409bf60`
and all nine raw tensor files were independently integrity-checked. A/A′ full
block-0-input and outputs 0/25 matched, sampled outputs 26–44 matched, and
sampled output 45 was the first observed mismatch. The A/A′ native velocity
mismatch was entirely in the generated 50 temporal suffix tokens; the protected
12 matched bitwise. This is **not** proof that the complete output of block 44
matched or that block 45 caused the first difference. The full-byte pair mode
exists to resolve that precise ambiguity without an attention-provider change.

Block receipts read VDN's active lease, bounded cache keys/storage metadata,
prefetch generation/target, CUDA stream, active Sol request, callable sources and
hook inventory. They never read scratch contents, consume a prefetch result,
query an event, acquire a buffer lease or change dispatch. Absence of an active
lease/request is an observation of scope, not evidence that a backend is disabled.
Source files are hashed at observation time; this does not attest loaded weights
or every compiled primitive. Cache identity/occupancy and hook counts do not prove
state equivalence. Host copies introduce synchronization and can perturb timing;
a failure that disappears under observation remains inconclusive.

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

Only production A invokes the user low callback; B/A′ invoke the trace callback.
`callback_provenance` reports that asymmetry and accessible effects: sampler
tensor hashes and known RNG states before/after A's preview/user callback.
Observed mutation or incomplete tensor inspection prevents qualification.
The callback still runs exactly once and its effects are preserved; the observer
does not rewind them or invoke it in shadows. Hidden callback/model effects are
unobserved even when these receipts match. Full sampler-side execution-history
equivalence is never asserted from tensor or RNG receipts alone.

## Offline control

An explicitly unqualified three-decode Frozen Low A/B Audit also reports
`replay_A` static-background tracks and
`temporal_increment_delta_A_to_A_replay`, using the existing A′ decode. It adds
zero H3 or VAE calls. Absolute same-frame gradients and each arm's own temporal
sharpness retention remain distinct. Independently selected feature landmarks
cannot establish a paired physical camera-scale improvement.

## Review and GPU validation

After updating the PR overlay through ComfyUI Patcher, fully restart ComfyUI.
Custom-node loading prints `[H3 Flow] diagnostic_sources=` followed by the
`h3_flow_low_trace_sources_v1` receipt. Compare its package path and SHA256 for
`low_sampler_trace.py`, `runtime.py`, `partitioned_scheduler.py`,
`low_sampler_ab_audit.py`, `native_execution_trace.py`,
`partitioned_transformer.py`, and `partitioned_node.py` with the intended revision. Missing or differing
sources require resolving the installation before spending GPU time. This
receipt hashes source files at startup; it does not verify weights, other
repositories or physical backend state. Keep the startup log with the run.

Review observer ordering, restoration, bounds and exact-head Python/native
contract tests before a GPU replay. Keep selected Sol/Spectrum/VDN controls,
model/LoRA/reference inputs, masks, sigmas and seed fixed. Tracing adds zero
sampling lifetimes beyond the existing two shadow low/probe pairs and zero
high/VAE calls. Its measured host-transfer/storage overhead is not yet known.
Return the metrics, complete ComfyUI log and exported frozen-low manifest with
its first-divergence operands. A′ divergence is a diagnostic result requiring
source-backed analysis; it is not authorization to change brightness, geometry,
prefix policy, sampler mathematics or attention-provider selection.
