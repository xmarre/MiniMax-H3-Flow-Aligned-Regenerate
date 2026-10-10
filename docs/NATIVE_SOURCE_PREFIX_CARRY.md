# Native source-prefix carry experiment

This opt-in continuation test preserves source context from the preceding chunk's
actual low/probe clean prediction, instead of reconstructing that context from
its refined target output. It tests whether changing representations between
chunks contributes to background drift and detail loss. It is an experimental
input policy, not an established quality fix.

## Patcher workflow

1. Refresh Flow PR #99 through ComfyUI Patcher and restart ComfyUI.
2. In **MiniMax H3 Partitioned Exact-Prefix Handoff**
   (`H3PartitionedExactPrefixDiagnosticHandoff`), retain
   `spatial_stage_control=progressive_uniform_source` and select
   `source_prefix_projection=native_source_carry`.
3. Execute the full sequence with this same patched MODEL, starting with its
   initial chunk. The initial chunk must execute under this mode to populate
   source context. A cached initial chunk from another mode is insufficient.
   No saved boundary bundle is a runtime input. No video VAE is required for
   this mode; an existing connection can remain.
4. Keep the established seed, checkpoints, geometry, schedule, guidance, audio
   settings, attention backends and Continuum **Auto 3** fixed. Enable
   `capture_boundary_witness=true` and save Flow metrics.
5. Save the rendered continuation and run the existing Extended Local Boundary
   Audit on its capture. Compare matching join and later frames for background
   geometry, bookshelf/picture detail and tone; inspect pose, clothing, face,
   flicker, end-of-chunk behavior and speech. A useful result needs rendered
   improvement without unacceptable regressions.

The widget default remains `latent_bicubic`. `vae_rgb_roundtrip` remains
available as a separate reconstruction experiment documented in
[PREFIX_PROJECTION_AB.md](PREFIX_PROJECTION_AB.md).

## State and numerical contract

The initial generation path records its already-computed source clean prediction
at the handoff sigma and pairs it with suffix hashes of the target video returned
by that sampling transaction. A continuation requires an exact float32 hash
match between its authoritative target prefix and the preceding returned target
suffix. Target and source grids must match the recorded grids, both clips must
use native `5k+2` temporal lengths, and the prefix must end on the same native
phase. When Continuum sequence metadata is present, session identity and the
immediately preceding chunk index must also match. Continuum's initial Flow
pass is unlabeled; the first labeled request (chunk 2) can bind that initial
pair by its exact target-suffix hash. An unlabeled continuation or skipped chunk
cannot use this exception. Later requests must advance the labeled session.

The matching tail of that source clean prediction replaces only the protected
source prefix before low sampling. Low/probe and learned transfer reuse those
bytes. The generated source suffix retains its existing initialization and
sampling path. The authoritative target prefix, protected audio, masks, sigma
schedule, three sampler lifetimes and selected backend contracts are preserved.
Generated audio can change through joint AV conditioning and needs listening.

No additional H3, provider or VAE call is introduced. This carries a **clean
prediction at a nonzero handoff sigma**, not a fully sampled low-resolution
video. It can disagree with the high-refined target's pose, texture or tone;
rendered seam quality remains an empirical question.

Missing previous context, a changed prefix, foreign/skipped sequence, changed
geometry or invalid temporal phase stops before the first low sampler. There is
no silent bicubic fallback. Latent processing between generation and continuation
that changes the carried target bytes will fail the pairing check. This includes
post-sampling refinement that changes the authoritative prefix.

## Ownership and failure behavior

A model-local owner retains one detached CPU float32 source clip and a small
mapping of target-suffix hashes. It retains no source tensor on CUDA and no
full target clip between chunks. For example, `[1,24,62,44,44]` retains
11,523,072 source bytes (about 11 MiB). Copying and hashing use temporary CPU
buffers; while a continuation runs, the prior clip and newly staged clip can
both be present. This does not measure the full process peak or model-cache
lifetime.

Nested Flow wrappers for the same guider share one transaction. Another active
guider cannot sample using that owner. A new source/target pair is published
only when the enclosing Flow sampling transaction succeeds, including its
post-sampling audio checks. A failed continuation discards the staged pair and
leaves the preceding successful pair available for a retry. Beginning a new
initial chunk clears the previous pair. Selecting another projection mode
removes this owner from the newly configured MODEL; existing cached MODEL
clones may retain their own references until ComfyUI releases them.

## Evidence receipts

`partitioned_source_prefix_projection` records policy `native_source_carry_v1`,
source/target prefix fingerprints, previous source token span, generation and
sequence pairing, and zero extra H3/VAE calls. `source_prefix_carry_prepared`
records the next candidate pair, CPU bytes and its publication condition;
"prepared" is not proof that a later enclosing check succeeded.

The saved boundary manifest records this policy and projection receipt.
Extended Local Boundary Audit verifies the saved source prefix against that
receipt and uses the actual captured source state. Its geometry fits are
stage-specific image measurements, not additive causal transforms. Compare
source, provider, pre-high, first-high and final results with their fit support,
then inspect the rendered outputs.

CPU tests cover initial capture, exact suffix pairing, native phase, sequence
lineage, bounded replacement, failure rollback and concurrent-owner rejection.
Native Core/Sol integration tests exercise the selected source bytes, unchanged
suffix initialization and protected AV, sampler masks/schedules, call budgets
and capture replay. Trained-weight CUDA quality and repeated-run residency
require the full rendered sequence; these CPU checks cannot establish them.
