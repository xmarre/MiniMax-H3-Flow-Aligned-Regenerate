# Experiment R Job-2 queued-graph contract

This document is the operational contract for the **fresh-process high-only Job 2** in experiment R. It exists to prevent progressive-only diagnostic extractors from being left in the queued graph after the replay path has replaced the normal low → probe → learned-upscale → high invocation with the captured high suffix.

The PR #35 and PR #36 overlay code must remain installed because strict provenance validates that stack. **Installed is not the same as executed.** Their progressive-only extraction nodes are not valid Job-2 outputs.

## Required Job-2 path

After a complete ComfyUI restart, rebuild the same production model/companion stack and original inputs, with Untwist absent. Then queue this path:

1. `MiniMax H3 Execution Contract Diagnostics`
   - `strict_provenance=true`
   - `capture_mib=512`
2. `MiniMax H3 Same-State Cold High Replay`
   - `bundle=[latest]` or the explicit Job-1 manifest
3. the unchanged normal sampler, using the original full schedule and seed
4. `MiniMax H3 Same-State Replay Report`
   - `replay` must come from the replay node's replay handle
   - `trigger` must come from the sampled replay LATENT
5. save the Same-State Replay Report JSON and the matching decoded replay media.

`MiniMax H3 Execution Contract Report` is optional additional evidence. It is **not** a substitute for `MiniMax H3 Same-State Replay Report`, because only the latter emits experiment-R attribution validity.

## Nodes that must not be evaluated in Job 2

The following nodes may exist on the canvas, and the overlay source that defines them may remain installed, but they must be bypassed/disconnected so they are absent from the queued execution path:

- `H3HandoffCheckpointDiagnostic` / **MiniMax H3 Progressive Handoff Checkpoint Diagnostic** (PR #35)
- `H3GuidanceAnchorTransportValidation` / **MiniMax H3 Learned-Anchor Guidance Validation** (PR #36)
- `H3SameStateReplayCapture` / **MiniMax H3 Same-State Replay Capture**
- `H3SameStateReplayBundleSave` / **Save MiniMax H3 Same-State Replay Bundle**

The first two require a complete normal progressive invocation. Job 2 intentionally does not run the low stage, exact handoff probe, or learned upscaler, so those extractors cannot produce a valid record. If either executes after a successful high-only sampler run, its `no complete ... capture` exception is an unrelated downstream workflow error, not a replay failure.

Job-1 capture/save nodes are likewise not part of the cold replay process. Reusing them in Job 2 would mix the two experiment roles.

## Queue-level acceptance check before spending CUDA time

Before pressing Queue, verify all of the following on the actual queued graph, not merely by looking at installed custom nodes:

- the PR #35/#36 overlays are installed;
- neither PR #35 nor PR #36 progressive extraction node has an active downstream output path;
- no Job-1 capture/save node is active;
- `MiniMax H3 Same-State Replay Report` is present and connected to both the replay handle and the replayed sampler LATENT;
- the generic Execution Contract Report, if retained, is treated only as additional evidence;
- Untwist is absent;
- the original seed, schedule, model/checkpoint, references, prompt, geometry, Flow/Spectrum/Sol/VDN/DiffAid configuration are unchanged.

If this graph contract is wrong, correct the workflow and **restart ComfyUI again before the next attempt**. A requeue in a process that already executed the high-only sampler is not the fresh-process R condition.

## Required result

A causally interpretable Job 2 must reach `MiniMax H3 Same-State Replay Report` and report all attribution gates, including:

- exactly `3L / 2A / 1F`;
- zero learned-upscaler calls;
- `attribution_valid=true`;
- matching checkpoint/provenance/conditioning/guidance/Spectrum/runtime/companion policy;
- Untwist absent;
- matching first-high sampler video/audio hashes;
- matching first-high H3 video/audio hashes.

A sampler run that completes `3L / 2A / 1F` but is followed only by a generic Execution Contract Report, or is then aborted by a progressive-only PR #35/#36 extractor, is **not yet an experiment-R result**. Preserve the log as structural evidence, fix the queued graph, restart, and rerun Job 2 only. Do not rerun Job 1 solely for this workflow correction.
