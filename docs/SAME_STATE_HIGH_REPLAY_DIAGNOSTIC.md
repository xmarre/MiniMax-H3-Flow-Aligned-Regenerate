# Same-state cold high replay diagnostic

This diagnostic implements experiment R from `FIRST_HIGH_STAGE_ARTIFACT_ARCHITECTURE.md`. It is not a production sampling mode and does not authorize a production fix.

## Purpose

R distinguishes a high-stage result caused by the explicit high-entry contract from one caused by retained low/probe/reload lifecycle state. The capture job records the exact high-entry state and the declarative information required to reconstruct the same high suffix. A second ComfyUI process rebuilds the normal installed model stack and runs only that high suffix.

The replay must not execute the low stage, exact probe, learned latent upscaler, stochastic-state transport, or weighted Mixed-Grid research. The expected replay accounting is exactly `3 logical / 2 actual / 1 forecast`, with zero learned-upscaler calls.

The artifact has also been reproduced with Untwist absent from the workflow. Untwist is therefore not required for this failure mode. Use the already-failing no-Untwist configuration for both R jobs so Untwist is outside the attribution surface. Do not install the rejected full-trajectory Untwist clock trial.

## Preconditions

Use the same checkpoint, model loader and model precision, adapters, prompt and references, seed, original full sigma schedule, source/target geometry, masks, Flow configuration, Spectrum configuration, Sol/VDN/DiffAid stack, decoder settings and output ordering in both jobs. The execution-contract diagnostic must keep strict installed-runtime provenance enabled.

R requires a resolvable checkpoint file. Each process computes the checkpoint's full SHA-256 during diagnostic-node preflight and compares capture with replay before sampling. A cheap path/size/mtime guard immediately before sampling rejects ordinary file changes that occur after preflight without hashing a multi-gigabyte checkpoint again on the hot path.

The controlled progressive capture must reproduce the expected topology before a bundle is accepted:

- low: `5L / 4A / 1F`
- exact probe: `1L / 1A / 0F`
- high: `3L / 2A / 1F`
- total: `9L / 7A / 2F`
- learned upscaler: exactly one call

R is attribution-invalid if those values differ.

## Job 1: capture

Start a clean ComfyUI process on the R implementation branch. Use the normal failing progressive Target Input workflow with `learned_3d` and `exact_prefix_mode=fallback`. Keep Untwist absent.

Apply the diagnostic model nodes in this order:

1. normal production model/companion patches;
2. `MiniMax H3 Execution Contract Diagnostics` with `strict_provenance=true` and `capture_mib=512`;
3. `MiniMax H3 Same-State Replay Capture`;
4. the normal sampler path.

Connect the sampled LATENT to `Save MiniMax H3 Same-State Replay Bundle` as its trigger. The save node writes two files under `output/h3_flow_replay/`:

- a JSON declarative manifest;
- a `.pt` payload containing tensors only.

The bundle records the exact high-stage input state, original target latent, high mask when present, high suffix, seed, conditioning identity, Flow guidance configuration, Spectrum configuration and the committed low/probe guidance trajectory. It also records exact checkpoint identity, installed-runtime provenance, first-high execution policy, first-high Sol/VDN/Spectrum companion policy and the corresponding runtime observation evidence.

`first_high_sampler_input_*` specifically means the raw high-entry sampler state **before** `KSamplerX0Inpaint` may rewrite masked model input. `first_high_model_input_*` is the separate downstream post-inpaint boundary. The bundle also stores the original, hash-validated high-child `OUTER_SAMPLE` noise argument that the progressive Job-1 invocation actually supplied.

Replay does **not** attempt to recover that noise argument with an algebraic inverse. The forward initialization is finite-precision arithmetic and is not generally bit-reversible. Job 2 therefore reuses the captured high-child noise argument directly. Exact high-entry state equivalence is then checked at the real installed-runtime `SAMPLER_SAMPLE` boundary, after Comfy has performed its own load/cast/`process_latent_in` path and immediately before the underlying sampler may execute H3. Video and audio must match Job 1 in shape, dtype, original device, stride and SHA-256. A mismatch aborts before any high H3 evaluation.

The raw VDN observation additionally records bounded retained cache-key topology and the one-block async prefetch lifecycle (generation, target, pending/completed status). These fields are evidence, not equality policy: retained counts, cache contents/keys and prefetch state are intentionally allowed to differ between progressive capture and the cold process. VDN configuration, layout, wrapper identity and whether retention is enabled remain attribution gates.

Do not reuse a bundle if the capture run did not visibly reproduce the artifact.

## Cross-process source provenance

The full Job-1 provenance manifest remains hash-validated and auditable. Cross-process attribution must not, however, equate `sys.modules` population with production source identity: a warm progressive Job 1 can have many more lazily imported Sol/VDN/Spectrum/etc. modules than a fresh high-only Job 2 before its first H3 call.

For companion repositories, Job 1's recorded source list is therefore treated as a **source-byte inventory**, not an import-count contract:

- every capture-listed companion source file must still exist at the same resolved path in Job 2 and its on-disk SHA-256 must still match Job 1;
- every source imported in both processes must report the same SHA-256;
- any replay-imported source that was not present in the Job-1 inventory is rejected;
- capture-only modules are allowed to remain unimported in the fresh process when their exact recorded source bytes are still present unchanged on disk;
- all non-companion provenance remains exact, including checkpoint identity, model/runtime fingerprint, effective production wrappers/hooks/providers/compiler policy and the separate first-high Sol/VDN/Spectrum runtime-policy gates.

This is intentionally narrower than ignoring companion provenance. It removes only import-population/lazy-import lifecycle as an equality requirement while retaining exact source-byte validation. The strict equality path used to verify a persisted `provenance_equivalence_identity` against the bundle's full provenance is unchanged, so a modified or truncated persisted identity cannot override the authoritative bundle provenance.

Existing Job-1 bundles remain usable for this repair because all required capture source paths and hashes are already stored in the full provenance manifest. Job 1 does not need to be rerun solely because the cross-process comparison was repaired.

## Job 2: cold high replay

Exit ComfyUI completely and start a new process. Rebuild the same model/companion stack and the same original workflow inputs. Keep Untwist absent. Do not run the capture node in this process.

Apply:

1. normal production model/companion patches;
2. `MiniMax H3 Execution Contract Diagnostics` with `strict_provenance=true` and `capture_mib=512`;
3. `MiniMax H3 Same-State Cold High Replay`, selecting the Job-1 manifest in `bundle` (`[latest]` picks the most recent saved bundle);
4. the normal sampler node with the original full schedule and seed.

Keep the PR #35 checkpoint-diagnostic **overlay installed**, because strict provenance validates the #35/#36 stack. But do **not** evaluate the `H3HandoffCheckpointDiagnostic` output branch in Job 2. That output node requires a complete normal progressive learned-3D low → probe → upscale → high invocation, while R intentionally executes only the high suffix. Evaluating that output branch in Job 2 produces an unrelated "no complete learned_3d ... capture" error after an otherwise successful replay.

The replay node validates the caller schedule, target geometry, seed, pristine target conditioning, Flow guidance configuration, Spectrum configuration, exact checkpoint identity and installed provenance. It then suppresses Flow's progressive split for that invocation, restores the captured high-stage continuation contract, reuses the original captured high-child noise argument, installs the captured guidance trajectory, validates exact first-high X at the actual `SAMPLER_SAMPLE` boundary, and executes only the captured high suffix.

Connect the replayed LATENT to `MiniMax H3 Same-State Replay Report` as its trigger. The replay report branch, not the PR #35 handoff-checkpoint output branch, is the Job-2 diagnostic output.

## Required report gates

`attribution_valid=true` requires all of the following:

- replay completed without error;
- exact `3L / 2A / 1F` high-only accounting;
- zero learned-upscaler calls;
- target conditioning identity matches the capture;
- installed-runtime provenance matches under the source-byte/lazy-import contract above;
- exact checkpoint identity matched at replay preflight and remained unchanged through the pre-sampling stat guard;
- Flow guidance configuration matches;
- Spectrum configuration matches;
- first-high runtime policy matches;
- first-high Sol/VDN/Spectrum companion policy matches;
- Untwist remains absent;
- first-high sampler video/audio input hashes match;
- first-high H3 video/audio input hashes match.

Do not interpret output differences when `attribution_valid=false`.

## Decision rule

If attribution is valid and the cold replay produces the same broken first-high raw/pre-guidance output, retained low/probe/reload lifecycle state is weakened as the causal explanation. Continue with a different contract owner rather than adding resets.

If attribution is valid and the cold replay becomes clean while the capture was broken, retained lifecycle state is implicated. Inspect the first diverging owner/receipt before running any buffer experiment. Only then is a bounded owner-local scratch test such as experiment B justified.

A different final output without a first-high difference is not sufficient to blame retained state because guidance and later forecast history can amplify downstream differences.

## Rollback

Remove the four R diagnostic nodes and delete the generated bundle files. R does not modify production defaults, source noise policy, sampler schedules, model weights or companion repositories.
