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
2. `MiniMax H3 Execution Contract Diagnostics` with `strict_provenance=true`;
3. `MiniMax H3 Same-State Replay Capture`;
4. the normal sampler path.

Connect the sampled LATENT to `Save MiniMax H3 Same-State Replay Bundle` as its trigger. The save node writes two files under `output/h3_flow_replay/`:

- a JSON declarative manifest;
- a `.pt` payload containing tensors only.

The bundle records the exact high-stage input state, original target latent, high mask when present, high suffix, seed, conditioning identity, Flow guidance configuration, Spectrum configuration and the committed low/probe guidance trajectory. It also records exact checkpoint identity, installed-runtime provenance, first-high execution policy, first-high Sol/VDN/Spectrum companion policy and the corresponding runtime observation evidence.

`first_high_sampler_input_*` specifically means the raw high-entry sampler state **before** `KSamplerX0Inpaint` may rewrite masked model input. `first_high_model_input_*` is the separate downstream post-inpaint boundary. R serializes the raw sampler state and applies Flow/Core's exact noise-initialization inverse to that raw state; it must never use the post-inpaint model input as replay X.

The raw VDN observation additionally records bounded retained cache-key topology and the one-block async prefetch lifecycle (generation, target, pending/completed status). These fields are evidence, not equality policy: retained counts, cache contents/keys and prefetch state are intentionally allowed to differ between progressive capture and the cold process. VDN configuration, layout, wrapper identity and whether retention is enabled remain attribution gates.

Do not reuse a bundle if the capture run did not visibly reproduce the artifact.

## Job 2: cold high replay

Exit ComfyUI completely and start a new process. Rebuild the same model/companion stack and the same original workflow inputs. Keep Untwist absent. Do not run the capture node in this process.

Apply:

1. normal production model/companion patches;
2. `MiniMax H3 Execution Contract Diagnostics` with `strict_provenance=true`;
3. `MiniMax H3 Same-State Cold High Replay`, pointing `bundle_manifest` to the JSON file from job 1;
4. the normal sampler node with the original full schedule and seed.

The replay node validates the caller schedule, target geometry, seed, pristine target conditioning, Flow guidance configuration, Spectrum configuration, exact checkpoint identity and installed provenance. It then suppresses Flow's progressive split for that invocation, restores the captured high-stage continuation contract, reconstructs sampler noise through Flow/Core's exact initialization inverse, installs the captured guidance trajectory, and executes only the captured high suffix.

Connect the replayed LATENT to `MiniMax H3 Same-State Replay Report` as its trigger.

## Required report gates

`attribution_valid=true` requires all of the following:

- replay completed without error;
- exact `3L / 2A / 1F` high-only accounting;
- zero learned-upscaler calls;
- target conditioning identity matches the capture;
- installed-runtime provenance matches after excluding only O/C and R instrumentation identity;
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
