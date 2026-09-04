# MiniMax H3 Flow-Aligned Regenerate v0.3.0

v0.3.0 preserves exact Continuum Native Masked prefixes in Progressive Target Input and adds opt-in OpenVDN-topology attention diagnostics/reference tooling. The Continuum fix is default-path correctness behavior; the VDN modes remain experimental diagnostics and correctness oracles, not production acceleration.

## Exact Continuum prefix safety

- Progressive Target Input now detects exact-zero values in the prepared **video** denoise mask and treats that prefix as authoritative target-grid state.
- When exact video protection is present, the private low-resolution stage is bypassed for that invocation: no target latent resize, private low-grid noise, learned transfer, low/probe/high split, or same-invocation Flow guidance is applied.
- The original target-grid noise, latent, denoise mask, sampler, schedule, callback, and mutable shape metadata are forwarded through one ordinary sampler lifetime.
- Audio-only protection and fractional video masks retain the existing progressive path.
- Metrics emit `progressive_target_fallback` and record the reason, target-grid forwarding, sampler invocation count, history-boundary count, and whether progressive guidance was applied.

This fixes the prior semantic mismatch where generated rows could attend to a spatially downsampled approximation of a Continuum prefix that was supposed to remain exact, even if protected values were restored later at the target-grid boundary.

## Decoded H3 runtime validation

The release gate used matched two-chunk Continuum Native Masked runs with ordinary `sa_solver_pece`, the normal H3 workflow stack, and a known-good VAE path.

- Progressive off: `440.32 s` complete workflow; each chunk used 11 logical calls = 8 actual transformer calls + 3 Spectrum forecasts; no full-frame jitter/flashing.
- Progressive on: chunk 1 used 7 low-grid logical calls, one exact handoff probe, and 3 target-grid high-stage actual calls; chunk 2 detected the protected prefix and used the exact target-grid fallback.
- Progressive-on complete workflow: `398.64 s`, about **9.5% lower** than the matched control even though only chunk 1 was progressively accelerated.
- For chunk 1's principal sampling path, native Spectrum wall time was `176.991 s` versus approximately `135.960 s` for progressive low + exact probe + high, about **23.2% lower** in that run.
- Progressive metrics recorded `progressive_target_fallbacks=1`, `handoff_exact_probe_nfe=1`, `progressive_history_boundaries=2`, `progressive_sampler_invocations=4`, 18 actual transformer evaluations, and 4 Spectrum forecasts.
- The decoded progressive result had no new jitter/flashing and no new Continuum seam regression was reported. The exact-prefix fallback retained the material seam/cut improvement observed in the tested seed/reference setup.

These timings are workflow-, resolution-, prompt-, and hardware-specific measurements, not universal speed claims.

The earlier full-frame flashing/jitter report was misattributed to Flow. It reproduced with the external **MiniMax H3 RefDelta Stability Sampler** and disappeared when replaced with ordinary SA-PECE in both progressive-off and progressive-on controls. RefDelta jitter is therefore a separate issue and is not presented as a v0.3.0 Flow regression.

## Attention Lab / VDN reference tooling

- `diagnostic` remains output-neutral and now reports native dense-attention retention/outside mass, modality mass, boundary mass, exact analytic allowed-pair density, and optional Continuum seam diagnostics.
- Adds `vdn_reference_dense`, an independently implemented all-head, query-chunked dense-mask correctness oracle using complete 5-latent-frame chunks, radius-1 previous/current/next temporal neighborhoods, globally visible non-video tokens, and symmetric first/last row-and-column anchors.
- Optional Continuum seam anchoring fails closed unless authoritative `h3_continuum.protected_video_prefix_latent_slots` metadata exists. Pixel/frame counts are never guessed into latent slots.
- The reference path does **not** implement OpenVDN's trained long-range linear branch, compiled FlexAttention/BlockMask kernel, cache design, or separately licensed VDN-H3 weights, and it makes no OpenVDN speed/quality claim.

## Validation and compatibility

- PR gate: Ruff check and format check passed; `pytest` reported **150 passed**; `compileall`, sdist/wheel build, built-wheel import, pinned sibling/source contracts, and GitHub CI passed.
- Existing Progressive Target Input workflows without exact-zero video protection retain the prior low/probe/high path.
- Existing audio handling, fractional masks, target geometry, learned handoff provider behavior, Spectrum accounting, and trajectory metrics remain compatible outside the protected-prefix fallback.
- v0.3.0 is a minor release because it adds a new user-selectable Attention Lab reference mode in addition to the Continuum correctness fix.

---

# MiniMax H3 Flow-Aligned Regenerate v0.2.1

v0.2.1 adds CI-gated Comfy Registry publication. Runtime and sampling behavior are unchanged from v0.2.0.

## Distribution

- Adds a dedicated **Publish to Comfy registry** workflow using the pinned `Comfy-Org/publish-node-action` v1 revision already used by the companion MiniMax H3 latent-upscaler release path.
- Registry publication runs only after a successful `CI` push run on `main` and only when the tested commit changes the package version relative to its first parent.
- Manual `workflow_dispatch` remains available for explicit operator-controlled publication/retry.
- The workflow checks out the exact CI-tested commit, uses read-only repository permissions and non-persistent checkout credentials, and consumes `REGISTRY_ACCESS_TOKEN` only in the publish step.
- Publication is serialized per tested commit so overlapping workflow-run/manual attempts are not cancelled mid-publish.

## Versioning

The package version is bumped to `0.2.1` rather than republishing `0.2.0` from a different source commit. This keeps the GitHub release/tag and Comfy Registry package version aligned to the same source revision.

## Compatibility

No Python runtime, sampler, handoff, guidance, Continuum, Spectrum, metrics, or learned-upscaler integration code changes are included in v0.2.1.

---

# MiniMax H3 Flow-Aligned Regenerate v0.2.0

v0.2.0 adds an optional learned 3D transfer at the Progressive Target Input handoff boundary, backed by the versioned API-v1 provider from `Comfyui_Minimax_h3_latent_Upscaler`.

## Learned progressive handoff

- Adds `handoff_transfer=learned_3d` to **MiniMax H3 Progressive Handoff (Target Input)**.
- Keeps `handoff_transfer=bicubic` as the compatibility default for existing workflows.
- Consumes the API-v1 `H3_LATENT_UPSCALER` clean-video provider rather than importing sibling-node internals.
- Replaces only the exact-probe clean-video spatial transfer. Deterministic target noise, conditional re-noising, caller audio, masks, target conditioning, sampler/Spectrum history boundaries, and the mandatory first high-grid actual H3 call remain unchanged.
- Learned CNN work is tracked separately and does not increase H3 model-call/NFE counters.

The coordinated provider is released in `Comfyui_Minimax_h3_latent_Upscaler` v0.2.0. The exact provider revision validated by this release is `bdc670e5926bcefbe4022e17fe8b171fbfcf15de`.

## Decoded-media validation

The learned-transfer path was validated on aggressive progressive transitions around 1 MP rather than only on synthetic shape/contract tests.

- Around a `1152×864` (~0.995 MP) target, an aggressive bicubic handoff from roughly a ~0.55 MP private source showed substantial body/spatial handoff artifacts in the tested difficult prompt.
- Replacing only that boundary with `learned_3d` fixed the majority of the observed artifacts.
- `source_scale=0.70` resolved to `800×608 → 1152×864` and was judged excellent.
- `source_scale=0.65` resolved to `736×576 → 1152×864` and began losing reference likeness / tonal stability, so it is not promoted.
- The final higher-resolution gate used `source_scale=0.70` at `832×640 → 1184×896` (~1.061 MP). The generated action differed, but decoded quality was again judged very good.
- That final run preserved `38 logical / 28 actual H3 NFE / 10 Spectrum forecast` calls across two physical chunks, with 2 exact probes, 6 progressive sampler invocations, 4 history boundaries, copied audio, rebuilt high-grid conditioning, and an actual first high-grid H3 call in both chunks.
- BF16 CUDA learned inference took about 0.60 s and 0.77 s for the two chunks and added zero H3 NFEs.

For this prompt around 1 MP, `source_scale=0.70` is the current tested quality/compute point and `0.65` is below the observed fidelity floor. These values are not claimed as universal optima.

## Guidance interpretation

The latest learned-transfer media sweep used `direction+acceleration`. It does not establish an acceleration advantage over direction-only. The earlier matched evidence still supports `direction` as the conservative preferred guidance mode; acceleration, temporal correspondence, downsample consistency, auto handoff, and resolution-aware refine sigmas remain experimental controls unless future matched media shows a clear benefit.

## Compatibility and validation

- Existing bicubic Progressive Target Input workflows retain their prior behavior.
- Full 1-to-0 H3 sigma schedules remain required for progressive handoff.
- CI validates Python 3.10–3.13, Ruff and formatting, 142 unit/synthetic tests, `compileall`, package build, isolated wheel import, and pinned native/sibling source contracts.
- The source-contract gate pins the merged learned-provider revision and checks its API/kind constants, helper, provider classes, callable surface, and ComfyUI type.

---

# MiniMax H3 Flow-Aligned Regenerate v0.1.0

Initial public release of MiniMax H3 Flow-Aligned Regenerate.

## Highlights

- H3-native low-resolution trajectory capture and time-aligned high-resolution guidance.
- Progressive Target Input handoff for Continuum: early H3 sampling can run on a private lower-resolution grid before switching to the final grid without a separate learned-refine replay.
- Integrated Continuum refine-state guidance for the existing MiniMax H3 Latent Upscaler + Refine path.
- Experimental resolution-aware refine sigmas, temporal correspondence, acceleration, downsample consistency, reference-budget diagnostics, and attention diagnostics.
- Runtime metrics distinguish logical sampler calls, actual H3 NFEs, Spectrum forecasts, handoff probes, guidance events, and resolution-map events.

## Tested operating point

The strongest matched difficult-motion result tested during v0.1.0 development used 14 SA-Solver-PECE outer steps, a 736×736 private source to 896×896 target, `source_scale=0.83`, fixed handoff 0.35, direction guidance 0.25, and 54 logical / 36 actual H3 NFE / 18 Spectrum forecast calls across two chunks.

D12 was judged better than D10, and D14 slightly better again. These are tested quality/speed tradeoffs, not universal optima.

## Scope

This project is an independent, training-free research implementation informed by public work. It does not reproduce MiniMax's closed H3-Regenerate-2K model or unreleased sparse-attention topology. Decoded media remains the quality gate.