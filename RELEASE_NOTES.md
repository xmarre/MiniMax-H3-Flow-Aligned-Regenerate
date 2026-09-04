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
