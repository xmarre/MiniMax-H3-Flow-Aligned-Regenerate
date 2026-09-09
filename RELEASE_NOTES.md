# MiniMax H3 Flow-Aligned Regenerate v0.3.3

v0.3.3 fixes the decode boundary exposed by the coordinated Continuum learned-upscale/refine path and records the measured performance limit of additive two-pass refinement against an already-short native H3 schedule.

## Native joint AV decode compatibility

`MiniMax H3 Latent Upscaler + Refine (3D)` correctly reconstructs native H3 joint `NestedTensor([video, audio])` state before sampler 2 and returns that native AV state after refinement. `MiniMax H3 Continuum Decode Context` historically accepted only Continuum's earlier split plain-video LATENT representation, so a fully completed refine run could fail immediately before Video VAE Decode with:

```text
decode group 1 requires native [1,24,T,H,W] video
```

Decode Context now accepts both representations:

- split Continuum video `samples: [1,24,T,H,W]`;
- native joint H3 `NestedTensor([video, audio])` sampler output.

For joint AV input it validates both members, extracts the existing 24-channel video tensor into a minimal decode-only LATENT view, and never mutates the source AV wrapper, source audio, masks, or assembly plan. Terminal/unextended joint-AV video extraction is zero-copy. Existing split-video identity behavior remains unchanged.

The temporal right-context logic is otherwise unchanged: only exact physical overlaps receive the five real future latent tokens required by the native H3 temporal VAE window, and the original assembly plan still trims the 17 decode-only output frames.

Regression coverage exercises valid joint AV input, zero-copy terminal extraction, source/audio/mask/plan immutability, malformed AV member counts and malformed audio shape, while retaining the existing split-video and native temporal-decoder oracles.

## Two-pass performance contract

A controlled hot comparison around a ~1.1 MP target showed that the complete low-resolution H3 pass remains materially cheaper, but the additive high-resolution refine can exceed that saving when the native control already uses a short schedule:

- native Spectrum-stage total: `282.11 s`;
- Flow low-resolution subtotal: `182.88 s`, saving `99.23 s / 35.2%`;
- added high-resolution refine: `150.34 s`;
- Flow Spectrum-stage total: `333.22 s`, `+51.11 s / +18.1%` versus native;
- native call topology: `16 logical / 11 actual / 5 forecast`;
- additive Flow topology: `24 logical / 17 actual / 7 forecast`.

The Flow final grid was about 2.7% larger than the native control, which biases the comparison slightly but cannot explain the full penalty. The result is documented as a topology/property of the two-pass path, not as a failure of the low-resolution stage: completing the same short logical trajectory at low resolution and then launching a second high-resolution sampler lifetime does not replace native intervals.

No public defaults are silently reduced to manufacture a speed claim. Reduced first-pass/refine budgets require matched decoded-media validation. For speed-oriented execution, use a sampler-internal progressive handoff where low-grid work replaces target-grid intervals inside one trajectory.

## Distribution

The package version is bumped to `0.3.3`. Existing CI, GitHub release, checksum and Comfy Registry workflows remain unchanged; publication occurs only after the exact `main` commit passes CI.

---

# MiniMax H3 Flow-Aligned Regenerate v0.3.2

v0.3.2 fixes the `KeyError: 'layout'` reported in issue #22 on MiniMax H3 block-patch callers that predate ComfyUI #15975, and adds complete executable workflow examples instead of leaving only topology overlays under `workflows/`.

## MiniMax H3 packed-layout compatibility

The failure was a versioned ComfyUI block-patch contract mismatch. Older MiniMax H3 cores invoked `double_block` replacements without a direct `layout` entry, while this extension's layout wrapper indexed `args["layout"]` unconditionally. ComfyUI #15975 later added the direct layout argument, which is why the repository's newer pinned source-contract lane did not reproduce the reporter's older-core failure.

The wrapper now resolves the packed H3 layout in this order:

1. the current direct block argument, `args["layout"]`;
2. `transformer_options["minimax_h3_layout"]`, published by newer ComfyUI for MiniMax H3 attention integrations;
3. if neither contract is available, output-neutrally forward to the previously installed block wrapper or native H3 block without installing Flow attention context.

The third path deliberately does not fabricate a layout. Generic Flow/Progressive sampling therefore keeps the native or previously patched attention path instead of failing before the first transformer evaluation. A `packed_layout_unavailable_calls` metric counter records the compatibility fallback. Layout-dependent research paths still require a core that exposes a real packed layout.

Regression coverage exercises the pre-layout block-argument shape, the newer `transformer_options` fallback, direct-layout precedence, wrapper chaining, context restoration, and the output-neutral legacy path.

## Executable workflow examples

The existing `workflows/*.overlay.json` files are wiring/specification overlays, not serialized ComfyUI graphs. That distinction is now documented explicitly in `workflows/README.md`.

Two loadable LiteGraph workflows are included under `workflows/examples/`:

- `progressive-target-input.workflow.json` — target-grid input with an early `source_scale=0.70` stage and dependency-free bicubic handoff;
- `progressive-source-input.workflow.json` — source-grid input with an in-sampler `1.20x` progressive target handoff.

Matching API-format prompt graphs are provided as `progressive-target-input.api.json` and `progressive-source-input.api.json`.

All four examples use the stock MiniMax H3 loader/conditioning/decode chain, `res_multistep`, 19 sampling steps, and no Turbo LoRA or optional companion integration. They wire the patched model into both `BasicScheduler` and `BasicGuider`. Structural tests parse both formats, verify every graph link, enforce the intended sampler and patch wiring, validate the LiteGraph node/link topology, and guard against accidentally adding a LoRA dependency.

## Distribution

The package version is bumped to `0.3.2`. Existing CI, GitHub release, checksum and Comfy Registry workflows remain unchanged; publication occurs only after the exact `main` commit passes CI.

---

# MiniMax H3 Flow-Aligned Regenerate v0.3.1

v0.3.1 fixes an exact-prefix failure exposed by `res_multistep` and hardens the final exact-mask contract for every target-input progressive path.

## Exact-mask sampler return canonicalization

ComfyUI's inpaint wrapper restores `mask == 0` values to the caller's `latent_image` after every model evaluation. Some numerical solvers can still perform a final arithmetic update after that last evaluation. In particular, `res_multistep` reaches its terminal state through an Euler-form expression that is algebraically equal to the final denoised prediction but can differ by a few floating-point ULPs.

Mixed-Grid correctly treated the protected prefix as immutable, but v0.3.0 then required the sampler's returned packed state to be bitwise identical without canonicalizing those post-model solver roundoff differences. A valid run could therefore finish all H3 work and then fail the final `torch.equal` exact-prefix assertion.

The Comfy compatibility boundary now restores only exactly protected `mask == 0` packed elements from the authoritative target input after the final target-grid sampler lifetime returns. This happens before Mixed-Grid's strict final-prefix assertion and seam diagnostics. The hard bitwise invariant remains unchanged; it is not weakened to an `allclose` tolerance.

The same final-return rule also covers Target-Sparse high-stage sampling and the generic Target Input exact-prefix fallback. Generated/unmasked values are left unchanged, already-exact sampler returns stay zero-copy, and no additional H3 transformer evaluation is introduced.

New telemetry records an `exact_mask_output` event with sampler/source identity, protected and changed element counts, pre-restore exactness, finite/non-finite drift counts, maximum absolute drift and RMS drift. `exact_mask_output_canonicalizations` counts returns that required restoration.

Regression coverage includes a deterministic reproduction of the `res_multistep` terminal Euler roundoff, verifies that only protected values are repaired, verifies a zero-copy already-exact path, and exercises both Mixed-Grid and conservative target-input fallback through the actual Flow outer-wrapper boundary.

## Distribution

The package version is bumped to `0.3.1`. Existing CI, GitHub release, checksum and Comfy Registry workflows remain unchanged; publication occurs only after the exact `main` commit passes CI.

---

# MiniMax H3 Flow-Aligned Regenerate v0.3.0

v0.3.0 makes the real low-grid **Mixed-Grid Continuum** path the recommended exact-prefix progressive workflow, adds the one-token suffix DC seam bridge that fixed the observed Continuum boundary flash, adds native temporal decode context, and ships the VDN API-2 interoperability needed by the mixed sequence.

## Mixed-Grid Continuum

- Adds **MiniMax H3 Progressive Mixed-Grid Continuum [Experimental]**.
- Keeps the authoritative target-grid protected prefix for H3 transformer conditioning while sampling a genuine lower-resolution generated suffix.
- Runs the exact handoff probe under the same mixed topology.
- Requires the versioned `learned_3d` provider from `Comfyui_Minimax_h3_latent_Upscaler` for suffix transfer.
- Restores the authoritative target-grid prefix before a fresh full-grid refinement stage.
- Requires an actual first target-grid H3 evaluation after the handoff.
- Preserves audio, mask/noise semantics, sampler boundaries and exact final-prefix protection.

## Continuum suffix DC bridge

`Progressive Mixed-Grid Continuum` exposes `suffix_dc_bridge`, enabled by default.

The bridge is intentionally narrow:

- computes a per-batch/per-channel spatial-mean offset from the learned-upscaler prefix boundary to the authoritative exact prefix;
- applies that offset to exactly the **first generated suffix latent token**;
- uses fixed weight `1.0`;
- never edits the protected prefix;
- leaves every later suffix token unchanged at bridge application;
- performs no video-space crossfade;
- adds no H3 transformer NFE.

Matched real-media testing on the production RTX Pro 6000 Continuum workflow removed the brief boundary flash, including multi-boundary continuation. The validated run reported approximately:

- uncorrected exact-boundary DC RMS: `0.207763`;
- corrected exact-boundary DC RMS: `0.093093`;
- learned-upscaler native boundary DC RMS: `0.093093`;
- corrected tokens: `1`;
- bridge weight: `1.0`;
- `final_prefix_exact=true`.

The correction fixed the artifact rather than merely moving it to another boundary.

The dedicated **Target-Sparse Continuum** research node also exposes the same one-token control at its full-grid high-stage boundary, but Target-Sparse is not the recommended quality path; see below. The generic **Progressive Handoff (Target Input)** node is not Continuum-specific and does **not** expose or apply this bridge.

## Target-Sparse quality result

Target-Sparse remains in the package as an architectural/control experiment but is not promoted for production output.

Real decoded-media testing showed that its target-grid hidden-row sparsification can produce cascading quality errors because it does not perform the learned latent upscale used by Mixed-Grid. Observed failures included skin imperfections, odd clothing changes and spurious background additions that propagated through later continuation.

This negative media result closes the previous Target-Sparse acceptance question: the preferred Continuum acceleration path is Mixed-Grid, where the low-resolution suffix is followed by learned 3D latent transfer and full-grid refinement.

## Continuum Decode Context

Adds **MiniMax H3 Continuum Decode Context** for native H3 temporal VAE boundaries.

- validates exact physical Continuum joins;
- supplies real right-context latent tokens to the preceding chunk's decode-only tensor;
- leaves accepted sampling latents, continuation state, mask, plan and audio unchanged;
- relies on unchanged Continuum assembly trimming to discard the decode-only tail.

This addresses the native temporal-decoder window boundary and is independent from the latent DC bridge above.

## VDN-H3 interoperability

- Adds external-sequence API 2 / `mixed_grid_low_suffix` integration for `ComfyUI-VDN-H3`.
- During the mixed sequence VDN retains the learned dense softmax gate while disabling only geometry-dependent local-window/linear-complement processing that cannot be interpreted on mixed spatial lattices.
- The fresh target-grid stage receives no external mixed-sequence contract and resumes ordinary VDN behavior.
- API 1 target-sparse compatibility remains available.

The coordinated VDN release is `xmarre/ComfyUI-VDN-H3` v1.5.0.

## Validation

The release is gated by:

- Python 3.10, 3.11, 3.12 and 3.13 CI;
- Ruff + format checks;
- full unit/synthetic suite;
- source-contract/native-oracle lane against pinned ComfyUI, Continuum, Spectrum, DiffAid, RefDelta, Untwisting RoPE, the learned upscaler and the coordinated VDN release;
- package build, Apache-2.0 metadata validation and isolated wheel import;
- real multi-boundary Mixed-Grid Continuum media validation on RTX Pro 6000.

## Distribution

The package version is bumped to `0.3.0`. After the exact `main` commit passes CI, the existing release workflow creates the GitHub source ZIP + `SHA256SUMS`, and the registry workflow publishes the same tested version to the Comfy Registry.

---

# MiniMax H3 Flow-Aligned Regenerate v0.2.1

v0.2.1 added CI-gated Comfy Registry publication. Runtime and sampling behavior were unchanged from v0.2.0.

---

# MiniMax H3 Flow-Aligned Regenerate v0.2.0

v0.2.0 added the optional `learned_3d` handoff provider for Progressive Target Input. Around ~1 MP, the learned transfer clearly outperformed aggressive bicubic handoff in difficult decoded-media testing; `source_scale=0.70` was the strongest tested quality/compute point in that sweep.

---

# MiniMax H3 Flow-Aligned Regenerate v0.1.0

Initial public release: H3 trajectory capture, flow-aligned second-pass guidance, progressive target-input handoff, Continuum refine-state guidance, runtime metrics and the initial experimental research controls.
