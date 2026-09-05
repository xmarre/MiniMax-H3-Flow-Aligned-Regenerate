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
