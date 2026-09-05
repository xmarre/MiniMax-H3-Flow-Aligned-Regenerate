# MiniMax H3 Flow-Aligned Regenerate

Training-free ComfyUI nodes for reusing lower-resolution MiniMax H3 structure while producing a higher-resolution result, with dedicated Continuum paths for exact-prefix continuation.

The project has two main approaches:

1. **Flow-aligned two-pass guidance** — capture the low-resolution H3 denoising trajectory and use it to guide a later learned-upscale/refine pass.
2. **Progressive handoff** — spend early H3 work on a smaller video grid, then switch to the target grid inside one sampling schedule.

For Continuum exact-prefix continuation, the recommended accelerated path is now **Progressive Mixed-Grid Continuum**: it keeps the authoritative target-grid prefix for H3 conditioning, generates the continuation suffix on a real lower-resolution grid, performs the learned 3D latent upscale, then starts a fresh full-grid refinement stage.

> This is an independent research implementation informed by public work. It does not reproduce MiniMax's closed H3-Regenerate-2K implementation or an unreleased sparse-attention model.

See [CREDITS.md](CREDITS.md) for research and implementation attribution.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate.git
```

Restart ComfyUI.

The core package has no mandatory sibling-node dependency. The learned transfer paths require the companion [MiniMax H3 Latent Upscaler](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler).

## Recommended Continuum path

### Progressive Mixed-Grid Continuum

Use **MiniMax H3 Progressive Mixed-Grid Continuum [Experimental]** for exact-prefix continuation when you want progressive speedup without giving up the learned latent upscale.

The execution contract is:

```text
authoritative target-grid prefix
            +
real low-grid generated suffix
            |
       mixed H3 sequence
            |
       exact handoff probe
            |
      learned 3D upscale
            |
  restore exact target prefix
            |
 fresh target-grid refinement
```

The protected prefix is never spatially resized for transformer conditioning. The low-grid suffix is real low-resolution H3 sampling, not a sparse subset of target-grid hidden rows.

When VDN is enabled, this path uses VDN external-sequence API 2 (`mixed_grid_low_suffix`) during the mixed low stage and returns to normal VDN execution for the fresh target-grid stage.

### Suffix DC bridge

**Mixed-Grid Continuum** exposes `suffix_dc_bridge`, enabled by default.

At the exact-prefix handoff, the learned 3D upscaler produces a complete target-grid clean sequence before its learned prefix is discarded. The bridge measures the per-channel DC difference between that learned prefix boundary and the authoritative exact prefix, then applies the corresponding constant offset to **only the first generated suffix latent token**.

It does not:

- edit the authoritative prefix;
- alter later suffix tokens at bridge application;
- perform a video-space crossfade;
- add an H3 transformer evaluation.

The bridge fixed the brief Continuum boundary flash in matched real-media testing, including multi-boundary continuation. In the validated run, the uncorrected exact-boundary DC RMS was about `0.207763`; the corrected boundary was about `0.093093`, matching the learned-upscaler native boundary at about `0.093093`, while `final_prefix_exact=true` remained intact.

### Continuum Decode Context

**MiniMax H3 Continuum Decode Context** can be placed immediately before the normal Video VAE Decode. It supplies real future latent context to the native H3 temporal decoder at exact chunk joins while leaving accepted sampling latents, continuation state, masks, audio and the assembly plan unchanged.

This solves a separate decoder-window boundary problem. It is not the mechanism that fixed the mixed-grid DC flash above.

See [docs/CONTINUUM_DECODE_CONTEXT.md](docs/CONTINUUM_DECODE_CONTEXT.md).

## Target-Sparse Continuum status

**MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** remains available as a research/control path, but it is **not recommended for production quality**.

It keeps the sampler latent on the target grid and sparsifies only the early H3 hidden-token stream over generated video rows. Because it does **not** perform the learned latent upscale used by the Mixed-Grid path, real decoded-media testing showed cascading quality errors: skin imperfections, odd clothing changes and spurious background additions could appear and propagate through later continuation.

That negative result is the reason the release recommendation is Mixed-Grid rather than Target-Sparse. Target-Sparse remains useful for architectural experiments and controlled comparisons, not as the preferred Continuum acceleration path.

## Generic Progressive Handoff (Target Input)

**MiniMax H3 Progressive Handoff (Target Input)** is the generic target-input progressive node. It is not a Continuum-specific exact-prefix node.

- Unprotected calls can run early H3 work privately at lower resolution before handing off to the target grid.
- Exact protected video prefixes conservatively fall back to one ordinary target-grid sampler lifetime.
- It supports `bicubic` and optional `learned_3d` clean-video transfer on the normal handoff path.
- It does **not** expose or apply the Continuum `suffix_dc_bridge`.

## Flow-aligned two-pass guidance

The two-pass path keeps the established low-resolution generation -> learned upscale/refine workflow but reuses the full low-resolution denoising trajectory instead of only the endpoint.

```text
low-resolution H3
      |
Trajectory Capture
      |
H3_FLOW_TRAJECTORY ------------------+
                                      |
learned upscale / refine input -------+
                                      |
                         Flow-Aligned Regenerate
                                      |
                             high-resolution H3
```

Use:

1. **MiniMax H3 Flow Trajectory**
2. **MiniMax H3 Trajectory Capture** on the first pass
3. your learned latent upscale/refine initialization
4. **MiniMax H3 Flow-Aligned Regenerate** on the second pass

For Continuum `refine_state`, use **MiniMax H3 Flow-Aligned Refine State**.

`direction` remains the conservative guidance recommendation. `direction+acceleration`, `direction+temporal`, `downsample_consistency`, resolution-aware sigma remapping and the Attention Lab remain research controls.

## Nodes

### Main nodes

| Node | Purpose |
|---|---|
| **MiniMax H3 Flow Trajectory** | Shared trajectory handle for capture, guidance and progressive execution. |
| **MiniMax H3 Trajectory Capture** | Records first-pass H3 predicted-clean trajectory states. |
| **MiniMax H3 Flow-Aligned Regenerate** | Guides a later H3 pass from the matching captured trajectory state. |
| **MiniMax H3 Flow-Aligned Refine State** | Continuum `refine_state` version of flow-aligned guidance. |
| **MiniMax H3 Progressive Handoff** | Generic source-sized progressive resolution handoff. |
| **MiniMax H3 Progressive Handoff (Target Input)** | Generic target-input progressive handoff; exact protected prefixes fall back to the target grid. |
| **MiniMax H3 Progressive Mixed-Grid Continuum [Experimental]** | Recommended accelerated exact-prefix Continuum path: real low-grid suffix, target-grid prefix conditioning, learned 3D transfer, DC bridge and fresh target-grid refine. |
| **MiniMax H3 Continuum Decode Context** | Supplies right context to the native temporal VAE at exact Continuum joins. |

### Research/control nodes

| Node | Purpose |
|---|---|
| **MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** | Target-grid hidden-row sparsification experiment. Real media showed cascading quality artifacts; not recommended for final output. |
| **MiniMax H3 Refine Target Geometry [Experimental]** | Mirrors learned-refiner target sizing metadata. |
| **MiniMax H3 Resolution-Aware Sigmas [Experimental]** | Resolution-dependent refine-sigma experiments; default remains off. |
| **MiniMax H3 Reference Budget [Experimental]** | Reference-row diagnostics and guarded direct-reference cap. |
| **MiniMax H3 Attention Lab [Experimental]** | Attention/retention diagnostics and topology oracles. |

### Diagnostics

| Node | Purpose |
|---|---|
| **MiniMax H3 Runtime Metrics Probe** | Passive sampler/model-call instrumentation. |
| **MiniMax H3 Metrics JSON** | Writes structured H3/Spectrum/sampler/geometry metrics. |

## Patch order and interoperability

For the tested Continuum stack, keep model patches in this order when present:

```text
DiffAid
  -> Untwisting RoPE
  -> Spectrum
  -> Progressive node
  -> Continuum
```

DiffAid, Untwisting RoPE, Spectrum and VDN are optional integrations.

Important contracts:

- **Spectrum:** actual/forecast provenance and sampler-history boundaries are preserved. Fresh target-grid stages start with an actual H3 evaluation where required.
- **VDN-H3:** Mixed-Grid uses API 2 only during the external mixed sequence and resumes ordinary VDN behavior at full target resolution.
- **SA-Solver/PECE, SEEDS, ER-SDE, Euler/RES:** sampler objects are preserved; separate sampler lifetimes are used where geometry/history boundaries require them.
- **Audio:** progressive spatial transfer affects video only. Audio is never spatially resized.
- **Learned upscaler:** Mixed-Grid requires `learned_3d`; the generic Target Input node can use either `bicubic` or `learned_3d`.

## Practical status

The paths with the strongest real-media support are:

- two-pass flow-aligned guidance with the learned upscale/refine workflow;
- generic Progressive Handoff for unprotected calls;
- **Mixed-Grid Continuum with learned 3D transfer and the suffix DC bridge** for exact-prefix continuation.

The Mixed-Grid path has been exercised on a real RTX Pro 6000 workflow with VDN API 2, Spectrum + SA-PECE, DiffAid, Untwisting RoPE, learned 3D transfer, exact handoff probing and multiple Continuum boundaries. The previously observed boundary flashing is fixed by the suffix DC bridge in that tested workflow.

Target-Sparse is deliberately not promoted because its no-latent-upscale design produced cascading decoded-media defects in testing.

Quality and speed still depend on prompt, references, geometry, sampler, Spectrum policy, model residency and hardware. Use decoded media rather than structural metrics alone for new workflow variants.

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — wiring and parameter details
- [docs/MIXED_GRID_CONTINUUM.md](docs/MIXED_GRID_CONTINUUM.md) — Mixed-Grid contract and diagnostics
- [docs/TARGET_SPARSE_CONTINUUM.md](docs/TARGET_SPARSE_CONTINUUM.md) — Target-Sparse research path
- [docs/CONTINUUM_DECODE_CONTEXT.md](docs/CONTINUUM_DECODE_CONTEXT.md) — decoder right-context helper
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — decoded-media validation ledger
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — workflow-specific timing evidence
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — internal contracts
- [docs/RESEARCH.md](docs/RESEARCH.md) — research-transfer rationale
- [CREDITS.md](CREDITS.md) — attribution and provenance
- [RELEASE_NOTES.md](RELEASE_NOTES.md) — release history

## License

MiniMax H3 Flow-Aligned Regenerate is licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 xmarre.

Referenced papers and third-party repositories retain their own copyrights and licenses.
