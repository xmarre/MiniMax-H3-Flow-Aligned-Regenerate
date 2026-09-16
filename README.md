# MiniMax H3 Flow-Aligned Regenerate

Training-free ComfyUI nodes for reusing lower-resolution MiniMax H3 structure while producing a higher-resolution result.

The project has two main generation paths:

1. **Flow-aligned two-pass guidance** — capture a lower-resolution H3 trajectory and use it to guide a later learned-upscale/refine pass.
2. **Progressive handoff** — run early H3 work on a smaller video grid, then transition once to the final target grid inside the same sampling schedule.

For target-input workflows, including Continuum, the standard progressive node is **MiniMax H3 Progressive Handoff (Target Input)**.

> This is an independent research implementation informed by public work. It does not reproduce MiniMax's closed H3-Regenerate-2K implementation or an unreleased sparse-attention model.

See [CREDITS.md](CREDITS.md) for research and implementation attribution.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate.git
```

Restart ComfyUI.

The core package has no mandatory sibling-node dependency. The intended learned-transfer workflow uses the companion [MiniMax H3 Latent Upscaler-Plus](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus).

## Example workflows

Loadable examples are under [`workflows/examples/`](workflows/examples/):

- [`progressive-target-input.workflow.json`](workflows/examples/progressive-target-input.workflow.json) — standard target-input progressive workflow using `source_scale=0.70`, fixed `0.35` handoff, `direction+temporal`, and `learned_3d` transfer;
- [`progressive-source-input.workflow.json`](workflows/examples/progressive-source-input.workflow.json) — dependency-minimal source-input progressive control using a `1.20x` target handoff.

The target-input example configures `minimax_h3_latent_upscaler_3d_bf16.safetensors`, CUDA, bf16 precision, and `offload_after_upscale=false`. Matching `.api.json` graphs are included for API execution.

## Standard Target Input path

Use **MiniMax H3 Progressive Handoff (Target Input)** when the surrounding workflow is already defined at the final target geometry.

Canonical controls remain:

```text
source_mode          = scale
source_scale         = 0.70
source_width         = 864
source_height        = 640
handoff_coordinate   = 0.35
handoff_selection    = fixed
guidance_mode        = direction+temporal
direction_weight     = 0.25
acceleration_weight  = 0.25
consistency_weight   = 0.25
low_frequency_cutoff = 0.25
temporal_weight      = 0.20
handoff_transfer     = learned_3d
```

With `guidance_mode=direction+temporal`, `acceleration_weight` and `consistency_weight` are stored but inactive. Acceleration is used only by `direction+acceleration`; consistency is used only by `downsample_consistency`.

### Unprotected or fractional-mask calls

The normal progressive path:

1. derives a private low video grid;
2. runs the early sampler lifetime there;
3. performs one exact handoff probe;
4. transfers the predicted-clean video state to the target grid;
5. keeps audio on its existing joint-H3 path;
6. resets sampler/Spectrum history across the geometry boundary;
7. starts the target-grid stage with an actual H3 evaluation.

`learned_3d` is the intended transfer. `bicubic` remains available as a dependency-free control.

### Exact protected video prefixes

Exact Native Masked protection takes precedence over progressive resizing. If the prepared video denoise mask contains any exact-zero value, Target Input uses the conservative target-grid fallback:

- the original target-grid latent/noise/mask remain authoritative;
- no private low-grid sampler lifetime is created;
- no exact handoff probe or learned-upscaler call is added;
- no geometry/history boundary is introduced;
- the returned protected video/audio values are restored exactly at the framework boundary.

The production fallback also uses a bounded **four-audio-tick guided overlap** at canonical continuation boundaries. During sampler lifetime only, the tail of the carried audio mask becomes the 1/256-aligned ramp `0.203125, 0.40234375, 0.6015625, 0.80078125`. The video mask is byte-for-byte unchanged, and the original exact mask still owns the returned output. Set `H3_FLOW_AUDIO_GUIDED_OVERLAP_TICKS=0` only for controlled disable/bisect work.

This is the standard exact-prefix continuation behavior. It intentionally favors the exact target-grid contract over a low-resolution continuation speedup.

## Mixed-Grid retirement

**MiniMax H3 Progressive Mixed-Grid Continuum [Deprecated]** is retained for one compatibility release so existing serialized workflows still load with their original semantics.

It is no longer a recommended production path and no Mixed-Grid compatibility render is a release gate. The old node ID is **not** remapped to Target Input because that would silently change serialized workflow behavior.

New workflows should use **MiniMax H3 Progressive Handoff (Target Input)**. The historical Mixed-Grid attention-measure, VDN external-sequence and seam-repair material remains documented only as compatibility/research history during the deprecation window; it is scheduled for removal after that window.

## Target-Sparse status

**MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** remains a research/control path, not a production recommendation. Real decoded-media testing showed cascading quality defects on its hidden-row sparsification path.

## Continuum Decode Context

**MiniMax H3 Continuum Decode Context** is independent of progressive sampling. It supplies real future latent context to the native H3 temporal VAE at exact Continuum joins while leaving accepted sampling latents, continuation state, masks, audio and assembly ownership unchanged.

See [docs/CONTINUUM_DECODE_CONTEXT.md](docs/CONTINUUM_DECODE_CONTEXT.md).

## Flow-aligned two-pass guidance

The two-pass path keeps a conventional low-resolution generation -> learned upscale/refine workflow and reuses the full low-resolution denoising trajectory.

Use:

1. **MiniMax H3 Flow Trajectory**
2. **MiniMax H3 Trajectory Capture** on the first pass
3. the learned latent upscale/refine initialization
4. **MiniMax H3 Flow-Aligned Regenerate** on the second pass

For Continuum `refine_state`, use **MiniMax H3 Flow-Aligned Refine State**.

## Nodes

### Main nodes

| Node | Purpose |
|---|---|
| **MiniMax H3 Flow Trajectory** | Shared trajectory handle for capture, guidance and progressive execution. |
| **MiniMax H3 Trajectory Capture** | Records first-pass H3 predicted-clean trajectory states. |
| **MiniMax H3 Flow-Aligned Regenerate** | Guides a later H3 pass from the matching captured trajectory state. |
| **MiniMax H3 Flow-Aligned Refine State** | Continuum `refine_state` version of flow-aligned guidance. |
| **MiniMax H3 Progressive Handoff** | Generic source-sized progressive resolution handoff. |
| **MiniMax H3 Progressive Handoff (Target Input)** | Standard target-input path; exact protected video prefixes use the conservative target-grid fallback. |
| **MiniMax H3 Continuum Decode Context** | Supplies right context to the native temporal VAE at exact Continuum joins. |

### Compatibility / research nodes

| Node | Status |
|---|---|
| **MiniMax H3 Progressive Mixed-Grid Continuum [Deprecated]** | Compatibility-only for one release; do not use for new production workflows. |
| **MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** | Architectural control; not recommended for final output. |
| **MiniMax H3 Refine Target Geometry [Experimental]** | Mirrors learned-refiner target sizing metadata. |
| **MiniMax H3 Resolution-Aware Sigmas [Experimental]** | Resolution-dependent refine-sigma experiment; default remains off. |
| **MiniMax H3 Reference Budget [Experimental]** | Reference-row diagnostics and guarded direct-reference cap. |
| **MiniMax H3 Attention Lab [Experimental]** | Attention/retention diagnostics and topology oracles. |

## Patch order and interoperability

For the tested Continuum stack, keep model patches in this order when present:

```text
DiffAid
  -> Untwisting RoPE
  -> Spectrum
  -> Progressive Target Input
  -> Continuum
```

Important contracts:

- **Spectrum:** actual/forecast provenance and history boundaries are preserved; a new target-grid progressive stage starts with an actual evaluation.
- **VDN-H3 / Sol-H3:** ordinary production attention backends remain independent of Target Input's exact-prefix fallback. Mixed-Grid-specific weighted compatibility is not a production requirement.
- **Audio:** progressive spatial transfer affects video only. The exact-prefix fallback may guide the last four carried audio ticks during sampling, but final protected AV ownership remains exact.
- **Learned upscaler:** used by the normal progressive handoff; exact protected-video fallback performs zero learned-upscaler calls.

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — current wiring and parameter details
- [docs/CONTINUUM_DECODE_CONTEXT.md](docs/CONTINUUM_DECODE_CONTEXT.md) — decoder right-context helper
- [docs/MIXED_GRID_CONTINUUM.md](docs/MIXED_GRID_CONTINUUM.md) — deprecated compatibility-path contract/history
- [docs/mixed-grid-seam-repair.md](docs/mixed-grid-seam-repair.md) — historical Mixed-Grid seam investigation
- [docs/TARGET_SPARSE_CONTINUUM.md](docs/TARGET_SPARSE_CONTINUUM.md) — Target-Sparse research path
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — decoded-media validation ledger; historical configurations remain historical evidence
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — workflow-specific timing evidence
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — internal contracts
- [CREDITS.md](CREDITS.md) — attribution and provenance
- [RELEASE_NOTES.md](RELEASE_NOTES.md) — release history

## License

MiniMax H3 Flow-Aligned Regenerate is licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 xmarre.

Referenced papers and third-party repositories retain their own copyrights and licenses.