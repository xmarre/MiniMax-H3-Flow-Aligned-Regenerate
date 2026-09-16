# MiniMax H3 Flow-Aligned Regenerate

Training-free ComfyUI nodes for reusing lower-resolution MiniMax H3 structure while producing a higher-resolution result, with dedicated Continuum support for exact-prefix continuation.

The project has two main approaches:

1. **Flow-aligned two-pass guidance** — capture the low-resolution H3 denoising trajectory and use it to guide a later learned-upscale/refine pass.
2. **Progressive handoff** — spend early H3 work on a smaller video grid, then switch to the target grid inside one sampling schedule.

For target-input workflows, including Continuum exact-prefix continuation, the standard production path is **MiniMax H3 Progressive Handoff (Target Input)**.

> This is an independent research implementation informed by public work. It does not reproduce MiniMax's closed H3-Regenerate-2K implementation or an unreleased sparse-attention model.

See [CREDITS.md](CREDITS.md) for research and implementation attribution.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate.git
```

Restart ComfyUI.

The core package has no mandatory sibling-node dependency. The intended learned-transfer workflows use the companion [MiniMax H3 Latent Upscaler-Plus](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus).

## Example workflows

Loadable examples are under [`workflows/examples/`](workflows/examples/):

- [`progressive-target-input.workflow.json`](workflows/examples/progressive-target-input.workflow.json) — standard target-input progressive workflow using `source_scale=0.70`, fixed `0.35` handoff, `direction+temporal`, and `learned_3d` transfer through a connected 3D latent-upscaler provider;
- [`progressive-source-input.workflow.json`](workflows/examples/progressive-source-input.workflow.json) — dependency-minimal source-input progressive control using a `1.20x` target handoff.

The target-input example configures the provider with `minimax_h3_latent_upscaler_3d_bf16.safetensors`, CUDA, bf16 precision, and `offload_after_upscale=false`. Matching `.api.json` prompt graphs are included for API execution. Both examples use stock MiniMax H3 loading/conditioning/decoding, `res_multistep`, and no Turbo LoRA. The `workflows/*.overlay.json` files are topology/specification documents rather than loadable ComfyUI workflows; see [`workflows/README.md`](workflows/README.md) for the format distinction.

## Standard Target Input path

Use **MiniMax H3 Progressive Handoff (Target Input)** when the surrounding workflow is already defined at the final target geometry.

The canonical defaults are:

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

`acceleration_weight` and `consistency_weight` are staged values with these defaults: the current guidance implementation does not use them while `guidance_mode=direction+temporal`. Acceleration is active only in `direction+acceleration`; consistency is active only in `downsample_consistency`.

For unprotected or fractional-mask calls, Target Input runs the normal progressive path: private low-grid sampling, one exact handoff probe, predicted-clean video transfer to the target grid, preserved joint-H3 audio, fresh sampler/Spectrum history, and an actual first target-grid H3 evaluation.

For exact protected video prefixes, the exact Native Masked contract takes precedence. Target Input forwards the original target-grid noise/latent/mask/schedule through one ordinary target-grid sampler lifetime. It creates no private low-grid sampler, exact handoff probe, learned-upscaler call, geometry boundary, or progressive history boundary.

On canonical exact-prefix continuation, the sampler-time audio mask additionally uses the validated four-tick 1/256-aligned overlap ramp:

```text
0.203125
0.40234375
0.6015625
0.80078125
```

The video mask remains byte-for-byte unchanged and the original exact video/audio mask remains authoritative at output. `H3_FLOW_AUDIO_GUIDED_OVERLAP_TICKS=0` is the explicit disable/bisect hook.

## Mixed-Grid compatibility window

**MiniMax H3 Progressive Mixed-Grid Continuum [Deprecated]** remains registered for one compatibility release so existing serialized workflows continue to load with their original semantics.

The old `H3ProgressiveMixedGridHandoff` node ID is intentionally not remapped to Target Input. Mixed-Grid is no longer the recommended production path, is no longer part of the production Patcher topology, and is not a release/promotion or compatibility-render gate.

Its historical low-grid suffix, learned-transfer, VDN API-2, attention-measure, suffix-bridge and seam-repair contracts remain documented in [docs/MIXED_GRID_CONTINUUM.md](docs/MIXED_GRID_CONTINUUM.md) and [docs/mixed-grid-seam-repair.md](docs/mixed-grid-seam-repair.md). Runtime compatibility machinery is retained for the deprecation window and can be removed separately afterward.

### Continuum Decode Context

**MiniMax H3 Continuum Decode Context** can be placed immediately before the normal Video VAE Decode. It supplies real future latent context to the native H3 temporal decoder at exact chunk joins while leaving accepted sampling latents, continuation state, masks, audio and the assembly plan unchanged.

This solves a separate decoder-window boundary problem. It is independent of Target Input's sampler-time exact-prefix/audio-overlap behavior and of the deprecated Mixed-Grid seam mechanisms.

See [docs/CONTINUUM_DECODE_CONTEXT.md](docs/CONTINUUM_DECODE_CONTEXT.md).

## Target-Sparse Continuum status

**MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** remains available as a research/control path, but it is **not recommended for production quality**.

It keeps the sampler latent on the target grid and sparsifies only the early H3 hidden-token stream over generated video rows. Because it does not perform the learned latent upscale used by ordinary progressive transfer, real decoded-media testing showed cascading quality errors: skin imperfections, odd clothing changes and spurious background additions could appear and propagate through later continuation.

Target-Sparse remains useful for architectural experiments and controlled comparisons, not as a production continuation path.

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

`direction` remains the conservative guidance recommendation for the explicit two-pass path. Target Input uses the canonical progressive defaults above. `direction+acceleration`, `direction+temporal`, `downsample_consistency`, resolution-aware sigma remapping and the Attention Lab remain available as research controls outside that default configuration.

## Nodes

### Main nodes

| Node | Purpose |
|---|---|
| **MiniMax H3 Flow Trajectory** | Shared trajectory handle for capture, guidance and progressive execution. |
| **MiniMax H3 Trajectory Capture** | Records first-pass H3 predicted-clean trajectory states. |
| **MiniMax H3 Flow-Aligned Regenerate** | Guides a later H3 pass from the matching captured trajectory state. |
| **MiniMax H3 Flow-Aligned Refine State** | Continuum `refine_state` version of flow-aligned guidance. |
| **MiniMax H3 Progressive Handoff** | Generic source-sized progressive resolution handoff. |
| **MiniMax H3 Progressive Handoff (Target Input)** | Standard target-input progressive path; exact protected prefixes use the target-grid fallback. |
| **MiniMax H3 Continuum Decode Context** | Supplies right context to the native temporal VAE at exact Continuum joins. |

### Compatibility / research nodes

| Node | Purpose |
|---|---|
| **MiniMax H3 Progressive Mixed-Grid Continuum [Deprecated]** | One-release compatibility path preserving historical serialized-workflow semantics. |
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
  -> Progressive Target Input
  -> Continuum
```

DiffAid, Untwisting RoPE, Spectrum and VDN are optional integrations.

Important contracts:

- **Spectrum:** actual/forecast provenance and sampler-history boundaries are preserved. Fresh target-grid stages start with an actual H3 evaluation where required.
- **VDN-H3 / Sol-H3:** ordinary production attention backends remain independent of Target Input's exact-prefix fallback. Deprecated Mixed-Grid external-sequence/measure behavior remains compatibility-only.
- **SA-Solver/PECE, SEEDS, ER-SDE, Euler/RES:** sampler objects are preserved; separate sampler lifetimes are used where geometry/history boundaries require them.
- **Audio:** progressive spatial transfer affects video only. Exact-prefix Target Input may guide the last four carried audio ticks during sampler lifetime, while final protected AV ownership remains exact.
- **Learned upscaler:** the normal unprotected Target Input handoff can use `bicubic` or `learned_3d`; the shipped target-input workflow uses `learned_3d`. Exact protected-video fallback performs zero learned-upscaler calls.

## Practical status

The paths with the strongest real-media support include:

- two-pass flow-aligned guidance with the learned upscale/refine workflow;
- Progressive Handoff for unprotected calls;
- Target Input's conservative exact-prefix target-grid fallback, with the four-tick guided-audio overlap carried by the production PR #33 line.

Historical Mixed-Grid validation remains evidence for that retired architecture rather than a current recommendation. The suffix DC bridge removed its brief tone/flash boundary, and its later framing work addressed unequal protected-prefix versus suffix attention sampling measure. The source-space affine/trajectory correction family remains retired because finite corrections only moved the discontinuity to the corrected-to-untouched transition.

Target-Sparse is deliberately not promoted because its no-latent-upscale design produced cascading decoded-media defects in testing.

Quality and speed still depend on prompt, references, geometry, sampler, Spectrum policy, model residency and hardware. Use decoded media rather than structural metrics alone for new workflow variants.

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — wiring and parameter details
- [docs/MIXED_GRID_CONTINUUM.md](docs/MIXED_GRID_CONTINUUM.md) — deprecated Mixed-Grid compatibility contract and historical diagnostics
- [docs/mixed-grid-seam-repair.md](docs/mixed-grid-seam-repair.md) — historical framing-defect evidence chain, retired source warp and attention-measure repair
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
