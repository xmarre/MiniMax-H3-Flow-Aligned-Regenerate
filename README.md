# MiniMax H3 Flow-Aligned Regenerate

Training-free ComfyUI nodes for reusing low-resolution MiniMax H3 generation structure while moving toward a higher-resolution result.

The project has two main goals:

- **guide a high-resolution/refine pass with the actual low-resolution H3 denoising trajectory**, instead of treating the low-resolution result as only a final latent;
- **move from a smaller video grid to the final grid inside one sampling schedule**, so more H3 work can happen at the cheaper resolution before the expensive high-resolution stage begins.

Native H3 generation directly on the final high-resolution grid can be more expensive and, depending on the workflow, less robust than generating smaller and refining after a learned latent upscale. The ordinary upscale/refine approach solves that by starting a second H3 sampling pass. These nodes explore two alternatives: make that second pass reuse the first pass's trajectory, or avoid a complete second pass by carrying one trajectory across a controlled spatial-resolution handoff.

> [!IMPORTANT]
> This is an independent research implementation informed by public work. It does **not** reproduce MiniMax's closed H3-Regenerate-2K implementation or its unreleased sparse-attention topology.

Full node-by-node research and implementation attribution is in [CREDITS.md](CREDITS.md).

## What the nodes do

### 1. Flow-aligned second-pass guidance

The low-resolution H3 pass is captured as a time-indexed trajectory. A later high-resolution or learned-refine pass can then be guided toward the matching low-resolution predicted-clean state at the same flow coordinate.

```text
low-resolution H3
      |
      +-> Trajectory Capture -> H3_FLOW_TRAJECTORY
                                  |
learned upscale / refine input ---+
                                  |
                         Flow-Aligned Regenerate
                                  |
                         high-resolution H3
```

This keeps the existing two-pass workflow, but lets the second H3 pass reuse information from the full first-pass trajectory rather than only the upscaled endpoint.

For H3 Continuum refinement, **Flow-Aligned Refine State** applies the same guidance directly to each Continuum `refine_state`.

### 2. Progressive resolution handoff

The progressive nodes keep early denoising on a smaller video grid and switch to the target grid later in the same schedule.

```text
early schedule                     late schedule

private/source grid  ------------>  target grid
        H3            handoff           H3
```

The handoff is not a blind resize of noisy state. The wrapper performs an exact low-grid probe, transfers the predicted-clean video state, reconstructs the target-grid conditional state, resets sampler/forecast histories that cannot safely cross the geometry change, and continues sampling at the final resolution.

For **Continuum**, use **MiniMax H3 Progressive Handoff (Target Input)**. Continuum stays configured for the final target size while the node privately runs the early stage on a smaller video grid. If Continuum's native mask exactly protects any video prefix (`mask == 0`), Flow automatically preserves that stronger contract by skipping the private resize/handoff and running the original target-grid sampler once.

### 3. Optional learned handoff

**Progressive Handoff (Target Input)** supports:

- `bicubic` — built-in compatibility path;
- `learned_3d` — one learned clean-video spatial transfer using the companion [MiniMax H3 Latent Upscaler](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler).

The learned provider replaces only the clean-video spatial transfer at the handoff. It does not add a second H3 sampling pass, does not spatially transform audio, and adds no H3 transformer NFE by itself.

## Install

From `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate.git
```

Restart ComfyUI.

The core package has no runtime dependency on the sibling H3 custom nodes. The optional `learned_3d` transfer requires the companion latent-upscaler package above.

## Recommended Continuum path

When the surrounding H3 patches used by the validated workflow are present, keep this model-patch order:

```text
DiffAid
  -> Untwisting RoPE
  -> Spectrum
  -> Progressive Handoff (Target Input)
  -> Continuum
```

DiffAid, Untwisting RoPE, and Spectrum are optional external integrations, not requirements of this package. Omit any that are not part of your workflow; when Spectrum is used, keep Progressive Handoff downstream of it.

Create one **MiniMax H3 Flow Trajectory** and connect it to **Progressive Handoff (Target Input)**. A separate **Trajectory Capture** node is not required on this path because the progressive wrapper captures its private low-grid trajectory internally.

Continuum remains on the target geometry. Only the video grid changes internally; audio remains in the native joint H3 path.

Detailed parameter guidance, tested starting points, `source_scale` behavior, handoff selection, learned-provider setup, and sampler requirements are in [docs/USAGE.md](docs/USAGE.md).

## Two-pass path

Use the explicit two-pass nodes when you want to keep an existing low-resolution generation + learned upscale/refine workflow:

1. Create one **Flow Trajectory**.
2. Patch the first-pass model with **Trajectory Capture**.
3. Run the low-resolution generation.
4. Perform the existing learned latent upscale / refine initialization.
5. Patch the second-pass model with **Flow-Aligned Regenerate**.
6. For Continuum-integrated refinement, patch the emitted `refine_state` with **Flow-Aligned Refine State** instead.

The same trajectory handle must be used by capture and guidance.

## Nodes

### Core generation nodes

| Node | What it is for |
|---|---|
| **MiniMax H3 Flow Trajectory** | Shared mutable trajectory handle used by capture, guidance, and progressive sampling. Storage can live in system RAM or VRAM. |
| **MiniMax H3 Trajectory Capture** | Patches an H3 model so first-pass predicted-clean trajectory states and provenance are recorded. |
| **MiniMax H3 Flow-Aligned Regenerate** | Guides a later H3 pass from the matching captured low-resolution trajectory. |
| **MiniMax H3 Flow-Aligned Refine State** | Continuum version of Flow-Aligned Regenerate; patches each `H3_CONTINUUM_REFINE_STATE`. |
| **MiniMax H3 Progressive Handoff** | Starts from a source-sized workflow input and grows the video grid to a target resolution during sampling. |
| **MiniMax H3 Progressive Handoff (Target Input)** | Target-sized/Continuum-safe variant: the public workflow stays at final geometry while the early H3 stage runs privately on a smaller grid. Supports optional `learned_3d` transfer. |

### Experimental research nodes

| Node | What it is for |
|---|---|
| **MiniMax H3 Refine Target Geometry [Experimental]** | Mirrors the companion learned-refiner's target sizing so schedule experiments can use the same geometry metadata. It does not upscale latents. |
| **MiniMax H3 Resolution-Aware Sigmas [Experimental]** | Tests a resolution-dependent remap of the downstream learned-refine sigma schedule. Default/recommended mode remains `off`. |
| **MiniMax H3 Reference Budget [Experimental]** | Reports direct-reference row growth and provides a guarded experimental direct-reference cap. |
| **MiniMax H3 Attention Lab [Experimental]** | Output-neutral H3/VDN retention diagnostics, a dense-mask VDN topology oracle, and the earlier guarded spatial-local experiment. No path is presented as production sparse acceleration. |

### Diagnostics

| Node | What it is for |
|---|---|
| **MiniMax H3 Runtime Metrics Probe** | Passive sampler/model-call instrumentation without enabling trajectory guidance or progressive handoff. |
| **MiniMax H3 Metrics JSON** | Saves structured H3/Spectrum/sampler/geometry metrics to a JSON artifact. |

## Guidance modes

| Mode | Purpose | Current posture |
|---|---|---|
| `direction` | Low-frequency alignment toward the matched captured predicted-clean state | **Preferred/default guidance path** |
| `direction+acceleration` | Adds adjacent denoising-time velocity-change alignment inspired by HiFlow | Experimental; structurally valid, no consistent media advantage established |
| `direction+temporal` | Adds conservative adjacent-frame latent correspondence | Experimental; functioning, currently neutral in matched decoded-media testing |
| `downsample_consistency` | Compares the target clean estimate against the captured low-grid state after downsampling | Experimental; measurable but not currently preferred |
| `off` | Disable trajectory correction while retaining the surrounding wrapper/metrics path | Control/debug use |

Exact settings and evidence are intentionally kept out of the main README. See [docs/USAGE.md](docs/USAGE.md) for practical configuration and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the decoded-media evidence ledger.

## Current practical status

The core paths are usable and have been exercised with real decoded H3 media:

- **two-pass flow-aligned guidance** is functional with the learned upscale/refine workflow;
- **Progressive Handoff (Target Input)** is the main single-schedule coarse-to-fine path for Continuum;
- **direction-only guidance** remains the conservative recommendation;
- **learned `learned_3d` handoff** has shown a clear benefit over bicubic for aggressive transitions in the tested ~1 MP workflow;
- the resolution-aware sigma, temporal, acceleration, reference-budget, and attention experiments remain research features rather than promoted defaults.

The current observed performance comparison for the learned progressive path is documented separately in [docs/PERFORMANCE.md](docs/PERFORMANCE.md). It is workflow-specific and not a universal speed or quality claim.

## Compatibility

The implementation is designed around native MiniMax H3 joint audio/video sampling rather than treating video as an isolated tensor path.

- **Continuum:** use **Progressive Handoff (Target Input)** for multi-chunk progressive generation. **Flow-Aligned Refine State** is available for explicit two-pass Continuum refinement.
- **Spectrum:** actual/forecast provenance is preserved. Feature-history state is reset across a progressive geometry boundary, and the first target-grid call is forced actual.
- **SA-Solver/PECE, SEEDS, ER-SDE, Euler/RES:** sampler objects are preserved; progressive stages use separate sampler lifetimes where required by the geometry change.
- **DiffAid / Untwisting RoPE:** keep these upstream of the progressive wrapper when used.
- **Learned H3 latent upscaler:** optional provider for `learned_3d`; not bundled with this repository.

More detailed wiring rules and failure conditions are in [docs/USAGE.md](docs/USAGE.md).

## Documentation

- **[docs/USAGE.md](docs/USAGE.md)** — practical wiring, settings, handoff behavior, guidance modes, learned transfer, metrics, and compatibility rules.
- **[CREDITS.md](CREDITS.md)** — node-by-node research and implementation attribution.
- **[docs/RESEARCH.md](docs/RESEARCH.md)** — research-transfer rationale and empirical conclusions.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — internal H3 contracts and implementation architecture.
- **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** — decoded-media validation ledger and benchmark protocol.
- **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** — measured workflow-level timing evidence and limits.
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — development setup, CI/test scope, source pins, and maintenance rules.
- **[RELEASE_NOTES.md](RELEASE_NOTES.md)** — release history.
- **[LICENSE](LICENSE)** — Apache License 2.0 terms for this project.

## License

MiniMax H3 Flow-Aligned Regenerate is licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 xmarre.

Research references and implementation provenance are documented in [CREDITS.md](CREDITS.md). Those references are attribution, not relicensing: third-party projects retain their own copyrights and licenses, and referenced research repositories are not bundled into this project unless explicitly stated otherwise.

## Scope and limitations

- This is research-grade tooling, not an official MiniMax implementation.
- Progressive handoff changes only the **video** spatial grid; audio is never spatially resized.
- Progressive sampling requires a complete H3 sigma schedule whose absolute flow origin is known.
- Arbitrary external sampler RNG/history closures cannot be safely carried across a geometry reset and may be rejected.
- Mutable capture/guidance/progressive state currently fails closed for unsupported parallel multi-GPU model-call ordering.
- Quality and speed depend on prompt, references, geometry, sampler, Spectrum policy, model residency, and hardware. Use decoded media rather than metrics alone as the final quality test.
