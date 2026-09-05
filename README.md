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

For **Continuum**, use **MiniMax H3 Progressive Handoff (Target Input)**. Continuum stays configured for the final target size while the node privately runs the early stage on a smaller video grid. If Continuum's Native Masked path exactly protects any video prefix (`mask == 0`), the ordinary Target Input node preserves that stronger contract by skipping the private resize/handoff and running the original target-grid sampler once. This means the conservative node does **not** progressively accelerate exact-prefix continuation chunks; normally only chunk 1 receives the low-grid handoff.

For exact-prefix chunk boundaries, place **MiniMax H3 Continuum Decode Context**
immediately before Video VAE Decode, after all latent processing. It supplies
real future context to H3's overlapping decoder at the join. Keep the original
assembly plan and audio path. See [wiring and limitations](docs/CONTINUUM_DECODE_CONTEXT.md).

All three Continuum progressive nodes expose `suffix_dc_bridge`, enabled by default on canonical whole-frame exact-prefix boundaries. The bridge changes only the first generated suffix latent token with a per-channel DC offset; it never edits the authoritative prefix or later suffix tokens. Mixed-Grid derives the offset from its discarded learned-upscaler prefix, while Target Input fallback and Target-Sparse derive it from the first actual H3 predicted-clean boundary before native mask restoration. Disable it only for matched seam A/B testing.

### 3. Experimental exact-prefix Continuum acceleration

**MiniMax H3 Progressive Mixed-Grid Continuum [Experimental]** samples a real
low-resolution continuation suffix while H3 attends to the original target-grid
protected prefix. It then performs an exact probe, learned 3D suffix transfer,
and fresh full-grid refinement. Requires `learned_3d`, an upscaler provider, and
VDN external-sequence API 2 when VDN is enabled. See
[mixed-grid setup and validation](docs/MIXED_GRID_CONTINUUM.md). GPU/media
acceptance is pending.

**MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** investigates continuation-chunk acceleration without resizing the protected Native Masked prefix.

On an exact-prefix chunk:

```text
full target-grid sampler latent + mask
               |
               +-- exact protected video rows: keep every row
               +-- generated video rows: keep a coarse target-grid anchor lattice
               +-- text / refs / audio: keep all rows
               |
         reduced early H3 hidden stream
               |
       lift hidden video field to full target grid
       + overwrite every retained row exactly
               |
       fresh full-grid H3 sampler lifetime
```

The sampler latent, original mask, protected prefix, and retained RoPE coordinates stay on the target grid. Only the **early transformer hidden-token stream** is reduced. The full hidden sequence is restored before H3's native final layer and before Spectrum's final-block observation. Because the early dense-attention computation is approximated, this is not semantic parity with ordinary H3 and remains opt-in until real decoded-media and timing gates pass.

For exact-prefix chunks, `source_scale` or explicit source dimensions control the coarse **anchor density**; they do not describe a private sampler latent geometry. The learned handoff provider is not called on these exact-prefix chunks. Chunk 1, which has no protected continuation prefix, still uses the normal Target Input low/probe/high path and can still use `learned_3d` if configured.

### 4. Optional learned handoff

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

For the conservative validated path, keep this model-patch order when the surrounding H3 patches are present:

```text
DiffAid
  -> Untwisting RoPE
  -> Spectrum
  -> Progressive Handoff (Target Input)
  -> Continuum
```

DiffAid, Untwisting RoPE, and Spectrum are optional external integrations, not requirements of this package. Omit any that are not part of your workflow; when Spectrum is used, keep the progressive node downstream of it.

Create one **MiniMax H3 Flow Trajectory** and connect it to **Progressive Handoff (Target Input)**. A separate **Trajectory Capture** node is not required on this path because the progressive wrapper captures its private trajectory internally.

Continuum remains on the target geometry. Only chunk 1 or other unprotected Target Input calls use a private lower-resolution sampler grid. Exact Native Masked continuation chunks stay on the target geometry; the conservative node uses a full-grid fallback, while the separate experimental target-sparse node reduces only early H3 hidden tokens.

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
| **MiniMax H3 Progressive Handoff (Target Input)** | Target-sized/Continuum-safe variant. Unprotected calls run the early H3 stage privately on a smaller grid; exact protected video prefixes conservatively fall back to one ordinary target-grid sampler lifetime. Supports optional `learned_3d` transfer on the normal handoff path. |

### Experimental research nodes

| Node | What it is for |
|---|---|
| **MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** | Exact-prefix Continuum experiment: keeps the full target-grid sampler state and every protected video row while reducing only early H3 hidden-token computation over generated video rows. Structural/CI validation exists; decoded-media quality and speed are not yet established. |
| **MiniMax H3 Progressive Mixed-Grid Continuum [Experimental]** | Real low-grid continuation suffix with original target-grid prefix conditioning, exact probe, learned suffix upscale, and fresh target refinement. Requires the learned upscaler and VDN API 2 when VDN is enabled. GPU/media acceptance is pending. |
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
- **Progressive Handoff (Target Input)** is functional for chunk 1/unprotected calls and uses a semantically safe target-grid fallback for Native Masked exact-prefix continuation chunks;
- **direction-only guidance** remains the conservative recommendation;
- **learned `learned_3d` handoff** has shown a clear benefit over bicubic for aggressive transitions in the tested ~1 MP workflow;
- **target-sparse exact-prefix Continuum acceleration is not yet promoted**: the implementation is structurally tested, but real multi-chunk decoded-media and timing validation remain required;
- the resolution-aware sigma, temporal, acceleration, reference-budget, and attention experiments remain research features rather than promoted defaults.

The current observed performance comparison for the learned progressive path is documented separately in [docs/PERFORMANCE.md](docs/PERFORMANCE.md). It is workflow-specific and not a universal speed or quality claim.

## Compatibility

The implementation is designed around native MiniMax H3 joint audio/video sampling rather than treating video as an isolated tensor path.

- **Continuum:** use **Progressive Handoff (Target Input)** for the conservative path. Exact Native Masked continuation chunks use target-grid fallback. **Progressive Mixed-Grid Continuum [Experimental]** adds real low-grid suffix generation and learned transfer; **Progressive Target-Sparse Continuum [Experimental]** retains the previous sparse-row experiment as a control. Both require explicit runtime validation. **Flow-Aligned Refine State** is available for explicit two-pass Continuum refinement.
- **Spectrum:** actual/forecast provenance is preserved. Feature-history state is reset across a progressive boundary, and the first full target-grid call is forced actual. The target-sparse wrapper restores the full target hidden stream before Spectrum's final-block observation.
- **SA-Solver/PECE, SEEDS, ER-SDE, Euler/RES:** sampler objects are preserved; progressive stages use separate sampler lifetimes where required by geometry or target-sparse history boundaries.
- **DiffAid / Untwisting RoPE:** keep these upstream of the progressive wrapper when used. The target-sparse path retains all non-video rows and reduces H3 target-video modulation metadata consistently; runtime media validation of the complete external-patch stack is still required.
- **Learned H3 latent upscaler:** optional provider for `learned_3d`; not bundled with this repository. It is not invoked for an exact-prefix target-sparse continuation stage.

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
- Normal progressive handoff changes only the **video** spatial grid; audio is never spatially resized. Target-sparse exact-prefix mode does not resize the sampler video grid at all.
- Progressive sampling requires a complete H3 sigma schedule whose absolute flow origin is known.
- Arbitrary external sampler RNG/history closures cannot be safely carried across a progressive split and may be rejected.
- Mutable capture/guidance/progressive state currently fails closed for unsupported parallel multi-GPU model-call ordering.
- The target-sparse path approximates early full-dense H3 transformer computation. Passing structural tests does not establish decoded-media quality or a net speedup.
- Quality and speed depend on prompt, references, geometry, sampler, Spectrum policy, model residency, and hardware. Use decoded media rather than metrics alone as the final quality test.
