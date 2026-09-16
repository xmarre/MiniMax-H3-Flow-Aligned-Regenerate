# Usage guide

This document describes the current production-facing wiring and parameter contracts. Historical Mixed-Grid and Target-Sparse experiments remain documented separately as evidence/compatibility material.

## Generation paths

The repository exposes two main generation paths:

1. **Flow-aligned two-pass guidance** — capture a lower-resolution H3 trajectory and use it to guide a later learned-upscale/refine pass.
2. **Progressive handoff** — keep the early part of one schedule on a smaller video grid, then transition once to the final target grid.

For target-input workflows, including Continuum, use **MiniMax H3 Progressive Handoff (Target Input)**.

## Standard Target Input wiring

Use this patch order when the optional integrations are present:

```text
DiffAid
  -> Untwisting RoPE
  -> Spectrum
  -> Progressive Handoff (Target Input)
  -> Continuum
```

Create one **MiniMax H3 Flow Trajectory** and connect it to Target Input. Connect **MiniMax H3 Latent Upscaler Provider (3D)** when using `handoff_transfer=learned_3d`.

Do not add a separate Trajectory Capture node to the progressive path. The wrapper owns its private low-grid capture when that path is eligible.

### Canonical controls

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

`acceleration_weight` and `consistency_weight` are inactive while `guidance_mode=direction+temporal`. Acceleration is used only by `direction+acceleration`; consistency is used only by `downsample_consistency`.

The intended learned-upscaler provider configuration is:

```text
model_name            = minimax_h3_latent_upscaler_3d_bf16.safetensors
device                = cuda
precision             = bf16
offload_after_upscale = false
```

The provider is supplied by `xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus`.

## Unprotected progressive execution

When exact protected video values are absent, Target Input runs the progressive path:

1. derive a private lower-resolution video grid;
2. run the early sampler lifetime on that grid;
3. run one exact low-grid H3 handoff probe;
4. obtain the predicted-clean video state;
5. transfer that clean video state to final target H/W;
6. reconstruct the target conditional state using deterministic target noise;
7. preserve audio on its existing joint-H3 path;
8. start a fresh target-grid sampler/Spectrum lifetime;
9. require the first target-grid H3 call to be actual.

The geometry/history boundary is deliberate: solver history from the low grid is not valid after the spatial transition.

### Source geometry

`source_mode=scale` treats `source_scale` as a linear H/W scale, not an area fraction:

```text
source_MP ~= target_MP * source_scale^2
```

For approximately fixed source area while changing target MP:

```text
source_scale ~= sqrt(desired_source_MP / target_MP)
```

Resolved H3 geometry is snapped to the model-safe latent grid, so nearby decimal values can resolve to the same physical dimensions. Use the `handoff_plan` metrics event when exact geometry matters.

### Handoff position

`handoff_selection=fixed` uses `handoff_coordinate` and snaps it to an available schedule point. `auto_compute` remains available for controlled experiments.

Progressive handoff requires a complete H3 1-to-0 schedule. Arbitrary partial tail schedules are rejected because the wrapper cannot infer their absolute H3 flow coordinate safely.

## Transfer modes

### `learned_3d`

`learned_3d` is the intended production transfer for ordinary progressive Target Input execution. It changes only the exact-probe **clean video** spatial transfer.

It does not change:

- audio;
- target noise semantics;
- masks;
- the required fresh target-grid history boundary;
- the first target-grid actual-call requirement.

One learned CNN inference is added per progressive handoff. It does not add an H3 transformer NFE.

### `bicubic`

`bicubic` remains a dependency-free control/compatibility transfer. It directly resizes the predicted-clean video state before target-state reconstruction.

## Exact protected video: conservative fallback

If the prepared video denoise mask contains any exact-zero value, the exact Native Masked contract takes precedence over progressive resizing.

Target Input then forwards the original target-grid noise, latent, mask, schedule, sampler, callback, conditioning and shape metadata through **one ordinary target-grid sampler lifetime**.

That path has:

```text
private low-grid sampler       = no
exact handoff probe            = no
learned-upscaler calls         = 0
geometry boundary              = no
progressive history boundary   = no
progressive guidance           = no
```

A `progressive_target_fallback` metrics event records the path. Audio-only exact zeros do not independently trigger the video fallback; fractional video masks remain intentional blends rather than exact protection.

### Four-tick guided audio overlap

On a canonical exact-prefix continuation, the fallback guides only the last four carried **audio latent ticks** during sampler lifetime. The audio latent rate is 40 Hz, so the default window is about 100 ms.

Core MiniMax-H3 uses a 1/256 mask grid. The four values are:

```text
0.203125
0.40234375
0.6015625
0.80078125
```

The video mask is byte-for-byte unchanged. The caller-owned original exact mask is retained separately and every originally protected video/audio value is restored exactly at the sampler boundary.

All-generated first chunks, fully protected audio, and exact prefixes too short to leave at least one fully protected audio tick are no-ops. Other non-canonical partially protected audio layouts fail closed rather than being guessed.

`H3_FLOW_AUDIO_GUIDED_OVERLAP_TICKS=0` explicitly disables the overlap for controlled bisection. Values 1-16 are accepted when deliberately testing a different width.

## Continuum

Continuum remains configured for the final target width/height. The standard Target Input node sits in the model patch chain before Continuum and obeys the same final-grid session contract.

Exact protected Continuum chunks use the conservative target-grid fallback above. Unprotected calls may use the normal progressive handoff.

### Continuum Decode Context

**MiniMax H3 Continuum Decode Context** is independent of progressive sampling. Place it immediately before the normal Video VAE Decode when using the matching Continuum physical-group contract.

It supplies real future latent context to the native temporal decoder at exact chunk joins. It does not change accepted sampling latents, continuation state, masks, audio or assembly ownership.

See [CONTINUUM_DECODE_CONTEXT.md](CONTINUUM_DECODE_CONTEXT.md).

## Mixed-Grid compatibility window

**MiniMax H3 Progressive Mixed-Grid Continuum [Deprecated]** remains registered for one compatibility release.

The old node ID `H3ProgressiveMixedGridHandoff` is intentionally preserved as a **distinct implementation**. It is not aliased/remapped to Target Input because doing so would silently change existing serialized workflow behavior.

During this window:

- old serialized workflows still load with their original Mixed-Grid semantics;
- Mixed-Grid is not recommended for new workflows;
- Mixed-Grid is not part of the production Patcher topology;
- its weighted attention/external-sequence companions are not release gates;
- no Mixed-Grid compatibility render is required;
- runtime machinery remains only so compatibility is real rather than a broken placeholder.

The runtime/measure machinery is scheduled for removal after the compatibility window. Historical architecture and media evidence remain in [MIXED_GRID_CONTINUUM.md](MIXED_GRID_CONTINUUM.md), [mixed-grid-seam-repair.md](mixed-grid-seam-repair.md), `BENCHMARKS.md`, `PERFORMANCE.md`, and release history.

## Target-Sparse compatibility/research path

**MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** remains available as an architectural control. It keeps sampler state on the target grid while reducing only the early H3 hidden-token stream.

Real decoded-media testing exposed cascading quality defects on that path, including skin/clothing/background errors. It is not a production recommendation.

## Flow-aligned two-pass guidance

The explicit two-pass path keeps a conventional low-resolution generation + learned upscale/refine workflow and adds trajectory guidance to the later H3 pass.

Generic wiring:

1. create one **MiniMax H3 Flow Trajectory**;
2. apply **MiniMax H3 Trajectory Capture** to the low-resolution model;
3. run the first pass;
4. perform the learned latent upscale/refine initialization;
5. apply **MiniMax H3 Flow-Aligned Regenerate** to the later model;
6. use the same trajectory handle for capture and guidance.

For Continuum integrated refine state, use **MiniMax H3 Flow-Aligned Refine State** with the matching trajectory handle.

### Forecast capture

`capture_forecasts=False` is the conservative default. Exact H3 evaluations are preferred trajectory anchors. If forecast capture is deliberately enabled for research, forecast provenance remains explicit and is not treated as an exact evaluation.

## Guidance modes

- `direction` — low-frequency alignment toward the time-matched captured predicted-clean state.
- `direction+acceleration` — adds denoising-time velocity-change alignment.
- `direction+temporal` — adds bounded adjacent-frame correspondence on the captured low-grid clean trajectory. This is the Target Input UI default but previous matched tests did not establish universal perceptual superiority over direction-only.
- `downsample_consistency` — compares target clean state against the low-grid reference after downsampling; historical tests did not establish a useful general media improvement.
- `off` — disables trajectory correction while leaving wrapper/metrics infrastructure available.

## Resolution-aware refine SIGMAS

**MiniMax H3 Resolution-Aware Sigmas [Experimental]** is a separate downstream learned-refine experiment, not part of standard progressive Target Input execution.

`mode=off` remains the recommendation. The completed matched E0/E1 pair did not show a relevant quality improvement from the remap.

## Metrics

**MiniMax H3 Runtime Metrics Probe** installs passive logical-call / actual-vs-forecast / layout / wall-time instrumentation without enabling trajectory capture, guidance, progressive handoff or attention changes.

**MiniMax H3 Metrics JSON** writes the accumulated structured metrics artifact.

For behavioral changes, treat structural counters and unit tests as necessary but insufficient. Decoded media remains the acceptance evidence for visual/audio quality.
