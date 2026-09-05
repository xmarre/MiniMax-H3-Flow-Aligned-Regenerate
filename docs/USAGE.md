# Usage guide

This document contains the detailed wiring and configuration information intentionally kept out of the main README.

The repository exposes two main generation paths:

1. **flow-aligned two-pass guidance** — capture a low-resolution H3 trajectory and use it to guide a later high-resolution/refine pass;
2. **progressive handoff** — keep the early part of one schedule on a smaller video grid, then transition once to the target grid and continue sampling.

For H3 Continuum, **Progressive Handoff (Target Input)** is the preferred progressive topology because the public workflow/session stays at target geometry while only the node's private early video state is smaller.

## Common concepts

### Flow trajectory

**MiniMax H3 Flow Trajectory** is a mutable execution handle shared by the nodes participating in one generation. It stores captured predicted-clean trajectory samples and provenance.

- `storage=system_ram` keeps captured trajectory tensors off VRAM where practical;
- `storage=vram` avoids host transfer at the cost of additional VRAM;
- `max_runs` bounds retained trajectory runs inside one handle.

Create one handle per workflow execution and reuse that same handle across capture/guidance nodes that are supposed to communicate.

### Flow coordinate

Guidance and progressive handoff match states by H3's shared flow coordinate rather than by raw sampler call index. This matters for samplers such as SA-Solver/PECE where predictor/corrector calls can occur at the same sigma and where logical calls are not equivalent to distinct denoising coordinates.

### Video vs audio

H3 is a joint audio/video model. The progressive path changes only the **video spatial grid**. Audio is never spatially resized and remains on the native joint H3 path.

## Progressive Handoff (Target Input)

### Wiring with Continuum

Use this order for the model patch chain:

```text
DiffAid
  -> Untwisting RoPE
  -> Spectrum
  -> Progressive Handoff (Target Input)
  -> Continuum
```

Create one **Flow Trajectory** and connect it to **Progressive Handoff (Target Input)**.

Do **not** add a separate Trajectory Capture node on this path. The progressive wrapper captures the private low-grid trajectory internally.

Continuum remains configured for the final target width/height. The wrapper creates a private smaller video state for the early stage, performs the handoff, rebuilds target-grid conditioning, and continues at the final grid.

If the prepared video denoise mask contains any exact-zero values, the exact native-masked contract takes precedence over progressive resizing. Flow forwards the original target noise, latent, mask, schedule, sampler, callback, and shape metadata through one ordinary target-grid sampler lifetime. No private noise, spatial transfer, exact probe, history boundary, or progressive guidance is used. A `progressive_target_fallback` metrics event records this path. Audio-only zero masks do not trigger it, and fractional video masks remain intentional blends rather than exact protection.

### Exact-prefix suffix DC bridge

The Continuum-facing **Progressive Handoff (Target Input)**, **Progressive Target-Sparse Continuum**, and **Progressive Mixed-Grid Continuum** nodes expose `suffix_dc_bridge`. It defaults to **on** and is only defined for a canonical whole-frame exact prefix followed by a whole-frame generated suffix. Partial/fractional/noncontiguous masks are skipped rather than guessed.

The correction is intentionally one latent token wide. Flow computes a per-batch/per-channel spatial-mean offset and applies it only to the first generated suffix token. The authoritative prefix and every later suffix token remain unchanged at bridge application. There is no video-space crossfade and no additional H3 transformer evaluation.

The calibration source depends on the progressive topology:

- **Mixed-Grid:** use the discarded learned-upscaler prefix to preserve the learned transfer's native DC boundary relation when the exact target-grid prefix is restored.
- **Target Input fallback / Target-Sparse high stage:** use the first **actual** H3 predicted-clean output at the exact boundary before native inpaint masking restores the protected prefix. Spectrum forecasts are not used as calibration; the bridge waits for the first actual model output. Target-Sparse already requires its first high-stage call to be actual. The conservative fallback reports `exact_prefix_suffix_dc_bridge_skipped` with `reason=no_actual_model_output` if a surrounding policy never produces an actual H3 evaluation.

The mixed-grid placement passed the matched decoded-media seam test that motivated enabling the control by default. The generalized Target Input and Target-Sparse placements are structurally covered by CI but should still be included in matched decoded-media validation for those topologies.

### What happens at the handoff

At the selected handoff point the wrapper:

1. completes an exact low-grid H3 probe at the boundary;
2. obtains the predicted-clean video state;
3. transfers that clean video state to the target H/W;
4. reconstructs the target conditional state with deterministic target noise;
5. keeps audio on its existing path;
6. resets sampler history that is invalid after a spatial-shape change;
7. resets Spectrum feature history;
8. starts the target-grid stage with a forced actual H3 evaluation.

This is why the progressive path is not equivalent to resizing a noisy latent in place.

### Source geometry

`source_mode` controls how the private early-stage grid is selected:

- `scale` — derive the private grid from the final target geometry;
- `pixels` — request an explicit private source width/height, which is then snapped to H3-safe latent geometry.

With `source_mode=scale`, `source_scale` is a **linear width/height scale**, not an area or megapixel fraction:

```text
source_MP ~= target_MP * source_scale^2
```

For example, a `source_scale` of `0.70` means the private H/W are roughly 70% of the final dimensions before H3 geometry snapping, so the private area is roughly 49% of the target area.

If the goal is to keep approximately the same private area while increasing target megapixels:

```text
source_scale ~= sqrt(desired_source_MP / target_MP)
```

A rough geometry-only guide for keeping the private stage near ~0.54 MP is:

| Target MP | Approx. `source_scale` |
|---:|---:|
| 0.84 | 0.80 |
| 0.90 | 0.78 |
| 1.00 | 0.74 |
| 1.10 | 0.70 |
| 1.20 | 0.67 |

These are starting points, not quality guarantees. H3-safe geometry snapping can make nearby decimal values resolve to the same or non-linearly different W/H. Inspect the `handoff_plan` metrics event when exact resolved geometry matters.

### Handoff position

`handoff_selection` supports:

- `fixed` — use the requested `handoff_coordinate` and snap it to an available schedule point;
- `auto_compute` — derive a geometry-aware earlier/later handoff from the source/target area relationship.

`fixed` at `handoff_coordinate=0.35` is the current tested reference. `auto_compute` is functional but did not show a clear decoded-media advantage in the matched smoke used during development, so it is not the preferred quality setting.

### Sigma schedule requirement

Progressive handoff requires a complete H3 schedule whose absolute flow origin is known. Use a full 1-to-0 schedule.

Partial low-sigma refinement schedules are rejected for progressive sampling because the wrapper cannot safely infer the original absolute flow coordinate from an arbitrary tail schedule.

## Handoff transfer modes

### `bicubic`

`bicubic` is the built-in compatibility/default transfer. It resizes the exact-probe predicted-clean video state to the target grid before target-state reconstruction.

### `learned_3d`

The optional learned path uses the companion repository:

https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler

Create **MiniMax H3 Latent Upscaler Provider (3D) [Experimental]**, connect its `H3_LATENT_UPSCALER` output to the progressive Target Input node, and select:

```text
handoff_transfer = learned_3d
```

The learned provider changes only the exact-probe **clean video** spatial transfer.

It does not change:

- audio;
- target noise semantics;
- the conditional re-noise equation;
- masks;
- sampler/Spectrum history boundaries;
- the mandatory first target-grid actual call;
- H3 model-call semantics.

One learned CNN inference is added per physical chunk and it adds no H3 transformer NFE by itself.

Provider mode fails instead of silently falling back if the configured learned-upscaler device/model is unavailable.

### Current learned-transfer evidence

Around a ~1 MP target, decoded-media testing found a meaningful benefit from `learned_3d` when the private-to-target transition was aggressive enough for bicubic to introduce visible body/spatial handoff artifacts.

The strongest tested quality/compute point for that specific difficult-motion prompt used approximately:

```text
source_scale = 0.70
handoff_selection = fixed
handoff_coordinate = 0.35
handoff_transfer = learned_3d
outer_steps = 10 SA-Solver-PECE
```

The final gate resolved approximately `832x640 -> 1184x896`. A `0.65` source scale at the same general target regime began losing reference likeness and tonal stability in that prompt.

This is not a universal source-scale optimum. Larger source scales are safer when source-grid fidelity is more important than reducing high-resolution work.

The latest learned-transfer media runs used `direction+acceleration`; that fact must **not** be interpreted as evidence that acceleration guidance is better than direction-only.

See [BENCHMARKS.md](BENCHMARKS.md) and [PERFORMANCE.md](PERFORMANCE.md) for the exact evidence and timing accounting.

## Progressive Handoff (source-input variant)

**MiniMax H3 Progressive Handoff** is the source-sized version of the same general idea. The incoming workflow starts on the smaller grid and the node grows the video state toward either:

- a target scale; or
- explicit target pixel dimensions.

Use this variant when the surrounding workflow can legitimately begin at source geometry.

For Continuum, prefer **Progressive Handoff (Target Input)** because Continuum's session, native masks, crop/decode bookkeeping, and external graph geometry are target-sized.

## Flow-aligned two-pass guidance

The explicit two-pass path preserves an existing low-resolution generation + learned upscale/refine workflow and adds trajectory guidance to the second H3 pass.

### Generic two-pass wiring

1. Create one **MiniMax H3 Flow Trajectory**.
2. Patch the low-resolution H3 model with **MiniMax H3 Trajectory Capture**.
3. Run the first-pass generation.
4. Perform the existing learned latent upscale / second-pass initialization.
5. Patch the high-resolution H3 model with **MiniMax H3 Flow-Aligned Regenerate**.
6. Use the same trajectory handle for capture and guidance.

When one combined metrics artifact is wanted, feed the `metrics` output from Trajectory Capture into the downstream Flow-Aligned Regenerate node.

### Continuum + integrated learned refinement

For Continuum's integrated learned-refine path:

1. place **Trajectory Capture** as the final H3 model patch before Continuum so the emitted `refine_state` carries the correct trajectory provenance;
2. run Continuum's first pass;
3. feed each `H3_CONTINUUM_REFINE_STATE` through **Flow-Aligned Refine State** using the same trajectory handle;
4. feed the patched refine state into the companion integrated latent upscaler/refiner.

The refine-state adapter checks the captured run identity and conditioning provenance and fails closed if the state cannot be matched safely.

### Capturing Spectrum forecasts

`capture_forecasts=False` is the conservative default. Exact H3 evaluations are the preferred trajectory anchors.

When forecast capture is deliberately enabled for research, forecast provenance remains marked and is not silently treated as an exact model evaluation.

## Guidance modes

### `direction`

The preferred mode. It aligns the target predicted-clean estimate toward the matched captured low-resolution predicted-clean state in low spatial frequencies and decays the correction through the later high-resolution stage.

Progressive nodes currently default to:

```text
direction_weight = 0.25
low_frequency_cutoff = 0.25
```

The explicit two-pass regenerate node exposes a slightly stronger public default (`direction_weight=0.35`). Treat these as starting points, not model-independent optima.

### `direction+acceleration`

Adds HiFlow-inspired adjacent denoising-time velocity-change alignment.

The H3 implementation reconstructs native flow velocity from the predicted-clean relationship and preserves the previous **distinct-coordinate** anchor across same-coordinate PECE predictor/corrector calls.

The implementation is structurally validated, but matched decoded-media testing has not shown a consistent quality advantage over direction-only. Keep `acceleration_weight=0` unless deliberately testing the research mode.

### `direction+temporal`

Adds bounded local adjacent-frame correspondence on captured low-grid H3 clean-state video latents.

The matcher uses:

- local cosine search;
- minimum similarity gating;
- best-vs-second-best uniqueness margin;
- reverse-cycle consistency;
- zero temporal copy for ambiguous/disoccluded locations.

It is functioning, but the final matched test showed extremely sparse valid support and no visible difference from direction-only. It remains experimental rather than a promoted default.

### `downsample_consistency`

Downsamples the target predicted-clean state to the low-grid reference geometry, measures the mismatch, and lifts the correction back to target resolution.

The term was measurable in telemetry but did not produce a useful decoded-media improvement in the matched smoke. It remains experimental.

### `off`

Disables trajectory correction while leaving the surrounding wrapper/metrics setup available for controls and debugging.

## Current tested direction-only reference

The final matched direction-only quality sweep used:

```text
outer_steps = 14
source_mode = scale
source_scale = 0.83
handoff_selection = fixed
handoff_coordinate = 0.35
guidance_mode = direction
direction_weight = 0.25
acceleration_weight = 0
temporal_weight = 0
consistency_weight = 0
low_frequency_cutoff = 0.25
```

That particular benchmark resolved roughly `736x736 -> 896x896` and improved subjectively from 10 to 12 to 14 SA-Solver-PECE outer steps.

This is a tested operating point for that difficult-motion setup, not a universal recommendation that every prompt needs 14 steps or `source_scale=0.83`.

## Resolution-aware refine SIGMAS

This is a separate downstream learned-refine experiment. It is **not** part of the recommended progressive handoff path.

Correct wiring:

```text
MP/base sizing ---------------------------> Continuum width/height
       \
        -> Refine Target Geometry --------> Resolution-Aware Sigmas target metadata

existing refine scheduler SIGMAS
        -> Resolution-Aware Sigmas
        -> MiniMax H3 Latent Upscaler + Refine (3D).sigmas
```

Do not feed **Refine Target Geometry** dimensions back into Continuum, and do not feed **Resolution-Aware Sigmas** into Continuum's main SIGMAS input.

`mode=off` is the current recommendation. The resolution-aware remap was structurally valid but did not show a relevant quality improvement in the completed matched E0/E1 media pair.

`source_width=source_height=0` means the node derives the H3-native analytic reference canvas for the target aspect ratio. It does **not** request a physical low-resolution sampling pass.

## Metrics

### Runtime Metrics Probe

**MiniMax H3 Runtime Metrics Probe** installs passive sampler/model-call instrumentation without enabling capture, guidance, progressive handoff, or attention changes.

Place it after Spectrum when exact/forecast provenance is required.

If a shared `H3_FLOW_METRICS` object already exists, connect it to the optional `metrics` input to append to that artifact. Otherwise the probe creates its own metrics object.

### Metrics JSON

**MiniMax H3 Metrics JSON** autosaves structured metrics under ComfyUI's output directory and refreshes the same file as later sampler events complete.

Useful events/counters include:

- logical sampler/model calls;
- actual H3 transformer evaluations;
- Spectrum forecasts/promotions;
- low/probe/high progressive stage;
- sigma and unshifted flow coordinate;
- resolved handoff/source/target geometry;
- trajectory commits and provenance;
- guidance component RMS ratios;
- sampler/history reset boundaries;
- sampler/stage wall time;
- resolution sigma-map diagnostics.

Metrics establish whether the intended path executed. They do not replace decoded video/audio review as the quality criterion.

## Compatibility and ordering

### Spectrum

Spectrum can remain upstream of the progressive wrapper. Across the handoff, feature-history state is reset because cached transformer features from one spatial shape cannot be reused at another. The first target-grid call is forced actual.

### SA-Solver/PECE and other samplers

SA-Solver/PECE, SEEDS, ER-SDE, Euler/RES sampler objects are preserved. Progressive low/probe/high stages use independent sampler lifetimes where the spatial geometry boundary invalidates multistep/RNG history.

A sampler object carrying an explicit external `noise_sampler` closure can be rejected because arbitrary mutable RNG/history semantics cannot be reconstructed safely across the geometry reset.

### DiffAid / Untwisting RoPE

Keep these patches upstream of the progressive wrapper. The target-grid stage publishes the exact refinement-anchor contract expected by compatible downstream patches.

### Continuum masks and chunks

Target Input keeps Continuum-facing geometry at the final target size. The wrapper reconstructs the private/target video state internally and preserves audio. Masked behavior is structurally tested, but it has less decoded-media coverage than the main unmasked difficult-motion path.

### Private low-grid noise

Target Input derives private low-grid video noise from a documented standard-Gaussian CPU generator keyed by the graph seed. Arbitrary custom/non-Gaussian private-grid video-noise semantics cannot be preserved across this internally generated source state.

### Parallel multi-GPU ordering

The capture/guidance/progressive state is mutable and ordered. Unsupported parallel multi-GPU model-call ordering currently fails closed rather than risking trajectory corruption.

## Reference Budget and Attention Lab

These are research/diagnostic nodes, not required for normal flow-aligned or progressive generation.

### Reference Budget

Modes:

- `native` — pass conditioning through unchanged;
- `diagnostic` — report direct-reference row growth without changing conditioning;
- `decoupled_direct_experimental` — apply the guarded experimental direct-video-row cap.

The node cannot retroactively change Qwen3-VL tokens that have already been encoded into conditioning.

### Attention Lab

Modes:

- `native` — no attention change;
- `diagnostic` — output-neutral measurement of native dense attention: entropy, modality mass, VDN-retained/outside mass, first/last boundary mass, exact mask density, and Continuum seam mass when authoritative seam metadata exists;
- `vdn_reference_dense` — query-chunked dense additive-mask correctness oracle for OpenVDN's chunk-local temporal topology; this changes attention output but is explicitly not a sparse-compute or acceleration path;
- `experimental_sparse` — the earlier guarded spatial-local/all-time dense-mask experiment with selected layers, local window size, global heads, and sequence cap.

The VDN reference defaults are 5-frame chunks, radius 1 (previous/current/next complete chunks), globally visible non-video tokens, and `both` first/last anchors: every video query sees both boundary frames and both boundary-frame query rows see all video frames. When `continuum_seam_anchor` is enabled, the same symmetric anchor is added at the last protected latent frame only if Continuum supplies `protected_video_prefix_latent_slots`. Flow deliberately does not infer a latent seam from `context_frames`.

The reference path still feeds ordinary attention backends dense Q×K masks in query chunks and can be slower than native attention. It is not OpenVDN's FlexAttention kernel, does not include OpenVDN's trained linear branch or weights, is not an implementation of MiniMax's unreleased H3 sparse-attention topology, and should not be treated as a production acceleration path.

## Evidence documents

For exact runs and measured outcomes, use the dedicated evidence documents rather than extending the main README:

- [BENCHMARKS.md](BENCHMARKS.md) — decoded-media smoke ledger, tested operating points, topology, metrics, and optional formal matrix;
- [PERFORMANCE.md](PERFORMANCE.md) — observed progressive learned-handoff versus proper two-pass timing comparison;
- [RESEARCH.md](RESEARCH.md) — research-transfer rationale and conclusions;
- [../CREDITS.md](../CREDITS.md) — full research and implementation provenance.
