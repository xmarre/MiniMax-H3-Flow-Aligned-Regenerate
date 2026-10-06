# Usage guide

This document contains the detailed wiring and configuration information intentionally kept out of the main README.

The repository exposes two main generation paths:

1. **flow-aligned two-pass guidance** — capture a low-resolution H3 trajectory and use it to guide a later high-resolution/refine pass;
2. **progressive handoff** — keep the early part of one schedule on a smaller video grid, then transition once to the target grid and continue sampling.

For target-input workflows, including H3 Continuum exact-prefix continuation, **MiniMax H3 Progressive Handoff (Target Input)** is the standard path. The former Mixed-Grid Continuum path remains executable for one compatibility release but is deprecated and is not part of the production acceptance topology.

## Canonical Target Input defaults

**MiniMax H3 Progressive Handoff (Target Input)** uses the same defaults as the shipped progressive target-input example:

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

The learned-transfer side input comes from **MiniMax H3 Latent Upscaler Provider (3D) [Experimental]** in [`xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus`](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus). The shipped example configures:

```text
model_name             = minimax_h3_latent_upscaler_3d_bf16.safetensors
device                 = cuda
precision              = bf16
offload_after_upscale  = false
```

`acceleration_weight=0.25` and `consistency_weight=0.25` are staged values in this default. With `guidance_mode=direction+temporal`, the current runtime uses direction and temporal guidance only. Acceleration becomes active only in `direction+acceleration`; consistency becomes active only in `downsample_consistency`.

Historical benchmark configurations in `BENCHMARKS.md`, `PERFORMANCE.md`, and `workflows/benchmark-matrix.json` retain the values that were actually measured. They are evidence records, not current-default declarations.

## Partitioned exact-prefix Continuum defaults

Use **MiniMax H3 Partitioned Exact-Prefix Handoff** with the coordinated
Sol-H3, VDN-H3-Plus, Continuum and Upscaler releases listed in the README.
New nodes use the following defaults; explicit saved controls remain active:

```text
source_mode                      = scale
source_scale                     = 0.70
source_width                     = 864
source_height                    = 640
handoff_coordinate               = 0.35
handoff_selection                = fixed
guidance_mode                    = direction+temporal
direction_weight                 = 0.25
acceleration_weight              = 0.25
consistency_weight               = 0.25
low_frequency_cutoff             = 0.25
temporal_weight                  = 0.20
vdn_linear_diagnostic            = normal
audio_guided_overlap_ticks       = 16
audio_guided_overlap_mode        = sampler_mask_exact_timestep
prefix_transformer_context       = exact_target_partitioned
audio_position_domain            = source_carrier
audio_handoff_source             = main_partitioned
av_handoff_source                = main_partitioned
guidance_trajectory_source       = main_exact_partitioned
low_probe_execution_source       = main_then_shadow
frame_gauge_repair               = true
frame_gauge_residual_mode        = off
provider_boundary_stabilization  = soft_support_v1
capture_boundary_witness         = false
vdn_temporal_carrier_policy      = native_grid_then_map_v1
handoff_transfer_control         = learned_3d
spatial_stage_control            = same_grid_target_control
softmax_diagnostic               = normal
video_guided_overlap_tokens      = 6
suffix_dc_bridge                 = true
target_band_tokens               = 4
```

Continuation low/probe and high share the target grid. Clean/residual transfer
is identity, paired-prefix checks resolve to identity, and the main source
selectors do not add shadow sampler lifetimes. All-generated first chunks
retain learned progressive transfer, so the upscaler input remains connected.
The historical serialized node ID remains unchanged.

`sampler_mask_exact_timestep` aliases coherent `exact_mask`: carried audio stays
protected by matching native input, timestep and velocity masks, with zero
effective overlap regardless of the stored width. Comparison modes accept any
non-negative overlap width and cap application at the available audio prefix.
The stored video width is provenance-only, with zero applied prefix release.
`soft_support_v1` observes the post-high boundary without its retired pre-high
correction. Residual tensor export and actual-feature witness capture are off.

The heterogeneous path retains the first-token channel-mean bridge and remains
opt-in with unresolved boundary quality. The accepted target-grid profile does
not qualify every model, scene or seed. Target-grid low/probe uses more spatial
work than a reduced-grid stage; compare continuation timings separately from
model loading and first-chunk work.

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

### Target-grid exact-prefix continuation

On **MiniMax H3 Partitioned Exact-Prefix Handoff**, select
`spatial_stage_control=same_grid_target_control` to run continuation low/probe
stages at the final target resolution. The carried prefix and generated suffix
then share one spatial grid, and clean/residual handoff uses identity transfer.
The configured handoff split and downstream high stage are retained. Keep
`handoff_transfer_control=learned_3d` and
`vdn_temporal_carrier_policy=native_grid_then_map_v1`; both are required by this
selector.

The selector applies to eligible exact-prefix continuations. An all-generated
first chunk retains the ordinary progressive path, so keep the learned-upscaler
provider connected. Increasing the low-stage spatial grid increases its work;
compare continuation stage timings when assessing cost. Audio can also change
because it is predicted jointly with the differently conditioned video.
New nodes default to `same_grid_target_control`. Explicit saved controls remain active.

### Target-band exact-prefix continuation

`spatial_stage_control=progressive_target_band` keeps the protected prefix and
the next `target_band_tokens` generated temporal latent tokens on the target grid
during low/probe, and runs every later generated token on the configured
reduced grid. The reduced-grid tokens use the learned 3D transfer. The aim is the
same-grid join next to the carried prefix at a lower low/probe cost.

- The band is an overlap region: its frames have both their own target-grid clean
  prediction and the learned transfer of their reduced-grid projection. The
  high-stage clean operand crossfades between them, with transfer weight `j/n`
  for band token `j` of `n`. The token next to the prefix keeps its target-grid
  prediction; the first tail token (pure transfer) continues the same ramp, so
  any disagreement between the two representations is spread over the band
  instead of switching between two adjacent tokens.
- Every generated token, band included, enters the high stage as its clean
  operand re-noised with one independent Gaussian noise field, regardless of
  `frame_gauge_repair`. The band's raw low-stage state is not resumed: it carries
  content-correlated low-stage residual next to a re-noised tail. Source residual
  transport is not used in this mode for the same reason.
- `partitioned_target_band_overlap` reports, per band frame, how the transfer
  differs from the target-grid prediction (total and high-pass energy ratios,
  raw, low-pass and channel-mean deltas). It adds no model or provider call.
  With `frame_gauge_residual_mode=measure`,
  `partitioned_target_band_same_frame_affine` also fits the same-frame
  displacement between the two representations.

- `target_band_tokens` counts H3 temporal latent tokens. It must leave at least
  one generated token on the reduced grid; otherwise the chunk fails before
  sampling.
- Low/probe video rows are `(prefix + band) x target rows + tail x source rows`.
  Every actual low/probe evaluation also processes the text, reference and audio
  rows. The video input projection and final layer process the full target-sized
  carrier, so the row reduction does not establish a wall-time or memory saving.
- The low trajectory used by Flow guidance is recorded on the uniform reduced
  grid. High-stage guidance binds to the actual clean operand: the crossfaded
  band and the learned tail.
- Required selectors: `handoff_transfer_control=learned_3d`,
  `vdn_temporal_carrier_policy=native_grid_then_map_v1`,
  `prefix_transformer_context=exact_target_partitioned`,
  `low_probe_execution_source=main_then_shadow`, the main audio/AV/guidance
  sources, `frame_gauge_residual_mode=off` (or `measure` for stage evidence) and
  `capture_boundary_witness=false`.
- Required companions: VDN-H3-Plus and Sol-H3 releases that accept a target-grid
  native partition carrier.

The paired-prefix frame gauge does not run in this mode. The prefix is followed
by the band's own target-grid prediction, so no transfer boundary sits next to it. With
`suffix_dc_bridge=true`, the one-token DC bridge moves to the band/tail boundary
and is measured against the target-grid head.

This mode is not qualified for rendered quality. Before relying on it, inspect
the prefix join, the band's far edge, tone and audio against
`same_grid_target_control` on the same seed.

### Suffix DC bridge selector

`suffix_dc_bridge=true` (default) keeps the historical one-token channel-mean
bridge for learned-transfer continuations. `false` leaves the first transferred
token exactly as the transfer produced it. Use it to test whether that bridge
contributes to a boundary artifact. Transfer receipts record
`suffix_dc_bridge_requested`, and the runtime evidence gate validates both
settings.

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

Create one **Flow Trajectory** and connect it to **Progressive Handoff (Target Input)**. Connect the 3D learned-upscaler provider to `learned_upscaler` when using `handoff_transfer=learned_3d`.

Do **not** add a separate Trajectory Capture node on this path. The progressive wrapper captures the private low-grid trajectory internally when the normal progressive path is eligible.

Continuum remains configured for the final target width/height.

### Exact protected video fallback

If the prepared video denoise mask contains any exact-zero values, the exact native-masked contract takes precedence over progressive resizing. Flow forwards the original target noise, latent, mask, schedule, sampler, callback, conditioning and shape metadata through one ordinary target-grid sampler lifetime.

That path performs:

```text
private low-grid sampler       = no
exact handoff probe            = no
learned-upscaler calls         = 0
geometry boundary              = no
progressive history boundary   = no
progressive guidance           = no
```

A `progressive_target_fallback` metrics event records this path. Audio-only zero masks do not independently trigger the video fallback, and fractional video masks remain intentional blends rather than exact protection.

On a canonical exact-prefix continuation, the default overlap guides the last four carried **audio latent ticks** during sampler lifetime. H3 audio latent rate is 40 Hz, so the default window is about 100 ms. Core MiniMax-H3 uses a 1/256 mask grid; the exact default ramp values are:

```text
0.203125
0.40234375
0.6015625
0.80078125
```

The video mask is byte-for-byte unchanged. The caller-owned original exact mask remains authoritative and every originally protected video/audio value is restored exactly at the sampler boundary.

All-generated first chunks and fully protected audio are no-ops. Other non-canonical partially protected audio layouts retain their existing rejection.

`H3_FLOW_AUDIO_GUIDED_OVERLAP_TICKS=0` explicitly disables the overlap for controlled bisection. Any non-negative integer is accepted. The applied width is the smaller of the requested ticks and the available carried audio prefix. The receipt separates `requested_ticks` and `applied_ticks` and records `width_limited_by_prefix` and `hard_prefix_ticks`.

### Guided overlap controls

Partitioned continuation applies a one-token channel-mean handoff correction.
High-stage direction, acceleration and temporal comparisons use that same DC-only
representation. The historical full-spatial guidance gauge remains implemented
for regression coverage; production does not select it. Source trajectory tensors
and source-grid temporal correspondence remain unchanged. This comparison logic
is separate from sampler overlap width.

The one-token DC handoff also rebases an unregistered direction reference when
its measured offset is nonzero. This path compares only per-channel spatial
means and applies the difference to the first generated token. It preserves
the DC handoff's one-token support and leaves the reference's spatially varying
component intact. A rejected structural registration does not authorize a full
residual correction. Registered references, guidance off, downsample consistency
and zero-offset DC handoffs retain their existing behavior.

The production exact-overlap fallback follows the same DC-only rule. It adds
only the per-channel spatial mean of the exact-vs-learned last-prefix residual to
the first generated token. It never copies the spatially varying residual or
tapers a correction through later tokens. The historical representation-bridge
primitive remains available for source regression and diagnostic evidence.

The `partitioned_exact_overlap_bridge` receipt reports
`policy=partitioned_exact_overlap_dc_only_v5`, `structural_support_tokens=0`,
`dc_support_tokens=1`, `dc_temporal_weights=[1.0]` and
`later_suffix_extrapolated=false`. An eligible nonzero DC fallback reports
`applied=true` and `suffix_support_tokens=1`; otherwise its fallback support is
zero. This receipt describes the correction's scope, not rendered acceptance.
The current DC-only path still exhibits the rendered frame-shift, shock and tone
defect; it is not a qualified visual continuity fix.

Production `guidance` receipts report `reference_gauge_used` and
`reference_gauge_policy=exact_prefix_guidance_reference_dc_v1` for a nonzero DC
handoff. The historical coupled oracle uses
`exact_prefix_guidance_reference_coupled_v1`.

When exact-prefix continuation uses the actual learned handoff and its matching
main low/probe trajectory, direction and acceleration reuse the learned provider
output as their transfer reference. Rebuilding it with bicubic interpolation
would compare a different temporal transition even after prefix rebasing.

The learned reference owns the exact source/target pair that the provider
actually executed. The source side is the post-probe sampler result after
authoritative source-grid prefix restoration; the target side is the learned
provider output before exact target-grid prefix replacement. The trajectory
capture still owns run identity, endpoint provenance and high-schedule
qualification. It is not required to be bit-identical to the provider source:
the one-call probe is captured at the PREDICT_NOISE boundary, then passes through
the sampler's mathematically cancelling `(1-sigma)` output scaling and FLOW_AV
inverse scaling before it reaches the provider. Floating-point round trips can
therefore differ without representing a different trajectory.

Temporal correspondence and innovations use the actual provider source tensor,
while the temporary target operand removes the learned-transfer difference and
the existing prefix representation residual before transport. This keeps source
and target in the same executed handoff pair instead of mixing the provider
target with a stale pre-sampler capture. The current high prediction continues
to evolve under the existing weights, schedule and RMS bound.

The source/target pair is retained only for the high lifetime and released on
success or failure. For the 62x40x44 -> 62x58x64 FP32 continuation geometry this
is about 31.1 MiB total retained video state. `handoff_reference_used` and
`temporal_handoff_reference_used` identify executed comparisons, while
`partitioned_handoff_guidance_reference` reports the captured-versus-provider
source deltas. Registered, same-grid, bicubic and independent shadow controls
retain their existing reference contracts. No additional H3, provider, sampler
or VAE evaluation is added. This comparison correction requires rendered
qualification; it is not a decoded boundary acceptance claim.

The Continuum handoff node accepts any non-negative integer for `video_guided_overlap_tokens` and `audio_guided_overlap_ticks`; neither widget sets a fixed maximum. Partitioned node defaults store video `6` and audio `16`; effective video release and exact-mode audio overlap remain zero. The separate Target Input node retains its four-tick audio default. Audio overlap retains its selected audio-mode semantics.

For partitioned exact-prefix video, the old target-high prefix-release interpretation is retired after run 01093. A positive `video_guided_overlap_tokens` request is still accepted and reported, but it no longer writes a denoise ramp into caller-owned carried-prefix tokens and no longer installs the temporary Core `denoise_mask_function` / `APPLY_MODEL` closure. The target-high video sampler mask and H3 video context remain the authoritative exact prefix for every evaluation. The receipt reports `policy=partitioned_video_high_exact_context_v4`, `applied_tokens=0`, `retired_prefix_release=true`, and the original requested width.

The retired `partitioned_video_high_sampler_overlap_exact_tail_v3` path repainted the last `N` protected prefix tokens early in target-high sampling, then decayed that release to zero for the final two evaluations and restored the exact prefix at output. Core correctly propagated the same changing mask through inpaint injection, H3 timestep labels and velocity conversion, but that coherence did not solve the ownership problem: the generated suffix could evolve against carried context that was later discarded. Increasing `N` increased the amount of temporary context. Run 01093 used six released prefix tokens and still reproduced the reported frame-shift/shock/tone defect; the first target-high H3 prediction recreated boundary displacement before Flow guidance materially changed it.

This retirement is not a claim that target-high is now visually accepted. It removes a hardware-falsified source of discarded-context conditioning and converts the next run into an exact-context test. A future video-overlap design must operate without repainting caller-owned prefix context or merely moving the discontinuity to a later protected suffix token; the previously rejected one-token high guard is not restored. No extra H3 evaluation, provider call, sampler lifetime, history boundary or VAE invocation is added.

### Boundary decoder-window evidence

`frame_gauge_residual_mode=measure` also saves the native seven-token boundary
decoder window, independently of registration acceptance. This captures failures
on the coupled fallback and DC paths as well as accepted registration. The
existing regional residual-fit gate remains unchanged; its `not_evaluated`
result does not suppress this separate stage evidence.

The bundle contains two authoritative prefix tokens; provider-native and
pre-high clean windows; the first actual high prediction before and after Flow;
that call's input after sampler inpaint; the initial video mask; and final
internal clean output. Later actual calls and forecasts do not overwrite the
first actual prediction. All snapshots own CPU storage. They are released on
success or failure and add no model, sampler, provider or VAE evaluation.

`partitioned_boundary_window_evidence` reports the relative output bundle path,
manifest hash, first actual call/sigma and CPU-copy time. The manifest records
tensor shapes, dtypes, byte hashes, decoder phase and domains. Model-predicted
prefix bytes remain native: for an offline comparison of the returned suffix,
replace the window's first two tokens with `authoritative_prefix`, then apply
the model's latent-output conversion before VAE decoding. The first retained
frame is local frame five; the next twelve frames precede the next native window
blend. Unsupported prefix phases or missing real right context report
`unsupported`; no context is invented. CPU copies and file I/O contribute
diagnostic overhead. Mode `off` creates no window snapshots or files. Capturing
these operands does not correct a rendered jump or qualify image quality.

### What happens at an unprotected handoff

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

`fixed` at `handoff_coordinate=0.35` is the current default. `auto_compute` remains available for controlled experiments.

### Sigma schedule requirement

Progressive handoff requires a complete H3 schedule whose absolute flow origin is known. Use a full 1-to-0 schedule.

Partial low-sigma refinement schedules are rejected for progressive sampling because the wrapper cannot safely infer the original absolute flow coordinate from an arbitrary tail schedule.

## Handoff transfer modes

### `learned_3d`

`learned_3d` is the intended transfer for the shipped target-input workflow. It uses the companion repository:

https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus

Create **MiniMax H3 Latent Upscaler Provider (3D) [Experimental]**, connect its `H3_LATENT_UPSCALER` output to the progressive node, and use:

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

The deprecated Mixed-Grid compatibility path also requires `learned_3d`; that requirement is part of the old serialized-workflow contract, not the current exact-prefix Target Input fallback.

### `bicubic`

`bicubic` remains available on Target Input as a dependency-free compatibility/control transfer. It resizes the exact-probe predicted-clean video state directly to the target grid before target-state reconstruction. It is not supported by the deprecated Mixed-Grid compatibility path, which retains its historical `learned_3d` requirement.

### Learned-transfer evidence

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

This is evidence for the chosen transfer/source-scale policy, not a claim that one source scale or step count is universally optimal.

See [BENCHMARKS.md](BENCHMARKS.md) and [PERFORMANCE.md](PERFORMANCE.md) for the exact historical evidence and timing accounting.

## Progressive Handoff (source-input variant)

**MiniMax H3 Progressive Handoff** is the source-sized version of the same general idea. The incoming workflow starts on the smaller grid and the node grows the video state toward either:

- a target scale; or
- explicit target pixel dimensions.

Use this variant when the surrounding workflow can legitimately begin at source geometry.

## Mixed-Grid compatibility window

**MiniMax H3 Progressive Mixed-Grid Continuum [Deprecated]** remains registered for one compatibility release.

The old node ID `H3ProgressiveMixedGridHandoff` is intentionally preserved as a **distinct implementation**. It is not aliased or remapped to Target Input because doing so would silently change existing serialized workflow behavior.

During this window:

- old serialized workflows still load with their original Mixed-Grid semantics;
- Mixed-Grid is not recommended for new workflows;
- Mixed-Grid is not part of the production Patcher topology;
- its weighted attention/external-sequence companions are not release gates;
- no Mixed-Grid compatibility render is required;
- runtime machinery remains only so compatibility is real rather than a broken placeholder.

The old path retains its low-grid suffix, exact target-prefix transformer conditioning, learned 3D transfer, VDN external-sequence API 2, suffix DC bridge and attention-measure/exact-overlap seam-repair semantics. Detailed historical contracts and evidence remain in [MIXED_GRID_CONTINUUM.md](MIXED_GRID_CONTINUUM.md) and [mixed-grid-seam-repair.md](mixed-grid-seam-repair.md).

Runtime/measure machinery is scheduled for removal after the compatibility window. Historical evidence remains historical evidence after that cleanup.

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

Aligns the target predicted-clean estimate toward the matched captured low-resolution predicted-clean state in low spatial frequencies and decays the correction through the later high-resolution stage.

The explicit two-pass nodes use direction-oriented defaults. Historical progressive direction-only sweeps also remain recorded because they are useful controls.

### `direction+acceleration`

Adds HiFlow-inspired adjacent denoising-time velocity-change alignment.

The H3 implementation reconstructs native flow velocity from the predicted-clean relationship and preserves the previous **distinct-coordinate** anchor across same-coordinate PECE predictor/corrector calls.

Historical matched decoded-media testing did not show a consistent quality advantage over direction-only. The current Target Input defaults still store `acceleration_weight=0.25`, but that value is inactive while the selected mode is `direction+temporal`.

### `direction+temporal`

Adds bounded local adjacent-frame correspondence on captured low-grid H3 clean-state video latents. This is the current Target Input default mode.

The matcher uses:

- local cosine search;
- minimum similarity gating;
- best-vs-second-best uniqueness margin;
- reverse-cycle consistency;
- zero temporal copy for ambiguous/disoccluded locations.

An earlier matched direction+temporal smoke had extremely sparse valid support and no visible difference from its direction-only control. That historical result remains valid; the current default is a policy choice and should not be misread as evidence that temporal guidance universally improves output.

### `downsample_consistency`

Downsamples the target predicted-clean state to the low-grid reference geometry, measures the mismatch, and lifts the correction back to target resolution.

The term was measurable in historical telemetry but did not produce a useful decoded-media improvement in the matched smoke. `consistency_weight=0.25` in the current Target Input defaults is inactive unless this mode is explicitly selected.

### `off`

Disables trajectory correction while leaving the surrounding wrapper/metrics setup available for controls and debugging.

## Historical direction-only reference

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

That benchmark resolved roughly `736x736 -> 896x896` and improved subjectively from 10 to 12 to 14 SA-Solver-PECE outer steps.

These values are retained because they describe the measured run. They are not the current Target Input node defaults.

## Resolution-aware refine SIGMAS

This is a separate downstream learned-refine experiment. It is **not** part of the standard progressive handoff path.

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

Spectrum can remain upstream of the progressive wrapper. Across an actual geometry handoff, feature-history state is reset because cached transformer features from one spatial shape cannot be reused at another. The first target-grid call is forced actual. The exact protected-video fallback has no geometry handoff and therefore creates no progressive history boundary.

### SA-Solver/PECE and other samplers

SA-Solver/PECE, SEEDS, ER-SDE, Euler/RES sampler objects are preserved. Progressive low/probe/high stages use independent sampler lifetimes where the spatial geometry boundary invalidates multistep/RNG history.

A sampler object carrying an explicit external `noise_sampler` closure can be rejected because arbitrary mutable RNG/history semantics cannot be reconstructed safely across the geometry reset.

### DiffAid / Untwisting RoPE

Keep these patches upstream of the progressive wrapper. The target-grid stage publishes the exact refinement-anchor contract expected by compatible downstream patches.

### Continuum masks and chunks

Target Input keeps Continuum-facing geometry at the final target size. Exact protected video takes the conservative one-sampler target-grid fallback; unprotected/fractional-mask calls may use the normal private low-grid handoff.

The deprecated Mixed-Grid compatibility node preserves its historical target-prefix/source-suffix mixed-sequence behavior only for existing workflows during the deprecation window.

### Private low-grid noise

Target-input progressive execution derives private low-grid video noise from a documented standard-Gaussian CPU generator keyed by the graph seed. Arbitrary custom/non-Gaussian private-grid video-noise semantics cannot be preserved across this internally generated source state.

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
- [MIXED_GRID_CONTINUUM.md](MIXED_GRID_CONTINUUM.md) — deprecated Mixed-Grid compatibility contract and historical implementation details;
- [mixed-grid-seam-repair.md](mixed-grid-seam-repair.md) — historical Mixed-Grid seam investigation;
- [../CREDITS.md](../CREDITS.md) — full research and implementation provenance.
