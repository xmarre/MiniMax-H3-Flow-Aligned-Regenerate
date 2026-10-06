# MiniMax H3 Flow-Aligned Regenerate

Training-free ComfyUI nodes for reusing lower-resolution MiniMax H3 structure while producing a higher-resolution result, with dedicated Continuum support for exact-prefix continuation.

The project has two main approaches:

1. **Flow-aligned two-pass guidance** — capture the low-resolution H3 denoising trajectory and use it to guide a later learned-upscale/refine pass.
2. **Progressive handoff** — spend early H3 work on a smaller video grid, then switch to the target grid inside one sampling schedule.

For Continuum exact-prefix continuation with Sol-H3 and VDN-H3-Plus, the production path is **MiniMax H3 Partitioned Exact-Prefix Handoff**. For general target-input workflows, use **MiniMax H3 Progressive Handoff (Target Input)**.

> This is an independent research implementation informed by public work. It does not reproduce MiniMax's closed H3-Regenerate-2K implementation or an unreleased sparse-attention model.

See [CREDITS.md](CREDITS.md) for research and implementation attribution.

## Install

Install this repository through ComfyUI Manager and restart ComfyUI.
For development updates, use ComfyUI Patcher's PR overlays in the declared
dependency order. After the coordinated versions are installed, the merged
custom-node overlays can be removed. Retain any required unmerged Core overlay.

The core package has no mandatory sibling-node dependency. The intended learned-transfer workflows use the companion [MiniMax H3 Latent Upscaler-Plus](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus).

## Example workflows

Loadable examples are under [`workflows/examples/`](workflows/examples/):

- [`partitioned-exact-prefix.workflow.json`](workflows/examples/partitioned-exact-prefix.workflow.json) — production-node wiring for **MiniMax H3 Partitioned Exact-Prefix Handoff**, including the complete 32-widget target-grid profile (`same_grid_target_control`, exact audio/video ownership, `main_then_shadow`, paired-prefix checks, the suffix DC bridge, the target-band width, and the current overlap/provenance values);
- [`progressive-target-input.workflow.json`](workflows/examples/progressive-target-input.workflow.json) — general target-input progressive control using `source_scale=0.70`, fixed `0.35` handoff, `direction+temporal`, and `learned_3d` transfer;
- [`progressive-source-input.workflow.json`](workflows/examples/progressive-source-input.workflow.json) — dependency-minimal source-input progressive control using a `1.20x` target handoff.

The partitioned and generic target-input examples both configure `MinimaxH3LatentUpscaler3DProvider` with `minimax_h3_latent_upscaler_3d_bf16.safetensors`, CUDA, bf16 precision, and `offload_after_upscale=false`. Matching `.api.json` prompt graphs are included for API execution. These compact examples deliberately use stock one-chunk H3 conditioning/sampling so the Flow patch wiring is inspectable in isolation; the partitioned exact-prefix runtime activates when the same patched `MODEL` is consumed by Continuum Native Masked continuation with a protected prefix. That production path requires the coordinated Sol-H3 and VDN-H3-Plus releases listed under [Coordinated H3 release set](#coordinated-h3-release-set).

All three examples use `res_multistep` and no Turbo LoRA. The `workflows/*.overlay.json` files are topology/specification documents rather than loadable ComfyUI workflows; see [`workflows/README.md`](workflows/README.md) for the format distinction.

## Partitioned exact-prefix Continuum path

Use **MiniMax H3 Partitioned Exact-Prefix Handoff** with the coordinated
Sol-H3/VDN-H3-Plus/Continuum releases. The historical serialized node ID
`H3PartitionedExactPrefixDiagnosticHandoff` remains unchanged.

New nodes use these defaults; existing explicit workflow values are retained:

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

Flow defaults exact-prefix continuation to `same_grid_target_control`: low/probe
and high operate at the target video resolution, with identity clean/residual
handoff and exact carried audio/video ownership. The first all-generated chunk
retains progressive learned transfer. The learned-upscaler provider therefore
remains connected. The selected profile has reported visual/audio acceptance;
heterogeneous continuation remains an opt-in path with unresolved boundary
quality. This does not establish acceptance for every model, scene or seed.

The stored audio width of 16 has zero effective overlap under
`sampler_mask_exact_timestep`, which aliases coherent `exact_mask`.
The stored video width of 6 is provenance-only and does not release protected
video tokens. Both stored overlap widths accept any non-negative integer;
comparison modes cap effective overlap at the available carried prefix. `main_then_shadow` executes a shadow only when a shadow source
is selected; the default main sources do not add duplicate sampler lifetimes.

`frame_gauge_repair=true` enables paired-prefix handoff checks; the same-grid
path resolves to identity. `soft_support_v1` records a bounded post-high
observation while its historical pre-high correction remains disabled. Neither
selector authorizes a decoded output warp. Diagnostic tensor export requires
`frame_gauge_residual_mode=measure`; it remains off by default.

Audio is never spatially resized. It is jointly predicted with video, so changing
the video grid can affect generated audio. On the default same-grid path,
source-carrier and target audio spatial positions coincide. Native masks remain
coherent throughout sampling and exact carried audio/video values are restored
at output.

The learned 3D upscaler input remains required for the first all-generated chunk.
Keep `handoff_transfer_control=learned_3d` and
`vdn_temporal_carrier_policy=native_grid_then_map_v1`. Selecting shadow sources
or heterogeneous controls can change work and output; their results are not
qualified by the target-grid profile's acceptance. Target-grid low/probe also
has more spatial work than a reduced-grid stage. Measure continuation timing
separately from loading and first-chunk work.

### Target-band continuation (opt-in)

`spatial_stage_control=progressive_target_band` is an experimental alternative
for exact-prefix continuations. The protected prefix and the first
`target_band_tokens` generated H3 temporal latent tokens after it stay on the
target grid through low/probe. Every later generated token runs on the
configured reduced grid. At the handoff the band keeps its own target-grid clean
prediction, and the remaining tokens use the learned 3D transfer of
`progressive_low_to_high`. Every generated token, band included, is then
re-noised with the same independent Gaussian noise regardless of
`frame_gauge_repair`. The band's raw low-stage sampler state is not resumed, and
source residual transport is not used in this mode.

The low/probe sampler state stays on the uniform target grid. Each reduced-grid
token stores its values in a source-sized window of its frame, and the rest of
that frame is masked padding excluded from the transformer blocks. Stochastic
samplers may add noise to raw padding; it is masked before model calls and
discarded when forming the reduced-grid view. The transformer blocks receive
`[target-grid prefix | target-grid band | reduced-grid tail]`. Their video row
count falls between the reduced-grid and target-grid controls. The video input
projection and final layer still process the target-sized carrier. Wall time
and peak memory therefore require measurement.
Spectrum forecasting, the native final layer and Flow guidance keep their usual
contracts. Guidance binds to the actual high-stage clean operand (the band's own
prediction plus the learned tail).

With `suffix_dc_bridge=true`, the one-token channel-mean bridge applies to the
first reduced-grid token after the band and is measured against the target-grid
head. The band must leave at least one generated token on the reduced grid.
The mode requires `handoff_transfer_control=learned_3d`,
`prefix_transformer_context=exact_target_partitioned`, the main handoff and
guidance sources and boundary-witness capture off.
It also requires VDN-H3-Plus and Sol-H3 releases that accept a target-grid
native partition carrier. Unsupported combinations fail before sampling.

The default `vdn_temporal_carrier_policy=native_grid_then_map_v1` applies each
neighbor's spatial short-convolution on that neighbor's grid before resampling
it for a temporal tap. The opt-in `destination_grid_stencil_v1` resamples the
raw projected neighbor onto the receiving frame's grid before applying the
checkpoint spatial convolution. This keeps a temporal tap's spatial stencil
on the receiving grid while retaining cross-grid temporal coupling. Same-grid
taps are unchanged. Use `vdn_linear_diagnostic=normal` with this policy; it
cannot be combined with cross-grid tap suppression. Paired VDN checkpoint
capability and completed carrier work are verified. High refinement restores
the native policy. The destination policy uses the batched linear path rather
than the fused short-convolution path, so wall time requires measurement.
Rendered quality of this target-band combination remains unqualified.

`vdn_linear_diagnostic=suppress_cross_grid_temporal_taps` remains available as
an ablation: it removes only temporal taps crossing the grid boundary, retaining
same-grid taps and the learned linear complement. Improvement under suppression
does not establish that the destination policy produces the same improvement.

For stage localization, set `frame_gauge_residual_mode=measure`. Target-band
runs export the native decoder window and full video snapshots of the low/probe
carrier, the uniform source-grid provider input, the learned provider output
before the band splice, the high-stage entry clean state, the first actual high
prediction before/after Flow, and the final clean video. The manifest identifies
the protected prefix, band/tail edge, tensor grids and stage domains. The native
low/probe carrier contains reduced-grid storage windows and padding and must
not be decoded as ordinary target-grid video. Full snapshots have a 256 MiB
CPU tensor budget; exceeding it reports an error. This opt-in mode adds CPU
copies and output-file I/O, with no additional model/provider/VAE evaluations.
The paired-prefix frame gauge remains inactive in band mode.

The boundary quality of this mode has not been qualified. Compare its join, the
band's far edge, tone and audio against `same_grid_target_control` before using
it for production output. The low/probe saving depends on the reduced-grid size:
a larger `source_scale` (or larger explicit source size) leaves less to save, and
the video-row reduction is not a wall-time measurement.

### Suffix DC bridge

`suffix_dc_bridge` controls the one-token channel-mean bridge on learned-transfer
continuations. When enabled (the default and historical behaviour), the first
transferred generated token receives the per-channel spatial-mean offset that the
learned transfer produced on its carried context. When disabled, that token stays
exactly as transferred, and transfer receipts report the bridge as disabled.
Same-grid continuation measures a zero offset either way.

## Continuum Decode Context

**MiniMax H3 Continuum Decode Context** can be placed immediately before the normal Video VAE Decode. It supplies real future latent context to the native H3 temporal decoder at exact chunk joins while leaving accepted sampling latents, continuation state, masks, audio and the assembly plan unchanged.

This solves a separate decoder-window boundary problem. It is independent of Target Input's sampler-time exact-prefix/audio-overlap behavior and of the deprecated Mixed-Grid seam mechanisms.

See [docs/CONTINUUM_DECODE_CONTEXT.md](docs/CONTINUUM_DECODE_CONTEXT.md).

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
| **MiniMax H3 Progressive Handoff (Target Input)** | General target-input progressive path; exact protected prefixes use the target-grid fallback. |
| **MiniMax H3 Partitioned Exact-Prefix Handoff** | Coordinated Continuum path with target-grid low/probe, identity continuation handoff, exact prefix ownership and coherent native audio masks. |
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

**MiniMax H3 Local Boundary Audit** replays a saved target-band bundle across
native temporal decoder windows spanning the prefix and band/tail joins. Use a
separate workflow containing the same video VAE loader/settings used for
production and this audit node. Set
`bundle_path` to the existing exported directory containing `manifest.json` and
the full binary operands. Set `chunk_join_frame` to the assembled output's
first new frame, or leave it at zero for frame labels relative to that join.
No diffusion-model or sampler connection is needed.
The path may also select `manifest.json` itself. In WSL, use a Linux path such
as `/home/...`, or an Explorer path beginning `\\wsl.localhost\<distribution>\`
or `\\wsl$\<distribution>\`. Explorer paths are translated only when the named
distribution matches `WSL_DISTRO_NAME`; otherwise use the Linux path. Surrounding
quotes from **Copy as path** are accepted.

The audit decodes the provider, pre-high state, first high prediction before and
after Flow, and final state sequentially using the connected native video VAE.
It reproduces the temporal blend at the band/tail edge, compares matching pixel
frames across stages, and reports translation, scale/shear gradients, RGB
differences and luminance statistics. Full-frame and upper-region affine fits
are diagnostic estimates; moving subjects, changed details and weak texture
can confound them. They do not establish rendered acceptance.
Adjacent-frame geometry and RGB/luminance changes are measured for every stage.
Stage comparisons also report how much the temporal pixel increment changes,
which helps distinguish a stable refinement difference from one that develops
between adjacent frames. These measurements do not classify motion or cuts as
defects. The extra comparisons use already decoded pixels and add CPU fitting
work without additional VAE calls.

Set `audit_scope=transfer_and_decoder_context` for the extended replay. It adds
the saved uniform reduced-grid handoff view and compares identical pixel times
from two independently decoded seven-token contexts near the band/tail edge for
each stage. The reduced view contains a projected target-grid head and a native
reduced-grid tail; it
does not represent uniform reduced-grid generation of the head. Its geometry
uses its own decoded pixel units, reported alongside the canvas dimensions.
The loader verifies its hash, exact tail ownership and the head projection
within float32 interpolation tolerance (`atol=1e-4`, `rtol=1e-5`).

Each window-context comparison uses the five overlapping pixel times. It can
show how the decoder's temporal context changes a frame without comparing two
different scene times. Standalone pixels are finalized and clamped separately;
production blends before clamping. The audit keeps the original native blended
replay and never assembles a replacement from standalone pixels. Extended mode
uses eighteen VAE calls, additional CPU fitting and one additional saved operand;
the default `stage_continuity` mode uses five calls. Both modes save numerical
JSON only and perform no diffusion generation or provider inference. Geometry
estimates and context disagreement still require comparison with visually
accepted output; they are not automatic defect classifications.

Only numerical JSON is saved in `output/h3_flow_regenerate/boundary_audits` and
returned by the node. Images, prompts and binary operands are not included in
the report. Original media and saved operands remain local and unmodified.
The replay needs complete full-video target-band snapshots and rejects
incomplete context or altered prefix/mask/band ownership. It uses
the connected VAE, including native quantized variants, rather than loading a
separate decoder checkpoint and does not modify generation behavior. It does not
compare against the prior chunk's retained pixels. Frames labeled before the
join are decoded chunk context discarded by assembly. Setting the
join frame to zero changes only labels. Crop length grows with band length;
VAE call count alone does not determine replay time.
Reports use `local_target_band_native_window_audit_v2`: the plan can contain
multiple shared-token pairs and blend intervals. Older v1 reports describe the
earlier two-window crop and remain historical evidence for their measured range.

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
- Partitioned exact-prefix continuation with the target-grid default profile;
- Target Input's conservative exact-prefix target-grid fallback with its separate four-tick audio overlap.

Historical Mixed-Grid validation remains evidence for that retired architecture rather than a current recommendation. The suffix DC bridge removed its brief tone/flash boundary, and its later framing work addressed unequal protected-prefix versus suffix attention sampling measure. The source-space affine/trajectory correction family remains retired because finite corrections only moved the discontinuity to the corrected-to-untouched transition.

Target-Sparse is deliberately not promoted because its no-latent-upscale design produced cascading decoded-media defects in testing.

Quality and speed still depend on prompt, references, geometry, sampler, Spectrum policy, model residency and hardware. Use decoded media rather than structural metrics alone for new workflow variants.

## Legacy and research continuation paths

These paths remain registered for existing workflows and controlled experiments. Neither is a production recommendation.

### Mixed-Grid compatibility window

**MiniMax H3 Progressive Mixed-Grid Continuum [Deprecated]** remains registered for one compatibility release so existing serialized workflows continue to load with their original semantics.

The old `H3ProgressiveMixedGridHandoff` node ID is intentionally not remapped to Target Input. Mixed-Grid is no longer the recommended production path, is no longer part of the production Patcher topology, and is not a release/promotion or compatibility-render gate.

Its historical low-grid suffix, learned-transfer, VDN API-2, attention-measure, suffix-bridge and seam-repair contracts remain documented in [docs/MIXED_GRID_CONTINUUM.md](docs/MIXED_GRID_CONTINUUM.md) and [docs/mixed-grid-seam-repair.md](docs/mixed-grid-seam-repair.md). Runtime compatibility machinery is retained for the deprecation window and can be removed separately afterward.

### Target-Sparse Continuum status

**MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** remains available as a research/control path, but it is **not recommended for production quality**.

It keeps the sampler latent on the target grid and sparsifies only the early H3 hidden-token stream over generated video rows. Because it does not perform the learned latent upscale used by ordinary progressive transfer, real decoded-media testing showed cascading quality errors: skin imperfections, odd clothing changes and spurious background additions could appear and propagate through later continuation.

Target-Sparse remains useful for architectural experiments and controlled comparisons, not as a production continuation path.

## Coordinated H3 release set

Flow-Aligned Regenerate is released together with the other H3 components.
Per-version details are in [RELEASE_NOTES.md](RELEASE_NOTES.md).

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.10](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.10) | [#96](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/96) |
| Sol-H3 | [v0.1.9](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.9) | [#39](https://github.com/xmarre/ComfyUI-Sol-H3/pull/39) |
| VDN-H3-Plus | [v1.5.8](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.8) | [#38](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/38) |
| H3 Continuum-Plus | [v3.4.6](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.6) | [#39](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/39), [#40](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/40) |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | unchanged |

[Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28)
is the unchanged companion. Separate Keyless, audio-training and rejected
decoded-geometry experiments are outside this release set.

The tested Core adapter repair is
[ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783).
For INT8 fused MLP runtime adapters, retain that ComfyUI Patcher PR overlay until
the repair is available upstream. The independent Core #16720 optimization is
not included in this release set.

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
