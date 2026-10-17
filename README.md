# MiniMax H3 Flow-Aligned Regenerate

## v0.3.14 — coordinated release (17 October 2026)

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.14](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.14) | [#102](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/102), [#103](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/103), [#105](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/105), [#107](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/107), [#108](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/108) |
| Sol-H3 | [v0.1.11](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.11) | [#41](https://github.com/xmarre/ComfyUI-Sol-H3/pull/41) |
| VDN-H3-Plus | [v1.5.10](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.10) | [#41–#44](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/44) |
| H3 Continuum-Plus | [v3.4.9](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.9) | [#44](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/44), [#46](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/46), [#47](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/47) |

The recommended **MiniMax H3 Native Source Continuation** workflow is unchanged. Flow #107 retains Sol-H3 grouped K/V fusion for the continuation path; full VDN trained-linear work must remain enabled. Diagnostic options from #102/#103/#108 capture full-stage tone, target-sized evidence and per-prompt lifecycle/paging receipts; #105 prevents old local audits holding model owners across prompts. The diagnostic example is [full-stage continuity](workflows/examples/full-stage-continuity.workflow.json). Set `H3_LIFECYCLE_TRACE=1` before ComfyUI startup to capture lifecycle data; it is opt-in. No Keyless implementations are included.


Training-free ComfyUI nodes for reusing lower-resolution MiniMax H3 structure while producing a higher-resolution result, with dedicated Continuum support for exact-prefix continuation.

The project has two main approaches:

1. **Flow-aligned two-pass guidance** — capture the low-resolution H3 denoising trajectory and use it to guide a later learned-upscale/refine pass.
2. **Progressive handoff** — spend early H3 work on a smaller video grid, then switch to the target grid inside one sampling schedule.

**MiniMax H3 Native Source Continuation** is the recommended progressive-generation node for single chunks and Continuum sequences. Connect VDN-H3-Plus and the learned 3D upscaler. Without a protected prefix, it runs ordinary low/probe/high progressive sampling; Continuum Native Masked selects its exact-prefix continuation path. The selected ComfyUI attention backend is supported; Sol-H3 is optional. [Defaults and wiring](#recommended-generation-defaults).

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

- [`native-source-continuation.workflow.json`](workflows/examples/native-source-continuation.workflow.json) — the recommended single-chunk and Continuum interface with native source carry, learned 3D transfer and exact protected audio/video;
- [`progressive-target-input.workflow.json`](workflows/examples/progressive-target-input.workflow.json) — general target-input progressive control;
- [`progressive-source-input.workflow.json`](workflows/examples/progressive-source-input.workflow.json) — source-input progressive control;
- [`partitioned-exact-prefix.workflow.json`](workflows/examples/partitioned-exact-prefix.workflow.json) — the preserved legacy control layout for existing workflows.

Matching `.api.json` graphs are included. These examples show loading, conditioning,
patch/provider wiring and one-chunk sampling. To continue a sequence, connect the
same patched **MODEL** to Continuum Native Masked for the initial chunk and every
continuation. A cached first chunk generated with another MODEL cannot supply
native source carry. See [workflow wiring](workflows/README.md).

The learned examples use the companion 3D provider with
`minimax_h3_latent_upscaler_3d_bf16.safetensors`, CUDA, bf16 and
`offload_after_upscale=false`. All examples use `res_multistep` without Turbo LoRA.
VDN-H3-Plus partitioned transport is required; Sol-H3 attention is optional.
The [coordinated release set](#coordinated-h3-release-set) lists companion versions. Wiring overlays are specifications, not loadable workflows.

## Recommended generation defaults

Use **MiniMax H3 Native Source Continuation**
(`H3NativeSourceContinuationHandoff`) with VDN-H3-Plus and the learned upscaler.
Connect the patched MODEL to both scheduler and guider. For a single chunk,
previous source context is not required: the normal progressive path computes
and retains the source prediction. For a Continuum Native Masked sequence,
reuse the same patched MODEL for the initial chunk and every continuation.
Native carry requires batch-one H3 video with native `5k+2` latent-token lengths,
an even spatial grid and a target larger than the reduced source grid.

### Recommended continuation: one uniform source trajectory

Low/probe samples one full-duration clip on the reduced source grid. Native source
carry reuses the preceding chunk's already-computed source clean prediction,
paired against the exact returned target suffix. By default its coarse latent
layout is first reconciled with the returned video (target band, below), so the
reduced-grid stage continues the scene that was actually rendered. The learned upscaler transfers
the generated trajectory to the target grid. High refinement uses the original
target prefix, and the returned audio/video prefix retains exact ownership.
Native carry adds no H3, provider or VAE calls. Missing or mismatched source context
stops before low sampling; there is no silent fallback.

The handoff retains the measured low-stage residual. Gaussian refinement couples
noise on the image grid; deterministic model drift uses a separate
constant-preserving lift. The low grid, sampling schedule and call budget remain
unchanged. [Native carry contract](docs/NATIVE_SOURCE_PREFIX_CARRY.md).

| Setting | New-node default |
| --- | --- |
| `source_mode` | `scale` |
| `source_scale` | `0.70` |
| `source_width`, `source_height` | `864`, `640` (used in `pixels` mode) |
| `handoff_coordinate`, `handoff_selection` | `0.35`, `fixed` |
| `guidance_mode` | `direction+temporal` |
| `direction_weight` | `0.25` |
| `acceleration_weight`, `consistency_weight` | `0.25`, `0.25` (inactive in `direction+temporal`) |
| `low_frequency_cutoff` | `0.25` |
| `temporal_weight` | `0.20` |
| `audio_guided_overlap_mode` | `sampler_mask_exact_timestep` (coherent exact mask) |
| `audio_guided_overlap_ticks` | `16` (zero effective release in exact mode) |
| `uniform_source_detail_transport` | `true` |
| `source_prefix_projection` | `native_source_carry_target_band` |
| `capture_boundary_witness` | `false` |

`native_source_carry_target_band` is the new-node default. The plain carry is
the preceding chunk's reduced-grid estimate at the handoff sigma, and target-high
refinement can still redraw background content after it. The default therefore
takes the low spatial band of the protected source prefix from the resized
returned target prefix. It keeps the carry's finer detail only where the latent
structure of the two agrees. In user-reviewed continuation renders this removed
the visible boundary and the background redraw a few frames after the join. It
adds no H3, provider or VAE call. See
[target-band reconciliation](docs/NATIVE_SOURCE_PREFIX_CARRY.md#target-band-reconciliation).

Saved workflows keep their stored `source_prefix_projection`. Select
`native_source_carry_target_band` in an existing workflow to use it.
`native_source_carry` remains available as the previous behavior. `latent_bicubic`
is available when paired native context is unsuitable; it reconstructs only the
source prefix. No mode changes the authoritative target prefix.
Detail transport is enabled for new production nodes. It adds a last-prefix
exact-minus-learned detail residual before high refinement only when its
calibration and holdout checks pass. Otherwise it reports a skipped correction.
It adds no H3, provider or VAE evaluation, but calibration uses compute and memory.
It has no prefix to correct in an all-generated single chunk. Existing saved
`false` values and legacy-node defaults remain unchanged.

Slight continuation darkening remains a known issue. Output quality and camera
or focus behavior depend on the model, prompts and references; the defaults do
not guarantee seam quality or timing for every configuration.

### Compatibility and diagnostics

Existing `H3PartitionedExactPrefixDiagnosticHandoff` workflows retain their node
ID, widget order and explicit continuation settings. Its display name now ends
in **[Legacy Controls]**. The older compatibility and mixed-grid interfaces are
also retained. New workflows should use Native Source Continuation.

Live low-sampler A/B/A replay, internal kernel observers, startup source dumps and
the Flow-side CUDA reproducer have been retired after the numerical investigation.
Their saved widget slots remain loadable but cannot add sampling calls or hooks.
A metrics receipt reports any requested retired flags and cleared stale options.
Saved audit bundles remain readable under **diagnostics/legacy**. Optional
boundary capture and the [full-stage continuity replay](docs/FULL_STAGE_CONTINUITY.md)
remain available for investigating tone without changing inference.

### Attention backends

The uniform source and target high stages use the selected ComfyUI backend with
VDN-H3-Plus partitioned transport. Sol retains its request-owned sparse path.
Non-Sol stages publish Flow's Spectrum history identity; Sol stages retain Sol's
history provider. Existing forecasting and exact-mask contracts are preserved.

## Other continuation modes (retained)

Existing spatial-stage modes remain loadable with their original semantics.
See [legacy continuation controls](docs/LEGACY_CONTINUATION_CONTROLS.md).

## Continuum Decode Context

**MiniMax H3 Continuum Decode Context** can be placed immediately before the normal Video VAE Decode. It supplies real future latent context to the native H3 temporal decoder at exact chunk joins while leaving accepted sampling latents, continuation state, masks, audio and the assembly plan unchanged.

This solves a separate decoder-window boundary problem. It is independent of Target Input's sampler-time exact-prefix/audio-overlap behavior and of the deprecated Mixed-Grid seam mechanisms.

See [docs/CONTINUUM_DECODE_CONTEXT.md](docs/CONTINUUM_DECODE_CONTEXT.md).

## Generic Target Input control

**MiniMax H3 Progressive Handoff (Target Input)** remains a general control for workflows defined at the final target geometry. Native Source Continuation is the recommended progressive interface with VDN-H3-Plus and learned transfer.

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
| **MiniMax H3 Native Source Continuation** | Uniform source trajectory, paired native source context, learned transfer and exact audio/video ownership. |
| **MiniMax H3 Partitioned Exact-Prefix Handoff [Legacy Controls]** | Preserved widget layout and explicit settings for existing workflows; live replay controls are retired. |
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

**MiniMax H3 Local Boundary Audit** replays saved target-band or uniform-source
bundles across native temporal decoder windows spanning the continuation join.
Connect the same resident video VAE instance and settings used for production.
The node reuses a VAE already on its decode device, including partially loaded
weights; it does not reload it or retry managed decode after OOM. It keeps
strong owners of already loaded model patchers before ComfyUI prunes the previous
workflow's cache, preserving them through audit completion and idle cleanup.
Those owners are released after the next non-audit prompt. Explicit unload and
ComfyUI's normal memory management remain available. A VAE on another device is
admitted only if free memory covers its weights and decode workspace; insufficient
decode memory stops the audit. Set
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
Saved boundary bundles can be replayed with the production native video VAE.
The full-stage scope saves matched PNGs and source/stage receipts without H3
sampling. Geometry and sharpness estimates need interpretation alongside rendered
output; they do not classify camera motion or bokeh as defects.
See [full-stage replay](docs/FULL_STAGE_CONTINUITY.md) and
[boundary audit details](docs/BOUNDARY_AUDIT_REPLAY.md).

**MiniMax H3 Metrics JSON** writes structured model, sampler, geometry and forecasting metrics.

## Patch order and interoperability

For the native-source Continuum stack, keep model patches in this order when present:

```text
DiffAid
  -> Untwisting RoPE
  -> Spectrum
  -> Native Source Continuation
  -> Continuum
```

DiffAid, Untwisting RoPE and Spectrum are optional. Native Source Continuation requires VDN-H3-Plus partitioned transport; Sol-H3 is optional. Generic Target Input retains its separate exact-prefix fallback.

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
- Partitioned exact-prefix continuation with the `progressive_uniform_source` default profile;
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

Flow v0.3.13 contains PR #101. The companion releases below are separate packages;
use their documented capabilities together. See [release context](RELEASE_CONTEXT.md)
for upstream patches that have not been released.

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.13](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.13) | [#101](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/101) |
| Sol-H3 | [v0.1.10](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.10) | [#40](https://github.com/xmarre/ComfyUI-Sol-H3/pull/40) |
| VDN-H3-Plus | [v1.5.9](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.9) | [#39](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/39), [#40](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/40) |
| H3 Continuum-Plus | [v3.4.6](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.6) | unchanged |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | unchanged |

[Spectrum MiniMax H3 v0.2.29](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.29)
includes [runtime compatibility PR #121](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/pull/121). Separate Keyless, audio-training and rejected
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
