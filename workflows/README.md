# Workflows

This directory contains executable examples and non-executable wiring overlays. The distinction is intentional and explicit.

## Loadable ComfyUI workflows

`examples/*.workflow.json` are ordinary ComfyUI LiteGraph workflow documents. Drag them onto the ComfyUI canvas or use **Workflow > Open**.

The examples intentionally cover the production partitioned node and the two simpler progressive controls:

- `examples/partitioned-exact-prefix.workflow.json` serializes the current **MiniMax H3 Partitioned Exact-Prefix Handoff** defaults exactly. It wires **MiniMax H3 Latent Upscaler Provider (3D)** and stores `progressive_uniform_source`, `main_then_shadow`, `frame_gauge_repair=false`, exact-mask audio semantics, the current 16/6 stored audio/video overlap widths, `suffix_dc_bridge=false` and the unused `target_band_tokens=4`.
- `examples/progressive-target-input.workflow.json` remains the general Target Input control. It uses `source_scale=0.70`, fixed `handoff_coordinate=0.35`, `direction+temporal`, and `learned_3d` transfer.
- `examples/progressive-source-input.workflow.json` remains the dependency-minimal source-input control. Generation starts from an `864x480` source-grid latent and hands off to a `1.20x` target scale in the same sampler run.

The partitioned and generic target-input examples configure the provider for `minimax_h3_latent_upscaler_3d_bf16.safetensors`, CUDA, bf16 precision, and no post-upscale offload. All three examples use stock ComfyUI MiniMax H3 loading, conditioning, decoding, and video output nodes with `res_multistep`; none adds a Turbo LoRA. The Flow-Aligned patched `MODEL` is wired to both `BasicScheduler.model` and `BasicGuider.model`.

In the target-input example, `acceleration_weight=0.25` and `consistency_weight=0.25` are stored to match the canonical node defaults, but they are inactive while `guidance_mode=direction+temporal`. The current guidance implementation activates acceleration only in `direction+acceleration` mode and consistency only in `downsample_consistency` mode.

The filenames in the H3 loader nodes match the public Comfy-Org MiniMax H3 model layout. The learned-upscaler checkpoint belongs in the companion node's normal latent-upscaler model directory. If your local files have different names, change the corresponding loader/provider values.

### Exact-prefix behavior

For the coordinated Continuum stack, use `H3PartitionedExactPrefixDiagnosticHandoff` (displayed as **MiniMax H3 Partitioned Exact-Prefix Handoff**). The historical serialized ID is intentional. The compact example in this repository demonstrates the patch/provider/default serialization; when that patched `MODEL` is consumed by Continuum Native Masked continuation, a supported protected prefix selects the partitioned runtime.

The production profile uses `progressive_uniform_source`: continuation low/probe runs one full-duration reduced-grid clip, the learned upscaler transfers the whole generated trajectory to the target grid, the exact target-grid prefix is restored before target-grid high refinement, exact carried audio/video ownership is restored at output, and the first all-generated chunk still uses progressive `learned_3d` transfer. `same_grid_target_control`, the previous default, remains selectable. `audio_guided_overlap_ticks=16` is stored with `sampler_mask_exact_timestep`, which aliases coherent exact-mask behavior and therefore applies zero effective audio release. `video_guided_overlap_tokens=6` is provenance-only and applies zero protected-video release.

The generic `H3ProgressiveTargetInputHandoff` example remains valid as a control. Its exact-prefix behavior is the conservative one-target-grid sampler fallback, including the historical four-tick sampler-time audio overlap. It is not the coordinated partitioned Continuum production path.

The deprecated `H3ProgressiveMixedGridHandoff` node remains registered for one compatibility release so existing serialized workflows retain their historical semantics. It is not part of the canonical overlay or a production release gate.

## API-format equivalents

`examples/*.api.json` contain the same three paths as complete ComfyUI API-format prompt graphs. Submit them with the normal ComfyUI API client or POST their JSON as the `prompt` payload to `/prompt`.

- `examples/partitioned-exact-prefix.api.json`
- `examples/progressive-target-input.api.json`
- `examples/progressive-source-input.api.json`

## Wiring overlays

The `*.overlay.json` files in the directory root are **not** serialized ComfyUI canvas workflows. They are versioned wiring/specification overlays used to document intended integration topology, current defaults, invariants, and historical benchmark variants without embedding a large machine-specific graph.

Use the suffixes as the contract:

- `*.workflow.json` = loadable ComfyUI canvas workflow;
- `*.api.json` = executable API prompt graph;
- `*.overlay.json` = topology/specification document only.

`progressive-handoff.overlay.json` describes the **Partitioned Exact-Prefix** Continuum topology and complete production defaults. The generic **Target Input** path is retained explicitly as a control, including its conservative exact-prefix fallback, while historical progressive validation records remain separate. The overlay also records the one-release deprecation contract for `H3ProgressiveMixedGridHandoff`: the old node ID remains loadable with its original semantics, is not remapped to Target Input, and is not a production release gate.
