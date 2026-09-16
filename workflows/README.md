# Workflows

This directory contains executable examples and non-executable wiring overlays. The distinction is intentional.

## Loadable ComfyUI workflows

`examples/*.workflow.json` are ordinary ComfyUI LiteGraph workflow documents. Drag them onto the ComfyUI canvas or use **Workflow > Open**.

- `examples/progressive-target-input.workflow.json` is the standard target-input progressive workflow. It wires **MiniMax H3 Latent Upscaler Provider (3D)** from `Comfyui_Minimax_h3_latent_Upscaler-Plus`, uses `source_scale=0.70`, fixed `handoff_coordinate=0.35`, `direction+temporal`, and `learned_3d` transfer.
- `examples/progressive-source-input.workflow.json` remains the dependency-minimal source-input control. Generation starts from an `864x480` source-grid latent and hands off to a `1.20x` target scale in the same sampler run.

Both examples use stock ComfyUI MiniMax H3 loading, conditioning, decoding, and video output with `res_multistep`; neither adds a Turbo LoRA. The Flow-patched `MODEL` is wired to both `BasicScheduler.model` and `BasicGuider.model`.

In the target-input example, `acceleration_weight=0.25` and `consistency_weight=0.25` are stored to match the node defaults, but they are inactive while `guidance_mode=direction+temporal`.

### Exact-prefix behavior

The target-input graph does not need a separate Continuum-specific progressive node. If the prepared video denoise mask contains any exact-zero value, **MiniMax H3 Progressive Handoff (Target Input)** preserves the exact target-grid contract by using one ordinary target-grid sampler lifetime. That path performs no private low-grid sampler, no handoff probe, and no learned-upscaler call.

For canonical carried audio, the fallback exposes a four-tick guided overlap during sampler lifetime. The video mask remains unchanged and the original exact video/audio mask remains authoritative at output.

## API-format equivalents

`examples/*.api.json` contain the same two paths as complete ComfyUI API-format prompt graphs:

- `examples/progressive-target-input.api.json`
- `examples/progressive-source-input.api.json`

## Wiring overlays

The root `*.overlay.json` files are topology/specification documents, not serialized ComfyUI canvas workflows.

- `*.workflow.json` = loadable ComfyUI canvas workflow;
- `*.api.json` = executable API prompt graph;
- `*.overlay.json` = topology/specification document only.

`progressive-handoff.overlay.json` now describes the standard **Target Input** topology and its exact-prefix fallback. It also records the one-release deprecation contract for `H3ProgressiveMixedGridHandoff`: the old node ID remains loadable with its original semantics, is not remapped to Target Input, and is not a production release gate.

Historical Mixed-Grid benchmark configurations remain in the benchmark/research documentation as evidence records. They are not current workflow recommendations.
