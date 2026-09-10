# Workflows

This directory contains executable examples and non-executable wiring overlays. The distinction is intentional and explicit.

## Loadable ComfyUI workflows

`examples/*.workflow.json` are ordinary ComfyUI LiteGraph workflow documents. Drag them onto the ComfyUI canvas or use **Workflow > Open**.

The target-input and source-input examples intentionally have different dependency profiles:

- `examples/progressive-target-input.workflow.json` uses the current learned-transfer setup. It wires **MiniMax H3 Latent Upscaler Provider (3D)** from the companion `Comfyui_Minimax_h3_latent_Upscaler-Plus` node, uses `source_scale=0.70`, a fixed `0.35` handoff, `direction+temporal`, and `learned_3d` transfer. The provider is configured for `minimax_h3_latent_upscaler_3d_bf16.safetensors`, CUDA, bf16 precision, and no post-upscale offload.
- `examples/progressive-source-input.workflow.json` remains the dependency-minimal source-input control. Generation starts from an `864x480` source-grid latent and hands off to a `1.20x` target scale in the same sampler run.

Both examples use stock ComfyUI MiniMax H3 loading, conditioning, decoding, and video output nodes with `res_multistep`; neither adds a Turbo LoRA. The Flow-Aligned patched `MODEL` is wired to both `BasicScheduler.model` and `BasicGuider.model`.

In the target-input example, `acceleration_weight=0.25` and `consistency_weight=0.25` are stored to match the canonical node defaults, but they are inactive while `guidance_mode=direction+temporal`. The current guidance implementation activates acceleration only in `direction+acceleration` mode and consistency only in `downsample_consistency` mode.

The filenames in the H3 loader nodes match the public Comfy-Org MiniMax H3 model layout. The learned-upscaler checkpoint belongs in the companion node's normal latent-upscaler model directory. If your local files have different names, change the corresponding loader/provider values.

## API-format equivalents

`examples/*.api.json` contain the same two paths as complete ComfyUI API-format prompt graphs. Submit them with the normal ComfyUI API client or POST their JSON as the `prompt` payload to `/prompt`.

- `examples/progressive-target-input.api.json`
- `examples/progressive-source-input.api.json`

## Wiring overlays

The `*.overlay.json` files in the directory root are **not** serialized ComfyUI canvas workflows. They are versioned wiring/specification overlays used to document intended integration topology, current defaults, invariants, and historical benchmark variants without embedding a large machine-specific graph.

Use the suffixes as the contract:

- `*.workflow.json` = loadable ComfyUI canvas workflow;
- `*.api.json` = executable API prompt graph;
- `*.overlay.json` = topology/specification document only.

`progressive-handoff.overlay.json` separates `canonical_defaults` from historical validation configurations so older measured results are not rewritten when the shipped defaults change.
