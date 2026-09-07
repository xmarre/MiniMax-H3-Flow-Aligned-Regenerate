# Workflows

This directory contains executable examples and non-executable wiring overlays. The distinction is intentional and explicit.

## Loadable ComfyUI workflows

`examples/*.workflow.json` are ordinary ComfyUI LiteGraph workflow documents. Drag them onto the ComfyUI canvas or use **Workflow > Open**.

They are deliberately minimal:

- no Turbo LoRA;
- no Spectrum, Continuum, VDN, DiffAid, Untwisting RoPE, or learned-upscaler dependency;
- `res_multistep` sampling, so the compatibility path reported in issue #22 is exercised;
- stock ComfyUI MiniMax H3 loaders, conditioning, decoding, and video output nodes;
- the Flow-Aligned patched `MODEL` is wired to both `BasicScheduler.model` and `BasicGuider.model`.

Available visual workflows:

- `examples/progressive-target-input.workflow.json`: target-grid latent is created up front, the early stage runs at `source_scale=0.70`, then hands off to the target grid. The example uses the dependency-free `bicubic` transfer mode.
- `examples/progressive-source-input.workflow.json`: generation starts from an `864x480` source-grid latent and hands off to a `1.20x` target scale in the same sampler run.

The filenames in the loader nodes match the public Comfy-Org MiniMax H3 model layout. If your local files have different names, change only the loader values.

## API-format equivalents

`examples/*.api.json` contain the same two minimal paths as complete ComfyUI API-format prompt graphs. Submit them with the normal ComfyUI API client or POST their JSON as the `prompt` payload to `/prompt`.

- `examples/progressive-target-input.api.json`
- `examples/progressive-source-input.api.json`

## Wiring overlays

The `*.overlay.json` files in the directory root are **not** serialized ComfyUI canvas workflows. They are versioned wiring/specification overlays used to document intended integration topology, optional companion nodes, invariants, and benchmark variants without embedding a large machine-specific graph.

Their previous location under `workflows/` made them easy to mistake for directly loadable workflows. Use the suffixes as the contract:

- `*.workflow.json` = loadable ComfyUI canvas workflow;
- `*.api.json` = executable API prompt graph;
- `*.overlay.json` = topology/specification document only.
