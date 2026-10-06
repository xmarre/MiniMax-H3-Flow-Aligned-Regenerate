## Coordinated release set

Update the coordinated components together. Every release links this same
version set and identifies its implementation PRs.

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
