## Coordinated release set

Update the coordinated components together. Every release links this same
version set and identifies its implementation PRs.

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.9](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.9) | [#89](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/89), [#93](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/93) |
| Sol-H3 | [v0.1.8](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.8) | [#37](https://github.com/xmarre/ComfyUI-Sol-H3/pull/37) |
| VDN-H3-Plus | [v1.5.7](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.7) | [#33](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/33), [#34](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/34), [#35](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/35), [#36](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/36), [#37](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/37) |
| H3 Continuum-Plus | [v3.4.5](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.5) | [#37](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/37), [#38](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/38) |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | [#16](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/pull/16) |

[Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28)
is the unchanged companion. Separate Keyless, audio-training and rejected
decoded-geometry experiments are outside this release set.

The tested Core adapter repair is
[ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783).
It remains an upstream review item, with upstream workflow approval and merge
controlled by Comfy-Org maintainers. For INT8 fused MLP runtime adapters,
retain that ComfyUI Patcher PR overlay until the repair is available upstream.
The independent Core #16720 optimization is not included in this release set.
