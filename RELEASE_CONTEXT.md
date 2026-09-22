## Coordinated production release

Flow v0.3.8 is the current Flow companion for [ComfyUI-Sol-H3 v0.1.6](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.6), [ComfyUI-VDN-H3-Plus v1.5.6](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.6), and [H3 Continuum v3.4.4](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.4). [Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28) remains unchanged.

The coordinated production set is:

```text
MiniMax H3 Flow-Aligned Regenerate v0.3.8
ComfyUI-Sol-H3                    v0.1.6
ComfyUI-VDN-H3-Plus              v1.5.6
H3 Continuum                     v3.4.4
Spectrum MiniMax H3              v0.2.28 (unchanged)
```

Flow v0.3.8 promotes the partitioned exact-prefix handoff used by the coordinated stack, including the one-token video DC continuity correction and the validated `sampler_mask_exact_timestep` / 4-tick audio default. The overlap width remains configurable from 0 through 16.

No companion release is required for Sol-H3, VDN-H3-Plus, Continuum or Spectrum for this Flow-only promotion. Keyless research and unrelated historical A/B branches remain outside the release set.
