## Coordinated production release

Flow v0.3.6 is coordinated with [ComfyUI-Sol-H3 v0.1.6](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.6), [ComfyUI-VDN-H3-Plus v1.5.6](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.6), and [H3 Continuum v3.4.4](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.4). [Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28) remains unchanged.

Production consolidation PRs: [Flow #73](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/73), [Sol-H3 #32](https://github.com/xmarre/ComfyUI-Sol-H3/pull/32), [VDN-H3-Plus #32](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/32), and [Continuum #34](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/34). Flow #73 consolidates the independently hardware-validated Flow #70 source tree; the diagnostic branches remain unmerged.

The coordinated release set is:

```text
MiniMax H3 Flow-Aligned Regenerate v0.3.6
ComfyUI-Sol-H3                    v0.1.6
ComfyUI-VDN-H3-Plus              v1.5.6
H3 Continuum                     v3.4.4
Spectrum MiniMax H3              v0.2.28 (unchanged)
```

Sol supplies the production partitioned SM120 attention route, VDN supplies the heterogeneous exact-prefix contract plus the final sampler-admission VRAM fix, Flow owns the fast source-uniform progressive continuation, and Continuum owns physical prompt/audio assembly at chunk and terminal boundaries.

Keyless research and historical diagnostic/A-B PR families are deliberately outside the release set.
