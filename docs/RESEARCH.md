# Research transfer notes

This design uses public work as evidence and inspiration while keeping H3 assumptions explicit.

| Source | Useful transfer | Boundary for H3 |
|---|---|---|
| [HiFlow](https://arxiv.org/abs/2504.06232) | A low-resolution flow trajectory contains time-varying structure; its separate initialization, direction, and acceleration alignments motivate the experiments here. | Published experiments are image models. H3 two-pass guidance operates on video denoised estimates and preserves packed audio. It does not expose a prediction-space correction as HiFlow initialization alignment, and the current acceleration option is a first-difference proxy rather than HiFlow's Eq. 9-13 construction. |
| [FrescoDiffusion](https://arxiv.org/abs/2603.17555) | Video high-resolution generation benefits from a global low-resolution prior and separately testable consistency constraints. | Its tiled/posterior mechanics are not copied into H3's packed transformer. |
| [RALU](https://arxiv.org/abs/2507.08422) | Naively resizing noisy state causes distribution/timestep mismatch; transitions need explicit noise semantics. | Its correlated-noise/JSD result assumes a particular upsampler and schedule. H3 uses a conditional RF state plus native exact probe. |
| [CineScale](https://arxiv.org/abs/2508.15774) | Self-cascade video regeneration and high-resolution attention/position degradation are relevant hypotheses. | Wan-specific position methods are not transferred to H3 MM-RoPE. |
| [FreeSwim](https://arxiv.org/abs/2511.14712) | Local-detail and global-semantic paths motivate selective spatial locality. | H3 is one multimodal stream; non-video and temporal paths remain global. |
| [HRDiT](https://arxiv.org/abs/2608.07003) | Per-head/layer heterogeneous scopes and position diagnostics are worthwhile. | No head policy is recommended before H3 measurements. |
| [Just-in-Time](https://arxiv.org/abs/2603.10744) | Fine spatial compute can be deferred while structure forms. | JiT micro-flow token activation is not an arbitrary latent-resize license. |
| [Self-Cascade Diffusion](https://arxiv.org/abs/2402.10491) | Semantic pivots and staged resolution adaptation support coarse-to-fine design. | Trained/model-specific modules are outside this plugin. |
| [simple diffusion](https://arxiv.org/abs/2301.11093) | Resolution changes the appropriate signal-to-noise relationship. | Its argument is not reduced to an unexplained H3 sigma multiplier. |
| [SD3 / Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) | The constant-observation derivation yields `alpha=sqrt(m/n)` and a fractional-linear map. | The constant-image assumption is unrealistic, as the paper notes; H3 composition is only a probe. |

## Source implementations

- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) describes a progressive 2K product path
  and sparse-attention capability. H3-Regenerate-2K and its sparse topology are not public.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) supplies the executable H3 contract for packing,
  shifts, sampling, layout, padding, and wrapper lifecycle.
- [Spectrum](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3),
  [RefDelta](https://github.com/xmarre/ComfyUI-MiniMax-H3-RefDelta-Solver),
  [Continuum](https://github.com/xmarre/ComfyUI-H3-Continuum),
  [the latent upscaler](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler),
  [DiffAid](https://github.com/xmarre/ComfyUI-DiffAid-Patches), and
  [Untwisting RoPE](https://github.com/xmarre/ComfyUI-Untwisting-RoPE) informed integration contracts.
  No implementation was copied from repositories with unclear licensing.

## Claims deliberately withheld

- No quality or speed improvement is claimed without matched decoded-media runs.
- Progressive handoff is a defensible local approximation, not MiniMax's trained regenerate model.
- The attention mask is an experiment, not MiniMax's proprietary sparse attention.
- A direct-reference row cap is not inferred from PAB or generic token-pruning results.

## Validated source revisions

CI checks the executable contracts at ComfyUI `567275141678c9fd65bafef6aa9dcb4ac9bd70e3`,
Spectrum `beb32dd210ef9e95520453107f158241d4f2ecf3`, Continuum
`bf25353d8bec44afea22c89717c4301ce13c2036`, DiffAid
`ba9d9efbcf7e64c755e068cb76547d8cc85481eb`, RefDelta
`034e4c4c14c56bf76813cee4765e7164b0c7e0db`, and Untwisting RoPE
`299d4c56a3f057a97b3140d2136189bcd1e7d6bb`. The audit also inspected current MiniMax-H3 main
`d21241f0a4b3acbb34c97dae47fa417b7065e438`. Spectrum PR #98 and its companion Untwist PR #5
were reviewed as open compatibility work; this package does not make either unmerged PR a dependency.
Updating a pin requires re-running the source audit and decoded-media compatibility matrix.
