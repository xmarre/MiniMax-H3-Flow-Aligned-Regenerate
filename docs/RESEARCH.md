# Research transfer notes

This design uses public work as evidence and inspiration while keeping H3 assumptions explicit.

| Source | Useful transfer | Boundary for H3 |
|---|---|---|
| [HiFlow](https://arxiv.org/abs/2504.06232) | A low-resolution flow trajectory contains time-varying structure; its separate initialization, direction, and acceleration alignments motivate the experiments here. | Published experiments are image models. H3 preserves packed audio and does not mislabel a prediction-space correction as HiFlow initialization alignment. Direction remains a low-frequency predicted-clean correction; acceleration now adapts HiFlow's adjacent full-spectrum velocity update by reconstructing H3 velocity from the native `x0 = x - sigma * v` sampling contract. Video-quality transfer remains an empirical question. |
| [FrescoDiffusion](https://arxiv.org/abs/2603.17555) | Video high-resolution generation benefits from a global low-resolution prior; its regional variant shows that prior strength should vary spatially instead of forcing every region equally. | Its tiled fusion and SAM-derived activity masks are not copied into H3. Here the low-resolution H3 clean-state trajectory supplies the prior, while correspondence confidence supplies the spatial gate. |
| [TokenFlow](https://arxiv.org/abs/2307.10373) | Diffusion-space features can expose useful inter-frame nearest-neighbor correspondences for training-free temporal propagation. | TokenFlow is a video-editing method built on image diffusion internals. This project does not replace H3 attention tokens; it independently tests local correspondence on H3's captured clean-state video latent. |
| [FRESCO](https://arxiv.org/abs/2403.12962) | Temporal correspondence needs an explicit validity/occlusion notion; forward/backward consistency is useful evidence for rejecting unreliable propagation. | FRESCO uses optical flow and image/video translation features. This project adds no flow-network dependency and does not copy its implementation; it uses mutual local H3-latent matching as a conservative experimental visibility gate. |
| [RALU](https://arxiv.org/abs/2507.08422) | Naively resizing noisy state causes distribution/timestep mismatch; transitions need explicit noise semantics. | Its correlated-noise/JSD result assumes a particular upsampler and schedule. H3 uses a conditional RF state plus native exact probe. |
| [CineScale](https://arxiv.org/abs/2508.15774) | Self-cascade video regeneration and high-resolution attention/position degradation are relevant hypotheses. | Wan-specific position methods are not transferred to H3 MM-RoPE. |
| [FreeSwim](https://arxiv.org/abs/2511.14712) | Local-detail and global-semantic paths motivate selective spatial locality. | H3 is one multimodal stream; non-video and temporal paths remain global. |
| [HRDiT](https://arxiv.org/abs/2608.07003) | Per-head/layer heterogeneous scopes and position diagnostics are worthwhile. | No head policy is recommended before H3 measurements. |
| [Just-in-Time](https://arxiv.org/abs/2603.10744) | Fine spatial compute can be deferred while structure forms. | JiT micro-flow token activation is not an arbitrary latent-resize license. |
| [Self-Cascade Diffusion](https://arxiv.org/abs/2402.10491) | Semantic pivots and staged resolution adaptation support coarse-to-fine design. | Trained/model-specific modules are outside this plugin. |
| [simple diffusion](https://arxiv.org/abs/2301.11093) | Resolution changes the appropriate signal-to-noise relationship. | Its argument is not reduced to an unexplained H3 sigma multiplier. |
| [SD3 / Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) | The constant-observation derivation yields `alpha=sqrt(m/n)` and a fractional-linear map. | The constant-image assumption is unrealistic, as the paper notes; H3 composition is only a probe. |

## Occlusion-aware temporal experiment

The difficult-motion D10 smoke isolated a failure that denoising-time acceleration did not clearly
improve: moving clothing and newly revealed background can become malformed even when the overall
trajectory and shot remain good. The new `direction+temporal` mode addresses a different axis:
**video-frame time**.

At each denoising coordinate, the exact low-resolution H3 clean-state prior is lightly smoothed and
adjacent latent frames are matched with a bounded local cosine search. Both directions are computed.
A match is trusted only to the degree that similarity, best-vs-second-best margin, and reverse-match
cycle consistency agree. This produces a spatial confidence/visibility gate without RAFT/GMFlow or
another learned network.

For a trusted current-to-neighbor match, high-resolution detail is motion-transported from that
neighbor and augmented with the low-resolution prior's same-time innovation:

[
P_i = W(H_j) + U(R_i - W(R_j)).
]

The temporal correction is confidence-weighted `P_i - H_i` and is combined with the existing
same-time low-frequency direction term before the global RMS guard. Regions without a trustworthy
predecessor/successor receive no temporal copy; they fall back to same-time low-resolution direction
guidance so genuinely disoccluded content remains free to be synthesized.

The matcher is bidirectional and non-recursive: every frame uses the current model call's unmodified
neighbor estimates, so temporal propagation cannot accumulate sequential warp error through the clip.
Its correspondence tensors are cached only across repeated PECE evaluations at the same denoising
coordinate and are discarded with the sampler guidance state.

This mode is **structurally tested but not decoded-media validated**. It is deliberately separate from
HiFlow acceleration so the next D10 run can isolate video-time correspondence from denoising-time
curvature.

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

CI checks the executable contracts at ComfyUI `1af040bf022569d7a890241c8dd79b296cda483f`,
Spectrum `beb32dd210ef9e95520453107f158241d4f2ecf3`, Continuum
`bf25353d8bec44afea22c89717c4301ce13c2036`, DiffAid
`ba9d9efbcf7e64c755e068cb76547d8cc85481eb`, RefDelta
`034e4c4c14c56bf76813cee4765e7164b0c7e0db`, Untwisting RoPE
`299d4c56a3f057a97b3140d2136189bcd1e7d6bb`, and the integrated H3 latent upscaler/refine
`2c707492084962f7ed665e8817a05a11b14dab27`. The audit also inspected current MiniMax-H3 main
`d21241f0a4b3acbb34c97dae47fa417b7065e438`. Spectrum PR #98 and its companion Untwist PR #5
were reviewed as open compatibility work; this package does not make either unmerged PR a dependency.
 The current ComfyUI master was re-audited after the pin and is two commits ahead; the delta is limited to `comfy_api/feature_flags.py` and `main.py`, with no H3, sampler, model-sampling, or ModelPatcher contract file changed.
Updating a pin requires re-running the source audit and decoded-media compatibility matrix.
