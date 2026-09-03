# Research transfer notes

This design uses public work as evidence and inspiration while keeping H3 assumptions explicit.

| Source | Useful transfer | Boundary for H3 |
|---|---|---|
| [HiFlow](https://arxiv.org/abs/2504.06232) | A low-resolution flow trajectory contains time-varying structure; its separate initialization, direction, and acceleration alignments motivate the experiments here. | Published experiments are image models. H3 preserves packed audio and does not mislabel a prediction-space correction as HiFlow initialization alignment. Direction remains a low-frequency predicted-clean correction; acceleration now adapts HiFlow's adjacent full-spectrum velocity update by reconstructing H3 velocity from the native `x0 = x - sigma * v` sampling contract. Video-quality transfer remains an empirical question. |
| [FrescoDiffusion](https://arxiv.org/abs/2603.17555) | Video high-resolution generation benefits from a global low-resolution prior; its regional variant shows that prior strength should vary spatially instead of forcing every region equally. | Its tiled fusion and SAM-derived activity masks are not copied into H3. A local H3-latent correspondence-transfer adaptation was tested twice and retired after patterned-media regressions; future transfer should preserve the paper's global/regional prior principle without direct nearest-neighbor detail copying. |
| [TokenFlow](https://arxiv.org/abs/2307.10373) | Diffusion-space features can expose useful inter-frame nearest-neighbor correspondences for training-free temporal propagation. | Direct local nearest-neighbor transport on H3's clean-state video latent was tested and failed decoded-media validation twice. That negative result is retained rather than assuming TokenFlow's editing-time feature correspondence transfers to H3 progressive regeneration. |
| [FRESCO](https://arxiv.org/abs/2403.12962) | Temporal correspondence needs an explicit validity/occlusion notion; forward/backward consistency is useful evidence for rejecting unreliable propagation. | FRESCO uses optical flow and image/video translation features. A dependency-free H3-latent analogue with strict reverse-cycle gating still produced a visible patterned regression, so that transport mechanism is removed rather than treated as a successful FRESCO transfer. |
| [RALU](https://arxiv.org/abs/2507.08422) | Naively resizing noisy state causes distribution/timestep mismatch; transitions need explicit noise semantics. | Its correlated-noise/JSD result assumes a particular upsampler and schedule. H3 uses a conditional RF state plus native exact probe. |
| [CineScale](https://arxiv.org/abs/2508.15774) | Self-cascade video regeneration and high-resolution attention/position degradation are relevant hypotheses. | Wan-specific position methods are not transferred to H3 MM-RoPE. |
| [FreeSwim](https://arxiv.org/abs/2511.14712) | Local-detail and global-semantic paths motivate selective spatial locality. | H3 is one multimodal stream; non-video and temporal paths remain global. |
| [HRDiT](https://arxiv.org/abs/2608.07003) | Per-head/layer heterogeneous scopes and position diagnostics are worthwhile. | No head policy is recommended before H3 measurements. |
| [Just-in-Time](https://arxiv.org/abs/2603.10744) | Fine spatial compute can be deferred while structure forms. | JiT micro-flow token activation is not an arbitrary latent-resize license. |
| [Self-Cascade Diffusion](https://arxiv.org/abs/2402.10491) | Semantic pivots and staged resolution adaptation support coarse-to-fine design. | Trained/model-specific modules are outside this plugin. |
| [simple diffusion](https://arxiv.org/abs/2301.11093) | Resolution changes the appropriate signal-to-noise relationship. | Its argument is not reduced to an unexplained H3 sigma multiplier. |
| [SD3 / Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) | The constant-observation derivation yields `alpha=sqrt(m/n)` and a fractional-linear map. | The constant-image assumption is unrealistic, as the paper notes; H3 composition is only a probe. |

## Occlusion-aware temporal experiment — negative transfer result

The difficult-motion D10 smoke isolated a failure that denoising-time acceleration did not clearly
improve: moving clothing and newly revealed background can become malformed even when overall action
and shot structure remain good. Two experiments therefore tested a separate **video-frame-time**
hypothesis using the captured low-resolution H3 clean-state trajectory.

V1 matched adjacent low-grid latent frames with bounded local cosine nearest neighbors and transported
high-grid neighbor detail only where forward/backward correspondence appeared valid. It failed decoded
media: parts of the grass developed a repeated/patterned texture and motion artifacts became
different/more broken. Telemetry showed that high cosine similarity was not enough in repetitive
texture: mean best-vs-second-best margins were only ~0.00381/~0.00269, yet ~46%/~51% of locations
still received some temporal support.

V2 fixed that concrete gating defect rather than changing the strength. It made minimum similarity and
uniqueness margin hard admission rules, required exact reverse-cycle consistency, and keyed the
correspondence cache by the **resolved** low-grid reference coordinate. That last correction mattered
because progressive low-grid sampling ends at the handoff; all later high-stage requests below ~0.4
resolve to the same final exact low-grid anchor.

The conservative v2 gate behaved exactly as intended structurally, but the decoded pattern remained.
Its valid temporal support collapsed to ~0.2202%/~0.2235% for the two chunks, mean confidence to
~0.00118/~0.00120, and the mean temporal RMS ratio across all 14 high-stage guidance calls to only
~0.00137. The correspondence cache correctly reused the final ~0.4 anchor on 12/14 calls. Even with
that tiny global support, the localized repeated texture remained visible.

This is decisive negative evidence for **nearest-neighbor latent detail transport in this H3 progressive
setup**. The experiment is closed:

- no `direction+temporal` public mode is retained;
- no temporal transport cache or widget remains in runtime;
- no further weight/radius/margin/cycle sweep is justified;
- the metrics and media result remain documented as a failed transfer.

The broader temporal problem is not declared solved or impossible. The failure is narrower: sparse
local copying is unsafe here. FrescoDiffusion's evidence instead favors coherent global/regional prior
regularization, while FreeSwim shows that local-detail mechanisms need a global branch to prevent
repetition. CineScale likewise treats repetition as a multi-scale/global-context problem rather than
simply propagating local detail. These sources motivate any future motion-specific work toward a
**non-copying global/local prior formulation** or attention/representation study, not another
nearest-neighbor transport implementation.

For the current PR, the active sequence therefore returns to the already-planned
**resolution-shift-only path E**. That path changes the shared H3 AV flow coordinate without adding
video-time detail propagation and remains separately media-gated.


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
