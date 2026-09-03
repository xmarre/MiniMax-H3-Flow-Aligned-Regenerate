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

## Resolution-shift experiment

The E path is deliberately narrower than the progressive/regeneration literature above. It changes
only the sampler SIGMAS while generating directly at the target grid.

The reference resolution is now tied to the public H3 contract rather than to D10's private low grid.
MiniMax documents H3-Base as a 768p generator and states that the shorter output side defaults to
768 pixels. The pinned ComfyUI H3 node contract matches this exactly (`BASE_SHORT_EDGE=768`,
`CANVAS_MULTIPLE=32`, `adapt_canvas`). For the current 896x928 target, preserving aspect ratio at a
768px short side and applying that 32px canvas alignment yields a 768x800 analytic reference regime.

SD3 Eq. 23 derives the relative flow-time shift

\[
\alpha=\sqrt{m/n},\qquad
t_m=\frac{\alpha t_n}{1+(\alpha-1)t_n},
\]

under a constant-image uncertainty model. For 768x800 -> 896x928, the area ratio is ~1.35333333 and
\(\alpha\approx1.16332856\). simple diffusion independently argues that increasing spatial resolution
requires a lower-SNR/more-noisy schedule at a comparable global-structure stage. Both results support
the *direction* of the experiment, but neither proves that MiniMax H3 should use this exact mapping.

H3 already uses video flow shift 12 and audio shift 3. The implementation therefore composes the SD3
factor **relatively** with shift 12 rather than replacing H3's native schedule. At strength 1 the
effective video shift is ~13.95994. Because H3 is a packed joint AV flow model, the same transformed
base coordinate also changes audio sigma. A video-only interpretation would be false; decoded audio is
part of the gate.

The E0/E1 pair cannot obtain its target geometry from the actual downstream refine node because that
node executes after Continuum sampler 1. Instead, the existing pre-Continuum MP width/height remain the
base geometry and a new geometry-only helper mirrors the pinned LBH refine node's
scale-by-multiplier calculation **before** sampler 1. With scale 1.20 / align 32 / keep_proportion true,
the current 736x768 MP base resolves to the same 896x928 canvas the downstream refine would have
produced. Those helper outputs drive both Continuum width/height and Resolution-Aware Sigmas target
width/height. The learned Latent Upscale Refine execution itself is bypassed. Resolution-Aware Sigmas
uses source_width=source_height=0 to derive the native H3 reference automatically for the target aspect;
896x928 resolves to 768x800. E0 passes SIGMAS through in `off` mode as an exact-parity control; E1
changes only to `resolution_aware` at strength 1. This isolates the schedule hypothesis without
manual per-image resolution calculation or a hidden sampler-2 refinement confound.

If E1 wins clearly, only then combine the shift with the accepted progressive+temporal path. If E1 is
neutral or worse, the mapping remains experimental/off and no strength sweep is justified from this
single pair.

## Occlusion-aware temporal experiment

The difficult-motion D10 smoke isolated a failure that denoising-time acceleration did not clearly
improve: moving clothing and newly revealed background can become malformed even when the overall
trajectory and shot remain good. `direction+temporal` tests a different axis: **video-frame time**.

The first implementation used adjacent-frame local cosine matching directly in the exact low-resolution
H3 clean-state latent, followed by bidirectional cycle gating and confidence-weighted high-resolution
detail transport. The first matched D10-temporal media run showed repeated/patterned grass and different/more-broken motion artifacts, but that media result is now **confounded**: the workflow had accidentally switched to the TensorRT H3 VAE and compiled the `w4a16_awq` decoder. The user subsequently identified that decoder as the source of the visible pattern, so the media cannot be used to attribute the artifact to temporal guidance.

The metrics still exposed a real matcher defect independent of the VAE confound. The two physical chunks reported high mean accepted cosine similarity (~0.938/~0.959)
but extremely small best-vs-second-best margins (~0.00381/~0.00269), while the configured
`temporal_min_margin` was 0.02. V1 mistakenly used that value only to scale confidence rather than
to reject ambiguous matches. Consequently ~46%/~51% of candidate locations were labeled "valid"
even though mean confidence was only ~0.056/~0.048, and accepted flow magnitude averaged ~2.55-2.57
low-grid cells with a radius-4 diagonal maximum ~5.657. Repetitive grass is precisely the kind of
region where a weakly unique nearest-neighbor field can create patterned copying.

The current v2 keeps the dependency-free H3-latent hypothesis but makes it conservative:

- minimum cosine similarity and best-vs-second-best margin are hard admission tests;
- reverse correspondence is exact by default (zero-cell cycle tolerance);
- rejected/ambiguous locations contribute exactly zero temporal copy;
- post-handoff reference clamping is explicit and cache identity follows the resolved low-grid anchor.

The last point matters because progressive low-grid sampling stops at the handoff. In D10, the exact
low-grid support ends at ~0.4, so requested high-stage coordinates ~0.3/~0.2/~0.1 all resolve to the
same ~0.4 prior. V1 recomputed the identical map for each requested coordinate; v2 reuses it and
reports the resolved coordinate/clamp state directly. The temporal correction still decays with the
existing high-stage schedule.

For a trusted current-to-neighbor match, the candidate target remains

\[
P_i = W(H_j) + U(R_i - W(R_j)).
\]

The important change is **where this operation is allowed**. V2 applies it only where the match is
unique and mutually consistent; otherwise the ordinary same-time low-frequency direction term remains
the only guidance.

The clean standard-VAE D10-temporal-v2 rerun is now complete. It preserved the expected 38 logical / 28 actual / 10 forecast topology and the conservative v2 matcher remained sparse: ~0.1598% valid support in chunk 1 and ~0.0542% in chunk 2, with mean temporal RMS ratio ~0.00090 across the 14 high-stage guidance calls. The decoded result was good, the VAE-induced repeated pattern was absent, and the user's perceptual verdict was **maybe slightly better than the non-temporal D10 control**. This is the first clean positive evidence for the hypothesis, but the effect is small and only one corrected-VAE case has been judged, so the result remains tentative rather than a general claim.

The experimental path is therefore retained at temporal weight 0.20 without a parameter sweep. The next active gate is the orthogonal resolution-shift-only experiment; if temporal promotion is considered later, use an additional matched seed/control pair rather than inferring broad benefit from this single weak positive.

This direction is consistent with the broader source evidence: FrescoDiffusion explicitly warns that
an unreliable prior can propagate incorrect structure or motion and decays/releases prior strength;
its regional variant applies different prior constraints to active and background regions. The present
H3 adaptation does not copy FrescoDiffusion's SAM activity map or tiled velocity fusion, but adopts the
same conservative principle that prior regularization must be spatially trustworthy rather than global.

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
