# Research transfer notes

This design uses public work as evidence and inspiration while keeping H3 assumptions explicit. Decoded-media results determine which transferred ideas are retained as practical recommendations. Full author, paper, code-repository, inspected-snapshot, license, and node-by-node attribution is maintained in [../CREDITS.md](../CREDITS.md).

| Source | Useful transfer | Boundary for H3 |
|---|---|---|
| [HiFlow](https://arxiv.org/abs/2504.06232) | Low-resolution flow trajectories contain time-varying structure; direction and acceleration alignment motivate the guidance experiments. | Published experiments are image models. H3 is packed AV. Direction is a low-frequency predicted-clean correction; acceleration reconstructs H3 velocity from `x0 = x - sigma * v`. Correct transfer does not imply a media win. |
| [FrescoDiffusion](https://arxiv.org/abs/2603.17555) | High-resolution video can benefit from a global low-resolution prior and spatially selective trust. | Its tiled fusion and activity masks are not copied. H3 uses captured clean-state trajectories and a conservative correspondence gate. |
| [TokenFlow](https://arxiv.org/abs/2307.10373) | Diffusion-space features can expose useful inter-frame correspondences. | This project does not replace H3 attention tokens; it tests local correspondence directly on captured H3 video latents. |
| [FRESCO](https://arxiv.org/abs/2403.12962) | Explicit spatial/temporal correspondence motivates stronger validity constraints than unconstrained feature copying. | No external optical-flow network is added; mutual local latent matching is used as a bounded visibility gate. |
| [MoVideo](https://arxiv.org/abs/2311.11325) | Warped latent guidance plus an explicit occlusion signal reinforces that correspondence should be trusted selectively. | The H3 temporal mode uses no trained flow/depth generator and no MoVideo conditioning pipeline. |
| [Upscale-A-Video](https://arxiv.org/abs/2312.06640) | Flow-guided recurrent latent propagation is evidence that latent-space correspondence can support temporal stability. | The current mode is local adjacent-frame matching rather than a recurrent propagation network. |
| [LatentWarp](https://arxiv.org/abs/2311.00353) | Warping latent features along motion correspondence motivates latent-level temporal constraints. | H3 correspondence is inferred locally from captured clean-state latents rather than external optical flow. |
| [RALU](https://arxiv.org/abs/2507.08422) | Naively resizing noisy state causes distribution/timestep mismatch; spatial transitions need explicit noise semantics. | H3 uses a conditional rectified-flow state, exact transition probe, and fresh sampler lifetime instead of copying RALU's noise construction. |
| [CineScale](https://arxiv.org/abs/2508.15774) | Self-cascade video regeneration and high-resolution degradation are relevant hypotheses. | Wan-specific positional changes are not transferred to H3 MM-RoPE. |
| [FreeSwim](https://arxiv.org/abs/2511.14712) | Separate global and local-detail paths motivate selective spatial locality. | H3 is one multimodal stream; non-video and temporal paths remain global. |
| [OpenVDN](https://openvdn.github.io/) | Complete 5-frame temporal chunks, previous/current/next chunk reach, global non-video tokens, and symmetric boundary anchors define a concrete H3 attention topology to measure. | Flow implements an independent output-neutral diagnostic and dense-mask oracle only. It does not include the trained linear branch, FlexAttention kernel, or separately licensed VDN-H3 weights. |
| [HRDiT](https://arxiv.org/abs/2608.07003) | Positional-capacity analysis and per-head/layer scope diagnostics are worth investigating. | No sparse-head policy is recommended without H3 measurements. |
| [ResDiT](https://arxiv.org/abs/2512.01426) | Separating global-layout positional behavior from local-detail receptive-field behavior supports the Attention Lab's local/global diagnostic framing. | No ResDiT PE scaling, Gaussian patch splicing, or spectral fusion is implemented. |
| [Just-in-Time](https://arxiv.org/abs/2603.10744) | Fine spatial compute can be deferred while global structure forms. | JiT SAG-ODE/DMF token activation is not treated as a license for arbitrary latent resizing. |
| [Self-Cascade Diffusion](https://arxiv.org/abs/2402.10491) | Semantic pivots and staged resolution adaptation support coarse-to-fine design. | Trained/model-specific cascade modules are outside this plugin. |
| [simple diffusion](https://arxiv.org/abs/2301.11093) | Resolution changes the appropriate signal-to-noise relationship. | This does not by itself justify an H3 sigma multiplier. |
| [SD3 / Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) | The constant-observation derivation gives `alpha=sqrt(m/n)` and a fractional-linear time map. | The constant-image assumption is only a probe; H3 already has native shifts and packed AV coupling. |
| [Spectrum](https://arxiv.org/abs/2603.01623) | Feature forecasting makes actual-vs-forecast provenance and history lifetime relevant to H3 trajectory capture and geometry changes. | Spectrum forecasting is implemented by the sibling Spectrum repository, not this package. |
| [SA-Solver](https://arxiv.org/abs/2309.05019) | Predictor/corrector and multistep-history semantics constrain capture, PECE acceleration history, and progressive reset behavior. | H3 sampler integration is supplied by the sibling RefDelta/solver work; this package preserves rather than reimplements the solver. |
| [RACER](https://arxiv.org/abs/2608.01740) | Forecast disagreement/trust/refresh is useful secondary context for acceleration robustness. | No RACER controller or disagreement-based refresh policy is implemented here. |
| [PAB](https://arxiv.org/abs/2408.12588), DiffCR / VideoDiffCR | Attention/token efficiency is useful secondary context when investigating H3 reference growth and attention cost. | These works do **not** establish a fixed H3 reference-token budget; no published pruning/broadcast policy is copied by Reference Budget. |

## Progressive spatial handoff: empirical result

The strongest practical result is the Continuum-safe **Progressive Handoff (Target Input)** path. It keeps the public workflow/session at target geometry while running early H3 denoising on a private smaller video grid, then performs one exact low-grid probe and starts a fresh target-grid sampler lifetime.

The reset is essential: SA-Solver/PECE Adams history and Spectrum hidden-feature history do not cross the spatial geometry boundary. Audio stays at target representation and is never spatially resized.

The Target Input boundary now also permits an experimental learned 3D transfer supplied by the
companion latent-upscaler package. It consumes the exact-probe clean 24-channel video latent and emits
the exact target H/W before the existing conditional re-noise equation. This is deliberately narrower
than the companion integrated refine node: it adds one CNN call, no second H3 sampling pass, and never
touches audio. Bicubic remains the default and the prior evidence baseline.

Decoded media now validates the learned boundary beyond the originally planned D14 46×46→56×56
control. Around a 0.995 MP target, an aggressive bicubic transition showed substantial body/spatial
artifacts in the tested difficult prompt; changing only the exact-probe clean-video transfer to the
versioned learned 3D provider fixed the majority of them. At that target, `source_scale=0.70` resolved
to 800×608→1152×864 and was judged excellent, whereas `0.65` resolved to 736×576→1152×864 and began
losing reference likeness / tonal stability.

A final `source_scale=0.70` higher-resolution gate resolved 832×640→1184×896 (~1.061 MP). The generated
action was different but the decoded result was again judged very good. The run preserved the expected
38 logical / 28 actual / 10 forecast topology, two exact probes, six sampler invocations, four history
boundaries, copied audio, and actual high-grid anchors. Learned BF16 CUDA inference cost about 0.60 s
and 0.77 s for the two physical chunks and added no H3 NFE. These learned-transfer media runs used
`direction+acceleration`, so they do not change the earlier conclusion that acceleration has not shown
a clear matched advantage over direction-only. Around 1 MP, 0.70 is the current tested learned-transfer
quality/compute sweet spot for this prompt; no cross-prompt optimum is claimed.

The original difficult-motion D10 reference established the topology at a rectangular 736x768 private source -> 896x928 target. A later matched square quality sweep used 736x736 -> 896x896 and showed a consistent subjective improvement from 10 -> 12 -> 14 SA-Solver-PECE outer steps with fixed handoff 0.35 and direction weight 0.25.

At 14 outer steps, fixed 0.35 snapped to unshifted coordinate ~0.358 / schedule index 9. Across two chunks the run recorded 54 logical model calls, 36 actual H3 NFEs, and 18 Spectrum forecasts, effectively allocating 9 low-grid and 5 high-grid outer steps per chunk. The user judged it again slightly better than the 12-step result. This is the current tested quality operating point, while 10/12 remain valid faster points.

This does **not** mean 14 progressive steps are mathematically equivalent to the historical 7+7 two-pass workflow. The useful comparison is compute placement: the progressive schedule spends most early work on the cheaper low grid and only the remaining tail on the target grid, whereas 7+7 performs a distinct second learned-refine pass.

## Handoff selection result

For the matched 46x46 -> 56x56 latent setup, geometry-based `auto_compute` produced

\[
0.35\sqrt{\frac{46^2}{56^2}} = 0.2875,
\]

which snapped to ~0.30/index 7. The topology remained 38 logical / 28 actual / 10 forecast calls, but work shifted toward the low stage: 26 low logical / 18 actual / 8 forecast and 10 high logical / 8 actual / 2 forecast, plus the two exact probes.

Decoded media was valid and different, but no clear quality advantage over fixed 0.35 was established. `auto_compute` is therefore retained as a functional optional policy, not the preferred quality setting.

## Direction, acceleration, and consistency

### Direction

Low-frequency direction guidance is the only guidance term with repeated useful media evidence and is the current practical recommendation at weight 0.25.

### HiFlow-style acceleration

The initial PECE adaptation exposed a real semantic bug: predictor and corrector occur at the same sigma, but the acceleration-history slot was advanced after the predictor. The corrected implementation keeps the previous **distinct-coordinate** anchor through both predictor and same-coordinate corrector, then promotes the final corrected endpoint only when the coordinate advances.

The corrected implementation is structurally validated. However, matched decoded media showed no clear improvement in fast-motion clothing, newly revealed background, or disocclusion behavior. The result is neutral/inconclusive; do not tune acceleration weight from this case or advertise a quality gain.

### Downsample consistency

`downsample_consistency` compares the downsampled target predicted-clean state to the low-grid reference and lifts the error back to the high grid. At weight 0.25 it was definitely active: corrections reached roughly 1.5-6.8% of baseline RMS without clamping. Decoded media nevertheless looked the same and did not fix the fast-motion clothing artifact. It remains experimental and is not preferred over direction.

## Occlusion-aware temporal correspondence: final result

`direction+temporal` tests video-frame time rather than denoising-time curvature. The v2 matcher uses:

- local cosine search on adjacent exact low-grid H3 clean-state video latents;
- hard minimum similarity;
- hard best-vs-second-best uniqueness margin;
- exact reverse-cycle consistency by default;
- zero temporal copy in ambiguous/disoccluded regions;
- cached correspondence keyed by the resolved low-grid reference coordinate.

Two early media runs are excluded from temporal attribution because the workflow accidentally used a TensorRT H3 VAE compiled as `w4a16_awq`; that decoder caused the repeated/patterned grass. The matcher bug found during those runs was still real and remains fixed.

A clean D10 standard-VAE rerun was a weak/tentative positive, so the hypothesis was rechecked at the final 14-step quality budget. The D14 direction-only and D14 `direction+temporal` runs had identical structural work: 54 logical calls, 36 actual H3 NFEs, 18 Spectrum forecasts, fixed handoff ~0.358/index 9.

The temporal term was active but extremely sparse:

- chunk 1 valid support ~0.2947%, confidence mean ~0.001567, similarity mean ~0.9213;
- chunk 2 valid support ~0.08445%, confidence mean ~0.000314, similarity mean ~0.7435;
- mean temporal RMS ratio across 18 guidance calls ~0.001039 (~0.104%);
- mean direction RMS ratio ~0.021881 (~2.19%);
- maximum temporal RMS ratio ~0.003480 (~0.348%).

The user saw **no perceptual difference** between D14 direction-only and D14 direction+temporal. This closes temporal as a functioning but currently neutral experimental feature. The limiting factor is matcher support, not evidence that a larger temporal weight is needed; a weight sweep is not justified.

## Resolution-aware refine SIGMAS: final result

The resolution experiment is a downstream learned-refine schedule test. Correct topology is:

```text
MP/base sizing ---------------------------> Continuum width/height
       \
        -> Refine Target Geometry --------> Resolution-Aware Sigmas target metadata

existing refine scheduler SIGMAS
        -> Resolution-Aware Sigmas
        -> MiniMax H3 Latent Upscaler + Refine (3D).sigmas
```

Refine Target Geometry never feeds Continuum, and Resolution-Aware Sigmas never feeds Continuum SIGMAS. The real learned latent upscaler/refiner remains enabled in both control and treatment.

The mapping uses the shared unshifted flow coordinate, composes an area-derived relative factor with H3's native video shift 12, and derives audio sigma from the transformed shared coordinate. `source_width=source_height=0` means the analytic H3-native reference; it does not mean the physical base latent.

The completed square E pair used a 928x928 refine target. For that aspect, the H3-native analytic reference is 768x768, so the actual manual 768x768 reference used in the smoke is numerically identical to auto 0/0. E1 therefore tested area ratio 1.46006944, relative factor 1.20833333, effective video shift 14.5, and a nonzero sigma remap; E0 was exact identity at native shift 12. Both completed with 52 logical calls / 34 actual H3 NFEs / 18 Spectrum forecasts.

Decoded E1 showed no relevant quality improvement over E0. A boundary hiccup appeared in E1 but not E0; the separate First Frame/start-composition recurrence appeared in both and is therefore not attributable to the resolution remap. No strength sweep is justified. Keep resolution-aware mode off/experimental.

## Claims deliberately withheld

- No broad quality or speed improvement is claimed from one difficult-motion workflow.
- Progressive handoff is a training-free local approximation, not MiniMax's trained regenerate model.
- The attention experiment is not MiniMax's proprietary sparse attention.
- A direct-reference row cap does not change already encoded Qwen3-VL tokens.
- Neutral experimental features remain in the code for research/diagnostics but are not promoted as defaults.

## Source implementations

- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) documents the public H3 model family and a progressive 2K product path; H3-Regenerate-2K internals are not public.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) supplies the executable H3 packing, shift, sampler, layout, and wrapper contracts.
- [Spectrum](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3), [RefDelta](https://github.com/xmarre/ComfyUI-MiniMax-H3-RefDelta-Solver), [Continuum](https://github.com/xmarre/ComfyUI-H3-Continuum), [the latent upscaler](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler), [DiffAid](https://github.com/xmarre/ComfyUI-DiffAid-Patches), and [Untwisting RoPE](https://github.com/xmarre/ComfyUI-Untwisting-RoPE) informed integration contracts.
- Official research-code repositories and the exact supplied research snapshots inspected during design are listed in [../CREDITS.md](../CREDITS.md).

## Validated source revisions

CI checks executable contracts at:

- ComfyUI `1af040bf022569d7a890241c8dd79b296cda483f`
- Spectrum `beb32dd210ef9e95520453107f158241d4f2ecf3`
- Continuum `bf25353d8bec44afea22c89717c4301ce13c2036`
- OpenVDN code `b8cb28fbfca0266d1c7742a9f25ab8b58191de97` (Apache-2.0 code inspected; weights not downloaded or used)
- DiffAid `ba9d9efbcf7e64c755e068cb76547d8cc85481eb`
- RefDelta `034e4c4c14c56bf76813cee4765e7164b0c7e0db`
- Untwisting RoPE `299d4c56a3f057a97b3140d2136189bcd1e7d6bb`
- H3 latent upscaler/refine and learned-handoff provider `bdc670e5926bcefbe4022e17fe8b171fbfcf15de` (merged provider revision; released by companion v0.2.0)

MiniMax-H3 main was additionally inspected at `d21241f0a4b3acbb34c97dae47fa417b7065e438`. Updating a source pin requires re-running the source audit and compatibility tests.
