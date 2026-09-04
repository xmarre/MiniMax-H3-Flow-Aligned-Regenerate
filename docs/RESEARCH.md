# Research transfer notes

This design uses public work as evidence and inspiration while keeping H3 assumptions explicit. Decoded-media results determine which transferred ideas are retained as practical recommendations.

| Source | Useful transfer | Boundary for H3 |
|---|---|---|
| [HiFlow](https://arxiv.org/abs/2504.06232) | Low-resolution flow trajectories contain time-varying structure; direction and acceleration alignment motivate the guidance experiments. | Published experiments are image models. H3 is packed AV. Direction is a low-frequency predicted-clean correction; acceleration reconstructs H3 velocity from `x0 = x - sigma * v`. Correct transfer does not imply a media win. |
| [FrescoDiffusion](https://arxiv.org/abs/2603.17555) | High-resolution video can benefit from a global low-resolution prior and spatially selective trust. | Its tiled fusion and activity masks are not copied. H3 uses captured clean-state trajectories and a conservative correspondence gate. |
| [TokenFlow](https://arxiv.org/abs/2307.10373) | Diffusion-space features can expose useful inter-frame correspondences. | This project does not replace H3 attention tokens; it tests local correspondence directly on captured H3 video latents. |
| [FRESCO](https://arxiv.org/abs/2403.12962) | Temporal propagation needs explicit occlusion/validity evidence and forward/backward consistency. | No external optical-flow network is added; mutual local latent matching is used as a bounded visibility gate. |
| [RALU](https://arxiv.org/abs/2507.08422) | Naively resizing noisy state causes distribution/timestep mismatch; spatial transitions need explicit noise semantics. | H3 uses a conditional rectified-flow state, exact transition probe, and fresh sampler lifetime instead of copying RALU's noise construction. |
| [CineScale](https://arxiv.org/abs/2508.15774) | Self-cascade video regeneration and high-resolution degradation are relevant hypotheses. | Wan-specific positional changes are not transferred to H3 MM-RoPE. |
| [FreeSwim](https://arxiv.org/abs/2511.14712) | Separate global and local-detail paths motivate selective spatial locality. | H3 is one multimodal stream; non-video and temporal paths remain global. |
| [HRDiT](https://arxiv.org/abs/2608.07003) | Per-head/layer scope diagnostics are worth investigating. | No sparse-head policy is recommended without H3 measurements. |
| [Just-in-Time](https://arxiv.org/abs/2603.10744) | Fine spatial compute can be deferred while global structure forms. | JiT micro-flow token activation is not treated as a license for arbitrary latent resizing. |
| [Self-Cascade Diffusion](https://arxiv.org/abs/2402.10491) | Semantic pivots and staged resolution adaptation support coarse-to-fine design. | Trained/model-specific cascade modules are outside this plugin. |
| [simple diffusion](https://arxiv.org/abs/2301.11093) | Resolution changes the appropriate signal-to-noise relationship. | This does not by itself justify an H3 sigma multiplier. |
| [SD3 / Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) | The constant-observation derivation gives `alpha=sqrt(m/n)` and a fractional-linear time map. | The constant-image assumption is only a probe; H3 already has native shifts and packed AV coupling. |

## Progressive spatial handoff: empirical result

The strongest practical result is the Continuum-safe **Progressive Handoff (Target Input)** path. It keeps the public workflow/session at target geometry while running early H3 denoising on a private smaller video grid, then performs one exact low-grid probe and starts a fresh target-grid sampler lifetime.

The reset is essential: SA-Solver/PECE Adams history and Spectrum hidden-feature history do not cross the spatial geometry boundary. Audio stays at target representation and is never spatially resized.

The Target Input boundary now also permits an experimental learned 3D transfer supplied by the
companion latent-upscaler package. It consumes the exact-probe clean 24-channel video latent and emits
the exact target H/W before the existing conditional re-noise equation. This is deliberately narrower
than the companion integrated refine node: it adds one CNN call, no second H3 sampling pass, and never
touches audio. Bicubic remains the default and the prior evidence baseline.

The intended D14 test is strict: 46×46→56×56, 14 outer steps, source scale 0.83, fixed 0.35 handoff,
direction 0.25, other guidance weights zero, cutoff 0.25, with transfer mode as the only variable. The
upscaler training includes arbitrary 1–4× scales, making the approximately 1.217× geometry supported,
but 2× was much more common. Synthetic contract validation cannot establish perceptual quality or
wall-time value; those claims remain withheld pending matched decoded video/audio.

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

## Validated source revisions

CI checks executable contracts at:

- ComfyUI `1af040bf022569d7a890241c8dd79b296cda483f`
- Spectrum `beb32dd210ef9e95520453107f158241d4f2ecf3`
- Continuum `bf25353d8bec44afea22c89717c4301ce13c2036`
- DiffAid `ba9d9efbcf7e64c755e068cb76547d8cc85481eb`
- RefDelta `034e4c4c14c56bf76813cee4765e7164b0c7e0db`
- Untwisting RoPE `299d4c56a3f057a97b3140d2136189bcd1e7d6bb`
- H3 latent upscaler/refine and learned-handoff provider `5256edceabf651bdd9094c224e1907b2f0edd941` (draft provider PR #12)

MiniMax-H3 main was additionally inspected at `d21241f0a4b3acbb34c97dae47fa417b7065e438`. Updating a source pin requires re-running the source audit and compatibility tests.
