# MiniMax H3 Flow-Aligned Regenerate

Research-grade ComfyUI nodes for H3-native coarse-to-fine generation: transactional low-resolution trajectory capture, time-aligned high-resolution guidance, and an in-flight spatial-resolution handoff.

> [!IMPORTANT]
> This project is an independent, training-free approximation informed by public research. It does **not** reproduce MiniMax's closed H3-Regenerate-2K model or its unreleased sparse-attention topology. Decoded media, not latent telemetry alone, is the quality gate.

## Research lineage and credits

This repository builds on published ideas rather than presenting flow-aligned guidance, mixed-resolution sampling, temporal correspondence, or high-resolution attention research as original inventions. **HiFlow** is the primary credit for trajectory/direction/acceleration alignment; **FrescoDiffusion** for a low-resolution video trajectory as a high-resolution prior; **RALU**, **Self-Cascade**, **CineScale**, and **Just-in-Time** for coarse-to-fine / spatial-transition reasoning; **simple diffusion** and **Scaling Rectified Flow Transformers** for resolution-dependent schedule research; and **FreeSwim**, **HRDiT**, and **ResDiT** for high-resolution DiT attention/receptive-field analysis. Temporal correspondence experiments additionally draw on **TokenFlow**, **FRESCO**, **MoVideo**, **Upscale-A-Video**, and **LatentWarp**.

See **[CREDITS.md](CREDITS.md)** for node-by-node attribution, paper authors, official paper/project/code links, inspected research-code commit snapshots, licensing notes, and the exact boundary between published methods and this H3-specific implementation. [docs/RESEARCH.md](docs/RESEARCH.md) records the corresponding transfer assumptions and experimental outcomes.

## Why this exists

Native H3 generation at the final high-resolution grid can be weaker than a lower-resolution generation followed by learned latent upscale/refinement, but the latter repeats part of the denoising trajectory. This repository tests whether H3 can reuse low-resolution trajectory information or change spatial resolution during one schedule without paying for a full second pass.

H3 is treated as its actual joint AV model: 24-channel video, 32-channel two-track audio, 16x spatial VAE compression, a `(1, 2, 2)` DiT patch, video/audio flow shifts 12/3, and a packed `text | conditioning/reference | audio | video` transformer sequence.

## Status

The implementation, guards, instrumentation, workflow specifications, and synthetic test suite are complete. Real decoded-media validation has now covered the integrated two-pass path and the important public Progressive Target Input settings.

### Validated media results

- **Integrated two-pass flow guidance:** C7=7+7, C6=7+6, C5=7+5, and exploratory C4=7+4 all completed acceptable difficult-motion smokes. This path remains useful when keeping the learned latent upscaler/refiner.
- **Progressive Target Input:** the accepted D10 direction-only run validated the Continuum-safe split topology and restored quality to roughly baseline level after the earlier under-budgeted 7-step progressive attempt.
- **Step-budget sweep:** on the later matched square difficult-motion setup (736x736 private source -> 896x896 target), direction-only improved perceptually from 10 -> 12 -> 14 SA-Solver-PECE outer steps. The 14-step run is the current tested direction-only quality operating point.
- **14-step topology:** fixed handoff 0.35 snapped to unshifted coordinate ~0.358 / index 9, giving 54 logical calls, 36 actual H3 NFEs, and 18 Spectrum forecasts across two chunks. The split is effectively 9 low-grid + 5 high-grid outer steps per chunk.
- **Learned handoff around 1 MP:** `source_scale=0.70` at 10 progressive outer steps was decoded-media positive through ~1.061 MP. In the final run, 832x640 -> 1184x896 completed in 621.63 s end-to-end with 38 logical / 28 actual H3 NFE / 10 Spectrum forecast calls. Against the historical proper 960x704 -> 1184x864 two-pass 7+6 upscale/refine workflow at ~777 s, this is an observed ~20% end-to-end wall-time reduction while producing ~3.7% more final pixels. This is a workflow-level reference, not a formal matched A/B.
- **Auto handoff:** functional. On the matched 46x46 -> 56x56 latent setup, `auto_compute` selected 0.2875 and snapped to ~0.30/index 7, reallocating work toward the low grid. Media changed but showed no clear quality advantage over fixed 0.35.
- **Downsample consistency:** functional at weight 0.25 and produced measurable latent corrections, but no meaningful decoded-media improvement. Do not prefer it over direction guidance from current evidence.
- **HiFlow-style acceleration:** corrected PECE semantics are validated, but the decoded-media result is neutral/inconclusive for the fast-motion clothing/newly revealed background artifact class.
- **Temporal correspondence:** the final matched 14-step standard-VAE `direction+temporal` run was perceptually indistinguishable from 14-step direction-only. The matcher was active but extremely sparse, so temporal remains experimental and is not promoted.
- **Resolution-aware refine SIGMAS:** matched E0/E1 learned-refine runs completed. The mapping worked structurally, but E1 showed no relevant quality improvement over the exact-identity control. Keep this mode off/experimental.

The conservative direction-only quality-sweep reference is therefore:

```text
outer_steps = 14
source_mode = scale
source_scale = 0.83
handoff_selection = fixed
handoff_coordinate = 0.35
guidance_mode = direction
direction_weight = 0.25
acceleration_weight = 0
temporal_weight = 0
consistency_weight = 0
low_frequency_cutoff = 0.25
```

This is a **tested operating point**, not a universal optimum. For lower latency, 10 or 12 outer steps remain valid operating points; 12 was already perceptually better than 10 in the matched direction-only quality sweep. The learned `learned_3d` path has a separate tested quality/compute point around 1 MP at 10 outer steps and `source_scale=0.70`, documented below; those latest learned runs used `direction+acceleration` and therefore do not replace the conservative direction-only guidance recommendation.

## Install

From `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate.git
```

Restart ComfyUI. PyTorch is supplied by ComfyUI. Sibling custom nodes are not imported by this package, although the benchmark workflow uses the surrounding H3 ecosystem described below.

## Nodes

Research attribution for every research-facing public node is mapped explicitly in [CREDITS.md](CREDITS.md); the node names below are implementation names, not claims of algorithmic originality.

| Node | Purpose | Posture |
|---|---|---|
| MiniMax H3 Flow Trajectory | Explicit `H3_FLOW_TRAJECTORY` handle with RAM/VRAM policy | Stable infrastructure |
| MiniMax H3 Trajectory Capture | Transactionally records denoised estimates and provenance | Stable infrastructure |
| MiniMax H3 Flow-Aligned Regenerate | Time-matched low-frequency guidance for an explicit second pass | Direction is conservative default |
| MiniMax H3 Flow-Aligned Refine State | Patches Continuum V3.4 per-chunk refine state for integrated learned refinement | Direction is conservative default |
| MiniMax H3 Progressive Handoff | Source-grid sampler input grows to target grid during one schedule | Experimental |
| MiniMax H3 Progressive Handoff (Target Input) | Continuum-safe target-sized topology with a private low-grid early stage; optional learned 3D clean-state transfer | Bicubic compatibility path; learned transfer decoded-media validated / experimental |
| MiniMax H3 Refine Target Geometry | Mirrors downstream learned-refine target sizing for sigma experiments | Experimental |
| MiniMax H3 Resolution-Aware Sigmas | Relative H3-native resolution/time remap | Off / experimental |
| MiniMax H3 Reference Budget | Direct-reference row diagnostics/cap | Native |
| MiniMax H3 Attention Lab | Packed-layout diagnostics and guarded sparse-attention experiment | Native |
| MiniMax H3 Runtime Metrics Probe | Passive sampler/model-call instrumentation | On demand |
| MiniMax H3 Metrics JSON | Structured counters/events and autosaved benchmark artifact | On demand |

## Progressive Target Input wiring

For Continuum, use the Target Input node rather than the source-input progressive node:

```text
DiffAid
  -> Untwist
  -> Spectrum
  -> Progressive Handoff (Target Input)
  -> Continuum
```

Create one **Flow Trajectory** handle and connect it to Progressive Handoff (Target Input). The progressive node captures the low-grid trajectory internally; a separate **Trajectory Capture** node is not required on this path.

Continuum remains configured on the final target grid. The wrapper creates a private low-grid video state for the early stage, performs an exact low-grid handoff probe, rebuilds target-grid conditioning, and starts a fresh sampler lifetime on the high grid. Audio is never spatially transformed. SA/PECE Adams history and Spectrum feature history are deliberately reset across the geometry boundary, and the first high-grid H3 call is forced actual.

`handoff_transfer=bicubic` remains the compatibility default. For the validated learned path,
install the companion latent upscaler, create **MiniMax H3 Latent Upscaler Provider (3D) [Experimental]**,
connect its `H3_LATENT_UPSCALER` output to the Target Input node, and select
`handoff_transfer=learned_3d`. The learned CNN replaces only the exact-probe clean-video spatial
transfer. Conditional re-noising, deterministic target noise, sampler/Spectrum history boundaries,
the mandatory first high-grid actual call, caller audio, masks, and all H3 model-call semantics remain
unchanged. One learned inference is added per physical chunk and no extra H3 NFE is added.

Decoded-media validation now covers substantially more aggressive transitions than the original
46×46→56×56 D14 plan. Around a 0.995 MP target, bicubic at roughly a 0.55 MP private source produced
visible body/spatial handoff artifacts in the tested difficult prompt; replacing only that boundary with
`learned_3d` fixed the majority of those artifacts. At the same target, `source_scale=0.70` resolved to
800×608→1152×864 and was judged excellent, while `0.65` resolved to 736×576→1152×864 and began losing
reference likeness / tonal stability. The final higher-resolution gate kept `source_scale=0.70` and
resolved 832×640→1184×896 (~1.061 MP target); the generated action was different but the result was
again judged very good. Its two BF16 CUDA learned calls took about 0.60 s and 0.77 s and added zero H3
NFEs. These latest learned-transfer media runs used `direction+acceleration`; they are not evidence for
promoting acceleration over direction-only.

### Observed performance versus proper two-pass upscale + refine

The useful legacy comparison is the established **7 first-pass + 6 learned-refine outer-step** workflow,
not the old 3-step high-ratio refine experiments. At nominal 0.7 MP, that proper two-pass path resolved
approximately 960×704 (0.676 MP) -> 1184×864 (1.023 MP) and took about 777 s end-to-end in the observed
run. The historical full-quality setup was 7+7; no equally clean ~1 MP raw timing has been recovered for
that exact 7+7 case, so no larger speedup is claimed from it.

| Path | Base/private grid | Final grid | Sampling structure | Observed full workflow |
|---|---:|---:|---|---:|
| Proper two-pass upscale + refine | 960×704 (0.676 MP) | 1184×864 (1.023 MP) | 7 base + 6 learned-refine outer steps | ~777 s (~12:57) |
| Progressive + `learned_3d` | 832×640 (0.532 MP) | 1184×896 (1.061 MP) | 10 progressive outer steps; 38 logical / 28 actual / 10 forecast | 621.63 s (10:21.6) |

Relative to that proper 6-step-refine baseline, the progressive learned-handoff run used ~21.2% fewer
private/source pixels, produced ~3.7% more final pixels, and saved about 155 s end-to-end: roughly a
**20% wall-time reduction / 1.25× observed workflow speedup**. The learned CNN itself was not the source
of the saving: its two calls totaled only ~1.37 s of inference (~1.43 s including transfer bookkeeping)
and added zero H3 NFEs. The saving comes from performing more of the trajectory on the cheaper private
grid and avoiding a complete second low-sigma H3 refine pass.

This timing comparison is a practical observed-workflow reference rather than a controlled same-seed
microbenchmark. The old and new runs differ slightly in final geometry and were recorded at different
points in development, so use the ~20% figure as measured evidence for this workflow, not a universal
speed claim or a quality-superiority percentage. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for the
full accounting and caveats.

For roughly 1 MP targets, `0.70` is therefore the current tested quality/compute sweet spot for this
prompt, not a universal optimum. `0.65` is below the current useful quality floor in this case, while
higher source scales remain the safer choice when preserving source-grid fidelity matters more than
compute. Keep bicubic available for compatibility and matched controls rather than silently changing
existing workflows. Provider mode fails instead of silently falling back when its configured inference
device is unavailable.

Use a complete 1-to-0 H3 sigma schedule. Partial low-sigma refinement schedules are rejected because their absolute flow origin is ambiguous for a progressive split.

### Choosing `source_scale` when target MP changes

With `source_mode=scale`, `source_scale` is a **linear H/W scale**, not a megapixel fraction. Before H3-safe geometry snapping, the private low-stage area therefore follows approximately:

```text
source_MP ~= target_MP * source_scale^2
```

If you want to preserve the same relative low/high resolution ratio as target MP increases, leave `source_scale` unchanged. The low stage then grows proportionally with the final target and becomes more expensive as well.

If instead you want to preserve roughly the tested private low-stage size (~736x736, ~0.54 MP) while increasing final MP, reduce `source_scale` approximately as:

```text
source_scale ~= sqrt(desired_source_MP / target_MP)
```

For a desired private stage around ~0.54 MP:

| Target MP | Approx. `source_scale` |
|---:|---:|
| 0.84 | 0.80 |
| 0.90 | 0.78 |
| 1.00 | 0.74 |
| 1.10 | 0.70 |
| 1.20 | 0.67 |

The actual source dimensions are snapped to valid H3 latent geometry, so these values are starting points rather than exact pixel guarantees. Check the `handoff_plan` metrics event for the resolved source/target latent dimensions. With fixed handoff, increasing MP alone does not require changing `handoff_coordinate`; keep the tested 0.35 initially and adjust `source_scale` separately to control the low-stage compute budget.

The table is a geometry heuristic, not a quality guarantee. At approximately 1 MP, real decoded media showed that `source_scale=0.70` can work very well with `learned_3d`, while `0.65` crossed the tested prompt's fidelity/tonal floor. Because H3 snaps both axes, compare the resolved `handoff_plan` geometry rather than assuming two nearby decimal scales produce a smooth change.

## Two-pass flow-aligned use

1. Create one **Flow Trajectory** handle.
2. Patch the low-resolution H3 model with **Trajectory Capture** and run the base sample.
3. Perform the existing learned latent upscale/initialization.
4. Patch the high-resolution model with **Flow-Aligned Regenerate**, or patch each Continuum `refine_state` with **Flow-Aligned Refine State** when using the integrated learned upscaler/refiner.
5. Feed the capture metrics object into the downstream adapter's optional `metrics` input when one combined benchmark artifact is desired.
6. Keep pass-one audio locked in the surrounding workflow.

Trajectory identity includes bounded conditioning fingerprints. Exact H3 evaluations are preferred as trajectory anchors; Spectrum forecasts retain provenance and are excluded from trustworthy anchors by default.

## Experimental guidance modes

### `direction`

The currently preferred mode. It aligns the target predicted-clean estimate toward the matched low-resolution trajectory only in low spatial frequencies and decays the correction through the high stage.

### `direction+acceleration`

Adapts HiFlow's adjacent velocity-difference idea to H3 by reconstructing native flow velocity from `x0 = x - sigma * v`. The PECE implementation explicitly keeps the previous **distinct-coordinate** anchor across predictor and same-coordinate corrector calls. The implementation is structurally correct, but current decoded-media evidence does not show a meaningful advantage over direction-only.

### `direction+temporal`

Uses bounded local adjacent-frame correspondence on captured low-grid H3 clean-state latents. Minimum similarity, best-vs-second-best margin, and exact reverse-cycle consistency are hard gates; ambiguous/disoccluded locations receive no temporal copy.

The final matched D14 temporal run kept the same 54 logical / 36 actual / 18 forecast topology as D14 direction-only, but valid correspondence support was only ~0.295% in chunk 1 and ~0.084% in chunk 2. Mean temporal correction RMS was ~0.104% of baseline RMS versus ~2.19% for direction. The user saw no perceptual difference, so temporal stays experimental at reference weight 0.20 rather than being promoted or weight-swept.

Earlier patterned temporal media from the TensorRT `w4a16_awq` VAE is excluded from attribution; that decoder, not temporal guidance, caused the repeated pattern.

### `downsample_consistency`

Compares the downsampled target predicted-clean estimate against the matched low-grid clean state and lifts the error back to the target grid. It is implemented and measurable, but the 0.25 decoded-media smoke was neutral.

## Resolution-aware refine SIGMAS

The resolution-aware node is a downstream learned-refine experiment. Correct placement is:

```text
existing refine scheduler SIGMAS
            -> Resolution-Aware Sigmas
            -> MiniMax H3 Latent Upscaler + Refine (3D).sigmas
```

Do **not** feed its output into Continuum SIGMAS. Continuum keeps its ordinary MP/base geometry. **Refine Target Geometry** may mirror the downstream refine node's scale/align calculation to supply target metadata only.

The mapping composes a relative area-derived factor with H3's native video shift 12 while deriving audio sigma from the same shared flow coordinate. E0/E1 matched media showed no relevant improvement, so `mode=off` remains the recommendation.

## Metrics and benchmarking

Runtime events distinguish:

- sampler logical calls;
- actual H3 transformer evaluations;
- Spectrum forecasts and promotions;
- low/probe/high progressive stage;
- sigma and unshifted coordinate;
- packed layout rows;
- trajectory transactions;
- guidance mode/component RMS ratios;
- handoff plan and exact probe;
- history/reset boundaries;
- sampler/stage wall time;
- resolution sigma maps and fallbacks.

Use [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/PERFORMANCE.md](docs/PERFORMANCE.md), and [workflows/benchmark-matrix.json](workflows/benchmark-matrix.json) for the evidence ledger, observed timing accounting, and the broader optional formal matrix. Decoded video and audio remain the pass/fail criterion.

## Compatibility

- **ComfyUI:** fails closed if the native H3 24/32-channel, `(1,2,2)`, shift-12/3 contract is absent.
- **SA-Solver/PECE, SEEDS, ER-SDE, Euler/RES:** sampler objects are preserved; progressive low/probe/high stages use independent sampler lifetimes.
- **Spectrum:** no import dependency. Actual/forecast and solver phase/outer-step metadata are consumed when available.
- **Continuum V3.4:** multi-chunk progressive use should use Target Input so session/native-mask geometry stays target-sized.
- **DiffAid / Untwisting RoPE:** patches stay upstream of the progressive wrapper. The high stage publishes the `h3_refinement` exact-anchor contract.
- **Parallel multi-GPU model calls:** mutable capture/guidance/progressive state currently fails closed until an ordering contract is reviewed.

## Validated source revisions

Pinned executable/source contracts:

- ComfyUI `1af040bf022569d7a890241c8dd79b296cda483f`
- Spectrum `beb32dd210ef9e95520453107f158241d4f2ecf3`
- Continuum `bf25353d8bec44afea22c89717c4301ce13c2036`
- DiffAid `ba9d9efbcf7e64c755e068cb76547d8cc85481eb`
- RefDelta `034e4c4c14c56bf76813cee4765e7164b0c7e0db`
- Untwisting RoPE `299d4c56a3f057a97b3140d2136189bcd1e7d6bb`
- H3 latent upscaler/refine and learned-handoff provider `bdc670e5926bcefbe4022e17fe8b171fbfcf15de` (merged provider revision; released by companion v0.2.0)

MiniMax-H3 main was additionally inspected at `d21241f0a4b3acbb34c97dae47fa417b7065e438`.

## Development

```bash
python -m pip install -e '.[test]' build
ruff check .
ruff format --check .
pytest
python -m compileall -q .
python -m build
```

See [CREDITS.md](CREDITS.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/RESEARCH.md](docs/RESEARCH.md), [docs/BENCHMARKS.md](docs/BENCHMARKS.md), and [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Limitations

- The current quality recommendation comes from a limited difficult-motion workflow and is not a broad cross-prompt benchmark claim.
- The 14-step quality sweep used the later matched square 736x736 -> 896x896 progressive setup; the earlier rectangular D10 reference remains a separate validated topology case.
- The ~20% observed end-to-end timing reduction is against the proper historical 7+6 two-pass workflow at similar final resolution; it is not a formal same-seed benchmark and must not be generalized to every prompt, target MP, model residency state, or reference budget.
- The progressive exact handoff probe costs one visible H3 NFE per chunk.
- Target Input derives private low-grid **video** noise from a documented standard-Gaussian CPU generator keyed by graph seed; arbitrary custom/non-Gaussian private-grid video-noise semantics cannot be preserved.
- Progressive handoff rejects sampler objects with an explicit external `noise_sampler` closure because mutable RNG/history cannot safely cross the geometry reset.
- Masked progressive behavior is synthetic-tested but still has less decoded-media coverage than the unmasked difficult-motion path.
- Reference budgeting cannot retroactively change already-encoded Qwen3-VL tokens.
- Sparse attention remains investigative and is not a reconstruction of MiniMax's proprietary sparse topology.
- Model assets and sibling custom nodes are not bundled.