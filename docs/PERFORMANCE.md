# Performance evidence: native, progressive Flow, and standalone two-pass H3 paths

## Evidence status

This document records observed workflow-level timing references around a ~1.1 MP final target. They are not universal speed claims. Comparisons are only treated as controlled when their seed, prompt, references, sampling settings, workflow structure, and resolved geometry are known. Decoded media remains the quality gate when denoising budgets change.

The September 9 runs below predate the v0.3.4 shipped-default/workflow alignment. They document the settings and geometry actually measured; current defaults are not retroactively projected onto historical benchmark evidence.

## Current controlled three-run comparison

The September 9 hot comparison contains **three distinct workflows**. They must not be collapsed into one another:

1. **Native direct target-grid generation** — the control.
2. **Flow progressive Mixed-Grid learned handoff** — low-grid work and target-grid continuation occur inside the progressive sampler path; the learned 3D upscaler is used only as the spatial handoff transform.
3. **Standalone learned upscale + refine** — a complete lower-resolution Continuum sampling pass is followed by the separate `MinimaxH3LatentUpscaler3DRefineHandoff` node and a second high-resolution H3 sampler lifetime.

The third workflow is **not Flow progressive sampling**. The Flow learned-handoff contract explicitly avoids `low sample -> complete Latent Upscaler + Refine -> second independent H3 pass`; that would defeat the progressive architecture.

### Run identity and wall time

| Path | Resolved generation geometry | H3 sampling/refine node wall | Prompt result |
|---|---|---:|---|
| Native direct | 1184×896 = 1.060864 MP | 290.41 s `H3ContinuumSamplerV34` | complete, 338.50 s |
| Flow progressive | private first-chunk grid 832×640 -> target 1184×896 | **247.63 s** `H3ContinuumSamplerV34` | complete, **298.59 s** |
| Standalone learned upscale + refine | base 992×736 -> final 1216×896 = 1.089536 MP | **341.36 s** = 188.57 s base sampler + 152.79 s refine | sampling/refine complete; decode failed at 342.71 s |

The standalone run cannot be compared to the other two by completed end-to-end wall time because it failed immediately after refinement at `H3ContinuumDecodeContext`, before video decode and assembly. Its sampler/refine wall remains valid for the sampling-cost comparison.

Direct deltas from the completed node walls:

- **Flow progressive vs native sampler:** 247.63 s vs 290.41 s = **-42.78 s / -14.73%**.
- **Flow progressive vs native prompt:** 298.59 s vs 338.50 s = **-39.91 s / -11.79%** end-to-end.
- **Standalone two-pass vs native sampler:** 341.36 s vs 290.41 s = **+50.95 s / +17.54%**.
- **Standalone two-pass vs Flow progressive sampler:** 341.36 s vs 247.63 s = **+93.73 s / +37.85%**.

The standalone final grid has about **2.7% more pixels** than the native/Flow target. That is a small adverse bias against the standalone run and should be retained when interpreting its timing, but it does not erase the large difference between the architectures.

### Model-call topology

| Path / stage | Logical calls | Actual H3 NFEs | Spectrum forecasts |
|---|---:|---:|---:|
| Native direct | 16 | 11 | 5 |
| Flow low/private | 10 | 8 | 2 |
| Flow exact handoff probes | 2 | 2 | 0 |
| Flow high/target | 6 | 4 | 2 |
| **Flow progressive total** | **18** | **14** | **4** |
| Standalone base pass | 16 | 11 | 5 |
| Standalone high-res refine | +8 | +6 | +2 |
| **Standalone two-pass total** | **24** | **17** | **7** |

The raw NFE count alone does not predict wall time because the calls occur on different geometries. Flow performs most of its trajectory work on the cheaper private grid, pays two explicit exact handoff probes, then performs only the remaining target-grid continuation. The learned handoff itself adds no H3 NFE.

The standalone path instead finishes a complete 8-logical-call-per-chunk lower-resolution trajectory and then launches a separate four-step high-resolution refinement sampler. That additive second sampler lifetime is the source of its extra cost.

### Standalone two-pass stage decomposition

The following stage table applies **only to native vs the standalone learned upscale+refine run**. It is not a decomposition of the Flow progressive run.

| Stage | Native direct | Standalone two-pass | Delta |
|---|---:|---:|---:|
| First/native base chunk | 116.84 s | 74.90 s | -41.93 s (-35.9%) |
| Later/native base chunk | 165.27 s | 107.97 s | -57.30 s (-34.7%) |
| Native/base Spectrum subtotal | 282.11 s | 182.88 s | -99.23 s (-35.2%) |
| Standalone high-res refine, first | — | 67.75 s | +67.75 s |
| Standalone high-res refine, later | — | 82.59 s | +82.59 s |
| Spectrum-stage total | 282.11 s | 333.22 s | +51.11 s (+18.1%) |
| Sampler/refine node total | 290.41 s | 341.36 s | +50.95 s (+17.5%) |

This shows two separate facts:

- moving the complete base pass to ~0.73 MP is effective by itself, saving about **99.23 s / 35.2%** at the Spectrum-stage boundary;
- the **standalone** four-step high-resolution refine costs about **150.34 s**, more than the base-pass saving, so that particular two-pass configuration loses to native overall.

It says nothing negative about the current Flow progressive path; that path is the 247.63 s sampler run above and is faster than native in this controlled comparison.

### Geometry and architectural interpretation

```text
native target:          74 x 56 latent -> 1184 x 896 = 1.060864 MP
Flow private (chunk 1): 52 x 40 latent ->  832 x 640 = 0.532480 MP
Flow target:            74 x 56 latent -> 1184 x 896 = 1.060864 MP
standalone base:        62 x 46 latent ->  992 x 736 = 0.730112 MP
standalone final:       76 x 56 latent -> 1216 x 896 = 1.089536 MP
```

The architectures therefore answer different questions:

- **Flow progressive:** can early target-trajectory work be executed on a cheaper grid and handed off inside the same progressive denoising process? In this run, yes: it reduced sampler wall by ~14.7% and end-to-end wall by ~11.8% relative to native.
- **Standalone upscale + refine:** can a complete cheaper base generation plus a second learned upscale/refine pass outperform native wall time? In this 8+4 configuration, no: the second high-resolution sampler costs more than the base pass saves.

Do not silently reduce standalone refine steps to manufacture a speed claim. A lower standalone refine budget changes the empirical quality point and requires matched decoded-media validation.

The failed standalone run also exposed a separate decode-boundary bug: the integrated refiner returns valid native joint H3 AV samples, while Continuum Decode Context historically accepted only a split plain-video tensor. This PR fixes that representation boundary without changing either sampling architecture.

## Historical proper two-pass baseline

The useful legacy baseline is the established proper learned upscale + H3 refine workflow:

- 7 first-pass SA-Solver-PECE outer steps;
- 6 learned-refine outer steps for the accepted practical baseline;
- denoise around 0.25;
- audio locked during the spatial refine;
- historical full-quality baseline: 7 + 7 outer steps.

Old 3-step high-ratio refine experiments are excluded from performance claims because three refine steps were a legacy under-refined setting and are not a quality-equivalent comparison.

### Historical proper two-pass timing

At the higher nominal 0.7 MP setting, the recorded workflow resolved approximately:

```text
base/source: 960 x 704 = 0.675840 MP
final:       1184 x 864 = 1.022976 MP
sampling:    7 base + 6 learned-refine outer steps
workflow:    ~777 s end-to-end (~12:57)
```

This is the proper 6-step-refine baseline used for the historical timing comparison below. The historical 7+7 workflow remains the fuller refine budget, but an equally clean ~1 MP raw timing for that exact case has not been recovered, so no exact 7+7 speedup is claimed.

## Historical progressive learned-handoff gate

An earlier progressive learned-transfer gate resolved:

```text
private/source: 832 x 640 = 0.532480 MP
final:          1184 x 896 = 1.060864 MP
source_scale:   0.70
outer_steps:    10 SA-Solver-PECE
handoff:        fixed 0.35 -> index 6
coordinate:     ~0.400000016
sigma:          ~0.888888896
transfer:       learned_3d
workflow:       621.63 s end-to-end (10:21.6)
Continuum node: 573.90 s
sampler walls:  239.935 s + 319.949 s = 559.884 s
```

The decoded video was different in action/content from earlier runs but was judged very good. The run used `direction+acceleration`; this does **not** establish an acceleration advantage over direction-only.

Exact H3/Spectrum telemetry across the two physical chunks:

| Stage | Logical calls | Actual H3 NFEs | Spectrum forecasts |
|---|---:|---:|---:|
| Low/private | 22 | 16 | 6 |
| Exact probes | 2 | 2 | 0 |
| High/final | 14 | 10 | 4 |
| **Total** | **38** | **28** | **10** |

The run also recorded 6 progressive sampler invocations, 4 history boundaries, copied audio, rebuilt high-grid conditioning, and an actual first high-grid H3 call in both chunks.

## Observed historical wall-time comparison

| Path | Base/private grid | Final grid | Sampling structure | Full workflow |
|---|---:|---:|---|---:|
| Proper standalone two-pass upscale + refine | 960×704 (0.676 MP) | 1184×864 (1.023 MP) | 7 base + 6 refine | ~777 s |
| Progressive + `learned_3d` | 832×640 (0.532 MP) | 1184×896 (1.061 MP) | 10 progressive outer steps | 621.63 s |

Derived from the actual resolved geometries and observed wall times:

- private/source area: 0.675840 -> 0.532480 MP, **21.2% fewer source pixels**;
- final area: 1.022976 -> 1.060864 MP, **3.7% more final pixels**;
- end-to-end wall time: ~777 -> 621.63 s, about **155 s saved**;
- observed reduction: approximately **20% less end-to-end wall time**;
- equivalent observed speedup: approximately **1.25x**.

The final output is therefore slightly larger, not smaller, despite the lower total observed wall time.

## Learned-transfer overhead

The learned 3D CNN itself is not responsible for most of the speedup. In the historical progressive gate its measured BF16 CUDA costs were:

```text
chunk 1 learned inference: 602.1 ms
chunk 2 learned inference: 771.1 ms
sum learned inference:     ~1.373 s

chunk 1 transfer wall:     627.8 ms
chunk 2 transfer wall:     801.9 ms
sum transfer wall:         ~1.430 s
```

It added **zero H3 NFEs**.

The architectural saving comes from carrying a progressive trajectory across the spatial handoff: more H3 work remains on the cheaper private grid, then the sampler performs only the required exact probe/high-grid continuation instead of finishing a complete lower-resolution generation and launching a separate low-sigma H3 refine pass.

## Model-call accounting relative to the old 7+6 path

The ~777 s historical run predates the current combined metrics artifact, so its total H3 call accounting is reconstructed from the validated SA-Solver-PECE budgets rather than claimed as direct telemetry from that exact log:

- 7-step base: later validated telemetry shows 9 actual + 4 Spectrum forecast calls per physical chunk (13 logical);
- 6-step refine: the established baseline contract is 11 logical calls per chunk, approximately 7 actual + 4 Spectrum forecasts;
- across two physical chunks, that reconstructs approximately **48 logical / 32 actual / 16 forecast** calls for a 7+6 standalone two-pass workflow;
- the historical progressive learned run directly measured **38 logical / 28 actual / 10 forecast**.

At target resolution specifically, the old 6-step refine budget implies about 14 actual H3 evaluations across two chunks. The progressive run directly measured 10 high-stage actual evaluations plus 2 exact target-grid probes = 12 target-resolution actual H3 evaluations. Treat the old totals as reconstructed accounting, not as exact counters from the ~777 s run.

## Excluded comparisons

Do not use the old ~1.75x / 3-step refine experiments as the headline baseline. They can be useful for implementation archaeology, but three refine steps are not enough for the established quality target and make the progressive path look artificially expensive.

Likewise, the repeated ~0.6 MP -> ~0.79 MP standalone two-pass runs around 544-546 s are useful scaling context but are not directly comparable to the ~1.06 MP final progressive gates because their final target is substantially smaller.

## Interpretation and limits

The current controlled evidence supports three separate statements:

> The September 9 Flow progressive run completed at 247.63 s sampler wall and 298.59 s prompt wall versus 290.41 s and 338.50 s for the native control: about **14.7% less sampler time** and **11.8% less end-to-end time** at the same 1184×896 target.

> The separate September 9 standalone 992×736 -> 1216×896 learned upscale + four-step refine path cost 341.36 s in sampler/refine nodes, about **17.5% more than native**. Its final decode failed, so no completed end-to-end or decoded-quality comparison is claimed for that run.

> In an earlier ~1 MP workflow, the progressive learned-handoff path completed in 621.63 s versus about 777 s for the proper historical 7+6 standalone upscale/refine workflow, an observed ~20% end-to-end wall-time reduction while producing ~3.7% more final pixels.

Do **not** turn these results into:

- a universal Flow speedup percentage;
- a universal standalone two-pass slowdown percentage;
- a claim that progressive output is universally higher quality;
- a matched-A/B quality percentage;
- an exact speedup over the historical 7+7 workflow without a comparable raw timing;
- evidence that acceleration guidance is better;
- evidence that a reduced standalone refine budget preserves quality without a decoded-media test.

Runtime depends on target geometry, private/source geometry, reference-conditioning load, model residency/loading, VAE/decode cost, sampler/Spectrum policy, chunk lengths, and hardware state. Decoded media remains the quality gate.
