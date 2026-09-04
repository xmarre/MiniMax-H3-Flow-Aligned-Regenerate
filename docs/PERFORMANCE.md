# Performance evidence: progressive learned handoff vs two-pass upscale + refine

## Evidence status

This document records the current **observed workflow-level timing reference** around a ~1 MP final target. It is not a formal same-seed microbenchmark and it is not a universal speed claim.

The useful legacy baseline is the established proper learned upscale + H3 refine workflow:

- 7 first-pass SA-Solver-PECE outer steps;
- 6 learned-refine outer steps for the accepted practical baseline;
- denoise around 0.25;
- audio locked during the spatial refine;
- historical full-quality baseline: 7 + 7 outer steps.

Old 3-step high-ratio refine experiments are excluded from performance claims because three refine steps were a legacy under-refined setting and are not a quality-equivalent comparison.

## Compared runs

### Historical proper two-pass baseline

At the higher nominal 0.7 MP setting, the recorded workflow resolved approximately:

```text
base/source: 960 x 704 = 0.675840 MP
final:       1184 x 864 = 1.022976 MP
sampling:    7 base + 6 learned-refine outer steps
workflow:    ~777 s end-to-end (~12:57)
```

This is the proper 6-step-refine baseline used for the timing comparison. The historical 7+7 workflow remains the fuller refine budget, but an equally clean ~1 MP raw timing for that exact case has not been recovered, so no exact 7+7 speedup is claimed.

### Progressive learned-handoff final gate

The final learned-transfer gate resolved:

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

## Observed wall-time comparison

| Path | Base/private grid | Final grid | Sampling structure | Full workflow |
|---|---:|---:|---|---:|
| Proper two-pass upscale + refine | 960×704 (0.676 MP) | 1184×864 (1.023 MP) | 7 base + 6 refine | ~777 s |
| Progressive + `learned_3d` | 832×640 (0.532 MP) | 1184×896 (1.061 MP) | 10 progressive outer steps | 621.63 s |

Derived from the actual resolved geometries and observed wall times:

- private/source area: 0.675840 -> 0.532480 MP, **21.2% fewer source pixels**;
- final area: 1.022976 -> 1.060864 MP, **3.7% more final pixels**;
- end-to-end wall time: ~777 -> 621.63 s, about **155 s saved**;
- observed reduction: approximately **20% less end-to-end wall time**;
- equivalent observed speedup: approximately **1.25x**.

The final output is therefore slightly larger, not smaller, despite the lower total observed wall time.

## Learned-transfer overhead

The learned 3D CNN itself is not responsible for the speedup. Its measured BF16 CUDA costs were:

```text
chunk 1 learned inference: 602.1 ms
chunk 2 learned inference: 771.1 ms
sum learned inference:     ~1.373 s

chunk 1 transfer wall:     627.8 ms
chunk 2 transfer wall:     801.9 ms
sum transfer wall:         ~1.430 s
```

It added **zero H3 NFEs**.

The architectural saving comes from carrying one progressive trajectory across the spatial handoff: more H3 work remains on the cheaper private grid, then the sampler performs only the required exact probe/high-grid continuation instead of finishing a complete low-resolution generation and launching a separate low-sigma H3 refine pass.

## Model-call accounting relative to the old 7+6 path

The ~777 s historical run predates the current combined metrics artifact, so its total H3 call accounting is reconstructed from the validated SA-Solver-PECE budgets rather than claimed as direct telemetry from that exact log:

- 7-step base: later validated telemetry shows 9 actual + 4 Spectrum forecast calls per physical chunk (13 logical);
- 6-step refine: the established baseline contract is 11 logical calls per chunk, approximately 7 actual + 4 Spectrum forecasts;
- across two physical chunks, that reconstructs approximately **48 logical / 32 actual / 16 forecast** calls for a 7+6 two-pass workflow;
- the new progressive learned run directly measured **38 logical / 28 actual / 10 forecast**.

At target resolution specifically, the old 6-step refine budget implies about 14 actual H3 evaluations across two chunks. The progressive run directly measured 10 high-stage actual evaluations plus 2 exact target-grid probes = 12 target-resolution actual H3 evaluations. Treat the old totals as reconstructed accounting, not as exact counters from the ~777 s run.

## Excluded comparisons

Do not use the old ~1.75x / 3-step refine experiments as the headline baseline. They can be useful for implementation archaeology, but three refine steps are not enough for the established quality target and make the progressive path look artificially expensive.

Likewise, the repeated ~0.6 MP -> ~0.79 MP two-pass runs around 544-546 s are useful scaling context but are not directly comparable to the ~1.06 MP final progressive gate because their final target is substantially smaller.

## Interpretation and limits

The defensible statement is:

> In the observed ~1 MP workflow, the 10-step progressive learned-handoff path completed in 621.63 s versus about 777 s for the proper historical 7+6 two-pass upscale/refine workflow, an observed ~20% end-to-end wall-time reduction while producing ~3.7% more final pixels.

Do **not** turn this into:

- a universal 20% speedup claim;
- a claim that progressive output is universally higher quality;
- a matched-A/B quality percentage;
- an exact speedup over the historical 7+7 workflow without a comparable raw timing;
- evidence that acceleration guidance is better.

Runtime depends on target geometry, private/source geometry, reference-conditioning load, model residency/loading, VAE/decode cost, sampler/Spectrum policy, chunk lengths, and hardware state. Decoded media remains the quality gate.
