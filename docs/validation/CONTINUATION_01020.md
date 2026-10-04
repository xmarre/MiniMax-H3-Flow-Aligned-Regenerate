# Continuation execution, throughput and boundary qualification: 01020

The capture executes weighted all-selected SM120 attention and coupled
guidance-reference rebasing. Its changed geometry and conditioning prevent a
matched speed comparison with 01000. It does not exercise the new one-token
DC-only reference policy, and no 01020 video or local boundary crops accompany
this capture. Neither GPU speedup nor local rendered tone acceptance is established.

## Workload and sampling time

| Workload | 01000 | 01020 |
|---|---:|---:|
| Source latent grid | 30x44 | 40x58 |
| Target latent grid | 42x64 | 56x82 |
| Initial low video patch rows | 17,160 | 30,160 |
| Continuation low video patch rows | 24,564 | 42,776 |
| Initial high video patch rows | 34,944 | 59,696 |
| Continuation high video patch rows | 41,664 | 71,176 |
| Initial text conditioning rows | 968 | 1,495 |

Initial low video rows increase 75.8%, continuation low rows 74.1% and target
video rows 70.8%. Both captures retain twelve exact target-grid prefix frames
and fifty source-grid suffix frames in continuation. Within 01020, low video
rows increase 41.8% from the initial chunk, and high rows increase 19.2%.
The conditioning lengths and compiled-text receipts also differ from 01000.

| Sampling stage | Initial chunk, s | Continuation, s | Difference, s |
|---|---:|---:|---:|
| Low | 180.461 | 165.894 | -14.567 |
| Exact probe | 12.903 | 23.874 | +10.971 |
| Transfer | 1.221 | 8.127 | +6.907 |
| High | 118.740 | 158.237 | +39.497 |
| Whole sampler | 313.464 | 358.395 | +44.931 |

High refinement contributes 87.9% of the continuation sampler increase. Low
and probe together decrease 3.596 s within this capture; this differs from
01000, where low/probe contributed 81.5% of the increase. The total prompt
receipt is `00:13:02`, with only whole-second precision. Approximately 110 s
lies outside the two sampler scopes. The capture does not separately time all
conditioning, VAE, assembly and other node work in that remainder.

Both captures have 22 actual H3 evaluations and twelve forecasts in total:
each chunk has six low actual calls, four low forecasts, one exact probe,
four high actual calls and two high forecasts. Sampler/history lifetimes remain
six/four respectively. Increased evaluation count does not explain the timing.

## Weighted attention execution and startup cost

Continuation low/probe executes 2,050/800 weighted dense calls under
`sm120-weighted-all-selected-v1`. Each scope admits ten request/layout arithmetic
identities against independent native SDPA. Their maximum relative L2 errors
are 0.000337/0.000364; all checks accept under the existing thresholds.

| Dispatcher receipt | 01000 low/probe | 01020 low/probe |
|---|---:|---:|
| Core dense calls, including arithmetic references | 2,061 / 800 | 21 / 10 |
| Weighted all-selected production calls | 0 / 0 | 2,050 / 800 |

The remaining 21 low Core calls are ten weighted dense references plus eleven
sparse all-selected references. The probe retains its own ten independent
checks. This establishes the dispatch change on hardware, not an isolated
kernel speedup. Weighted-check wall time is 25.769 s in low and 0.290 s in the
subsequent probe using the same shapes. The first cost includes cold execution,
preparation/compilation and queued work; neither figure is pure GPU kernel latency.
All arithmetic-gate wall times are already inside their sampling scopes.

Initial low executes zero weighted dense calls. Its 180.461 s duration cannot
be attributed solely to the new weighted route. Its Spectrum scope is 168.523 s,
and a cold model-profile lookup logs 11.919 s, consistent with much of the
11.938 s scope difference. These are observations from nested timing scopes,
not a complete attribution of the initial-stage cost.

## Memory admission

All six admissions preserve the required resident H3 clone. Continuation high
requests 31,754.563 MiB headroom with only 25,248.444 MiB available. One Core
memory-admission pass takes 15.266 s; an adjacent Core receipt reports a partial
unload freeing 7,139.56 MiB while retaining 18,744.27 MiB of that model. Available
memory rises to 32,528.158 MiB and the admission target is met.

`unloaded=0` counts fully unloaded model entries. It does not mean no weights
moved out of GPU memory. Allocated memory falls from 65,453.376 to 58,313.830 MiB
while reserved memory remains 69,056 MiB. This admission lies inside high-stage
and Spectrum timing, so its 15.266 s must not be added to the 158.237 s again.
The capture does not isolate transfer cost from synchronization or Core work,
and does not justify removing the finite workspace reservation.

## Boundary and reference ownership

Registration rejects `unsupported_sampler_contract` for `sample_euler` after
video-boundary acceptance. The existing fallback applies the coupled exact-overlap
bridge over four suffix tokens with weights `(1, 0.75, 0.5, 0.25)`. All six
continuation guidance calls, including two forecasts, execute
`exact_prefix_guidance_reference_coupled_v1`. Completion reports
`spatial_mean_only=false`, four-token support, unchanged source/temporal reference
and no extra H3/provider/VAE calls. The DC-only policy
`exact_prefix_guidance_reference_dc_v1` is not exercised in this realization.

Video overlap closes over its final two exact high calls with a final mask
maximum of zero. Exact output restoration changes values by at most `2.384e-7`.
All 22 sampled duplicate-overlap decoded frames are equal, including the five-frame
right-context tail. Video Seam Auto replaces zero frames.

At the first retained transition, whole-frame luma mean changes from 0.459670 to
0.461765 (+0.456%) and deviation from 0.214989 to 0.215998 (+0.470%). These
131x192 global samples do not qualify the previously reported local contrast
failure. Later frames also differ, and motion/shading affect the measurements.

The immediate decoded full-frame affine scale estimate is `(1.005654, 1.004124)`
with translation `(-3.572, -2.865)` pixels. Upper-region scale is
`(1.000908, 0.990707)`. Confidence is low (full X/Y 0.149/0.015; upper X/Y
0.055/0.093), with inconsistent regional estimates. These observations cannot
certify absent zoom or establish regression against a different scene/workload.

## Qualification status and evidence identity

The weighted production route and coupled comparison ownership execute and
pass their arithmetic/ownership checks. A matched comparison must hold geometry,
conditioning, seed, carried prefix and schedule constant and distinguish cold
from warm execution. Local video tone remains an independent rendered check;
DC-only ownership requires a realization in which the one-token bridge executes.
The logged policy signatures identify these executed paths, not an exact Git SHA.

- `metrics_01020_.json`: SHA256
  `8ea1fe4725fdbd59181a1480549cb737f61355be7295b1acba6c8c640c2432e2`.
- `Pasted text(20261003-190827).txt`: SHA256
  `019d2659752b3f1d7c1c68f24f89fd1e55c6f29807e07ef59be1c1f02de875d1`.
- Previous measurements and limits: [01000 throughput](CONTINUATION_THROUGHPUT_01000.md)
  and [local crop/DC evidence](BOUNDARY_TONE_01000.md).
