# Continuation throughput and boundary qualification: 01000

The prompt completes in 348.73 s. Its first sampler takes 136.314 s and the
continuation takes 182.523 s. The reported standalone single-chunk time is about
135 s; this capture does not contain a separate matched standalone run.

| Sampling stage | Initial chunk, s | Continuation, s | Increase, s |
|---|---:|---:|---:|
| Low | 55.500 | 88.121 | 32.622 |
| Exact probe | 7.787 | 12.823 | 5.036 |
| Transfer | 0.418 | 3.218 | 2.800 |
| High | 72.582 | 76.436 | 3.854 |
| Whole sampler | 136.314 | 182.523 | 46.209 |

Low/probe accounts for 37.657 s, or 81.5%, of the continuation sampler increase.
The 29.893 s outside both sampler invocations includes conditioning, decoding,
assembly and node work; the capture does not time each of those separately.
Both chunks have eleven actual H3 evaluations and six forecasts. All six
sampling admissions report zero evictions. No additional continuation NFE or
admission eviction explains this sampler increase.

The progressive source grid is active. Initial low sampling has 52 frames on a
30x44 latent grid, or 17,160 video patch rows. Continuation retains twelve exact
42x64 prefix frames and fifty 30x44 generated frames: 8,064 + 16,500 = 24,564
video patch rows, a 43.1% increase. It has 27,015 total transformer rows versus
19,384 initially. The target-grid prefix, its conditioning and temporal context
are part of the continuation workload. The earlier same-grid target override is
not active in this capture.

## Weighted dense attention

Continuation low/probe executes 2,061/800 Core dense dispatcher calls, including
eleven low-stage sparse arithmetic references. The VDN grouped plan has fourteen
local groups plus global and anchor calls. Five calls per block are always dense;
the first evaluation makes all sixteen dense. With fifty blocks, six low actual
evaluations and one dense probe, this gives 2,050 + 800 production dense calls.
Column anchors include a prefix key in every group, so all those calls carry the
nonzero physical key measure `log(330 / 672) = -0.711165686...`.

PyTorch's FlashAttention eligibility check rejects a non-null attention mask.
These repeated weighted dense calls therefore use another native SDPA backend;
the capture does not identify which backend or isolate its GPU latency. Routing
through Core preserves its priority but does not remove the real measure bias.
Arithmetic-gate wall totals are 24.137 s initially and 16.570 s for continuation;
they include compilation and queued work and cannot be treated as wholly
removable validation cost.

The paired Sol update executes weighted dense requests through its existing
SM120 union with every K/V block selected. It retains the complete grouped
domain, one normalizer, native scale and native query-dtype bias rounding, with
an independent Core SDPA check per request/layout/measure before completing the
new kernel receipt. Unit-measure dense dispatch and sparse attention retain
their current paths. This targets a concrete repeated execution cost. It does
not remove the extra physical context, establish the selected old fallback
backend, or prove a GPU speedup. See Sol's `docs/WEIGHTED_DENSE.md` for the
arithmetic, verification lifetime and receipt contract.

## Boundary evidence

The visual report describes the boundary as substantially improved. The immediate
decoded full-frame affine estimate is scale `(1.001518, 0.998984)`, or roughly
`(+0.152%, -0.102%)`, and translation `(+0.123, +0.139)` pixels. Upper-region
scale is `(1.002875, 0.997993)`. Affine confidence is low, so these are observations
rather than a calibrated proof of absent zoom. Later decoded motion differs
from the preceding cadence: upper-region median X changes from `+1.964` to
`-6.136` pixels/frame over the first three retained transitions.

Video Seam Auto classifies a clean boundary and replaces zero frames. All 22
sampled duplicate-overlap frames are equal. First retained luma deviation is
0.547% above the last preceding frame. Prefix overlap closes with a terminal
mask maximum of zero; final exact-prefix restoration changes values by at most
`2.384e-7`. These support a small immediate join but do not independently qualify
the whole successor window, tone or audio.

The exact-overlap structural bridge rejects with regional disagreement, and
every direction-guidance receipt reports `reference_gauge_used=false`. The
guidance-reference rebase therefore does not execute in this realization and
cannot be credited for its improved visual report.

Keep the geometry, prompt, seed, carried prefix, six-token video overlap,
0.5/0.5 strengths and sixteen scheduled steps for hardware qualification of the
paired Sol update. Confirm nonzero `partitioned_weighted_dense_calls`, successful
all-selected arithmetic gates and reduced repeated Core dense calls. Compare
low/probe and whole-prompt time, peak allocation, and rendered video/audio; cold
compilation and warm execution need to be distinguished.

Evidence: metrics SHA256
`b27cab732a625b04332fa400d086effdb798a402de9f0ea564636038aada5865`;
runtime log SHA256
`f42e236309b68c9e3eff8afc2fd34e4868536b525538cfcdf457ec9e24486c62`.
