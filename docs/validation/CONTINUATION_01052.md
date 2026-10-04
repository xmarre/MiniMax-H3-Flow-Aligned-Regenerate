# Boundary jump, shading and compile reuse: 01052

01052 fails rendered boundary acceptance according to the user's latest report.
The decoded receipts corroborate a sharp motion change and darker shadows.
Both preceding guidance-representation fixes execute; successful prefix
restoration and attention qualification do not establish rendered continuity.
This capture contains no new crops or full video for independent inspection.

## Decoded boundary

The continuation retains raw frames `[39,209)` at global `[175,345)`. Assembly
uses contiguous slices, and exact duration removes nine frames only from the
final output tail. All 22 sampled duplicate-overlap frames are identical,
including the five-frame right-context tail. Video Seam Auto applies no patch.
These checks do not establish the quality of newly generated frames.

| Decoded motion fit, pixels | Pre-boundary median | First transition | Second | Third |
|---|---:|---:|---:|---:|
| Full dx | -2.016 | +2.042 | +2.073 | -7.541 |
| Full dy | +1.706 | +1.661 | +2.106 | +0.208 |
| Upper 45% dx | -2.675 | +1.898 | +2.197 | -5.926 |
| Upper 45% dy | +1.999 | +1.834 | +1.795 | -5.053 |

The third transition is global frame 176 to 177. Its translation estimates are
not clipped. The third full-frame affine fit has scale x/y 1.018477/1.019810
and translation -15.589/-8.943 px. These describe an expansion in that fit,
not a calibrated geometry or quality verdict. The receipts do not show a
literal skipped assembly frame. Earlier 01041's large vertical estimate must
not be substituted for this realization's predominantly horizontal shock.

| Whole-frame luma | Last pre-boundary | First generated | Third generated |
|---|---:|---:|---:|
| Mean | 0.423539 | 0.422532 (-0.238%) | 0.414743 (-2.077%) |
| Standard deviation | 0.221586 | 0.226283 (+2.120%) | 0.229924 (+3.763%) |
| Fifth percentile | 0.081852 | 0.077974 (-4.737%) | 0.062471 (-23.679%) |
| Ninety-fifth percentile | 0.774574 | 0.788782 (+1.834%) | 0.782477 (+1.020%) |

These sampled pixel profiles support the reported shading/contrast change.
Composition and motion affect them; they cannot identify the cause of new hair,
garment deformation or other localized content changes.

## Executed guidance and high-stage contribution

The coupled handoff applies weights `(1,0.75,0.5,0.25)`. All six continuation
guidance calls report both `reference_gauge_used=true` and
`temporal_reference_gauge_used=true`, including forecasts. Temporal valid
fraction is 0.011047 and mean confidence 0.006171. The new path demonstrably
executes, but this render does not pass. Video release closes at sigma
0.631579; the final two Core entries see an exact prefix, and the terminal
prefix mask maximum is zero. Prediction and VAE repair actuators remain disabled.

| First full-frame latent translation fit | dx | dy |
|---|---:|---:|
| Exact-restored pre-high | -0.053743 | +0.082246 |
| First actual high, before Flow | -0.079674 | +0.232601 |
| First actual high, after Flow | -0.080040 | +0.232324 |
| Final post-high | -0.072487 | +0.206626 |

High prediction changes this latent transition before guidance. This is an
observed contributor, not proof of the root cause of the third decoded-frame
shock or tone/content changes. Global centered low-pass and gradient boundary
RMS decrease to 0.9123 and 0.8854 of their pre-high values. The post-high
stabilization shadow selects only `r0c1` and remains diagnostic-only. Neither
global latent scores nor that unrendered shadow justify promoting an actuator.

Measurement mode previously exported stage tensors only when the retired
registration candidate passed. Here it rejects with
`boundary_upper45_degraded_over_bound`, so the needed common-window operands
were absent. The new independent window evidence records the actual provider,
pre-high, first actual high before/after Flow, input/mask and final windows even
when registration rejects. It preserves the regional residual-fit gate and
production output. CPU byte copies and file I/O are diagnostic overhead; no
extra H3, provider, sampler or VAE work is added. This is a capture correction,
not a rendered repair. Native temporal-oracle checks verify the saved window's
first twelve retained frames against continuous decoding.

## Learned handoff and guidance reference correction

The previous comparison tests used `learned_output = bicubic(source)`. That
omitted the actual production transfer difference: the learned 3D provider can
change temporal transitions, while high guidance reconstructed its reference
with bicubic interpolation. Prefix rebasing alone therefore did not guarantee
that guidance preserved a correctly reconciled learned transition.

A finite nonlinear provider counterexample preserves the first transition to
`1.192093e-7` before guidance. The current coupled-gauge direction update creates
`0.0440133` maximum transition error and `-0.0436337` first-suffix mean change,
while preserving the prefix exactly. This establishes an implementation
inconsistency; it does not establish its contribution to this render's third
decoded-frame shock. The first high prediction's independently observed change
still requires rendered assessment.

High direction/acceleration now reuse the already executed actual learned
provider output for the matching main exact-probe endpoint. Binding verifies
the selected probe's generated suffix to native sampler-roundtrip precision
and the high schedule's endpoint identity.
The intentionally restored provider prefix is allowed to differ from the
pre-inpaint capture. Independent shadow and spatial controls are not rebound.
Native temporal correspondence remains unchanged; its temporary operand removes
the learned-versus-bicubic representation difference before applying the existing
prefix-gauge pullback. The existing scheduled guidance and RMS limit remain;
there is no fixed clean prediction, new fade or extra model/provider evaluation.

The new direction regression, moving/stationary temporal covariance, actual and
forecast wrapper, audio ownership, source/schedule rejection and exception
lifetime cases pass in the targeted suite (70 cases). One target video endpoint
is retained for high, approximately 21.82 MiB for this run's FP32 24x62x62x62
geometry, and released on success or failure. GPU overhead and rendered quality
remain unmeasured. The existing compile-reuse change is independent of this fix.

## Timing and actual GPU reuse receipts

| Seconds | Initial | Continuation | Increase |
|---|---:|---:|---:|
| Low | 70.603 | 125.450 | 54.847 |
| Exact probe | 10.254 | 19.149 | 8.895 |
| Transfer | 0.541 | 7.754 | 7.212 |
| High | 94.140 | 117.180 | 23.039 |
| Whole sampler | 175.561 | 271.302 | 95.741 |

The prompt takes 481.54 s; about 34.677 s lies outside these two samplers.
Low/probe contributes 63.742 s, or 66.58%, of the continuation increase.
There are still 22 actual H3 calls, twelve forecasts, six sampler scopes and
four progressive history boundaries. Grids are 44x44 source and 62x62 target.
Video rows increase from 25,168 to 35,732 in low (+41.98%) and from 49,972
to 59,582 in high (+19.23%). Text and audio rows also increase.

Sol's new machine-function reuse is observed on the actual SM120 GPU route:

| Partitioned scope | Compile calls | Cache hits | Compile wall seconds | Source scans |
|---|---:|---:|---:|---:|
| Continuation low | 2 | 4,809 | 2.275182 | 1 |
| Continuation probe | 0 | 800 | 0 | 1 |

The two source scans take 7.047 ms combined. Low/probe arithmetic-gate wall
times are 15.192/0.214 s and include qualification/reference work; they are
not compilation timers. All six Sol scopes report success. The 21 low and
ten probe recorded gate entries retain independent checks; worst recorded
relative L2 is 0.000544/0.000331 respectively. This qualifies observed reuse
in this capture, not all GPU layouts or a matched speedup. Different geometry
and conditioning hashes make comparison with 01041 unsuitable for isolating
the changes' timing effect. The remaining latency is not explained by two
source scans or repeated partitioned machine-function compilation.

## Evidence identity

- `Pasted text(20261003-225807).txt`, SHA-256
  `3df832b518b7e99c54ae362418c47c78c03519e8f1f2c30a0c3af2f8f9259732`.
- `metrics_01052_.json`, SHA-256
  `49771a09570354a19a047b1ad540a0c2a922fa822453c84ca2a8703763454825`.

Use the existing Flow #93 overlay for window evidence; Sol #37 remains at
`f59bdd15c4ef1d1893052298a663da3850ac1a48`. The current measurement setting
already selects the capture. The next comparison needs those exact window
operands decoded with the same VAE and authoritative prefix. Rendered
boundary acceptance and the root cause remain unresolved.
