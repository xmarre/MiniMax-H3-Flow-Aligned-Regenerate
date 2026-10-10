# 01770: high-stage redraw and image-grid residual coupling

The owner reports continued frame shift and multi-frame darkening on6dd02eb.
The three audit reports share manifest
`97673df5b3d604c5563d867bc24a024f1d3268487e4f260ce77ac9503e27dd20`.
The half-pixel prefix/provider routing did execute; its sufficiency is rejected.
Different prefix hashes between01768–01770 preclude matched causal conclusions.

| Same-run measurement | Pre-high | First actual high before Flow | Final |
| --- | ---: | ---: | ---: |
| Upper45 175→176 vertical estimate, decoder px | -0.633 | -2.848 | -2.796 |
| Adjacent 175→176 RGB RMS | 0.03768 | 0.06786 | 0.06958 |
| Frame181 mean luma | 0.32486 | 0.32159 | 0.32012 |

Immediate Flow guidance raises frame181 luma to0.32303 and slightly reduces
motion. The first actual native prediction already changes both; forecasts are
not their sole origin. PT227 reports zero duplicate-overlap sampled pixel
difference and unit gains for both the17-frame interior and five-frame context
tail. PT212 confirms assembled motion. Small same-frame provider/prefix fits
(<0.08 target latent cells) weaken a pure uncompensated translation explanation.
Source-grid decoded tone already varies; fixing high entry may not remove all
source-stage changes. These diagnostics locate onset, not a unique trained cause.

## What changes

The old uniform handoff takes the low clean prediction at sigma0.878 and builds
`(1-sigma)*provider(clean) + sigma*unrelated_noise`, discarding the measured
low-state residual. The proposed correction retains the actual residual
`r = (state - (1-sigma)*clean)/sigma`. It keeps reduced-grid low/probe, the
ordinary learned provider, original step budget and exact protected AV masks.
There are no extra model/provider/sampler/history/VAE calls.

For ordinary antialiased bicubic half-pixel analysis A, define
`Q=(A Aᵀ)^(-1/2) A`, so `Q Qᵀ=I`. Gaussian refinement is
`n_target=Qᵀ n_source + (I-QᵀQ) innovation`.
Thus `Q n_target=n_source`, added innovation cannot replace the coarse noise,
and independent unit Gaussian inputs produce unit Gaussian target covariance.
Operators are separable; small CPU matrices are cached, and tensor maps execute
on the input device. Source and innovation RNG seeds must differ.

For a deterministic low sampler, decompose
`r=noise_scale*initial_source_noise + model_drift`.
Only initial Gaussian noise uses Q. Drift uses a separate image right inverse
`P=B+Aᵀ(AAᵀ)^(-1)(I-AB)`, where B is ordinary half-pixel bicubic upsampling.
`AP=I` retains the measured drift under the actual image analysis; `P1=1`
preserves constants. It transfers each frame's own measured drift, not a static
residual copied from old prefix frames. Stochastic samplers cannot classify
new stochastic increments as model drift and therefore refine the complete
measured residual through Q. Their independently generated high-grid innovation
is multiplied by `model_sampling.noise_scale`, matching the carried source
noise scale rather than creating an under-variance fine component. A measured
residual is not claimed to be Gaussian. Unit Gaussian covariance statements
apply only to inputs with the stated independent Gaussian distribution.

This differs from the earlier physical patch-phase owner bins and physical
coordinate drift mapping. It also differs from simply enabling frame-gauge
repair: uniform coupling now operates with that feature off. Protected prefix
noise is still replaced with caller-owned noise, and returned video/audio
protected values retain the existing exact-mask canonicalization.

The real provider clean operand is captured unchanged for guidance and audit.
Independent-noise inverse recovery would fabricate a wrong clean operand after
a coupled handoff, so residual modes fail if that real operand is lost. Uniform
provider receipts previously incorrectly said the checkpoint was not invoked;
the selector set is corrected without changing provider invocation counts.

## Counterevidence and limits

Earlier01150 residual transport still shifted with heterogeneous geometry and
frame gauge enabled. Residual consistency is not sufficient proof of visual
continuity. Matching image coordinates, separating noise from deterministic
content drift and retaining coarse modes remove specific defects; they do not
prove the trained high model preserves clean layout, luma or watermark identity.
The H3/VDN distilled clean prediction is not asserted to be an exact Bayesian
posterior. Source/target denoisers differ and the learned provider is nonlinear.

The full-target single-pass comparison is preserved only on
`checkpoint/pr99-native-single-pass-01770-20261009`; it is not the chosen fix or
recommended workflow. It avoids the handoff but does not meet the progressive
speed objective. All changes taken toPR99 keep reduced-grid low/probe.

Refresh PR99 through ComfyUI Patcher and restart ComfyUI. Keep the existing
uniform selector (including exact_context if already selected),
`frame_gauge_repair=false`, `uniform_source_detail_transport=false` and
`suffix_dc_bridge=false`. The handoff receipt must report
`source_residual_image_drift_v1` for deterministic samplers or
`source_residual_image_refinement_v1` for stochastic/unverified samplers.
Compare the actual rendered boundary and first-high audit, not only scalar
projection residuals. Trained GPU acceptance and timing remain pending.

## Validation

Initial 18 operator/distribution/handoff tests passed on the first checkpoint, including independent full
2D resize-based whitened analysis, exact coarse-noise retention across innovation
seeds, empirical spatial/temporal covariance, constant/moving/random drift at
50x38→72x54 and other rounded-aspect grids, byte-exact identity, and provider/audio
ownership. Native Core and full regression validation are pending after the stochastic noise-scale correction.
