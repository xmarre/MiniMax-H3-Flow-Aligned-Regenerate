# Experimental geometric Continuum seam bridge

A progressive Mixed-Grid Continuum run exposed a second boundary failure after the released one-token DC bridge had removed the obvious flash: a slight whole-frame framing change at the chunk join, visible as a small shrink/zoom-out and top-edge reveal.

The DC bridge remains correct for its narrow purpose. It adjusts per-channel spatial means. It cannot correct global scale or translation.

The geometric bridge remains **experimental and off by default**. Latent diagnostics justify another decoded A/B test, but they do not establish that the visible seam is fixed.

## What the measurements show

The instrumented handoff uses source `40x54` and target `56x76` latent grids with `suffix_dc_bridge=true`.

The same-time protected-prefix / learned-prefix fit is still effectively identity:

- `sx ~= 1.0125`, `sy ~= 0.9950`;
- improvement only about `1.58%`.

That argues against the learned 3D upscaler itself being the main source of the framing jump.

The framing residual instead appears at the genuine low-grid protected-prefix -> generated-suffix boundary and remains independently measurable after learned upscaling on the target grid.

For the latest real vertical-scale (`sy`) case:

| domain | signed log-scale residual | residual scale | evidence |
| --- | ---: | ---: | ---: |
| source low grid | `-0.0253151371` | `0.9750026041` | `1.95311695` |
| target learned grid | `-0.0170032489` | `0.9831404905` | `1.83264785` |

The engineering concordance score is

```text
hypot(1.95311695, 1.83264785) ~= 2.67829
```

which exceeds the configured combined threshold `2.5`. Both domains independently exceed the minimum contribution `1.25`, the signs agree, and the magnitudes are compatible.

This score is an **engineering concordance score**. It is not a p-value and does not assume that source and target registrations are statistically independent.

The source grid nominates only `sy` for this defect. Target-only `sx`, `tx`, or `ty` candidates therefore cannot become corrections merely because a target registration happens to fit them.

## Why the old persistent-only policy was wrong

The previous bridge required the source residual to remain a constant offset across the suffix before target corroboration was even evaluated. The latest run falsified that model.

For source-grid `sy`, the boundary residual is:

```text
b = -0.0253151371
```

The subsequent genuine low-grid suffix transition residuals relative to expected natural motion are approximately:

```text
r1 = -0.0000920575
r2 = +0.0023740350
r3 = +0.0221887658
r4 = +0.0073062200
```

The framing state affecting each suffix token is integrated as:

```text
s0 = b
sk = b + sum(r1 .. rk)
```

which gives approximately:

```text
token 0  -0.025315   relative 1.00
token 1  -0.025407   relative 1.00
token 2  -0.023033   relative 0.91
token 3  -0.000844   below estimator floor
token 4  +0.006462   sign crossed / recovered
```

The defect is therefore a **transient framing excursion around the handoff**, not a permanent whole-chunk framing bias. The old persistence rejection prevented an unsafe full-suffix warp, but its binary `persistent -> full suffix / otherwise -> no correction` decision was too coarse.

## Current policy

The redesigned pipeline separates spatial evidence from temporal shape:

1. Fit recent protected-prefix motion on the source grid.
2. Measure the source boundary residual and build provisional source axes.
3. Independently fit recent protected-prefix motion on the target grid.
4. Measure the target boundary residual and build provisional target axes.
5. Cross-grid authorize only shared source/target axes with compatible sign, magnitude, and evidence.
6. Only for cross-grid-authorized source axes, inspect genuine source suffix transitions.
7. Integrate those transition residuals into a per-axis temporal framing state.
8. Classify each axis independently as `persistent`, `recovering`, `ambiguous`, or `inactive`.
9. Use the **target-grid residual magnitude** for the actual geometric transform.
10. Use the **source-grid temporal state only for the temporal envelope**.

Cross-grid corroboration therefore runs before temporal classification. A recovering source trajectory no longer suppresses an otherwise valid target corroboration result.

## Evidence gates

The existing conservative evidence/safety contract is retained:

- scale estimator floor: `0.005` in signed log scale;
- translation estimator floor: `0.25` source latent px;
- maximum scale residual: `log(1.03)`;
- maximum translation: `min(1.5 latent px, 2.5% of the spatial axis)`;
- minimum evidence from each domain: `1.25`;
- combined source/target evidence threshold: `2.5`;
- source/target magnitude compatibility ratio: `0.35 .. 2.85`.

The transient redesign does not weaken these thresholds.

## Temporal state and envelope

For an authorized axis with source boundary residual `b` and follow-up residuals `r_k`:

```text
s0 = b
sk = s(k-1) + rk
```

While the state has the same sign as `b`, a raw correction weight is:

```text
raw_k = clamp(sk / b, 0, 1)
```

If the state crosses zero, correction terminates. If the absolute state falls at or below that axis's estimator floor, correction also terminates because the remaining state is below the estimator's meaningful resolution.

The final envelope is the deterministic monotonic projection:

```text
w0 = 1
wk = min(w(k-1), raw_k)
```

This ensures:

- the boundary suffix token receives full correction;
- weights never overshoot `1`;
- correction never becomes negative;
- estimator noise cannot make a decayed correction grow again;
- recovery/sign crossing is terminal and never reintroduces correction later;
- no arbitrary handcrafted `(1, 0.5, 0.25)` fade is used.

For the real `sy` trajectory, the `0.000844` state is below the `0.005` log-scale estimator floor, so the measured policy snaps that token and the remainder to zero. The expected envelope is therefore approximately:

```text
1.0, 1.0, 0.91, 0.0, 0.0
```

The exact weights are derived from the measured trajectory at runtime; those values are not hard-coded.

## Persistent, recovering, and unsafe cases

### Persistent

At least two usable follow-up transitions are required. If the cumulative state remains within one estimator floor of the original boundary state, the residual is classified as persistent and the accepted axis keeps weight `1` through the suffix.

### Recovering

Recovery must be **observed**, not extrapolated. If cumulative state reaches the estimator floor or crosses sign inside the measured suffix window, the axis is classified as recovering and receives only the measured monotonic transient envelope up to recovery.

Immediate observed recovery is valid: only the boundary suffix token is corrected.

### Ambiguous / unsafe

The axis is rejected rather than guessed when evidence is insufficient or the measured state:

- grows beyond the boundary magnitude by more than estimator resolution;
- reverses materially away from an already observed recovery trend;
- remains in unresolved partial drift without observed recovery;
- has non-contiguous/unusable transition evidence before classification.

Uncertainty is never promoted to `persistent`.

## Target-grid transform semantics

The source trajectory decides **when** an authorized correction is active. It does not supply the final target transform magnitude.

For target residual scale `a` and weight `w_k`:

```text
effective_scale_k = exp(w_k * log(a))
```

Scale is therefore interpolated in log space.

For target residual translation `d`:

```text
effective_translation_k = w_k * d
```

Translation is interpolated linearly in target-grid latent units.

Source translation is projected into target units only for cross-grid magnitude comparison. It is not blindly reused as the correction.

## Vectorized suffix warp

The warp remains one batched `affine_grid` / `grid_sample` operation:

1. find the last suffix token for which any accepted axis has non-zero weight;
2. build one affine matrix per active temporal token;
3. flatten `B x active_T`;
4. perform one float32 resampling batch;
5. write back only the active corrected suffix prefix;
6. leave every later recovered suffix token untouched bit-for-bit.

Persistent axes keep the full suffix active. Recovering axes stop at their measured recovery point. Identity tokens after recovery are not resampled merely for bookkeeping.

The output is converted back to the original tensor dtype.

## Exact-prefix and runtime contracts

The authoritative target-grid protected prefix remains non-negotiable:

- no geometric transform is applied to protected prefix tokens;
- the learned prefix is diagnostic/calibration context only;
- final prefix restoration remains exact;
- canonical masking behavior is unchanged.

The runtime ordering remains:

```text
clean learned transfer
-> geometric correction
-> one-token DC bridge
-> conditional re-noise / high-stage state mapping
```

The geometric prefix diagnostic never replaces the authoritative prefix and does not change DC calibration semantics.

The bridge also does **not** change:

- audio;
- caller noise;
- masks;
- conditioning;
- VDN external-sequence API v2;
- Spectrum history or NFE accounting;
- generic Progressive or Target-Sparse node UI.

No optical flow network, VAE pass, crossfade, decode-space fix, or additional H3 transformer evaluation is introduced.

## No-op behavior

If the option is disabled, source context is unavailable, source or target candidates fail, cross-grid corroboration fails, temporal classification is ambiguous/unsafe, or no axis survives, `geometric_seam_bridge()` returns the original learned tensor object.

The released DC-only runtime arithmetic is then used unchanged. Diagnostic execution alone must not create numerical differences.

## Diagnostics

`mixed_grid_geometry` now separates the evidence stages instead of allowing temporal rejection to suppress cross-grid reporting.

Relevant fields include:

- `motion_residual`: source boundary residual, estimator floors, motion dispersion, evidence, provisional axes;
- `target_motion_residual`: equivalent independent target-grid measurements;
- `cross_grid_corroboration`: shared candidate axes, sign agreement, magnitude ratio, source/target evidence, combined score/threshold, per-axis authorization/reason;
- `temporal_profile`: per-axis boundary residual, follow-up residuals, cumulative state, raw and monotonic envelope, estimator floor, recovery token, mode and rejection/acceptance reason;
- `base_target_transform`: target-grid residual after both spatial and temporal axis selection;
- `effective_active_temporal_length` and `tokens_corrected`;
- `applied_transforms`, with a compact single-transform representation for a persistent constant policy plus `applied_transform_sequence_length`;
- `boundary_after_geometry` and `out_of_bounds_fraction`;
- `learned_prefix_unchanged` and runtime `final_prefix_exact`.

`suffix_persistence` is retained as a compatibility alias for the richer `temporal_profile`; new consumers should use `temporal_profile`.

## DC bridge remains separate

The latest rejected-geometry run still showed the one-token DC bridge doing useful work:

```text
exact-restored seam spatial-mean RMS  0.246803
after DC bridge                      0.174657
```

This is about a 29% reduction. Geometry and DC address different failure modes and remain separate operations.

## Validation status

Structural tests cover:

- the actual `sy` evidence and follow-up numeric regression;
- persistent and recovering synthetic seams;
- immediate recovery and sign crossing;
- monotonic projection under estimator noise;
- divergent/oscillatory/unresolved rejection;
- source-only, target-only, sign, magnitude, domain-floor, and combined-evidence failures;
- source-axis independence from target-only nuisance fits;
- natural pan/zoom motion;
- log-space scale and linear translation interpolation;
- source-to-target translation units;
- exact protected prefix and exact recovered suffix tail;
- float32/float16/bfloat16/float64 paths plus CUDA where available;
- runtime geometry -> DC -> re-noise ordering;
- unchanged caller noise, masks, audio, and exact-probe NFE;
- Mixed-Grid-only UI exposure with `suffix_geometric_bridge=false` by default.

These tests establish structural/runtime correctness only. They do not prove decoded perceptual improvement.

## Next decoded-media check

After CI is green on the exact final PR SHA, run one matched B case with:

```text
suffix_dc_bridge = true
suffix_geometric_bridge = true
```

Keep seed, prompt, references, source/target grids, sampler, scheduler, step counts, VDN, Spectrum, DiffAid, Untwist RoPE, learned latent upscaler, VAE/decode setup, and Continuum configuration unchanged.

The previous run where geometry was requested but rejected (`tokens_corrected=0`, `applied_transforms=[]`) is already an effective DC-only control if its decoded media is retained.

For the observed `sy` case, the new B metrics should show cross-grid authorization around the existing `2.68` combined score and a recovering temporal envelope affecting only early suffix tokens. The decisive release criterion is still the decoded seam: it must improve without introducing a delayed compensating zoom/wobble.
