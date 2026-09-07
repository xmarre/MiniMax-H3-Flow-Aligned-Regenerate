# Experimental geometric Continuum seam bridge

A progressive Mixed-Grid Continuum run exposed a second boundary failure after the released one-token DC bridge had already removed the obvious flash: a slight whole-frame framing change at the chunk join, visible as a small shrink/zoom-out and top-edge reveal.

The DC bridge remains correct for its narrow purpose. It adjusts per-channel spatial means. It cannot correct global scale or translation.

This geometric bridge remains **experimental and off by default**. No corrected decoded run has yet validated the current policy.

## Evidence from the instrumented runs

The real handoff uses source `40x54` and target `56x76` latent grids with `suffix_dc_bridge=true`.

The same-time protected-prefix / learned-prefix fit remains effectively identity:

- `sx=1.0125`, `sy=0.9950`, `tx=0`, `ty=0.25` latent px;
- identity error `0.171124`;
- aligned error `0.168418`;
- improvement only `1.58%`.

That argues against a systematic learned-upscaler prefix warp being the main cause.

The actual generated boundary repeatedly carries a vertical framing residual:

- genuine source-grid boundary relative to recent source-grid motion: `sy≈0.9750`;
- learned target-grid boundary relative to recent target-grid motion: `sy≈0.9831`;
- final accepted latent boundary after high-grid refinement: `sy≈0.9800`.

The final boundary fit improves about `30%` over identity. The repeated roughly `2%` vertical scale signal matches the observed top-edge reveal/zoom-out and exists before the target-grid high stage.

The second instrumented run also exposed a flaw in the previous authorization gate. Source vertical-scale motion dispersion was about `0.01296` in log-scale and target dispersion about `0.00928`. The source residual contributes about `1.95` dispersion units and target about `1.83`. Neither domain independently clears the previous `2.5x` dispersion rule, although the same axis/sign appears in both domains and survives the high stage.

Worse, `2.5 * 0.01296 ≈ 0.0324` exceeds the separate `log(1.03) ≈ 0.0296` maximum scale safety bound. For that real motion history, the old source rule made any scale correction mathematically impossible to authorize.

The fix is **not** to weaken the safety bound or simply lower `2.5x` to another hand-tuned standalone threshold. The bridge now separates provisional evidence from final authorization.

## Scope and invariants

Only `MiniMax H3 Progressive Mixed-Grid Continuum` exposes `suffix_geometric_bridge`.

- default: `false`;
- generic Progressive Target Input: unchanged;
- Target-Sparse: unchanged;
- authoritative target-grid protected prefix: never warped or blended;
- learned prefix output: never made authoritative;
- audio, masks, conditioning, caller noise, VDN API 2, Spectrum histories and H3 NFE count: unchanged;
- geometry is applied to the learned clean target-grid suffix before conditional re-noising;
- DC calibration remains after geometry and uses the unmodified learned-prefix relation;
- final exact-prefix restoration and the existing exactness checks remain mandatory;
- disabled or rejected geometry returns to the released v0.3.2 DC-only arithmetic path.

## The two clean sequences

Immediately before learned 3D upsampling, the Mixed-Grid runtime has a source-grid clean handoff sequence:

```text
[ downsample(P_exact) | S_low ]
```

`P_exact` is the authoritative target-grid protected prefix. A bicubic source-grid copy is used only as transient 3D-upscaler context. `S_low` is the genuinely generated low-grid suffix from the exact handoff probe.

The learned provider returns the target-grid clean sequence:

```text
[ U_prefix | U_suffix ]
```

`U_prefix` is discarded after calibration/diagnostics. `U_suffix` is the learned target-grid continuation that is conditionally re-noised and refined.

The provider wrapper exposes its invocation-local source-grid input to the geometric diagnostic on the returned tensor. It retains no trajectory tensor between calls.

## Registration model

The estimator fits only `sx, sy, tx, ty`; there is no rotation or shear.

Coordinates are output-to-input sampling coordinates about image centre:

```text
input_x = sx * (output_x - (W-1)/2) + (W-1)/2 + tx
input_y = sy * (output_y - (H-1)/2) + (H-1)/2 + ty
```

Warping uses float32 bilinear `grid_sample`, `padding_mode="border"`, `align_corners=false`, then restores the original dtype/device.

Registration uses all latent channels with:

1. 5x5 spatial low-pass;
2. area reduction capped at 48 pixels on the long axis;
3. per-channel spatial centring and RMS normalization;
4. a fixed three-pixel interior scoring margin;
5. clipped squared residuals to limit local-detail outliers.

The deterministic coarse-to-fine search runs without autograd.

## Natural-motion model

Both source and target domains model up to the last six protected-prefix transitions. Each transition receives a diagnostic `sx, sy, tx, ty` fit.

At least three usable transitions spanning at least two transition indices are required. Each parameter is extrapolated with indexed Theil-Sen regression; scale is modelled in log space. Rejected intermediate transitions keep their original time indices so trend estimation is not compressed.

The report includes measured transitions, expected next transform, trend and robust dispersion.

## Provisional source and target residuals

For each domain:

```text
C = T_observed ∘ inverse(T_expected)
```

Natural pan/zoom already explained by recent motion is therefore preserved.

A residual axis becomes **provisional** when all of the following hold:

- fitted boundary is better than the predicted natural-motion transform;
- aligned error is `<= 0.25`;
- residual exceeds the estimator floor;
- forcing that axis back to expected motion measurably worsens the objective;
- residual stays inside the existing safety bound.

Current estimator floors:

- scale: `0.5%` in log-scale magnitude;
- translation: `0.25` latent pixel.

Current maximum residual:

- scale: `3%` per axis;
- translation: `min(1.5 latent px, 2.5% of that spatial axis)`.

The previous `>=5%` whole-boundary improvement and `>=2.5x` motion-dispersion checks no longer stop the bridge before persistence/cross-grid evidence can be collected. Metrics still expose the old `2.5x` standalone threshold and whether each axis would have passed it.

For each axis the domain evidence score is:

```text
E_domain = |residual| / max(estimator_floor, robust_motion_dispersion)
```

This is an engineering evidence unit, not a statistical sigma or p-value.

## Persistent-offset evidence

A provisional source boundary residual does not justify warping the whole suffix. The bridge inspects up to the first four genuine low-grid suffix-to-suffix transitions.

Persistence is evaluated **per provisional source axis**. Later transitions must return close to extrapolated natural motion and must not contain a strong opposite compensating residual.

This distinguishes:

- one-time framing offset that persists in subsequent frames -> axis remains eligible;
- immediate recovery or continuing drift -> that axis is rejected;
- insufficient usable suffix evidence -> no authorization.

One failed provisional axis cannot veto another axis that is independently persistent.

The old three-token `(1, 0.5, 0.25)` decay policy remains removed because a persistent framing offset followed by decay-to-identity can create a delayed second zoom/wobble.

## Cross-grid corroboration and combined authorization

A persistent source axis must also exist independently in the target-grid boundary.

For translation, source projection uses the actual grid ratios:

```text
tx_target_projection = tx_source * target_W / source_W
ty_target_projection = ty_source * target_H / source_H
```

Scale is dimensionless and is not rescaled.

An axis can proceed only if:

- source and target both nominate it provisionally;
- source persistence supports it;
- source and target residual signs agree;
- target/source residual magnitude ratio is within `0.35x .. 2.85x`;
- both source and target contribute at least `1.25` evidence units.

The combined engineering concordance score is:

```text
E_combined = hypot(E_source, E_target)
```

and must be at least `2.5`.

The quadrature score is deliberately labelled an engineering concordance score. Source and target measurements are related and this formula is **not** a claim of statistical independence.

The per-domain minimum prevents one very strong measurement from carrying an almost absent signal in the other domain.

The applied correction magnitude comes from the independently measured target-grid residual, not from the projected source transform. The source domain establishes origin and persistence; the target domain establishes what correction is actually present on `U_suffix`.

## Persistent suffix correction

Only axes authorized by persistence plus combined cross-grid evidence are applied to the complete learned target-grid suffix.

The suffix is flattened over batch/time and resampled in one batched `grid_sample` call. There is no tensor crossfade and the authoritative prefix is untouched.

The full-suffix policy remains an empirical hypothesis pending decoded-media validation.

## Geometry -> DC -> re-noise ordering

When geometry is authorized:

1. measure same-time learned-prefix geometry diagnostically;
2. build source and target recent-motion models;
3. build provisional source/target residual axes;
4. measure per-axis source suffix persistence;
5. require same-axis/sign/magnitude cross-grid compatibility;
6. authorize only axes meeting per-domain and combined evidence thresholds;
7. warp only the learned target-grid suffix;
8. calibrate the existing DC relation from the unmodified learned-prefix boundary;
9. apply the existing one-token DC bridge;
10. conditionally re-noise;
11. restore the authoritative exact prefix;
12. run the fresh target-grid high stage and verify prefix exactness.

If geometry is disabled or rejected, the original learned tensor object is returned and the released inverse-recovery/DC/state-delta arithmetic is retained.

## Metrics

`mixed_grid_geometry` carries:

- `source_hw`, `target_hw`, grid scale and anisotropy;
- same-time `prefix_registration` and `transfer_bias_candidate`;
- source and target natural-motion models;
- source and target boundary fits;
- provisional source and target residual transforms;
- per-axis estimator floors, motion dispersion, evidence units and evidence scores;
- diagnostic standalone `2.5x` dispersion thresholds;
- per-axis suffix-persistence evidence and `axis_persistent`;
- projected source transform;
- source/target sign agreement and magnitude ratios;
- per-domain evidence-floor pass state;
- per-axis `combined_evidence_score` and authorization reason;
- final authorized axes and applied transform;
- corrected-token count and out-of-bounds fraction;
- DC information and exact-prefix state.

`mixed_grid_transfer` continues to report the existing raw, low-pass and spatial-mean seam/DC measurements. `mixed_grid_complete` continues to report the final accepted-latent boundary after target-grid refinement.

## Validation

Keep PR #24 draft and keep `suffix_geometric_bridge=false` as the default.

Use matched runs with the same workflow, seed, prompt, references, source scale, sampler, VDN, Spectrum and DC settings:

1. A: `suffix_geometric_bridge=false`, `suffix_dc_bridge=true`.
2. B: `suffix_geometric_bridge=true`, `suffix_dc_bridge=true`.

Return both decoded videos, full logs, complete H3 Flow metrics JSON and Decode Context reports.

For B, inspect:

- source provisional `motion_residual.axis_applied` and `axis_evidence_score`;
- `suffix_persistence.axis_persistent`;
- target provisional `target_motion_residual.axis_applied` and `axis_evidence_score`;
- `cross_grid_corroboration.sign_agreement` and `magnitude_ratio`;
- `cross_grid_corroboration.combined_evidence_score`;
- final `cross_grid_corroboration.axis_applied`;
- `tokens_corrected` and `applied_transforms`;
- final accepted-latent boundary registration;
- decoded top-edge reveal, whole-frame zoom/shift, natural camera motion, edge stretching, delayed wobble, detail and audio continuity.

Do not loosen a rejected gate before identifying which evidence failed. Do not call the visible issue fixed, enable the option by default, or merge until the corrected decoded-media run demonstrates it.
