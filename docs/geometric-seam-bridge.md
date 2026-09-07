# Experimental geometric Continuum seam bridge

A progressive Mixed-Grid Continuum run exposed a second boundary failure after the released one-token DC bridge had already removed the obvious flash: a slight whole-frame framing change at the chunk join, described as a small shrink/zoom-out and top-edge reveal.

The DC bridge remains correct for its narrow purpose. It adjusts per-channel spatial means. It cannot correct global scale or translation.

This geometric bridge remains **experimental and off by default**. No corrected decoded run has yet validated the current policy.

## Evidence that changed the design

The first instrumented real run used source `40x54` and target `56x76` latent grids with `suffix_geometric_bridge=true` and `suffix_dc_bridge=true`. The initial geometric bridge correctly rejected and changed zero geometric tokens.

The same-time learned-prefix fit was effectively identity:

- `sx=1.0125`, `sy=0.9950`, `tx=0`, `ty=0.25` latent px;
- identity error `0.171124`;
- aligned error `0.168418`;
- improvement only `1.58%`;
- no stable learned-prefix bias candidate.

That argues against a systematic learned-upscaler prefix warp being the main cause.

The actual protected-prefix -> generated-suffix boundary was different:

- before target-grid refinement: approximately `sx=1.0075`, `sy=0.9775`, `tx=-1.25`, `ty=0.5`;
- final accepted latent boundary: approximately `sx=0.9975`, `sy=0.9800`, `tx=-1.25`, `ty=0.25`;
- final fitted-vs-identity improvement was about `30%`.

The repeated `sy≈0.98` signal matches the observed slight vertical zoom-out/top-edge reveal. Recent exact-prefix motion already contained much of the leftward translation, so the raw boundary transform must not be cancelled wholesale.

The implementation therefore no longer lets same-time learned-prefix registration drive correction. It now asks two separate questions:

1. **Did the real low-grid generated suffix introduce a persistent framing step relative to recent prefix motion?**
2. **Did that same residual survive learned upsampling into the target-grid suffix that would actually be warped?**

Both must be true before correction is allowed.

## Scope and invariants

Only `MiniMax H3 Progressive Mixed-Grid Continuum` exposes `suffix_geometric_bridge`.

- default: `false`;
- generic Progressive Target Input: unchanged;
- Target-Sparse: unchanged;
- authoritative target-grid protected prefix: never warped or blended;
- learned prefix output: never made authoritative;
- audio, masks, conditioning, caller noise, VDN API 2, Spectrum histories and H3 NFE count: unchanged;
- geometry is applied to the learned clean target-grid suffix before conditional re-noising;
- DC calibration remains after geometry;
- final exact-prefix restoration and the existing exactness checks remain mandatory;
- disabled or rejected geometry returns to the released v0.3.2 DC-only arithmetic path.

## The two clean sequences

Immediately before learned 3D upsampling, the Mixed-Grid runtime has a source-grid clean handoff sequence:

```text
[ downsample(P_exact) | S_low ]
```

`P_exact` is the authoritative target-grid protected prefix. A bicubic source-grid copy is used only as transient 3D-upscaler context. `S_low` is the genuinely generated low-grid suffix from the exact handoff probe. The source-grid prefix copy is not authoritative and is never returned.

The learned provider then returns the target-grid clean sequence:

```text
[ U_prefix | U_suffix ]
```

`U_prefix` is discarded after calibration/diagnostics. `U_suffix` is the learned target-grid continuation that is conditionally re-noised and refined.

The provider wrapper exposes its invocation-local source-grid input to the geometric diagnostic on the returned tensor. The wrapper retains no trajectory tensor between calls; repeated chunks receive fresh context. This mechanism exists only for the Mixed-Grid learned-transfer node.

## Registration coordinates

The estimator fits only `sx, sy, tx, ty`; there is no rotation or shear.

Coordinates are output-to-input sampling coordinates about the image centre:

```text
input_x = sx * (output_x - (W-1)/2) + (W-1)/2 + tx
input_y = sy * (output_y - (H-1)/2) + (H-1)/2 + ty
```

Positive `tx` samples to the right and therefore moves displayed structure left.

Warping uses float32 bilinear `grid_sample`, `padding_mode="border"`, `align_corners=false`, then restores the original dtype/device. Border extension avoids invented zero-valued edges but can stretch edge content, so transforms are tightly bounded and the out-of-bounds sample fraction is reported.

## DC-insensitive registration representation

Registration uses all latent channels, not one selected channel:

1. 5x5 spatial low-pass;
2. area reduction capped at 48 pixels on the long axis;
3. per-channel spatial centring and RMS normalization;
4. a fixed three-pixel interior scoring margin so padding cannot win the fit;
5. clipped squared residuals to limit local-detail outliers.

The deterministic coarse-to-fine search runs without autograd. NumPy is used only for the small CPU candidate-scoring arrays; global thread settings are not modified.

## Same-time learned-prefix diagnostic

`P_exact[t] <-> U_prefix[t]` is still measured over the last up-to-three paired prefix tokens.

A stable nonidentity fit is reported as `transfer_bias_candidate`, but **it cannot authorize suffix warping**. The first real run showed why: the learned prefix was effectively aligned even though the generated boundary was not.

## Natural-motion model

Both the source and target domains build a local natural-motion model from up to the last six protected-prefix transitions.

Source domain:

```text
downsample(P_exact[-6]) -> downsample(P_exact[-5])
...
downsample(P_exact[-2]) -> downsample(P_exact[-1])
```

Target domain:

```text
P_exact[-6] -> P_exact[-5]
...
P_exact[-2] -> P_exact[-1]
```

Each transition receives a diagnostic `sx, sy, tx, ty` fit. At least three usable transitions spanning at least two transition indices are required.

For each parameter the motion model performs indexed Theil-Sen trend estimation and predicts the next transition. Scale is modelled in log space. Keeping the original transition indices matters: a rejected intermediate fit must not compress time and bias the extrapolation.

The report contains the measured transitions, usable count, expected next transform, trend, robust dispersion and fitted intercept.

## Source-grid trajectory residual

The primary cause diagnostic is the actual source-grid boundary:

```text
downsample(P_exact[-1]) -> S_low[0]
```

Let:

- `T_source_expected` = the next transform predicted from recent source-grid prefix motion;
- `T_source_observed` = the fitted source-grid boundary transform.

The unexpected residual is:

```text
C_source = T_source_observed ∘ inverse(T_source_expected)
```

Ordinary translation or zoom already explained by recent motion is therefore preserved.

Each of `sx`, `sy`, `tx`, `ty` is gated independently. An axis is selected only when:

- the fitted boundary is at least 5% better than the natural-motion prediction;
- aligned error is no worse than `0.25`;
- the residual exceeds a fixed estimator floor;
- it also exceeds `2.5x` recent-motion dispersion;
- forcing that axis back to the expected-motion value measurably worsens the objective;
- it remains inside conservative source-grid safety bounds.

Current floors:

- scale: `0.5%` in log-scale magnitude;
- translation: `0.25` latent pixel.

Current maximum residual:

- scale: `3%` per axis;
- translation: `min(1.5 latent px, 2.5% of that spatial axis)`.

These are engineering gates, not perceptual calibration constants. Do not loosen them merely because a real run rejects.

## Persistent-offset evidence

A boundary residual alone does not justify warping the whole suffix. A transient first-token mismatch corrected by the subsequent low-grid trajectory would make a full-suffix warp wrong.

The bridge therefore inspects up to the first four genuine low-grid suffix-to-suffix transitions after the boundary. For each axis selected at the boundary, those later transitions must return close to the extrapolated natural-motion transform and must not contain a strong opposite compensating residual.

This distinguishes:

- **constant framing offset introduced once at the boundary** -> eligible for persistent correction;
- **immediate recovery or continuing drift** -> reject with `suffix_recovery_or_drift_detected`;
- **too little usable suffix evidence** -> reject.

The previous three-token `(1, 0.5, 0.25)` decay policy is removed. A short decay could turn one framing discontinuity into a delayed second zoom/wobble.

## Target-grid corroboration

Even a persistent source-grid residual is not enough. The learned upscaler could attenuate, remove or change it. The bridge therefore independently evaluates the actual target-grid boundary:

```text
P_exact[-1] -> U_suffix[0]
```

It builds `T_target_expected`, computes the target residual `C_target`, and applies the same conservative per-axis gate.

A source-selected axis is eligible only when the target domain independently selects the same axis, the residual sign agrees, and the target magnitude is broadly compatible with the source residual projected into target-grid units.

For translation, source projection uses the actual grid ratios:

```text
tx_target_projection = tx_source * target_W / source_W
ty_target_projection = ty_source * target_H / source_H
```

Scale is dimensionless and is not rescaled.

The current corroboration band allows the target residual magnitude to be `0.35x` to `2.85x` the projected source magnitude. This intentionally broad gate is only a consistency test; it does not assume the learned upscaler preserves residual magnitude exactly.

**The applied correction magnitude comes from the independently measured target-grid residual, not from the projected source transform.** The source domain establishes cause and persistence; the target domain establishes what correction is actually present on `U_suffix`.

If the two domains disagree, geometry is a no-op.

## Persistent suffix correction

Only after all three gates succeed:

1. significant source-grid motion residual;
2. persistent-offset evidence in later low-grid suffix transitions;
3. same-axis target-grid corroboration;

is the target residual applied to the complete learned target-grid suffix.

The suffix is flattened over batch/time and resampled in one batched `grid_sample` call. There is no tensor crossfade. The authoritative prefix is untouched.

The full-suffix policy is still an empirical hypothesis and requires decoded-media validation before merge.

## Geometry then DC

If geometric correction is accepted:

1. measure same-time learned-prefix geometry diagnostically;
2. measure source-grid natural motion and boundary residual;
3. verify persistent-offset behaviour in the genuine low-grid suffix;
4. independently measure and corroborate the target-grid boundary residual;
5. warp only the learned target-grid suffix;
6. calibrate the existing DC relation from the unmodified learned-prefix boundary;
7. apply the existing one-token DC bridge;
8. conditionally re-noise;
9. restore the authoritative exact prefix;
10. run the fresh target-grid high stage and verify prefix exactness.

If geometry is disabled or rejected, the hook returns the original learned tensor object and the released inverse-recovery/DC/state-delta arithmetic runs unchanged.

## Metrics

`mixed_grid_geometry` carries:

- `source_hw`, `target_hw`, independent grid scale and anisotropy;
- `prefix_registration` and `transfer_bias_candidate`;
- `target_natural_motion_model`;
- target `boundary_before`;
- `target_motion_residual`;
- `low_grid_trajectory`, including source natural-motion model, source boundary, source residual and suffix-persistence evidence;
- top-level `motion_residual` and `suffix_persistence` aliases for the source decision;
- `cross_grid_corroboration`, including projected source transform, target transform, per-axis sign agreement, magnitude ratios and selected axes;
- `persistent_bias_candidate` and `corroborated_bias_candidate`;
- requested/accepted state and rejection reason;
- corrected-token count and applied transform;
- out-of-bounds fraction;
- `boundary_after_geometry`;
- pre/post geometry DC residual;
- learned-prefix unchanged flag;
- final exact-prefix state supplied by the runtime.

`mixed_grid_transfer` continues to report the existing raw, low-pass and spatial-mean seam/DC measurements. `mixed_grid_complete` continues to report the final accepted-latent boundary and exact-prefix result after target-grid refinement.

## Interpretation

### Low-grid trajectory cause supported

`low_grid_trajectory.motion_residual.accepted=true` means the real low-grid generated boundary differs materially from recent prefix motion. This is the direct evidence we were missing in the first implementation.

### Persistent framing offset supported

`suffix_persistence.accepted=true` means later low-grid suffix transitions do not immediately undo or continue accumulating the selected residual. This supports a one-time framing offset rather than a transient jump.

### Target-grid survival supported

`target_motion_residual.accepted=true` plus `cross_grid_corroboration.accepted=true` means the same residual axis survives learned upsampling and is present on the clean target-grid suffix. Only this case can activate correction.

### Transfer bias

`prefix_registration` may show a stable same-time nonidentity learned-prefix transform. It remains diagnostic and does not drive this bridge.

### High-stage/decode candidate

If the clean learned target boundary is normal but `mixed_grid_complete.final_boundary_registration` changes, inspect target-grid high-stage evolution. If accepted latents remain normal while decoded media jumps, inspect temporal VAE/decode/assembly.

## Validation

Keep PR #24 draft.

Use the same workflow, seed, prompt, references, source scale, sampler, VDN, Spectrum and DC settings:

1. A: `suffix_geometric_bridge=false`, `suffix_dc_bridge=true`.
2. B: `suffix_geometric_bridge=true`, `suffix_dc_bridge=true`.

Return both decoded videos, full logs, complete H3 Flow metrics JSON and Decode Context reports.

For B, inspect `mixed_grid_geometry` first. Do not loosen a rejected gate before understanding which evidence failed.

Inspect the decoded boundary for:

- top-edge reveal;
- whole-frame zoom/shift;
- preservation of ordinary camera/object translation;
- edge stretching from border sampling;
- persistent crop error;
- ghosting;
- delayed wobble or a second boundary;
- face/background detail;
- temporal motion;
- audio continuity.

Do not call the visible issue fixed until the corrected decoded-media run demonstrates it.
