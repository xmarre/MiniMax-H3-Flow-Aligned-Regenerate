# Mixed-Grid seam repair — historical evidence

## Status

Mixed-Grid is retired from the current production acceptance matrix. This document preserves the seam-investigation findings because they remain useful evidence about the old path; they are not current workflow instructions.

The compatibility node remains executable for one deprecation release, but no Mixed-Grid compatibility render is required. New target-input/Continuum workflows use **MiniMax H3 Progressive Handoff (Target Input)** and its conservative exact-prefix target-grid fallback.

## Findings retained from the Mixed-Grid investigation

The historical seam work separated several independent effects rather than treating every boundary defect as one problem.

### Target-grid affine correction was insufficient

`metrics_00276` reduced the measured signed `sy` residual from roughly `-0.01025` to `+0.00236`, but the decoded framing jump remained. A target-grid affine correction was therefore not sufficient.

### Exact-overlap representation repair was real but insufficient alone

`metrics_00281` reduced the learned-prefix replacement mismatch essentially to numerical noise before target-grid refinement:

```text
centered error: 0.400812 -> ~4.3e-08
corrected/native raw seam ratio:          ~1.0
corrected/native low-pass seam ratio:     ~1.0
corrected/native spatial-mean seam ratio: ~1.0
```

The decoded framing defect still remained in that isolated test. The replacement splice was a measurable seam amplifier, but not a complete explanation for the whole-frame discontinuity.

### Source-space warping was falsified under the preserved contracts

The source trajectory experiments moved correction before the learned upscaler and progressively tightened authorization, temporal classification, cumulative safety, per-axis verification and finite-horizon backoff.

`metrics_00318` exhausted every directly authorized finite source-warp horizon:

```text
axis_active_tokens_authorized     = [1, 1, 4, 5]
round 1                           = [1, 1, 4, 5]
round 2                           = [0, 0, 3, 4]
round 3                           = [0, 0, 2, 3]
round 4                           = [0, 0, 1, 2]
round 5                           = [0, 0, 0, 1]
axis_active_tokens_post_verified  = [0, 0, 0, 0]
source_trajectory_bridge_accepted = false
source_trajectory_bridge_tokens_corrected = 0
learned_upscaler_input_modified   = false
```

The final one-token translation attempt improved its immediate boundary residual but created a larger corrected-to-untouched transition immediately afterward. Shortening the finite warp only moved that compensating discontinuity earlier. The source-warp family was therefore retired rather than replaced with an unsupported fade or extrapolation.

### One-token DC correction addressed a separate transfer-boundary problem

The historical Mixed-Grid path used the learned upscaler's discarded prefix output as calibration for a one-token per-channel spatial-mean correction on the first generated suffix token. In the validated multi-boundary run, the uncorrected exact-boundary DC RMS was about `0.207763`; the corrected boundary was about `0.093093`, approximately matching the learned-upscaler native boundary, while the authoritative prefix remained exact.

This result applied to the Mixed-Grid transfer path. It is not part of the standard Target Input exact-prefix fallback, which does not perform a learned transfer at that boundary.

### Unequal Mixed-Grid carrier density required attention-measure treatment

The old mixed sequence could contain target-grid prefix frames with substantially more physical K/V rows per frame than genuine source-grid suffix frames. A representative validated geometry used:

```text
protected prefix: 28 x 38 = 1064 rows/frame
source suffix:    20 x 27 =  540 rows/frame
ratio:                         ~1.97037x
```

The historical attention-measure repair normalized this unequal physical sampling measure rather than treating every packed row as equal mass. In one validated accounting path:

```text
Q:   56029 -> 56029
K/V: 56029 -> 49741
removed protected-prefix K/V rows: 6288 per attention call
```

Later weighted implementations represented the same physical-measure requirement with additive key-score bias while preserving all Q/K/V rows. Those companion implementations are now retired from the current production Patcher topology because Mixed-Grid itself is no longer the production continuation path.

## What is current instead

For exact protected video, **Progressive Handoff (Target Input)** stays on the final target grid and executes one ordinary target-grid sampler lifetime. It adds no low-grid continuation lifetime, no exact handoff probe, no learned-upscaler call, and no mixed-grid attention contract.

Its production boundary intervention is the independently validated four-audio-tick guided overlap during sampler lifetime, with exact caller-owned video/audio restored at output.

## Preservation policy

The old metrics, release notes and closed PRs remain historical evidence. Their conclusions are not rewritten merely because the production architecture changed. After the one-release compatibility window, the remaining Mixed-Grid runtime machinery can be deleted separately without deleting this evidence trail.
