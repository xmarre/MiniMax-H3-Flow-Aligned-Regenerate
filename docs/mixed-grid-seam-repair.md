# Mixed-Grid seam repair

## Current status

The Mixed-Grid exact-prefix seam repair is validated and enabled by default through the legacy workflow input `suffix_geometric_bridge=true`.

The name is retained for workflow compatibility. The former source-space affine/trajectory warp family is **retired** and is not applied to the learned-upscaler input. The active repair has two independent pieces:

1. an upstream Mixed-Grid attention-measure contract during the low/probe H3 transformer stages; and
2. target exact-overlap representation reconciliation after learned 3D transfer.

The separate `suffix_dc_bridge=true` remains responsible for the one-token spatial-mean/DC correction. Both controls default on in **MiniMax H3 Progressive Mixed-Grid Continuum**.

## Evidence chain

### Target-grid affine was insufficient

`metrics_00276` authorized a target-grid affine correction and reduced the measured signed `sy` residual from roughly `-0.01025` to `+0.00236`. The decoded framing jump remained.

### Exact-overlap representation repair was real but insufficient alone

`metrics_00281` repaired the learned-prefix replacement mismatch essentially to numerical noise before target-grid refinement:

```text
centered error: 0.400812 -> ~4.3e-08
corrected/native raw seam ratio:          ~1.0
corrected/native low-pass seam ratio:     ~1.0
corrected/native spatial-mean seam ratio: ~1.0
```

The decoded framing defect still remained in that isolated test. The exact-prefix replacement splice is therefore a measurable seam amplifier, but was not a sufficient explanation for the whole-frame framing discontinuity.

### Source-space warping was falsified under the preserved contracts

The source trajectory experiments then moved the correction before the learned upscaler and successively fixed authorization, temporal classification, cumulative safety, per-axis verification and finite-horizon backoff.

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

The final one-token translation attempt improved its immediate boundary residual but created a larger corrected-to-untouched transition immediately afterward. Shortening the finite warp only moved that compensating discontinuity earlier. Removing it would require a handcrafted fade/crossfade or unmeasured extrapolation, neither of which follows from the measured source trajectory.

The runtime therefore no longer attempts source warping. It reports:

```text
source_trajectory_bridge_reason = retired_source_warp_family
source_trajectory_bridge_tokens_corrected = 0
learned_upscaler_input_modified = false
```

The original source-grid clean sequence reaches the learned 3D upscaler unchanged apart from the private resized authoritative-prefix context required by Mixed-Grid.

## Root cause: unequal Mixed-Grid attention measure

The surviving mismatch existed before the learned upscaler, inside the Mixed-Grid low-stage transformer sequence.

For the matched validation geometry, one protected target-grid prefix frame contains:

```text
28 x 38 = 1064 video rows
```

while one genuine source-grid suffix frame contains:

```text
20 x 27 = 540 video rows
```

The per-frame row-density ratio is therefore:

```text
1064 / 540 ~= 1.97037
```

Mixed-Grid already gives the two regions their correct native MiniMax-H3 spatial RoPE coordinates and preserves temporal-coordinate continuity. Ordinary attention, however, treats every K/V row as one equal discrete sample in the softmax sum. A protected-prefix frame consequently contributes almost twice as many K/V samples as a source-grid suffix frame.

That is an attention-integration-measure mismatch, not a RoPE-coordinate mismatch.

## Attention-measure contract

With `suffix_geometric_bridge=true`, the Mixed-Grid low/probe wrapper publishes:

```text
key  = h3_flow_mixed_grid_attention_measure_v1
api  = 1
mode = prefix_kv_stratified_subsample
```

This contract does **not** modify VDN external-sequence API 2. The existing VDN contract remains:

```text
key      = vdn_h3_external_sequence_v1
api      = 2
mode     = dense_gate_no_linear
topology = mixed_grid_low_suffix
```

The Flow contract describes the native source/target grids, temporal partition, packed-video start row and expected native-density K/V row count. Flow itself does not rewrite attention tensors. A compatible attention backend must explicitly validate and consume the contract.

The companion implementation is in ComfyUI-Sol-H3. It performs the correction after explicit full-domain Q/K/V preprocessing such as Untwisting RoPE and immediately before rectangular Sol-Attn execution.

### Query domain is unchanged

Every query row remains present. In the representative production geometry:

```text
Q: 56029 -> 56029
```

Protected-prefix queries are therefore not discarded or resampled.

### Only the denser protected-prefix K/V region is normalized

All packed rows before target video are preserved. Every source-grid suffix K/V row is preserved byte-for-byte and in order.

For each protected prefix frame, the compatible backend maps each source-grid spatial coordinate to the nearest target-prefix coordinate under MiniMax-H3's native area-normalized `_frame_grid` construction and keeps that representative K/V row. The mapping is deterministic and one-to-one.

For the representative geometry:

```text
K/V: 56029 -> 49741
removed protected-prefix K/V rows: 6288
```

`49741` is exactly the native low-carrier packed row count:

```text
video_start + temporal * source_rows_per_frame
```

This converts the protected-prefix K/V integration measure to the same per-frame spatial sampling density as the genuine low-grid suffix without changing Q ownership, suffix K/V content or temporal topology.

### Fail-closed behavior

A compatible consumer rejects the measure contract if it disagrees with the already validated Mixed-Grid API-2 stream, including row counts, temporal partition, source/target grids or expected native K/V rows. A malformed measure request is not silently interpreted as the previous square-attention path.

If no compatible consumer is present, the Flow metadata alone does not perform K/V normalization. The target representation stage still executes locally, but the attention-measure portion requires a compatible backend.

## VDN ownership

VDN API 2 is unchanged.

During an external Mixed-Grid sequence, VDN disables its geometry-dependent local-window/linear complement and evaluates the released learned dense softmax branch through Comfy attention. With Sol-H3 installed as the attention provider, the measure contract changes only that softmax branch's K/V integration domain. VDN still applies its released learned gate to the resulting query outputs afterward.

The repair therefore does not retrain, replace or reinterpret VDN's learned gate, and does not alter VDN's normal high-grid behavior after the Mixed-Grid external contract is removed.

## Independent target exact-overlap reconciliation

The target representation stage remains independent of the attention-measure repair.

After learned 3D transfer, let:

```text
L_p = learned-upscaler last prefix token
E_p = authoritative exact last prefix token
L_s = learned first suffix token
D   = E_p - L_p
```

The representation bridge transfers only the zero-spatial-mean component of `D` to `L_s`. `suffix_dc_bridge` independently owns the spatial-mean/DC component. The authoritative prefix is never edited, and target suffix token 1+ remains unchanged at bridge application.

This stage demonstrated near-exact algebraic closure in matched metrics. It is retained because it repairs a real representation splice, while the attention-measure correction addresses the separate whole-frame framing defect.

## Validated decoded-media result

The matched v0.3.3 production validation removed the earlier whole-frame shrink/zoom-out/top-edge reveal. The formerly problematic join remained approximately unit-scale both in the initial validation policy and after restoring the ordinary Spectrum quality schedule. The validation did not introduce a delayed framing pulse, NFE change, or exact-prefix violation.

The expected representative accounting remained:

```text
Q:   56029 -> 56029
K/V: 56029 -> 49741
```

and the ordinary Spectrum schedule closed with:

```text
18 logical calls
13 actual transformer NFE
5 Spectrum forecasts

low:   7 actual / 3 forecast
high:  4 actual / 2 forecast
probe: 2 actual
```

That decoded-media result is the basis for enabling `suffix_geometric_bridge` by default. It remains geometry/backend-specific empirical evidence rather than proof that arbitrary configurations cannot regress.

## Runtime order

With the default repair enabled, the relevant order is:

```text
genuine source-grid low-stage continuation
-> mixed target-prefix/source-suffix H3 sequence
-> publish API-2 external sequence + attention-measure metadata
-> compatible backend keeps all Q and normalizes protected-prefix K/V density
-> exact handoff probe through the same mixed topology
-> recover clean source-grid x0
-> restore resized authoritative prefix only as private upscaler context
-> source trajectory stage reports retired/no-op
-> learned 3D target-grid upscaler
-> exact-overlap target representation reconciliation
-> one-token DC bridge
-> discard learned prefix output
-> restore authoritative target-grid prefix exactly
-> fresh target-grid refinement
-> exact-mask final return canonicalization
```

## Preserved contracts

The repair:

- never modifies or warps the authoritative target-grid protected prefix;
- preserves every Mixed-Grid Q row;
- preserves non-video K/V rows and all generated source-grid suffix K/V rows;
- adds no H3 transformer evaluation/NFE;
- adds no VAE/model/optical-flow call;
- performs no image-space or latent-space crossfade;
- performs no decode-space repair;
- does not change audio, caller noise, masks or conditioning;
- does not change VDN API 2;
- does not change Spectrum history, forecast or NFE accounting;
- leaves `suffix_dc_bridge` arithmetic unchanged.

## Validation scope

Structural tests cover native/source and cross-repository contracts including representative row accounting, query preservation, exact suffix K/V preservation, deterministic protected-prefix representative selection, native `_frame_grid` coordinate agreement, rectangular Sol-Attn routing, malformed-contract rejection, Untwist ordering, VDN/Spectrum/DiffAid composition, retired source-warp non-mutation, and final exact-prefix/model-call accounting.

Those tests establish execution semantics. Decoded media remains required when changing geometry, attention backends, handoff policy, or repair semantics.
