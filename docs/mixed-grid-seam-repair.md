# Experimental Mixed-Grid seam repair

## Current status

The remaining Continuum defect is a small whole-frame framing discontinuity at the exact-prefix join: a shrink/zoom-out/top-edge reveal that survived the released one-token DC bridge.

This document records the current experimental repair architecture. The earlier source-space affine/trajectory warp family is **retired**. It is no longer applied to the learned-upscaler input. The active experiment now has two independent pieces:

1. an **upstream Mixed-Grid attention-measure contract** during the low/probe H3 transformer stages; and
2. the already validated **target exact-overlap representation reconciliation** after learned 3D transfer.

The legacy workflow input name `suffix_geometric_bridge` is retained for compatibility. It remains **off by default** and is exposed only by the Mixed-Grid node.

No decoded-media success is claimed yet. PR #24 remains draft until a matched real-GPU/media run passes the final gate.

## Evidence chain

### Target-grid affine was insufficient

`metrics_00276` authorized a target-grid affine correction and reduced the measured signed `sy` residual from roughly `-0.01025` to `+0.00236`. The decoded framing jump remained.

### Exact-overlap representation repair was real but insufficient

`metrics_00281` repaired the learned-prefix replacement mismatch essentially to numerical noise before target-grid refinement:

```text
centered error: 0.400812 -> ~4.3e-08
corrected/native raw seam ratio:          ~1.0
corrected/native low-pass seam ratio:     ~1.0
corrected/native spatial-mean seam ratio: ~1.0
```

The decoded framing defect still remained. The exact-prefix replacement splice is therefore a measurable seam amplifier, but not a sufficient explanation for the whole-frame framing discontinuity.

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

The final one-token translation attempt still improved its boundary residual but created a much larger corrected-to-untouched transition immediately afterward. Shortening the finite warp only moved that compensating discontinuity earlier. Removing it would require a handcrafted fade/crossfade or unmeasured extrapolation, both explicitly outside this experiment's contracts.

The runtime therefore no longer attempts source warping. When the legacy experimental option is requested, `mixed_grid_source_trajectory_bridge` reports a no-op with:

```text
source_trajectory_bridge_reason = retired_source_warp_family
source_trajectory_bridge_tokens_corrected = 0
learned_upscaler_input_modified = false
```

The original source-grid clean sequence reaches the learned 3D upscaler unchanged apart from the already required private resized authoritative-prefix context.

## Upstream attention-measure hypothesis

The surviving mismatch exists before the learned upscaler, inside the Mixed-Grid low-stage transformer sequence.

For the matched `00318` geometry, one protected target-grid prefix frame contains:

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

Mixed-Grid already gives the two regions their correct native MiniMax-H3 spatial RoPE coordinates and preserves temporal-coordinate continuity. Ordinary attention, however, still treats every K/V row as one equal discrete sample in the softmax sum. A protected-prefix frame consequently contributes almost twice as many K/V samples as a source-grid suffix frame.

That is an attention-integration-measure mismatch, not a RoPE-coordinate mismatch.

## Attention-measure contract

When `suffix_geometric_bridge=true`, the Mixed-Grid low/probe wrapper now publishes an independent contract:

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

The Flow contract describes the native source/target grids, temporal partition, packed-video start row and expected native-density K/V row count. Flow itself does not arbitrarily rewrite attention tensors. A compatible attention backend must explicitly validate and consume the contract.

The companion implementation is in ComfyUI-Sol-H3. It performs the correction after any explicit full-domain Q/K/V preprocessing such as Untwisting RoPE and immediately before rectangular Sol-Attn execution.

### Query domain is unchanged

Every query row remains present. In the production `00318` geometry:

```text
Q: 56029 -> 56029
```

Protected-prefix queries are therefore not discarded or resampled.

### Only the denser protected-prefix K/V region is normalized

All packed rows before target video are preserved. Every source-grid suffix K/V row is preserved byte-for-byte and in order.

For each protected prefix frame, the compatible backend maps each source-grid spatial coordinate to the nearest target-prefix coordinate under MiniMax-H3's native area-normalized `_frame_grid` construction and keeps that representative K/V row. The mapping is deterministic and one-to-one.

For `00318`:

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

If no compatible consumer is present, the Flow metadata alone does not perform the K/V normalization. The independent target representation stage can still execute, but the upstream attention-measure experiment has not been exercised.

## VDN ownership

VDN API 2 remains unchanged.

During an external Mixed-Grid sequence, VDN already disables its geometry-dependent local-window/linear complement and evaluates the released learned dense softmax branch through Comfy attention. With Sol-H3 installed as the attention provider, the measure contract changes only that softmax branch's K/V integration domain. VDN still applies its released learned gate to the resulting query outputs afterward.

The experiment therefore does not retrain, replace or reinterpret VDN's learned gate, and does not alter VDN's normal high-grid behavior after the Mixed-Grid external contract is removed.

## Independent target exact-overlap reconciliation

The target representation stage remains independent of the attention-measure experiment.

After learned 3D transfer, let:

```text
L_p = learned-upscaler last prefix token
E_p = authoritative exact last prefix token
L_s = learned first suffix token
D   = E_p - L_p
```

The representation bridge transfers only the zero-spatial-mean component of `D` to `L_s`. The released `suffix_dc_bridge` independently owns the spatial-mean/DC component. The authoritative prefix is never edited, and target suffix token 1+ remains unchanged at bridge application.

This stage has already demonstrated near-exact algebraic closure in matched metrics, but it did not by itself remove the decoded framing defect. It remains because it is a real exact-prefix splice correction, not because it is treated as proof of perceptual success.

## Runtime order

With the experimental option enabled, the relevant order is now:

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
-> released one-token DC bridge
-> discard learned prefix output
-> restore authoritative target-grid prefix exactly
-> fresh target-grid refinement
-> exact-mask final return canonicalization
```

## Preserved contracts

The experiment:

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
- leaves released `suffix_dc_bridge` arithmetic unchanged;
- remains Mixed-Grid-only and off by default.

## Code-side validation

The implementation is covered by native/source and cross-repository tests that verify:

- `00318` row accounting: `56029` Q rows, `49741` K/V rows;
- all Q rows preserved;
- non-video K/V and every suffix K/V row preserved exactly;
- deterministic one-to-one protected-prefix representative selection;
- representative coordinates agree row-for-row with pinned native ComfyUI MiniMax-H3 `_frame_grid` geometry;
- rectangular Q/KV reaches the Sol-Attn path;
- malformed contracts fail closed;
- Untwist preprocessing occurs on the original full mixed domain before K/V selection;
- existing VDN/Spectrum/DiffAid/Flow composition tests remain green;
- the retired source warp cannot modify learned-upscaler input;
- final exact-prefix and model-call accounting contracts remain intact.

These tests establish execution semantics, not perceptual success.

## Decoded-media gate

The next matched run should leave the workflow otherwise unchanged and use:

```text
suffix_dc_bridge        = true
suffix_geometric_bridge = true
```

A compatible ComfyUI-Sol-H3 build is required for the upstream measure experiment.

The expected runtime evidence is:

```text
Flow:
  attention_measure_requested = true
  source_trajectory_bridge_reason = retired_source_warp_family
  source_trajectory_bridge_tokens_corrected = 0
  learned_upscaler_input_modified = false

Sol-H3:
  external_mixed_measure_calls > 0
  external_mixed_measure_q_rows unchanged at the full mixed Q domain
  external_mixed_measure_kv_rows_before > external_mixed_measure_kv_rows_after
```

For a `00318`-equivalent mixed call, the expected per-call geometry is `56029 -> 49741` K/V rows while Q stays `56029`.

The release gate is the decoded boundary itself: the shrink/top-edge reveal must be gone or materially reduced without a new pulse, delayed wobble, detail loss, motion regression, NFE change or exact-prefix violation. Until then, the option remains off by default and PR #24 remains draft.
