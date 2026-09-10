# Mixed-Grid Continuum continuation

**MiniMax H3 Progressive Mixed-Grid Continuum** is the recommended accelerated exact-prefix Continuum path.

It requires `handoff_transfer=learned_3d` and a connected companion `H3_LATENT_UPSCALER` provider. Continuum stays configured for the final target geometry.

## Canonical defaults

```text
source_mode             = scale
source_scale            = 0.70
source_width            = 864
source_height           = 640
handoff_coordinate      = 0.35
handoff_selection       = fixed
guidance_mode           = direction+temporal
direction_weight        = 0.25
acceleration_weight     = 0.25
consistency_weight      = 0.25
low_frequency_cutoff    = 0.25
temporal_weight         = 0.20
handoff_transfer        = learned_3d
suffix_dc_bridge        = true
suffix_geometric_bridge = true
```

The intended provider configuration in the shipped workflow is:

```text
node                    = MinimaxH3LatentUpscaler3DProvider
model_name              = minimax_h3_latent_upscaler_3d_bf16.safetensors
device                  = cuda
precision               = bf16
offload_after_upscale   = false
```

The provider node is supplied by [`xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus`](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus).

With `guidance_mode=direction+temporal`, `acceleration_weight` and `consistency_weight` are stored but inactive. The runtime activates acceleration only in `direction+acceleration` and consistency only in `downsample_consistency`.

## Why Mixed-Grid exists

The conservative generic Target Input node cannot safely resize an exact protected prefix, so exact-prefix continuation falls back to one target-grid sampler lifetime. The earlier Target-Sparse experiment avoids resizing the prefix but also avoids the learned latent upscale; real decoded-media testing showed cascading quality defects on that path.

Mixed-Grid instead keeps the exact target-grid prefix authoritative while generating the new suffix on a genuine smaller sampler grid, then uses the learned 3D latent transfer before target-grid refinement.

## Execution contract

1. Snapshot the original model-domain target-grid prefix and corresponding target-grid sampler noise.
2. Run the private early sampler on a normal low-grid carrier.
3. Immediately before H3 transformer block 0, independently patchify/project the original target-grid protected prefix using native H3 masked-conditioning semantics and replace only the carrier prefix rows.
4. Keep genuine source-grid suffix patch embeddings. Prefix positions use the native target grid; suffix positions use the native source grid while preserving temporal slot continuity.
5. Run all transformer blocks on the mixed sequence with matching per-row modulation metadata.
6. Before native output projection, discard target-prefix transformer outputs and restore the normal low-carrier layout so native projection/unpatchification remains valid.
7. Run the exact handoff probe through the same mixed topology in a separate sampler lifetime.
8. Reconstruct the clean source-grid probe sequence and restore the resized authoritative prefix only as private learned-upscaler context.
9. Feed that clean source-grid sequence to the learned 3D upscaler.
10. Apply the target exact-overlap representation reconciliation and suffix DC bridge, discard the upscaler's prefix output, and restore the authoritative target-grid prefix.
11. Rebuild the high-stage conditional state/noise and start a fresh full-grid sampler lifetime. Require its first call to be an actual H3 evaluation. At the final Comfy sampler-return boundary, restore only exact `mask == 0` packed elements from the authoritative target input, then verify the returned protected prefix bitwise and measure the final seam.

The authoritative target-grid prefix is never spatially resized for H3 transformer conditioning.

When `suffix_geometric_bridge=true`, the low/probe mixed stages also publish the validated attention-measure contract described below. The retired source-warp implementation does **not** modify the clean source sequence before step 9.

### Final exact-mask return canonicalization

The final restoration in step 11 is intentionally narrower than a tolerance check. ComfyUI already restores protected values after each model evaluation, but some solvers can perform a terminal arithmetic update after the final evaluation. `res_multistep`, for example, reaches its zero-sigma endpoint through an Euler-form expression that is mathematically equal to the final denoised estimate but can differ by a few floating-point ULPs.

Flow therefore canonicalizes exactly protected `mask == 0` packed elements at the Comfy sampler-return boundary before the existing strict `torch.equal` contract and before final seam diagnostics. It does not use `allclose`, does not edit any generated/unmasked value, performs no extra H3 NFE, and avoids a clone when the returned protected values are already bitwise exact.

The runtime records an `exact_mask_output` event for final target-grid returns with the sampler/source identity, protected and changed element counts, pre-restore exactness, non-finite drift count, maximum absolute drift and RMS drift. `exact_mask_output_canonicalizations` counts returns that required restoration.

## Suffix DC bridge

`suffix_dc_bridge` defaults **on** for Mixed-Grid.

The learned 3D transfer produces a complete target-grid clean sequence before its learned prefix is discarded. Let that learned output be `[U_prefix | U_suffix]` and the authoritative protected prefix be `P_exact`.

Flow computes the per-batch/per-channel spatial-mean offset:

```text
delta = mean(P_exact[-1]) - mean(U_prefix[-1])
```

and adds `delta` to **only** `U_suffix[0]`.

The bridge:

- corrects exactly one generated suffix latent token;
- uses fixed weight `1.0`;
- never edits the authoritative prefix;
- leaves later suffix tokens unchanged at bridge application;
- performs no video-space crossfade;
- adds no H3 transformer NFE.

For the conditional flow state, the clean-space delta is mapped through the same affine construction:

```text
x_sigma = (1 - sigma) * x0 + sigma * noise
```

The deterministic noise is unchanged.

### Validated DC result

Matched decoded-media testing removed the earlier brief exact-prefix tone/flash boundary, including multi-boundary continuation.

The validated metrics were approximately:

- uncorrected exact-boundary DC RMS: `0.207763`;
- corrected exact-boundary DC RMS: `0.093093`;
- native learned-upscaler boundary DC RMS: `0.093093`;
- corrected suffix tokens: `1`;
- weight: `1.0`;
- `final_prefix_exact=true`.

## VDN-H3 contract

With VDN enabled, Mixed-Grid requires external-sequence capability API 2 using:

```text
key      = vdn_h3_external_sequence_v1
api      = 2
mode     = dense_gate_no_linear
topology = mixed_grid_low_suffix
```

VDN validates target-prefix/source-suffix row identities, full sequence length, temporal partition and explicit RoPE.

During the mixed sequence, VDN retains the released learned dense softmax gate and disables only geometry-dependent local-window/linear-complement processing. The fresh target-grid high stage receives no external contract and resumes ordinary VDN behavior.

The attention-measure contract does **not** change this API-2 contract or VDN ownership.

## Attention-measure framing repair

`suffix_geometric_bridge` is the legacy workflow input name for two independent operations and now defaults **on** for Mixed-Grid:

1. publish the protected-prefix K/V spatial-measure contract during low/probe Mixed-Grid attention;
2. apply the target exact-overlap representation reconciliation after learned 3D transfer.

The old source-space affine/trajectory warp family is retired and remains a diagnostic no-op.

### Unequal spatial K/V density

For the matched validation geometry:

```text
protected target-grid prefix: 28 x 38 = 1064 video rows/frame
genuine source-grid suffix:   20 x 27 =  540 video rows/frame
ratio:                                  ~1.97037x
```

Mixed-Grid already uses native target-grid RoPE positions for the prefix and native source-grid positions for the suffix. Ordinary softmax, however, treats each K/V row as one equal discrete sample, so a protected-prefix frame contributes almost twice as many K/V samples as a source-grid suffix frame.

That is an attention-integration-measure mismatch, not a RoPE-coordinate mismatch.

Flow publishes:

```text
key  = h3_flow_mixed_grid_attention_measure_v1
api  = 1
mode = prefix_kv_stratified_subsample
```

This is independent of the VDN API-2 sequence contract.

### Compatible backend behavior

Flow does not itself shorten the transformer sequence. A compatible attention backend must validate and consume the measure metadata.

The companion ComfyUI-Sol-H3 implementation:

- preserves every Q row;
- preserves all non-video K/V rows;
- preserves every genuine source-grid suffix K/V row exactly and in order;
- for each protected prefix frame, retains one K/V representative nearest each source-grid spatial coordinate under native MiniMax-H3 area-normalized `_frame_grid` geometry;
- performs the selection only after explicit full-domain preprocessing such as Untwisting RoPE;
- executes the resulting rectangular Q/KV domain through Sol-Attn;
- fails closed if the measure metadata disagrees with the validated Flow API-2 sequence.

For the validated representative geometry:

```text
Q:   56029 -> 56029
K/V: 56029 -> 49741
```

`49741` is exactly the native low-carrier packed sequence row count. The operation changes K/V integration density, not query ownership or suffix content.

If no compatible backend consumes the contract, the metadata alone does not normalize K/V.

### Decoded-media result

The matched v0.3.3 production validation removed the previous whole-frame shrink/top-edge reveal. The problematic join remained approximately unit-scale both with the validation Spectrum policy and after restoring the ordinary quality schedule. No delayed framing pulse, NFE change, or exact-prefix violation was observed in those matched runs.

This result is why the framing-repair path is now enabled by default. It does not imply that every backend consumes the attention-measure contract; that remains a capability requirement of the selected attention backend.

## Retired source-trajectory warp

The source trajectory experiment is a diagnostic no-op. Flow reports:

```text
source_trajectory_bridge_reason = retired_source_warp_family
source_trajectory_bridge_tokens_corrected = 0
learned_upscaler_input_modified = false
```

Matched testing showed that finite source warps moved the discontinuity to the corrected-to-untouched transition rather than resolving it. The clean source-grid continuation therefore reaches the learned upscaler unchanged except for the private authoritative-prefix context that Mixed-Grid already requires.

See [mixed-grid-seam-repair.md](mixed-grid-seam-repair.md) for the evidence chain.

## Independent exact-overlap target reconciliation

After learned 3D transfer, let the discarded learned prefix boundary be `L_p`, the authoritative exact boundary be `E_p`, and the first learned suffix token be `L_s`. Flow transfers only the zero-spatial-mean component of `E_p - L_p` onto `L_s`; the released `suffix_dc_bridge` independently owns the spatial-mean component.

Matched metrics showed the centered splice error collapse from about `0.400812` to `4.3e-08`. This algebraic closure is a real splice correction and remains independent from the K/V attention-measure correction.

The target bridge never edits the authoritative prefix or target suffix token 1+.

## Continuum Decode Context

**MiniMax H3 Continuum Decode Context** addresses a separate native temporal-VAE issue: independently decoding chunks can miss real right context that a continuous decode would have seen.

Place it immediately before the normal Video VAE Decode when using the native H3 temporal decoder. It changes only the decode-only tensor; accepted sampling latents, continuation state, masks, audio and the assembly plan remain unchanged.

The suffix DC bridge fixes the latent tone flash. The attention-measure path addresses the separate framing discontinuity. Decode Context addresses neither of those sampler-space defects.

## Diagnostics

Mixed-Grid records the established target seam states:

- **A** — native learned-upscaler boundary `[U_prefix | U_suffix]`;
- **B** — authoritative prefix restored without bridge `[P_exact | U_suffix]`;
- **C** — authoritative prefix plus corrected first suffix token;
- **D** — final boundary after target-grid refinement and exact-mask return canonicalization.

The framing-repair path additionally records:

- Flow plan metadata showing whether attention measure was requested;
- retired source-trajectory no-op telemetry;
- target exact-overlap representation metrics;
- compatible Sol-H3 external mixed-measure counters, including full Q rows and K/V rows before/after normalization.

For the representative validated mixed call, the expected Sol-side K/V accounting is `56029 -> 49741` while Q remains `56029`.

## Preserved contracts

The default Mixed-Grid repair path:

- never modifies or warps the authoritative target-grid protected prefix;
- preserves every Mixed-Grid Q row;
- preserves non-video K/V rows and all generated source-grid suffix K/V rows;
- adds no H3 transformer NFE;
- adds no VAE/model/optical-flow call;
- performs no image-space/latent-space crossfade or decode-space repair;
- does not change audio, caller noise, masks or conditioning;
- does not change VDN API 2 or its learned gate ownership;
- does not change Spectrum history, forecasts or NFE accounting;
- leaves `suffix_dc_bridge` arithmetic unchanged.

These contracts are structurally covered by tests. Decoded-media validation remains necessary when changing geometry, backend combinations, or seam-repair semantics; passing structural tests alone does not establish output quality.
