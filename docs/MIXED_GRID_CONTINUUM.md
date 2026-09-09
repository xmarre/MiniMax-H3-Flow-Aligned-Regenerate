# Mixed-Grid Continuum continuation

**MiniMax H3 Progressive Mixed-Grid Continuum [Experimental]** is the recommended accelerated exact-prefix Continuum path in v0.3.x.

It requires `handoff_transfer=learned_3d` and the companion `H3_LATENT_UPSCALER` provider. Continuum stays configured for the final target geometry.

## Why Mixed-Grid exists

The conservative Target Input node cannot safely resize an exact protected prefix, so exact-prefix continuation falls back to one target-grid sampler lifetime. The earlier Target-Sparse experiment avoids resizing the prefix but also avoids the learned latent upscale; real decoded-media testing showed cascading quality defects on that path.

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
10. Independently apply the optional exact-overlap target representation reconciliation, then the existing suffix DC bridge, discard the upscaler's prefix output, and restore the authoritative target-grid prefix.
11. Rebuild the high-stage conditional state/noise and start a fresh full-grid sampler lifetime. Require its first call to be an actual H3 evaluation. At the final Comfy sampler-return boundary, restore only exact `mask == 0` packed elements from the authoritative target input, then verify the returned protected prefix bitwise and measure the final seam.

The authoritative target-grid prefix is never spatially resized for H3 transformer conditioning.

When the optional legacy-named `suffix_geometric_bridge` is enabled, the low/probe mixed stages additionally publish the experimental attention-measure contract described below. The retired source-warp implementation does **not** modify the clean source sequence before step 9.

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

Matched production testing on RTX Pro 6000 removed the earlier brief exact-prefix tone/flash boundary. Multi-boundary continuation was also clean.

The validated metrics were approximately:

- uncorrected exact-boundary DC RMS: `0.207763`;
- corrected exact-boundary DC RMS: `0.093093`;
- native learned-upscaler boundary DC RMS: `0.093093`;
- corrected suffix tokens: `1`;
- weight: `1.0`;
- `final_prefix_exact=true`.

This DC result is separate from the smaller whole-frame framing discontinuity currently under investigation.

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

The experimental attention-measure contract does **not** change this API-2 contract or VDN ownership.

## Experimental attention-measure normalization

`suffix_geometric_bridge` remains **off by default**. The input name is retained for workflow compatibility, but the source-space affine/trajectory warp family has been retired after matched `00318` testing exhausted every directly authorized finite correction horizon without producing a verified source modification.

The active upstream experiment addresses a different structural mismatch inside low/probe Mixed-Grid attention.

### Unequal spatial K/V density

For the matched `00318` geometry:

```text
protected target-grid prefix: 28 x 38 = 1064 video rows/frame
genuine source-grid suffix:   20 x 27 =  540 video rows/frame
ratio:                                  ~1.97037x
```

Mixed-Grid already uses native target-grid RoPE positions for the prefix and native source-grid positions for the suffix. However, ordinary softmax still gives each K/V row one equal discrete contribution. A protected-prefix frame therefore contributes almost twice as many K/V samples as a source-grid suffix frame.

When the experimental option is enabled, Flow publishes:

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

For `00318`-equivalent geometry:

```text
Q:   56029 -> 56029
K/V: 56029 -> 49741
```

`49741` is exactly the native low-carrier packed sequence row count. The operation changes K/V integration density, not query ownership or suffix content.

If no compatible backend consumes the contract, the metadata alone does not normalize K/V.

## Retired source-trajectory warp

The source trajectory experiment is now a diagnostic no-op. When the legacy experimental option is requested, Flow reports:

```text
source_trajectory_bridge_reason = retired_source_warp_family
source_trajectory_bridge_tokens_corrected = 0
learned_upscaler_input_modified = false
```

The clean source-grid continuation reaches the learned upscaler unchanged except for the private authoritative-prefix context that Mixed-Grid already requires.

See [mixed-grid-seam-repair.md](mixed-grid-seam-repair.md) for the evidence chain and the reason finite source warps were abandoned rather than weakened with an invented fade or extrapolation.

## Independent exact-overlap target reconciliation

The experimental option also enables the target exact-overlap representation bridge after learned 3D transfer.

Let the discarded learned prefix boundary be `L_p`, the authoritative exact boundary be `E_p`, and the first learned suffix token be `L_s`. Flow transfers only the zero-spatial-mean component of `E_p - L_p` onto `L_s`; the released `suffix_dc_bridge` independently owns the spatial-mean component.

Matched metrics showed the centered splice error collapse from about `0.400812` to `4.3e-08`. The decoded whole-frame framing jump nevertheless remained, so this algebraic closure is kept as a real splice correction but is not treated as a perceptual fix by itself.

The target bridge never edits the authoritative prefix or target suffix token 1+.

## Continuum Decode Context

**MiniMax H3 Continuum Decode Context** addresses a separate native temporal-VAE issue: independently decoding chunks can miss real right context that a continuous decode would have seen.

Place it immediately before the normal Video VAE Decode when using the native H3 temporal decoder. It changes only the decode-only tensor; accepted sampling latents, continuation state, masks, audio and the assembly plan remain unchanged.

The suffix DC bridge, not Decode Context, fixed the earlier latent tone flash. Decode Context is also not the current framing-measure experiment.

## Diagnostics

Mixed-Grid records the established target seam states:

- **A** — native learned-upscaler boundary `[U_prefix | U_suffix]`;
- **B** — authoritative prefix restored without bridge `[P_exact | U_suffix]`;
- **C** — authoritative prefix plus corrected first suffix token;
- **D** — final boundary after target-grid refinement and exact-mask return canonicalization.

The experimental path additionally records:

- Flow plan metadata showing whether attention measure was requested;
- retired source-trajectory no-op telemetry;
- target exact-overlap representation metrics;
- compatible Sol-H3 external mixed-measure counters, including full Q rows and K/V rows before/after normalization.

For a `00318`-equivalent mixed call, the expected Sol-side K/V accounting is `56029 -> 49741` while Q remains `56029`.

## Preserved contracts

The current experiment:

- never modifies or warps the authoritative target-grid protected prefix;
- adds no H3 transformer NFE;
- adds no VAE/model/optical-flow call;
- performs no image-space/latent-space crossfade or decode-space repair;
- does not change audio, caller noise, masks or conditioning;
- does not change VDN API 2 or its learned gate ownership;
- does not change Spectrum history, forecasts or NFE accounting;
- leaves released `suffix_dc_bridge` arithmetic unchanged;
- remains Mixed-Grid-only and off by default.

## Current validation gate

Code-side validation covers native MiniMax-H3 coordinates, production row accounting, exact preservation domains, rectangular Sol-Attn routing, malformed-contract rejection and the existing VDN/Spectrum/DiffAid/Untwist/Flow composition tests.

Decoded media is still the release gate for the framing repair. The next matched run must keep the workflow otherwise unchanged and use:

```text
suffix_dc_bridge        = true
suffix_geometric_bridge = true
```

with the compatible ComfyUI-Sol-H3 companion revision.

Success requires the whole-frame shrink/top-edge reveal to disappear or materially reduce without a new pulse, delayed wobble, motion/detail regression, NFE change or exact-prefix violation. Until that result exists, `suffix_geometric_bridge` remains off by default and the work remains experimental.
