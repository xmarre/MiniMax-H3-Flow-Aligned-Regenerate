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
8. Reconstruct the clean source-grid probe sequence and restore the resized authoritative prefix only as private learned-upscaler context. If the experimental legacy-named `suffix_geometric_bridge` is enabled, measure recent source motion and correct only strongly evidenced, temporally observed source-suffix drift before the learned transfer.
9. Feed that clean source-grid sequence to the learned 3D upscaler.
10. Independently apply the optional exact-overlap target representation reconciliation, then the existing suffix DC bridge, discard the upscaler's prefix output, and restore the authoritative target-grid prefix.
11. Rebuild the high-stage conditional state/noise and start a fresh full-grid sampler lifetime. Require its first call to be an actual H3 evaluation. At the final Comfy sampler-return boundary, restore only exact `mask == 0` packed elements from the authoritative target input, then verify the returned protected prefix bitwise and measure the final seam.

The authoritative target-grid prefix is never spatially resized for H3 transformer conditioning.

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

This preserves the learned upscaler's native boundary relation while keeping the exact prefix authoritative.

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

## Real-media result

The matched production test on RTX Pro 6000 removed the brief exact-prefix boundary flash. Multi-boundary continuation was also clean.

The validated metrics were approximately:

- uncorrected exact-boundary DC RMS: `0.207763`;
- corrected exact-boundary DC RMS: `0.093093`;
- native learned-upscaler boundary DC RMS: `0.093093`;
- corrected suffix tokens: `1`;
- weight: `1.0`;
- `final_prefix_exact=true`.

This is the basis for making the bridge default on for Mixed-Grid in v0.3.0.

## VDN-H3 contract

With VDN enabled, Mixed-Grid requires external-sequence capability API 2 using:

```text
key      = vdn_h3_external_sequence_v1
api      = 2
mode     = dense_gate_no_linear
topology = mixed_grid_low_suffix
```

VDN validates both target-prefix and source-suffix row identities, full sequence length, temporal partition and explicit RoPE.

During the mixed sequence, VDN retains the learned dense softmax gate and disables only geometry-dependent local-window/linear-complement processing. The fresh target-grid high stage receives no external contract and resumes ordinary VDN behavior.

The coordinated release is `xmarre/ComfyUI-VDN-H3` v1.5.0.

## Continuum Decode Context

**MiniMax H3 Continuum Decode Context** addresses a separate native temporal-VAE issue: independently decoding chunks can miss real right context that a continuous decode would have seen.

Place it immediately before the normal Video VAE Decode when using the native H3 temporal decoder. It changes only the decode-only tensor; accepted sampling latents, continuation state, masks, audio and the assembly plan remain unchanged.

The suffix DC bridge, not Decode Context, is what fixed the mixed-grid latent tone flash in the validated workflow.

## Diagnostics

Mixed-Grid records four seam states:

- **A** — native learned-upscaler boundary `[U_prefix | U_suffix]`;
- **B** — authoritative prefix restored without the bridge `[P_exact | U_suffix]`;
- **C** — authoritative prefix plus corrected first suffix token;
- **D** — final boundary after target-grid refinement and exact-mask return canonicalization.

Diagnostics include raw RMS, spatial low-pass RMS, per-channel spatial-mean/DC RMS, bridge magnitude/count/weight and final/B/final/C ratios. Exact-mask return telemetry separately reports whether final protected values needed canonicalization and the magnitude of any pre-restore solver drift.

## Current status

The mixed-grid path is still labeled Experimental because it is an independent research topology, but its previously open production acceptance gate is closed for the tested stack: real GPU/media validation, multiple Continuum boundaries, VDN API 2, Spectrum + SA-PECE, DiffAid, Untwisting RoPE, learned 3D transfer, exact probe, fresh target-grid refinement and the suffix DC bridge have all been exercised together successfully.

The v0.3.1 exact-mask return fix is structurally regression-tested against `res_multistep` endpoint roundoff. A real workflow rerun remains the empirical check for that sampler/configuration; the prior real-media validation above used the v0.3.0 stack before this patch.

Quality/speed remain workflow dependent; the documented result is evidence for this implementation and tested stack, not a universal model guarantee.

## Experimental source-trajectory + exact-overlap repair

The optional `suffix_geometric_bridge` input name is retained for workflow compatibility and remains **off by default**. It now enables two independent Mixed-Grid experimental stages.

### Why correction moved upstream

The `metrics_00276` affine experiment reduced the independently authorized target `sy` residual to the estimator floor but did not remove the decoded whole-frame shrink/top-edge reveal. The next matched B, `metrics_00281`, then made the exact-prefix replacement splice essentially identical to the learned upscaler's native boundary before high refinement: the centered representation mismatch fell from about `0.400812` to `4.3e-08`, with corrected/native raw, low-pass and spatial-mean seam ratios all approximately `1.0`. The decoded framing defect still remained.

That falsifies the exact-prefix splice as a sufficient root cause. The same vertical-scale signature is measurable earlier in the genuine source-grid continuation, before learned 3D transfer, so the current experiment acts there first. `metrics_00294` additionally held all reference inputs at the same 1.1 MP budget and still showed the preview framing shift. With the current source-only scalar threshold already removed, that run measured provisional `sx` and `sy` source residuals whose directly observed cumulative states continued away from the natural-motion prediction; the remaining no-op was therefore a temporal-classification limitation rather than a reference-resolution or evidence-threshold result.

### Stage 1: measured source trajectory

Immediately before learned upscaling, Flow estimates recent natural prefix motion from up to six genuine source-grid transitions and measures the first generated transition against that prediction. Candidate axes retain the established estimator floors and safety bounds (`0.005` log scale, `0.25` latent px translation, `log(1.03)` scale safety, and `min(1.5 px, 2.5%)` translation safety) plus objective-gain gating.

Because no independent target-domain observation exists before the upscaler, the earlier two-domain quadrature threshold is not reused as a source-only scalar gate. A source axis must instead clear the existing estimator floor, objective-gain floor, and safety bound; then directly observed suffix evolution must authorize a safe recovering, persistent, or measured same-direction drift state; finally the applied warp must strictly reduce the measured residuals. `axis_evidence_score` is retained for diagnostics only. Existing cross-grid thresholds elsewhere are unchanged.

Up to four genuine suffix transitions are measured for temporal classification. Recovering weights are derived from the measured cumulative state. Persistent correction is limited to token 0 plus the directly observed follow-up transitions. A measured-drift state uses each directly observed cumulative signed residual as that suffix token's correction rather than extrapolating or scaling the boundary correction. No mode extends correction over unmeasured later suffix tokens, and there is no handcrafted fade.

After correction the source boundary is registered again. Every authorized axis must reduce in residual magnitude. Measured-drift axes additionally re-register every corrected directly observed follow-up transition and require residual reduction there as well. A sign crossing is allowed only within the estimator floor. Otherwise the exact original source tensor is returned. The protected source prefix and all suffix tokens outside the measured active window remain byte-identical.

### Stage 2: independent exact-overlap reconciliation

After learned 3D transfer, the discarded learned prefix still provides exact same-frame overlap calibration against the authoritative target prefix. The representation stage transfers only the zero-spatial-mean part of that residual to the first target suffix token; `suffix_dc_bridge` independently owns the spatial-mean component. This second stage runs whenever the experimental option is requested, regardless of whether the source stage accepted a correction.

The target stage still never modifies the authoritative prefix or target suffix token 1+, and the complete experimental path adds no H3 NFE, VAE pass, crossfade, decode-space fix, audio/noise/mask/conditioning change, VDN API change, or Spectrum-history change.

`mixed_grid_source_trajectory_bridge` records source evidence, temporal classification, applied source transforms, corrected-token count, post-warp residual reduction, and whether learned-upscaler input was modified. `mixed_grid_representation_bridge` independently records target exact-overlap reconciliation. Decoded media remains the release gate; the experimental option stays off by default.
