# Mixed-Grid Continuum continuation

**MiniMax H3 Progressive Mixed-Grid Continuum [Experimental]** is the recommended accelerated exact-prefix Continuum path in v0.3.0.

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
8. Feed the complete clean low-grid sequence plus resized prefix context to the learned 3D upscaler.
9. Discard the upscaler's prefix output, restore the authoritative target-grid prefix, rebuild the high-stage conditional state/noise, and start a fresh full-grid sampler lifetime.
10. Require the first high-stage call to be an actual H3 evaluation and verify the returned protected prefix remains exact.

The authoritative target-grid prefix is never spatially resized for H3 transformer conditioning.

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
- **D** — final boundary after target-grid refinement.

Diagnostics include raw RMS, spatial low-pass RMS, per-channel spatial-mean/DC RMS, bridge magnitude/count/weight and final/B/final/C ratios.

## Current status

The mixed-grid path is still labeled Experimental because it is an independent research topology, but its previously open production acceptance gate is closed for the tested stack: real GPU/media validation, multiple Continuum boundaries, VDN API 2, Spectrum + SA-PECE, DiffAid, Untwisting RoPE, learned 3D transfer, exact probe, fresh target-grid refinement and the suffix DC bridge have all been exercised together successfully.

Quality/speed remain workflow dependent; the documented result is evidence for this implementation and tested stack, not a universal model guarantee.
