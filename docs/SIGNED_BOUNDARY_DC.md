# Signed boundary DC receipts

Diagnostic only. These receipts attribute a brightness or tone step at a
continuation boundary to the stage that introduces it. They are always on in
partitioned exact-prefix sampling. They add no H3 evaluation, provider call or
VAE call, and generated output is unchanged.

## What is measured

Four existing video latents are measured at the boundary between the last
protected token `p-1` and the first generated token `p`:

| Stage | Tensor | Grid | Domain |
| --- | --- | --- | --- |
| `source_low_exact_context` | low/probe clean prediction with the protected source prefix written in, as handed to the learned transfer | source | model internal |
| `provider_native` | learned transfer output, including its own rendering of the prefix | target | model internal |
| `pre_high_exact_restored` | transfer output with the authoritative exact prefix restored, the clean operand of target-high | target | model internal |
| `final` | returned latent | target | caller output |

MiniMax-H3 video uses latent scale factor 1 with no shift, so the model-internal
and caller video domains are identical.

For each stage, the `partitioned_signed_boundary_dc` event records:

- per-channel signed spatial means of tokens `p-3 .. p+3` (batch item 0);
- the signed boundary step `mean_hw(token p) - mean_hw(token p-1)` per channel, with its norm and RMS;
- the RMS of each adjacent token pair's step, as a baseline for ordinary motion;
- a linear preview luma per token and its boundary step, computed through the
  latent format's RGB preview factors. This is an approximation of decoded
  brightness, not a decode.

Spatial means are comparable across the source and target grids. The boundary
step RMS of the three target-grid stages equals the existing
`upscaler_native_seam_spatial_mean_rms`, `corrected_exact_seam_spatial_mean_rms`
and `final_seam_spatial_mean_rms` fields; these receipts add the sign, the
channels and the source-grid stage.

## Attribution

The `partitioned_signed_boundary_dc_ledger` event splits the final step exactly:

```
s_final = s_source
        + (s_provider - s_source)      provider_transfer
        + (s_restored - s_provider)    exact_prefix_restore
        + (s_final - s_restored)       target_high
```

For each component, `signed_contribution_to_final_norm` is its projection onto
the direction of the final step. The four contributions sum to the final step's
norm. A positive contribution builds the final step, and a negative one opposes
it. `preview_luma_step` gives the same split in approximate luma.

## Cost

Each receipt is one spatial-mean reduction over seven tokens of an existing
tensor and one transfer of a `channels × 7` table to the CPU. No tensor
reference is retained.

## Limits

The receipts measure latent-domain DC only. They do not establish decoded tone,
geometry or motion, and they do not identify why a stage introduces a step, only
which stage does.
