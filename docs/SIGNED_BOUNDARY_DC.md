# Signed boundary DC receipts

Diagnostic only. These receipts attribute a brightness or tone step at a
continuation boundary to the stage that introduces it. They are always on in
partitioned exact-prefix sampling and need no extra node or setting: neither the
First-High Counterfactual node nor `capture_boundary_witness` is required. They
add no H3 evaluation, provider call or VAE call, and generated output is
unchanged.

## What is measured

Existing video latents are measured at the boundary between the last protected
token `p-1` and the first generated token `p`, in pipeline order:

| Stage | Tensor | Grid | Domain |
| --- | --- | --- | --- |
| `source_low_native` | low/probe clean prediction as the low stage returned it | source | model internal |
| `source_low_exact_context` | the same with the protected source prefix written in (and, when selected, the shadow clean video), as handed to the learned transfer | source | model internal |
| `provider_native` | raw learned-transfer output, including its own rendering of the prefix | target | model internal |
| `provider_boundary_pair_replaced` | only when the exact-overlap fallback replaces the provider's boundary pair: the provider output after that replacement | target | model internal |
| `pre_high_exact_restored` | the target-high clean operand: provider output after any configured correction, with the authoritative exact prefix restored | target | model internal |
| `final` | returned latent | target | caller output |

MiniMax-H3 video uses latent scale factor 1 with no shift, so the model-internal
and caller video domains are identical.

For each stage, the `partitioned_signed_boundary_dc` event records:

- per-channel signed spatial means of tokens `p-3 .. p+3` (batch item 0);
- the signed boundary step `mean_hw(token p) - mean_hw(token p-1)` per channel, with its norm and RMS;
- the RMS of each adjacent token pair's step, as a baseline for ordinary motion;
- a linear preview luma per token and its boundary step, computed through the
  latent format's RGB preview factors. This approximates decoded brightness; it
  is not a decode and cannot confirm a decoded luminance change.

The boundary-step RMS of `provider_native`, `pre_high_exact_restored` and
`final` equals the existing `upscaler_native_seam_spatial_mean_rms`,
`corrected_exact_seam_spatial_mean_rms` and `final_seam_spatial_mean_rms` fields
when the exact-overlap fallback is inactive. These receipts add the sign, the
channels and the source-grid stages.

With `native_source_carry_target_band`, the `target_band.signed_dc` field of the
`partitioned_source_prefix_projection` event records, for the last three prefix
tokens, the per-channel signed DC that the target-band reconciliation adds to the
carried prefix.

## Attribution

The `partitioned_signed_boundary_dc_ledger` event splits the final step exactly
into the change each recorded stage introduces relative to the previous one:

| Component | Change from | to |
| --- | --- | --- |
| `source_low` | nothing | `source_low_native` |
| `source_prefix_write` | `source_low_native` | `source_low_exact_context` |
| `source_prefix_write_and_shadow_clean_selection` (replaces the row above when the shadow AV handoff source is selected; the shadow clean video can also replace generated tokens) | `source_low_native` | `source_low_exact_context` |
| `provider_transfer` | `source_low_exact_context` | `provider_native` |
| `provider_boundary_pair_replacement` (optional) | `provider_native` | `provider_boundary_pair_replaced` |
| `pre_high_correction_and_exact_restore` | previous stage | `pre_high_exact_restored` |
| `target_high` | `pre_high_exact_restored` | `final` |

For each component, `signed_contribution_to_final_norm` is its projection onto
the direction of the final step. The contributions sum to the final step's
norm. A positive contribution builds the final step, and a negative one opposes
it. Each component also reports its own `norm` and `cosine_to_final`.
`component_norm_sum_over_final_norm` well above 1 means the components largely
cancel: large opposing changes, rather than one stage, then shape the final
step. Read the vectors and cosines in that case, not only the projections.
`preview_luma_step` gives the same split in approximate luma.

## Low-stage CFG branches

When a ledger places a step in `source_low`, the `partitioned_low_cfg_branch_dc`
event shows where in the low stage it appears. During the low and probe sampler
lifetimes, an observer on ComfyUI's `sampler_post_cfg_function` reduces, for
every evaluation, four tensors to per-channel spatial means of tokens
`p-1 .. p+1`:

- the sampler input;
- the conditional clean prediction;
- the unconditional clean prediction;
- the guided prediction.

Each is reported with its boundary step and preview luma, and with the sigma and
CFG scale. The predictions are the model's own, before the protected rows are
restored, so token `p-1` shows how the model reads the protected context.

- A step in the conditional prediction that the unconditional one lacks is selected by the prompt or conditioning and amplified by CFG.
- A step in both is not explained by guidance; the generated tokens do not follow the context's tone in either branch.
- The first evaluation, at the highest sigma, shows whether the step is decided before any generated content exists.

With CFG scale 1, the unconditional branch is absent. Guiders that do not call
ComfyUI's CFG function record `not_observed`. The guided prediction is returned
unchanged, and an observer failure is recorded as an error row.

## Cost and failure behaviour

Each receipt is one spatial-mean reduction over at most seven tokens of an
existing tensor and one transfer of a `channels × 7` table to the host, which
waits for that reduction. The existing unconditional trajectory and seam
diagnostics at the source, provider and final stages already wait on the device
at the same points. Each receipt records its own `elapsed_ms`. No tensor
reference is retained.

A receipt that cannot be computed is recorded with `status: "error"` and the
error text, and sampling continues. The ledger then reports `incomplete` with
the failed stages.

## Limits

The receipts measure latent-domain DC only. They do not establish decoded tone,
geometry or motion, and they show which stage introduces a step, not why.
