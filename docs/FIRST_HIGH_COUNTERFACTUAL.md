# First-high counterfactual capture

Diagnostic only. It tests which input or execution path determines the first
actual target-high prediction at a continuation join. Generated output is
unchanged.

## Setup

1. Add **MiniMax H3 First-High Counterfactual [Diagnostic]** to the MODEL chain
   that reaches the sampler, anywhere after the Flow continuation node.
2. Enable `capture_boundary_witness` on the continuation node. Large target grids
   also need the geometry-derived full-stage evidence budget.
3. Generate the continuation. The boundary bundle named by
   `partitioned_boundary_window_evidence.fields.bundle` then contains one native
   seven-token decoder window per arm. The
   `partitioned_first_high_counterfactual` metrics event holds the receipt.
4. Run **MiniMax H3 First-High Counterfactual Audit** with the production video
   VAE, that bundle and the assembled `chunk_join_frame`.

## Arms

Each arm repeats the first actual high prediction at the same sigma and with the
same conditioning, CFG, denoise-mask timestep labels and sample schedule, but
changes exactly one thing:

| Arm | Change |
| --- | --- |
| `replay` | Nothing. Establishes repeatability against the production prediction. |
| `provider_prefix` | Conditions on the learned provider's own prefix instead of the authoritative exact prefix. Applied to the injected prefix rows and to the partitioned stage plan. |
| `suffix_noise_reseed` | Replaces the generated-suffix state with the pre-high clean state re-noised with independent seeded noise. |
| `native_high_attention` | Removes Flow's partitioned stage runtime, so the prediction runs through the installed native attention chain, including Core block-sparse attention. |

Compare arms with `replay`, not only with the production prediction: arms skip
every predict-noise wrapper, while the production call ran inside them.

## Isolation

Arms call the guider's unwrapped CFG prediction. Predict-noise wrappers, Flow
guidance, boundary traces and Spectrum step accounting do not see them.
Spectrum's per-step diffusion-model keys are removed, so its diffusion-model
wrapper passes through. The partitioned runtime is replaced by a copy with
scratch metrics. CPU and CUDA RNG state is restored. A failed arm is recorded
with its error and does not stop generation. The production prediction is
verified unchanged.

Each arm costs one H3 evaluation per CFG branch at target resolution. Each saved
window adds 7 latent tokens of float32 to the bundle.

## Receipt and report

The receipt records sigma, the sample schedule and the PDD video head-bank count.
It also has a high-entry ledger: per-token suffix state RMS, clean-signal RMS,
and the mean, standard deviation and pre-high correlation of the implied noise.
For each arm it records the input change, the isolation applied and the per-token
latent difference from the production prediction.

The audit decodes `pre_high`, the production first prediction and every arm.
Each window is decoded standalone with the authoritative prefix tokens as
context. The `provider_prefix` arm is decoded a second time with the provider's
prefix as context. The audit reports adjacent-frame geometry (full frame and
upper 45%) and RGB change for frames join−3 to join+5, plus each stage's
same-frame difference from the production prediction.

Standalone window decodes do not reproduce production's blend of the preceding
decoder window. The frame before the join is therefore the window's own render,
not the assembled frame. All arms use the same decode, so they remain comparable
to each other. Confirm a verdict against matched ROIs or feature tracks, and
against assembled output, before attributing a cause.
