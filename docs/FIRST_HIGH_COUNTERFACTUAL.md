# First-high counterfactual capture

Diagnostic only. It tests which input or execution path determines the first
actual target-high prediction at a continuation join. Generated output is
unchanged, and no VAE is loaded.

## Setup

1. Add **MiniMax H3 First-High Counterfactual [Diagnostic]** to the MODEL chain
   that reaches the sampler, anywhere after the Flow continuation node. Every arm
   is off by default. With no arm selected, the node returns its input model
   unchanged and adds no evaluation.
2. Enable `capture_boundary_witness` on the continuation node. Large target grids
   also need the geometry-derived full-stage evidence budget.
3. Generate the continuation. The `partitioned_first_high_counterfactual`
   metrics event, also stored as `window.first_high_counterfactual` in the
   boundary bundle, holds the results. Each arm's native seven-token decoder
   window is also saved in the bundle.

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

## Isolation and memory

Arms call the guider's unwrapped CFG prediction. Predict-noise wrappers, Flow
guidance, boundary traces and Spectrum step accounting do not see them.
Spectrum's per-step diffusion-model keys are removed, so its diffusion-model
wrapper passes through. The partitioned runtime is replaced by a copy with
scratch metrics. CPU and CUDA RNG state is restored. A failed arm is recorded
with its error and does not stop generation. The production prediction is
verified unchanged.

Each selected arm costs one H3 evaluation per CFG branch at target resolution.
Arms run inside the same sampler call, reusing the production call's workspace.
Before the first arm, the device's free memory is compared with an estimate of
the arms' added state: four times the packed sampler state. If it falls short,
every arm is skipped and the receipt reports `skipped_insufficient_headroom`.
The estimate does not account for allocator fragmentation.

## Receipt

The receipt records sigma, the sample schedule, the PDD video head-bank count and
the device headroom check. It also has a high-entry ledger: per-token suffix
state RMS, clean-signal RMS, and the mean, standard deviation and pre-high
correlation of the implied noise.

For each arm it records:

- the input change and the isolation applied;
- the per-token latent difference from the production prediction;
- latent boundary trajectories (full frame and upper 45%).

The same trajectories are recorded for `pre_high` and the production prediction.
Trajectories are pairwise phase-correlation shifts in latent cells between
consecutive tokens, measured on the boundary window with the authoritative
prefix tokens as context. They are computed on CPU copies of the windows.

Latent trajectories resolve token-to-token motion only. One latent token decodes
to four frames, so they cannot establish decoded sub-frame geometry, such as
whether the first decoded frame of a token holds still. A token whose first
frame holds while its later frames move shows a reduced latent shift on the
boundary pair rather than a zero shift. Treat arm differences as evidence about
the first prediction's latent response.
