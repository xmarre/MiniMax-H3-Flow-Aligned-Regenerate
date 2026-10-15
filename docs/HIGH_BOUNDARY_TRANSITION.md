# High boundary transition (candidate)

Candidate production-path change for uniform-source continuations. It is off by
default and adds no H3 or VAE evaluation. Rendered quality is not validated.

## Behavior

Target-high starts every generated token at the handoff sigma, next to an exact
protected prefix that the model sees as clean conditioning. With
**MiniMax H3 High Boundary Transition [Candidate]** enabled, the first one or two
generated video tokens instead use a fractional Core denoise mask `m` for the
whole target-high stage:

| `transition_tokens` | mask weights |
| --- | --- |
| 1 | `0.5` |
| 2 | `0.5`, `0.75` |

Core's native masked-diffusion semantics apply to those rows at every high-stage
call:

- H3 labels them at `t = 1 - m * sigma`.
- The model input is `m * x + (1 - m) * inject(clean)`, which places them at the
  labelled noise level.
- Each prediction is restored as `m * out + (1 - m) * clean`.

The rows are still denoised by the model, at a reduced effective sigma. Every
other generated token, the exact prefix, the audio mask and the low/probe stages
are unchanged. The weights are exact multiples of 1/256, the grid Core uses for
H3 label masks, so the sampler mask and the model labels are identical.

## Clean state ownership

Core takes `clean` from the sampler's `latent_image`. For generated tokens the
caller's latent does not hold the handoff state. Target-high therefore receives
a private `latent_image` whose transition tokens hold the exact-prefix restored
pre-high clean state, which is the clean state the high entry state was built
from. The sampler noise is re-derived against it, so the high entry state, the
protected noise and audio are unchanged. The caller's latent is not modified.

## Cost and limits

Transition rows finish as a blend of refined output and the pre-high clean state.
Where the two differ in fine detail, those tokens can look softer or show double
edges. Changing tokens inside the boundary decoder window also changes how the
neighboring decoded frames render, including frames of the protected prefix,
although the prefix latents stay exact.

The policy fails before sampling for target-band continuations, a non-binary
incoming high mask, a `denoise_mask_function` owner or a non-identity video
latent-in transform. Its `partitioned_high_boundary_transition` receipt records
the transition tokens, weights, effective sigmas and the entry-state
reconstruction error.

Runs with the policy enabled report `sampler_latent_mask_unchanged=false` in
`handoff_transfer_wall`. A captured boundary bundle stores the graded initial
high mask, which the existing boundary replay audits reject because they require
every generated token at mask 1.

Evaluate it with an ordinary generation: the same seed and inputs with the node
off and on, judged on the rendered join.
