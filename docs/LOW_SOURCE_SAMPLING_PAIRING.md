# Low-source sampling input-pairing witness

This diagnostic records the inputs actually entering MiniMax-H3's uniform-source
low sampler. It supports the next investigation after the **01795 trained-VAE
decoder-only counterfactual** established that changing the protected decoder
prefix alters only a bounded number of decoded frames and does not remove the
sustained apparent background expansion/softening in the fixed source suffix.

**Status:** input-provenance instrumentation, not a paired-run executor or a
production geometry fix. The native-source-carry continuation remains
unaccepted for rendered seam quality. Existing sampler, prefix, mask,
attention, audio, and VAE behavior is unchanged.

## Enable and interpret

On **Flow PR #99**, connect the existing Partitioned Exact-Prefix Handoff node
with `spatial_stage_control=progressive_uniform_source` and
`capture_boundary_witness=true`. The node already enables the saved
boundary evidence for this mode. The Flow metrics JSON now also contains three
additional read-only events for each observed continuation:

- `partitioned_low_input_pairing`: version
  `h3_uniform_source_low_input_pairing_v1`. Hashes the exact preceding
  authoritative target prefix, chosen source protected prefix, entire
  source-grid initial video noise and generated suffix noise, audio noise,
  generated suffix initial latent, audio initial latent, video/audio masks
  and low sigma sequence. Also includes seed, sampler, source/target grids,
  temporal length, selected source projection, and Flow's conditioning
  signature. These are computed **after** constructing the actual low-stage
  tensors but **before** its first sampler call.
- `partitioned_low_first_actual_prediction`: SHA-256, sigma, outer-step and
  provenance of the earliest low-stage *actual* model prediction, taken from
  the completed Flow trajectory. If trajectory capture was not active,
  `status=unavailable_without_trajectory`; there is no fake substitute
  using a forecast or a sampler preview callback.
- `partitioned_low_probe_clean_pairing`: source-grid protected-prefix and
  generated-suffix hashes of the actual model-internal low/probe clean result,
  before learned 3D transfer or high refinement.

The events are emitted only while **uniform-source boundary witness capture**
is explicitly enabled. The baseline without that witness performs no extra
input hashing, no additional low/high H3 calls, and no VAE encode/decode.
Captured byte hashes are evidence of data identity, not evidence of rendered
geometry correctness.

The pure helper
`h3_flow_regenerate.low_input_pairing.compare_low_input_pairing_receipts(A, B)`
compares two extracted `partitioned_low_input_pairing.fields` dictionaries.
It rejects mismatched authoritative target prefix, initial video/audio noise,
latent suffix, masks, low sigmas, conditioning signature, sampler, seed,
shape/dtype and temporal phase. It expects the **chosen source protected
prefix and source projection policy to differ** between the two candidates.
It reports `input_pair_eligible` only when these known controls match.

## Required controlled experiment

Do **not** treat simply running 01794 and 01795 again with the same nominal
seed as a paired experiment: their saved authoritative prefixes differed in
SHA-256, so they did not have the same denoising starting context.

A meaningful next GPU experiment needs one frozen preceding-chunk state whose
**same authoritative target prefix and same previous source carry snapshot**
can be supplied to both low-sampler arms. Vary only the selected low-source
protected prefix (actual native carry versus projected authoritative target).
Both arms must use exactly the same source-grid suffix initial noise, target
audio noise, low latent suffix/audio inputs, video/audio mask, sigmas,
conditioning tensors and model, while preserving their own isolated
Spectrum/VDN/model histories and execution order. Capture the first real
low prediction, the final source low/probe clean suffix, and decoded
same-time static-ROI motion and Sobel/detail before comparing downstream
stages. A safe replay mechanism for this frozen state **is not implemented
by the receipt alone**. Do not run a claimed paired comparison without it.

The pairing receipt cannot hash the H3 model weights, replay the complete
runtime RNG/caches, prove equality of internal attention state or guarantee
the Flow conditioning signature covers all semantics. The report explicitly
sets `checkpoint_identity_verified=false`,
`rng_state_verified=false`, `runtime_model_cache_state_verified=false`
and `causal_pair_qualified=false`. The comparator also returns
`causal_pair_qualified=false` even when all known input hashes match.
Record the exact H3 checkpoint identity, backend/overlay versions and
state isolation separately. A difference in source suffix output after a
properly isolated paired run would support a **low-conditioning effect**,
not automatically identify the root cause of camera motion or detail loss.

This patch does not change low-stage random number generation, masks,
sampler iterations, VAE, source-carry ownership, Sol, VDN, Spectrum
forecast policy, Continuum Auto3, or runtime model options. Keep PR #99
draft until controlled GPU evidence demonstrates an actual quality fix.
