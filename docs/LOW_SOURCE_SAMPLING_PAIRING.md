# Frozen source low-sampler A/B/A experiment

This is an **opt-in real low/probe sampling experiment** for MiniMax-H3's
`progressive_uniform_source` continuation. It follows run 01795's native-VAE
decoder-only test: changing the decoded protected prefix did not remove
sustained expansion/softening in the identical generated latent suffix.

The experiment compares **two alternative inputs** to the actual low sampler
using the same authoritative previous-chunk target prefix, source-carried
prefix snapshot, seeded source-grid suffix noise, audio, masks, sigma schedule,
guider conditioning and selected attention/Spectrum/VDN implementation.
Unlike the preceding decoder-only diagnostic, the model **generates a new B
source suffix**.

## Configuration

Refresh **Flow PR #99** with ComfyUI Patcher and restart ComfyUI. On the
existing **MiniMax H3 Partitioned Exact-Prefix Handoff** node set:

| Control | Value |
|---|---|
| `spatial_stage_control` | `progressive_uniform_source` |
| `source_prefix_projection` | `native_source_carry` |
| `capture_boundary_witness` | `true` |
| `low_sampler_aba` | `true` |
| Flow trajectory/capture input | Connected and enabled |
| All other sampler/seed/geometry/backend settings | Keep existing tested configuration |

**Run from the initial chunk through the first continuation with the SAME
patched MODEL**, since native source carry relies on the preceding successful
chunk's actual source low/probe prediction and its hash-paired final target.
Do not start at chunk 2 with a different model clone or compare unmatched
renderings. A missing/incorrect snapshot fails *before* the low sampler.
The opt-in toggle defaults to **false** and adds no work otherwise.

## What runs

The ordinary continuation low+probe runs first (arm **A**) using the
hash-verified native source carry. At the low/probe boundary, before learned
transfer or high refinement, the diagnostic freezes the already resolved
low-stage initial video/audio state and then performs:

- **B:** a new, actual low+probe sampling lifetime, changing **only**
  the protected video prefix to the source-grid half-pixel-bicubic
  projection of the *same* authoritative target prefix. Its generated
  video suffix, audio, noise, masks, sigmas, model and seed are unchanged.
- **A'**: a second new low+probe sampling lifetime restoring exactly
  the original native source-carried protected prefix. This repeat checks
  reproducibility after B has executed; only an A→A' match on both the
  first actual model prediction and final generated clean suffix qualifies
  A→B as a low-input effect.

Python, NumPy, CPU torch and initialized CUDA RNG state are restored to
their position before the original A low sampler **at each shadow arm**.
On leaving an arm, all those RNGs are restored to the state they had after
A; the normal production high stage uses the **original A** low/probe
result and handoff state, never B or A'. Both shadow arms get independent
Flow trajectory/metrics owners and reset guider conditioning. No existing
attention backend is disabled, Sol is not gated, and no VAE runs inside
the shadow sampling experiment.

The diagnostic costs **two extra low and two extra probe sampler lifetimes**;
there are **zero extra high passes**, no implicit hidden transfer, and
no extra VAE calls. It may take substantial GPU time, particularly with
spectrum forecast acceleration. The extra operations are experimental:
A' reproducibility verifies the observed low output, but does **not**
prove all hidden CUDA/attention-model cache state or future-chunk residency
was restored. Do not treat a run with the diagnostic enabled as a routine
production/throughput benchmark.

## Outputs and interpretation

The existing Flow metrics JSON contains:

- `partitioned_low_input_pairing`: frozen exact-input SHA receipts.
- `partitioned_frozen_low_source_aba`: full `known_input_pairing` validation,
  changed protected-prefix hashes, A-vs-A' reproducibility
  (`repeat_a_clean_suffix` and `repeat_a_first_actual_prediction`),
  A-vs-B differences for the same stages, and actual/forecast counter totals.
- `partitioned_low_probe_clean_pairing`: original selected A source hashes.

The paired native source-grid tensors are exported under
`ComfyUI/output/h3_flow_regenerate/residual_geometry/session-.../`
with manifest kind `h3_flow_frozen_low_source_aba_v1` and byte SHA hashes:
`source_A_full`, `source_B_full`, `source_A_replay_full`,
`authoritative_target_prefix`, `first_actual_A`,
`first_actual_B` and `first_actual_A_replay`.

**Qualification:** `causal_low_input_effect_qualified=true` requires
matching known initial inputs and a maximum absolute A-to-A' replay error
no greater than `1e-5` in both first actual low prediction and final
model-internal clean source suffix. If the replay fails, the B results are
**not qualified** and must not be interpreted as a prefix-causation test.
This is an operational replayability check; internal state isolation remains
unproven and GPU execution must be inspected independently.

A valid, reproducible A–B difference establishes that changing the source
protected prefix changes the **low model trajectory** under these frozen
inputs. It does **not** prove that either arm prevents background zoom,
preserves texture, fixes tone, or improves speech. To score those outcomes,
decode the saved **A and B source clean tensors with the same native production
video VAE**, measure frame-matched static bookshelf/picture/curtain regions
with equal decoder/ROI conditions, and examine later generated frames as
well as the changed f174 anchor. The authoritative target prefix remains
unchanged between arms; the B source prefix is intentionally changed.

PR #99 remains draft. Do not merge or change default source-prefix policy
without same-prefix GPU visual acceptance.

## Failure and rollback

The diagnostic refuses incompatible stage, missing source carry,
missing witness/trajectory or mismatched initial inputs. It explicitly
reports an absent first actual prediction instead of treating a forecast
as ground truth. A replay nondeterminism result is a diagnostic rejection,
not an automatic geometry repair. Set `low_sampler_aba=false` and restart
ComfyUI to return to the preceding production-shape path.
