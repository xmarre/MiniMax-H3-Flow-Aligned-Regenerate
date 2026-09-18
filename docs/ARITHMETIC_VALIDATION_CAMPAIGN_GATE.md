# Arithmetic-validation campaign promotion gate

This repository includes an offline promotion gate for the staged MiniMax-H3
arithmetic-validation lifetime/performance work. The gate does not run H3,
CUDA, decoding, or benchmarks. It validates a campaign manifest and the
artifacts produced by real runs.

Use:

```bash
python tools/check_arithmetic_validation_campaign.py \
  --manifest /path/to/campaign.json
```

A passing report is evidence that the supplied campaign is structurally
complete. It is not a substitute for the recorded runs or decoded-media review.

## Required campaign shape

The manifest kind is `h3_arithmetic_validation_campaign_v1`. It must contain
one frozen identity covering workflow, prompt, reference media, model/adapter/
patch stacks, decoder, sampler/conditioning/geometry settings, seed, Continuum
revision, device/driver and compiler/runtime versions. Every run references the
canonical SHA-256 of that identity.

Three implementation arms are required:

- `released_target`: released Target Input semantics executed on the same
  candidate source stack used by the fixed partitioned arm;
- `partitioned_preserved`: the preserved partitioned implementation;
- `partitioned_fixed`: the candidate staged implementation.

Each arm must provide `cold`, `primed`, `numerical_invalidated`, and
`geometry_bias_mutated` evidence. Cold runs must use a fresh process and an
explicit isolated compiler-cache state. Non-cold runs must retain the same
process/compiler cache while forcing fresh Sol Requests. Mutation runs must
identify the changed contract with distinct before/after SHA-256 values and
must report revalidation.

Every run provides hash-pinned metrics, full log, decoded video and decoded
audio artifacts. The sampler time declared in the manifest is checked against
the metrics `sampler_wall` events. Partitioned runs must preserve the frozen
18 logical / 14 actual / 4 forecast whole-run count and pass the existing
partitioned runtime gate; diagnostic partitioned runs additionally require the
correlated host/CUDA performance-accounting evidence.

## Diagnostic versus measured runs

Diagnostic runs enable Sol CUDA attribution and same-input replay. At least one
diagnostic run is required for both `partitioned_preserved` and
`partitioned_fixed`. Its Sol diagnostic report must contain the ordinary-low,
ordinary-continuation-high and partitioned-suffix replay targets, with no
primed recompilation and a retained-proof hit. An `isolated_empty` cold run
must actually observe first-executable compilation.

The paired promotion timings are intentionally separate low-overhead runs:
CUDA/replay diagnostics must be disabled. At least three primed
`released_target` + `partitioned_fixed` pairs are required and each pair must
be adjacent in the declared benchmark order. The two runs in a pair must use
the same exact Flow/Sol/VDN/Continuum source stack.

The promotion boundary is deliberately strict: every supplied pair must show a
net advantage for the fixed partitioned run in both sampler wall and E2E wall.
If variability breaks that condition, collect more evidence rather than
promoting from a noisy aggregate.

## Decoded-media acceptance

Every run must explicitly pass all decoded checks:

- motion;
- continuity;
- prefix seam;
- prompt adherence;
- texture artifacts;
- audio seam;
- audio intelligibility.

The top-level `video_pass` and `audio_pass` flags must also be true. The
video/audio files themselves are hash-pinned by the manifest.

## Run object

A run has this shape:

```json
{
  "id": "fixed-primed-01",
  "implementation": "partitioned_fixed",
  "condition": "primed",
  "sequence": 3,
  "pair_id": "pair-01",
  "frozen_identity_sha256": "<64 hex>",
  "source_stack": {
    "flow": "<40 hex>",
    "sol": "<40 hex>",
    "vdn": "<40 hex>",
    "continuum": "<40 hex>"
  },
  "timing": {
    "sampler_s": 0.0,
    "e2e_s": 0.0
  },
  "fresh_process": false,
  "fresh_sol_requests": true,
  "compiler_cache_state": "retained_same_process",
  "diagnostic_mode": false,
  "decoded_media": {
    "video_pass": true,
    "audio_pass": true,
    "checks": {
      "motion": true,
      "continuity": true,
      "prefix_seam": true,
      "prompt_adherence": true,
      "texture_artifacts": true,
      "audio_seam": true,
      "audio_intelligibility": true
    }
  },
  "artifacts": {
    "metrics": {"path": "fixed-primed-01.metrics.json", "sha256": "<64 hex>"},
    "log": {"path": "fixed-primed-01.log.txt", "sha256": "<64 hex>"},
    "video": {"path": "fixed-primed-01.mp4", "sha256": "<64 hex>"},
    "audio": {"path": "fixed-primed-01.wav", "sha256": "<64 hex>"}
  }
}
```

Diagnostic runs add a hash-pinned `sol_diagnostics` artifact. Mutation runs add
a `mutation` object with `kind`, `before_sha256`, and `after_sha256`.
Supported mutation kinds are `lora_strength`, `preprocess_generation`,
`geometry`, `bias`, and `geometry_and_bias`.

This gate is specific to the frozen arithmetic-validation campaign. The
18/14/4 benchmark expectation is not part of generic Flow runtime policy.
