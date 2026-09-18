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
The source-provenance capture is a preflight artifact and is not included in
sampler/E2E benchmark timing.

## Required campaign shape

The manifest kind is `h3_arithmetic_validation_campaign_v1`. It must contain
one frozen identity covering workflow, prompt, reference media, model/adapter/
patch stacks, decoder, sampler/conditioning/geometry settings, seed, Continuum
revision, device/driver and compiler/runtime versions. The device field is the
SHA-256 of Sol's stable CUDA identity
(device/index/name/UUID/memory/SM/driver/context, excluding PID and process
nonce). Python/platform, PyTorch, CUDA, Triton, nvidia-cutlass-dsl, cuda-python
and apache-tvm-ffi values are copied from Sol's runtime lease. Every run
references the canonical SHA-256 of that identity.

## Installed source provenance

Do not populate `source_stack`, loaded module paths, or custom-node order from
branch names, expected checkouts, or manually typed `__file__` values. The
campaign uses a two-stage v2 provenance receipt.

First, in the **running ComfyUI process that loaded the campaign stack**, execute
the **MiniMax H3 Arithmetic Validation Runtime Source Receipt** output node once.
The node reads `nodes.LOADED_MODULE_DIRS` in insertion order, resolves the Flow,
Sol, VDN and Continuum roots from repository-specific runtime signatures, and
records the actual loaded custom-node entrypoint `__file__` paths. It also
records loaded critical runtime modules when they are already present. The
receipt is written to the ComfyUI output directory, outside the source trees.
If the running registry is absent or ambiguous, capture fails closed.

Then, outside benchmark timing, convert that runtime receipt into the
hash-pinned Git/source receipt:

```bash
python tools/capture_arithmetic_validation_provenance.py \
  --runtime-receipt /path/to/arithmetic_validation_runtime_source_00001_.json \
  --output candidate.source-provenance.json
```

The offline capture accepts no manual repository roots, loaded-file paths, or
overlay-order argument. It reopens the exact roots and files named by the
running-process receipt, requires their bytes to be unchanged since runtime
capture, records each Git HEAD and complete dirty-state fingerprint, and
requires every loaded file to be tracked and match HEAD exactly. The only
accepted transport normalization is Git-for-Windows CRLF materialization to LF.
The source-provenance v2 artifact embeds and hashes the runtime receipt; the
campaign gate rejects legacy v1/manual-path receipts and cross-checks every
captured root/file against the embedded running-process evidence.

Write the source receipt outside all five repositories so the receipt itself
cannot dirty a later capture. The campaign manifest hash-pins one provenance
artifact for each implementation arm:

```json
{
  "source_provenance": {
    "released_target": {"path": "candidate.source-provenance.json", "sha256": "<64 hex>"},
    "partitioned_preserved": {"path": "preserved.source-provenance.json", "sha256": "<64 hex>"},
    "partitioned_fixed": {"path": "candidate.source-provenance.json", "sha256": "<64 hex>"}
  }
}
```

The gate cross-checks those receipts against every run's `source_stack` and
`source_dirty` declarations. Released Target Input and the fixed partitioned
timing arm must resolve to the same canonical installed-source identity. The
runtime receipt is a preflight/source-freeze artifact; it is never included in
sampler or E2E timing.

## Diagnostic versus measured runs

Diagnostic runs enable Sol CUDA attribution and same-input replay. At least one
diagnostic run is required for both `partitioned_preserved` and
`partitioned_fixed`. Its Sol diagnostic report must contain the ordinary-low,
ordinary-continuation-high and partitioned-suffix replay targets, with no
primed recompilation and a retained-proof hit. Progressive low/probe/high
execution spans separate Sol Request lifetimes, so generate this report with
Sol's `--all-requests` mode; continuation-high is correlated to the same Flow
request as the partitioned suffix rather than inferred from summary order.
The promotion gate then cross-checks every diagnostic request ID plus its Sol
source/implementation generation against the hash-pinned run log, so a report
from another prompt or source generation cannot satisfy the campaign. An
`isolated_empty` cold run must actually observe first-executable compilation.

For example:

```bash
python custom_nodes/ComfyUI-Sol-H3/tools/check_arithmetic_validation_diagnostics.py \
  --log /path/to/run.log \
  --all-requests \
  --require-replay-target ordinary_low \
  --require-replay-target ordinary_continuation_high \
  --require-replay-target partitioned_suffix \
  --require-replay-cold-miss \
  > /path/to/run.sol-diagnostics.json
```

The paired promotion timings are intentionally separate low-overhead runs:
CUDA/replay diagnostics must be disabled. This is artifact-derived, not a
manifest assertion: every Sol summary in a run must report both
`cuda_diagnostics.enabled=false` and `replay_diagnostics.enabled=false` when
`diagnostic_mode=false`; diagnostic campaign runs must report both enabled.
Each implementation first needs an **unpaired same-arm primed repeat** after
its own cold run; this proves the cold→primed lifetime independently of
performance pairing.

Before measured pairing, the shared timing process must then execute one
`pair_warmup` run of **both** `released_target` and `partitioned_fixed`.
These warmups are recorded campaign runs but are never used as benchmark
measurements; compiler misses are allowed there because their purpose is to
make both mode-specific executable sets resident. At least three additional
`released_target` + `partitioned_fixed` primed timing pairs are then
required, and measured runs must report zero compilation misses. The complete
pair set must contain both control-first and fixed-first member order; a campaign
that always measures one implementation second is rejected rather than allowing
systematic within-pair thermal/cache/drift bias.

The two members of each timing pair must be adjacent in the **complete**
declared campaign order, use the same exact Flow/Sol/VDN/Continuum source
stack, and carry the same Sol process ID + process-generation nonce. All
measured pairs must also share that one warmed process/source identity. This
permits both timing modes to alternate inside one already-primed ComfyUI
process instead of requiring two resident model processes on one GPU. A
timing-only fixed run and its `pair_warmup` may therefore name the control
process's earlier cold anchor; non-paired primed/invalidation evidence must
still anchor to the cold run of its own implementation. Every run records a
`source_dirty` map for those repositories and promotion rejects any dirty
checkout; a commit SHA alone is not accepted as complete source identity.

The promotion boundary is deliberately strict: every supplied pair must show a
net advantage for the fixed partitioned run in both sampler wall and E2E wall.
If variability breaks that condition, collect more evidence rather than
promoting from a noisy aggregate. The passing JSON report includes every cold
run, every diagnostic/warmup setup run, absolute and percentage paired timing
deltas, and the paired median deltas. Setup and cold costs therefore remain
visible; they are not subtracted from a claimed E2E improvement.

The manifest must also include an explicit `cold_behavior` policy:

```json
{
  "cold_behavior": {
    "speedup_claim_scope": "unconditional",
    "minimum_primed_reuses": 0,
    "amortization_note": "No production-cold penalty was observed."
  }
}
```

The gate compares diagnostics-off `isolated_empty` production-cold medians for
the released Target Input control and fixed partitioned arm. If either fixed
sampler wall or fixed E2E is slower, `speedup_claim_scope` must be
`"amortized_only"`; an unconditional speedup claim is rejected. The gate
computes the conservative break-even reuse count as the larger of the sampler
and E2E cold penalties divided by their measured paired-primed median savings,
rounded up. `minimum_primed_reuses` may not understate that measured
break-even, and `amortization_note` must document the scope. This enforces the
design's cold-amortization boundary without inventing a separate numerical
"non-pathological" threshold. The report preserves the cold medians, penalties,
calculated break-even and declared claim scope for product review.

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
    "comfyui": "<40 hex>",
    "flow": "<40 hex>",
    "sol": "<40 hex>",
    "vdn": "<40 hex>",
    "continuum": "<40 hex>"
  },
  "source_dirty": {
    "comfyui": false,
    "flow": false,
    "sol": false,
    "vdn": false,
    "continuum": false
  },
  "timing": {
    "sampler_s": 0.0,
    "e2e_s": 0.0
  },
  "fresh_process": false,
  "fresh_sol_requests": true,
  "process_anchor_run_id": "partitioned_fixed-cold",
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
a `mutation` object with `kind`, `base_field`, `before_sha256`, and
`after_sha256`. The named `base_field` must be one of the frozen SHA-256
identity fields and `before_sha256` must equal that frozen digest, so the
mutation is anchored to the actual campaign base rather than a free-standing
claim. Supported mutation kinds are `lora_strength`,
`preprocess_generation`, `geometry`, `bias`, and
`geometry_and_bias`.

This gate is specific to the frozen arithmetic-validation campaign. The
17/13/4 control and 18/14/4 partitioned expectations are campaign acceptance
criteria, not generic Flow runtime policy.
