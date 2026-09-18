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

Do not populate `source_stack` from branch names or expected checkouts alone.
Before the hardware campaign, obtain the actual loaded module `__file__`
paths from the running ComfyUI process and capture each repository independently:

```bash
python tools/capture_arithmetic_validation_provenance.py \
  --repo comfyui=/path/to/ComfyUI \
  --repo flow=/path/to/MiniMax-H3-Flow-Aligned-Regenerate \
  --repo sol=/path/to/ComfyUI-Sol-H3 \
  --repo vdn=/path/to/ComfyUI-VDN-H3-Plus \
  --repo continuum=/path/to/ComfyUI-H3-Continuum-Plus \
  --loaded-file comfyui:comfy.model_management=/actual/ComfyUI/comfy/model_management.py \
  --loaded-file flow:h3_flow_regenerate.runtime=/actual/flow/runtime.py \
  --loaded-file sol:sol_h3.runtime=/actual/sol/runtime.py \
  --loaded-file vdn:vdn_h3.partitioned_runtime=/actual/vdn/partitioned_runtime.py \
  --loaded-file continuum:continuum.runtime=/actual/continuum/runtime_file.py \
  --overlay-order continuum,flow,vdn,sol \
  --output candidate.source-provenance.json
```

The `--loaded-file` paths above are placeholders for the real `__file__`
receipts; do not infer them from repository names. ComfyUI core is recorded as
a source repository but is not part of the custom-node overlay-order list. The
capture records HEAD, dirty-state fingerprints, loaded-file raw/canonical SHA-256 values,
corresponding HEAD-file SHA-256 values, match mode and overlay order. Loaded
files must be tracked and match HEAD exactly, except that Git-for-Windows
CRLF materialization may canonicalize CRLF to LF before comparison. No other
byte normalization is accepted. Write the receipt outside all five repositories
so a previous receipt cannot make a later source capture dirty. The campaign
manifest hash-pins one provenance artifact
for each implementation arm:

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
absolute checkout path is not part of that identity; repository HEAD, overlay
order and loaded relative-file bytes are.

Three implementation arms are required:

- `released_target`: released Target Input semantics executed on the same
  candidate source stack used by the fixed partitioned arm;
- `partitioned_preserved`: the preserved partitioned implementation;
- `partitioned_fixed`: the candidate staged implementation.

Each arm must provide `cold`, `primed`, `numerical_invalidated`, and
`geometry_bias_mutated` evidence. Every arm must include a diagnostics-off,
`isolated_empty` production-cold run whose Sol summary observes at least one
first-executable compile miss. The partitioned arms additionally require a
separate diagnostic/replay `isolated_empty` cold run; diagnostic replay is
setup evidence and cannot substitute for production-cold wall time. Every
non-cold run names its cold
`process_anchor_run_id`; the gate reads Sol's Request-owned runtime lease from
the log and requires both the OS process ID and Sol's random per-process
generation nonce to match that anchor. This makes the same-process claim
artifact-derived and remains valid even if the OS later recycles a PID. The
gate also derives the stable CUDA-device fingerprint and compiler/runtime
versions from every source-verified Sol Request and requires them to equal the
frozen campaign identity; manifest-only version claims are insufficient.
Non-cold runs must retain that process/compiler cache while forcing fresh Sol
Requests. Mutation runs must identify the changed contract with distinct
before/after SHA-256 values and must report revalidation. A
`numerical_invalidated` run must also contain an actual Sol invalidation and a
`numerical_transition` validation miss; an ordinary `new_request` miss does
not satisfy numerical invalidation evidence.

Every run provides hash-pinned metrics, one complete single-prompt log
segment, decoded video and decoded audio artifacts. The log segment must contain
exactly one final `Prompt executed in ... seconds` receipt while preserving all
Sol Request summaries emitted by that prompt. The sampler time declared in the
manifest is checked against the metrics `sampler_wall` events. Released Target
Input runs must preserve 17 logical / 13 actual / 4 forecast whole-run calls;
partitioned runs must preserve 18 / 14 / 4 and pass the existing partitioned
runtime gate. Diagnostic partitioned runs additionally require the correlated
host/CUDA performance-accounting evidence.

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
CUDA/replay diagnostics must be disabled. Each implementation first needs an
**unpaired same-arm primed repeat** after its own cold run; this proves the
cold→primed lifetime independently of performance pairing.

Before measured pairing, the shared timing process must then execute one
`pair_warmup` run of **both** `released_target` and `partitioned_fixed`.
These warmups are recorded campaign runs but are never used as benchmark
measurements; compiler misses are allowed there because their purpose is to
make both mode-specific executable sets resident. At least three additional
`released_target` + `partitioned_fixed` primed timing pairs are then
required, and measured runs must report zero compilation misses.

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
