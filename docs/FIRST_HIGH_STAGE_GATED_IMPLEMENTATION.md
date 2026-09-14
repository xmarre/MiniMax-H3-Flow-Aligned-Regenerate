# First-high artifact investigation: gated implementation

This document describes the implementation state for
`FIRST_HIGH_STAGE_ARTIFACT_ARCHITECTURE.md` at design commit
`3b86aa212743e972cdd922bd9f235c0824346569`. The architecture document remains
the authoritative specification.

## Scope implemented in this branch

This branch contains the **O (observation)** and **C (conditioning comparison)**
prerequisites plus the bounded **R (same-state cold high replay)** diagnostic.
It does **not** implement a production sampling change and does not activate U,
B, or A. R is an attribution experiment only; a production change remains gated
on valid CUDA/media evidence.

The artifact has been reproduced with Untwist absent from the workflow. Untwist
is therefore outside R's attribution surface. The rejected full-trajectory
Untwist clock trial remains uninstalled.

The O/C diagnostic is deliberately stacked on PR #36's branch. It does not
modify PR #33, PR #35, or PR #36 source files. PR #37 remains a separate
diagnostic branch. Closed PRs #39 and #40 are not incorporated. R composes the
existing O/C recorder rather than replacing those contracts.

### Native execution-contract recorder

`MiniMax H3 Execution Contract Diagnostics` clones the already-patched MODEL and
adds ordinary ComfyUI `ModelPatcher` wrappers around the existing Flow wrapper
keys. It uses a `ContextVar` owned by one outer Flow invocation. There are no
process-global function swaps in this implementation.

Wrapper placement is intentional:

- one `OUTER_SAMPLE` observer is immediately outside Flow and owns the record;
- one `OUTER_SAMPLE` observer is immediately inside Flow and sees the low,
  exact-probe, and high child calls;
- one `PREDICT_NOISE` observer is immediately outside Flow and sees the
  post-Flow result;
- one `PREDICT_NOISE` observer is immediately inside Flow and sees the result
  before Flow's suffix bridge/guidance processing;
- `SAMPLER_SAMPLE`, `APPLY_MODEL`, and `DIFFUSION_MODEL` observers capture the
  already-executing Comfy/H3 path without invoking it a second time.

The recorder captures bounded detached tensor clones and defers CPU summaries and
SHA-256 calculation until the outer sampling invocation has finished. The byte
budget is explicit (`capture_mib`, 256 MiB by default); an over-budget capture is
reported as incomplete rather than silently dropping the condition.

The report includes:

- original and child sigma schedules and schedule digest;
- low/probe/high geometry;
- low carried sampler state;
- exact probe internal clean result;
- fresh target-noise identity and learned target clean recovered from the
  already-constructed first-high state;
- first-high sampler input, pre-Flow model result, post-Flow result, core result,
  native H3 input and native H3 velocity when observable;
- exact carried-audio equality;
- per-channel mean/RMS, finite counts, bounded spatial/temporal correlations,
  shapes/strides/dtypes/devices, and full SHA-256 for captured snapshots;
- Flow/Spectrum logical/actual/forecast accounting and one-upscaler accounting;
- runtime contract fields exposed in `transformer_options` for the active
  companions;
- post-load injection/forward-hook lifecycle evidence for low, probe, and high,
  plus active VDN/Sol/Spectrum state at selected actual transformer calls;
- a promotion block that remains false until CUDA/media gates are satisfied.

PR #35 remains the owner of the existing decode checkpoint output names/order,
and PR #36 remains the owner of learned-anchor guidance transport validation.
The new observer intentionally does not replay guidance to synthesize those
outputs because doing so would mutate PR #36's validation TLS and confound the
run.

## Conditioning comparison adaptation

The architecture calls for a pristine target conditioning oracle with zero H3
sampling evaluations. Current ComfyUI 0.35.0 `MiniMaxH3.extra_conds()` invokes
`diffusion_model.preprocess_text_embeds()`. Re-running `extra_conds()` inside a
diagnostic would therefore execute H3 model-side text preprocessing a second
time and could perturb caches/state. That is not a valid observation-only oracle.

The implemented C comparison therefore uses two non-invasive observations:

1. It fingerprints the post-hook pristine target conditioning at the outer Flow
   boundary and compares it exactly with the high-stage conditioning immediately
   before Comfy's normal processing. A mismatch here is owned by Flow rebuild or
   an intervening wrapper.
2. It fingerprints the **already processed** high-stage MiniMaxH3 conditioning,
   including the resulting payload/layout, after Comfy has processed it once and
   before the sampler executes. It does not replay `extra_conds()`.

This preserves the zero-extra-H3 and zero-extra-upscaler contract while still
separating Flow rebuild from downstream Comfy/H3 condition processing. If the
pre-core fingerprints are equal but a fresh-process reproduction later differs
in the processed payload, the next experiment must target downstream processing
or retained state; the recorder does not infer a cause from that observation.

## Installed provenance gate

The provenance collector runs inside the actual ComfyUI process and records:

- imported `__file__`, resolved path, SHA-256, git HEAD and dirty status for
  participating core source;
- active wrapper key order and callable source/code identity;
- active PR #35/#36 overlay identities;
- DiT replacement-chain entries and attention-override callable identity;
- block forward/pre/post hook identities;
- learned-upscaler provider type, callable methods and scalar configuration;
- loaded H3 parameter names/shapes/dtypes plus bounded parameter-content
  fingerprint;
- checkpoint path/stat metadata when exposed by the loader;
- Torch/CUDA and relevant compiler/runtime flags.

The node's third string output is explicitly a **preflight** manifest captured
when the graph applies the diagnostic patch. At the outer sampling boundary, after
ComfyUI has registered conditioning hooks and merged effective wrappers/callbacks,
the recorder invalidates its source/git caches and replaces that preflight snapshot
with an authoritative runtime manifest. The completed report embeds this refreshed
manifest and is the provenance artifact used by the O/C gate.

`strict_provenance=true` fails before the Flow sampling executor is entered when a
participating imported/wrapper source cannot be tied to a file and git HEAD, or
when the expected PR #35/#36 validation overlays are not active. Dirty trees are
recorded, not silently rejected, because exact source-file SHA-256 is also
recorded. Callable provenance also records bounded nested closure/default captures
so runtime-selected providers such as a KJ attention closure are distinguishable
even when they share one source file and outer code object.

The loaded-model fingerprint is intentionally bounded. Hashing a 20+ GiB model
file during every O/C observation would be unacceptable. Experiment R therefore
adds an exact checkpoint file SHA-256 gate once per process at its diagnostic-node
preflight, then uses a cheap path/size/mtime check immediately before sampling to
reject ordinary post-preflight file changes without rehashing the model on the
hot path.

### Post-load lifecycle and active-companion observation

A preflight or pre-sampling manifest cannot prove the state of the shared H3
modules after Core has loaded and injected a particular child sampler lifetime.
The diagnostic therefore adds read-only observations at the actual runtime
boundaries rather than inferring them from node placement.

After Core `prepare_sampling` returns for each low/probe/high child, the report
records the loaded patcher's injection state, object patches, effective wrappers
and callbacks, native block hooks, and nested modules whose `forward` is replaced
on the instance or that own native forward hooks. Nested-module inspection is
bounded to 1024 modified modules; truncation invalidates the structural gate.
The resulting lifecycle digest is compared across low, probe, and high so an
adapter/hook path change is not silently treated as stable.

At actual `DIFFUSION_MODEL` execution, the recorder retains only the selected
low-last, probe, high-first, and high-last observations. It records:

- VDN closure-owned layout and runtime-pool state when present;
- Sol request/forward context, route counters, kernel identity and available
  backend receipts without importing another Sol copy;
- Spectrum active run/step/solver/history state from the already-shared runtime
  binding.

These observers call their executor exactly once. They do not reset VDN pools,
change Sol routes, touch Spectrum history, invoke the learned upscaler, or add an
H3 evaluation. An observation failure is recorded and makes the runtime evidence
gate fail, but it does not replace an otherwise successful model call with a
diagnostic exception.

The report preserves `observation_gate.structural_candidate=true` only when the
base O/C structural checks pass **and** all low/probe/high lifecycle observations
are present, injection state is coherent, the lifecycle digest is stable, nested
runtime modifications were observed without truncation, all four selected
companion-call slots are present, active Sol/VDN/Spectrum state is visible at the
first high actual, and no observation error occurred. This remains a structural
evidence gate; it is not visual approval.

## Controlled O acceptance criteria

Use the exact baseline workflow/settings from the architecture document. A valid
O run must report all of the following before its media is interpreted:

- 9 logical calls / 7 actual H3 NFE / 2 Spectrum forecasts;
- low 5L/4A/1F, probe 1L/1A/0F, high 3L/2A/1F;
- exactly one learned-3D upscaler event;
- no nonzero weighted/Mixed-Grid counters;
- exact carried-audio equality;
- first-high `h3_refinement.min_actual_prefix_steps == 1`;
- complete installed provenance;
- complete post-load runtime observation with no observer error or truncation;
- no capture-budget truncation affecting the compared boundary.

The observer itself is not evidence of transparency until those conditions and
paired decoded media agree with the uninstrumented baseline.

## Experiment R implementation

R is documented operationally in `SAME_STATE_HIGH_REPLAY_DIAGNOSTIC.md`. It uses
two separate ComfyUI processes and keeps Untwist absent in both jobs.

The capture job runs the normal failing progressive path and exports a JSON
manifest plus pure-tensor payload only after the controlled `9L/7A/2F` topology,
one-upscaler accounting, O/C structural gate, exact carried audio and first-high
runtime evidence pass. The bundle contains the exact high-entry tensors, high
suffix, guidance trajectory and declarative configuration/provenance identities.

The replay job validates the original schedule, target geometry, seed,
conditioning, Flow guidance configuration, Spectrum configuration, exact
checkpoint identity, installed runtime provenance and first-high companion policy.
It suppresses the progressive split for that invocation and runs only the captured
high suffix. It must produce exactly `3L/2A/1F` with zero learned-upscaler calls.
No low stage, exact probe, stochastic-state transport or weighted Mixed-Grid path
is executed by R.

`attribution_valid=true` additionally requires exact first-high sampler and H3
video/audio input hashes. Only then is the first-high raw/pre-guidance comparison
causally interpretable:

- same broken first-high output weakens retained low/probe/reload lifecycle state
  as the cause;
- clean/different first-high output implicates retained lifecycle state and
  requires locating the first diverging owner before any reset/buffer experiment.

R remains a diagnostic. Neither branch structure nor passing CPU CI is evidence
of visual correctness.

## Production workflow for the next evidence round

1. Keep the existing PR #35 checkpoint diagnostic and PR #36 learned-anchor
   validation in the workflow.
2. Keep Untwist absent. Apply `MiniMax H3 Execution Contract Diagnostics` **after
   all MODEL patching nodes** with `strict_provenance=true` and `capture_mib=256`,
   then apply `MiniMax H3 Same-State Replay Capture`.
3. Feed that MODEL to the unchanged failing progressive sampler. Do not change
   the seed, reference inputs, prompt, geometry, sampler, scheduler,
   Spectrum/Sol/VDN/DiffAid settings, guidance, or handoff settings.
4. Save the R bundle only if the run visibly reproduces the artifact and the
   controlled topology/provenance/runtime gates pass.
5. Exit ComfyUI completely. Start a fresh process, rebuild the same stack with
   Untwist still absent, apply `MiniMax H3 Execution Contract Diagnostics`, then
   `MiniMax H3 Same-State Cold High Replay` using the saved manifest.
6. Run the normal sampler node with the original full schedule and seed and save
   the `MiniMax H3 Same-State Replay Report` plus matching decoded media.

If strict provenance, checkpoint identity, topology or any replay contract blocks
execution, do not weaken the gate. The reported mismatch identifies evidence that
must be reconciled before R can be interpreted.

## External evidence still missing

The repository does not contain enough production-CUDA evidence to pass the
architecture's promotion gate. The following remain required:

- installed source/runtime manifest from the actual production process;
- paired baseline/capture media demonstrating that instrumentation preserves the
  failing artifact;
- the matching Sol/backend route receipts for the controlled run;
- a successful no-Untwist R capture bundle with exact checkpoint identity;
- a fresh-process no-Untwist R report with `attribution_valid=true` and matching
  decoded media.

Consequently this branch intentionally contains **no production fix**. CPU unit
tests, source reconstruction and algebra can validate the diagnostic machinery,
but cannot establish visual correctness or authorize promotion.
