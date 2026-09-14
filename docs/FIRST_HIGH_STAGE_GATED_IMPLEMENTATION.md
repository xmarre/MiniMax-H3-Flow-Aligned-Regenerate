# First-high artifact investigation: gated implementation

This document describes the implementation state for
`FIRST_HIGH_STAGE_ARTIFACT_ARCHITECTURE.md` at design commit
`3b86aa212743e972cdd922bd9f235c0824346569`. The architecture document remains
the authoritative specification.

## Scope implemented in this branch

This branch implements the **O (observation)** and **C (conditioning comparison)**
prerequisites only. It does **not** implement a production sampling change and it
does not select or activate U, R, B, or A. Selection of a controlled experiment
remains gated on the captured O/C evidence and the missing external artifacts
listed below.

The diagnostic is deliberately stacked on PR #36's branch. It does not modify
PR #33, PR #35, or PR #36 source files. PR #37 remains a separate diagnostic
branch. Closed PRs #39 and #40 are not incorporated.

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
file during every diagnostic run is unacceptable. Therefore an exact checkpoint
file SHA remains an explicit **replay/promotion gate** and is not claimed by the
runtime fingerprint.

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
- no capture-budget truncation affecting the compared boundary.

The observer itself is not evidence of transparency until those conditions and
paired decoded media agree with the uninstrumented baseline.

## Production workflow for the next evidence round

1. Keep the existing PR #35 checkpoint diagnostic and PR #36 learned-anchor
   validation in the workflow.
2. Apply `MiniMax H3 Execution Contract Diagnostics` **after all MODEL patching
   nodes** so its provenance manifest sees the effective wrapper/replacement
   stack. Leave `strict_provenance=true` and `capture_mib=256` initially.
3. Feed that node's MODEL output to the production sampler. Do not change the
   seed, reference inputs, prompt, geometry, sampler, scheduler, Spectrum/Sol/VDN
   settings, guidance, or handoff settings.
4. Feed the diagnostic handle and the sampler's output LATENT into
   `MiniMax H3 Execution Contract Report`. Save the preflight provenance string
   for setup diagnostics and the final report next to the normal Flow metrics JSON.
   The report's embedded `provenance` object is the authoritative runtime manifest.
5. Save the existing PR #35/#36 checkpoint videos/reports from the same run.
6. Provide the matching original-baseline media and the backend route log for
   run 00442 before selecting U or R.

If `strict_provenance` blocks execution, do not disable it for the controlled
run. The manifest/error identifies the unresolved source owner that must be
reconciled first.

## External evidence still missing

The repository does not contain enough evidence to pass the architecture's
promotion gate. The following remain required:

- installed source manifest from the actual production process (the new node
  produces it);
- paired **ORIGINAL** baseline media for the controlled first-high artifact run;
- the matching 00442 Sol/backend route receipts;
- a newly captured O/C report bundle from the production CUDA process;
- exact checkpoint file identity for any later fresh-process replay experiment.

Consequently this branch intentionally contains **no production fix**. CPU unit
tests, source reconstruction and algebra can validate the diagnostic machinery,
but cannot establish visual correctness or authorize promotion.
