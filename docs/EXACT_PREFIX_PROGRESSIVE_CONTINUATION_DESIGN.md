# Exact-Prefix Progressive Continuation Design

Status: design / implementation plan

Base: Flow v0.3.5 main (`b659b311fa548c7e3275db0e6f0a05c037dac730`)

## Problem

`H3ProgressiveTargetInputHandoff` currently detects any exact protected video value in the prepared Native Masked continuation mask and exits the progressive path before the low/probe/high split. The conservative fallback forwards the untouched target-grid latent/noise/mask and full sigma schedule through one sampler lifetime.

That contract is correct for exact preservation, but it removes the main performance benefit of Progressive Target Input from every Continuum continuation chunk after the first. A representative post-release run shows the second chunk receiving the full eight-step target-grid schedule, with Spectrum executing `8 logical / 6 actual / 2 forecast` on the target grid.

The behavior is intentional in v0.3.5, not a Spectrum, Sol-H3, VDN-H3, or scheduler regression.

## Why the current fallback exists

The pre-fallback Target Input implementation resized the complete target-grid video latent, including the exact protected prefix, into the private low grid. The final high-grid stage could restore protected output values, but generated suffix rows had already conditioned on a spatially altered version of the protected prefix during the early H3 transformer lifetime. Restoring the prefix later cannot undo that changed trajectory.

The v0.3.5 fallback therefore preserves exact semantics by keeping the entire continuation on the target grid.

Two existing alternatives are not acceptable as the production replacement:

- `mixed_grid_low_suffix`: keeps exact target-grid prefix rows beside low-grid suffix rows, but the historical Mixed-Grid path produced framing/composition discontinuities and has been retired from production.
- `target_sparse_lifter`: keeps target-grid latent geometry but approximates early transformer rows; decoded-media testing showed cascading quality changes and it remains a research/control path.

Simply restoring the old resize-all-prefix Target Input behavior would reintroduce the exact-context defect that motivated the fallback. Simply splitting the eight target-grid steps into two sampler lifetimes would not recover the low-grid compute saving.

## Required production properties

A replacement must satisfy all of the following simultaneously:

1. every caller-owned exact protected target-grid video value remains authoritative;
2. generated suffix work is genuinely cheaper than running the complete target-grid transformer for the full schedule;
3. no protected prefix latent is spatially resized and then used as if it were the exact prefix;
4. no heterogeneous packed video tensor is exposed as a single implicit spatial lattice;
5. no Mixed-Grid attention-measure heuristic is inherited silently;
6. audio remains on the native joint-H3 path and the four-tick guided overlap remains independent;
7. the low/probe/high schedule retains explicit sampler/Spectrum history boundaries and the first target-grid high call is actual;
8. no additional H3 transformer NFE is hidden beyond the existing exact handoff probe;
9. unsupported geometry/backends fail closed to the current target-grid fallback;
10. decoded media, not structural tests alone, is the promotion gate.

## Proposed architecture: partitioned exact-prefix progressive attention

The correct low-stage representation is two explicit video spatial domains rather than one mixed packed lattice:

- `P`: immutable protected prefix frames at the target spatial grid;
- `S`: generated suffix frames at the lower source spatial grid.

Text/reference/audio rows remain ordinary H3 non-video rows (`G`).

Each H3 block keeps `P` and `S` as separately described domains with their own physical spatial coordinates and RoPE. The implementation must not pretend that target-grid prefix rows and source-grid suffix rows share one uniform spatial measure.

### Attention

For each query partition, attention against the key partitions is evaluated as explicit rectangular pieces and merged by log-sum-exp so that the result is equivalent to one softmax over the union of the permitted key domains without concatenating them into an implicit single-grid sequence.

For a query partition `Q` and key partitions `K_i`, each backend piece returns output `O_i` and log-normalizer `L_i`. With any required physical key-measure bias included in `L_i`, combine:

```text
L = logsumexp_i(L_i)
O = sum_i exp(L_i - L) * O_i
```

This removes the old failure mode where unequal target-prefix/source-suffix carrier density changed softmax mass merely because one domain had more discrete spatial rows.

The partition metadata must be explicit and immutable for the block call. Query-position ownership remains with the producer of the restricted domain; the Sol mapped-neighbor v4 contract continues to apply where VDN owns rectangular restricted K/V positions.

### Block state

Protected prefix hidden rows are still propagated through the transformer blocks because their deeper-layer K/V state can influence generated suffix queries. They are not sampler outputs and remain subordinate to the caller-owned exact latent at the external boundary.

Generated suffix hidden rows remain on the source grid through the low stage. There is no source-grid copy of the prefix presented as the exact target prefix.

### Handoff

At the existing selected handoff coordinate:

1. execute the exact low-stage probe under the same partitioned-domain contract;
2. obtain the generated suffix clean state on the source grid;
3. apply the existing `learned_3d` transfer only to the generated suffix context needed by the target-grid continuation;
4. restore the original exact target-grid prefix directly from the authoritative target input;
5. rebuild the full target-grid conditional state/noise at the same sigma;
6. reset solver/Spectrum histories exactly as ordinary Progressive Target Input already does;
7. begin the target-grid high stage with a mandatory actual H3 evaluation;
8. retain the four-audio-tick guided overlap/final exact restore independently of the spatial handoff.

## Backend contract

This design should not revive the deprecated `mixed_grid_low_suffix` API or its node. Introduce a new versioned, narrowly scoped contract for partitioned progressive video domains.

The contract must carry at minimum:

- domain id and owner generation;
- query domain (`global/nonvideo`, `prefix_target`, `suffix_source`);
- target/source spatial shape and temporal extent;
- immutable physical query-position map inside any VDN-restricted K/V domain;
- per-domain physical key-measure scale or exact additive log-measure bias;
- explicit capability bit for backend log-sum-exp return/merge;
- fallback callback for unsupported cases.

No compatibility adapter may silently reinterpret the deprecated Mixed-Grid contract as this new one.

## Bounded implementation sequence

### Phase A — structural prototype

- keep v0.3.5 behavior unchanged by default;
- add a new internal exact-prefix mode, disabled unless the complete backend capability contract is present;
- implement partition metadata and fail-closed validation;
- preserve all current exact-mask and audio-overlap invariants;
- add unit tests proving that exact-prefix continuation chooses the new path only when every capability is present.

### Phase B — arithmetic oracle

Before any media claim, build a small dense reference that compares partitioned log-sum-exp merge against explicit concatenated dense attention on synthetic tiny domains. Required tolerance is ordinary BF16/FP32 numerical agreement, with explicit tests for unequal spatial carrier density.

### Phase C — real Sol/VDN backend

- extend the rectangular Sol backend with the bounded partition/LSE contract rather than reviving the old Mixed-Grid weighted path;
- extend VDN provider ownership only where its restricted-domain semantics require physical query-position transport;
- preserve native global/anchor fallbacks and mapped-neighbor v4 local routing;
- keep specialization keys structural: runtime descriptor values and per-request positions must not create unbounded kernel specializations.

### Phase D — production runtime gate

On the matched two-chunk workflow, require the later continuation chunk to show the same progressive topology class as the initial chunk:

```text
low source-grid lifetime
exact handoff probe
fresh target-grid high lifetime
```

The later chunk must no longer report one `steps=8` target-grid sampler lifetime.

Structural acceptance includes:

- exact protected output bitwise restored;
- no deprecated Mixed-Grid route/counters;
- no target-sparse lifter route;
- no square-Q expansion;
- no hidden corrective NFE;
- explicit low/probe/high history boundaries;
- first high call actual;
- four-tick audio overlap preserved;
- bounded backend descriptors/caches;
- no unsupported backend fallback in the matched production run.

### Phase E — media/performance gate

Compare against v0.3.5 fallback with seed, prompt, references, geometry, sampler, Spectrum, DiffAid, Untwisting RoPE, VDN, Sol, VAE and Continuum state held fixed.

Reject the candidate on any recurrence of:

- frame shift/shrink/zoom;
- top-edge reveal;
- pre-boundary or post-boundary composition displacement;
- flash/grid artifact;
- semantic replay attributable to transport/runtime;
- audio seam regression;
- reference-fidelity regression.

Performance promotion requires a material reduction in later-chunk sampler/model-call wall time versus the v0.3.5 full target-grid fallback. If the new path does not recover meaningful continuation-chunk compute, it should not replace the simpler fallback.

## Immediate conclusion

The eight continuous target-grid steps are the direct consequence of the v0.3.5 conservative exact-prefix fallback. There is no safe configuration toggle that turns the current standard Target Input node back into low/probe/high continuation without selecting one of the previously rejected approximations.

The production fix is therefore a new exact-prefix progressive backend contract, not a scheduler tweak and not a reactivation of deprecated Mixed-Grid.