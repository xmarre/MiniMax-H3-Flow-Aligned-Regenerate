# Exact-Prefix Progressive Continuation Design

Status: implementation candidate / validation plan

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

### Attention semantics and production execution

The mathematical reference is partitioned attention. For a query partition `Q` and permitted key partitions `K_i`, each dense reference piece returns output `O_i` and natural-log normalizer `L_i`. With the physical key-measure correction included as an additive log measure, the pieces combine exactly as:

```text
L = logsumexp_i(L_i + log_measure_i)
O = sum_i exp(L_i + log_measure_i - L) * O_i
```

This reference is algebraically equivalent to dense attention over one concatenated K/V union with the same per-key additive log-measure bias. It is the arithmetic oracle used to prove that unequal target-prefix/source-suffix carrier density does not change physical attention mass merely because one domain contains more discrete rows.

The production sparse Sol route deliberately does **not** execute each physical K/V partition through an independently prepared sparse kernel and then merge those results. Sol's sparse routing threshold is derived from the K/V domain presented to preprocessing. Preparing `G`, `P`, and `S` independently would therefore create independent routing thresholds and would no longer represent the same sparse decision as one permitted K/V domain.

For production sparse execution, VDN first constructs the exact permitted K/V domain for each query group. Sol then evaluates that **single physical K/V union** in one SM120 call, with:

- the target-prefix rows carrying the exact additive log-measure bias `log(source_rows_per_frame / target_rows_per_frame)`;
- the global/non-video and source-suffix rows carrying zero log-measure bias;
- VDN-owned provider-v4 query-position metadata applied to the already-gathered restricted K/V domain;
- one shared Sol preprocessing/routing threshold for the complete permitted union;
- target-prefix K/V rows kept exact where required by the partition contract.

The single union is only an execution transport. It does not make the heterogeneous video stream a uniform spatial lattice: target-prefix and source-suffix coordinates, RoPE, ownership, row counts, physical key measure, and mapped query positions remain explicit in the immutable contract.

Dense correctness therefore has two required equivalence checks:

1. explicit physical partitions + LSE merge must equal explicit dense weighted-union attention;
2. the production single-union additive-bias dense reference must equal that same partition-LSE oracle.

The partition metadata must be explicit and immutable for the block call. Query-position ownership remains with the producer of the restricted domain; the Sol mapped-neighbor v4 contract continues to apply where VDN owns rectangular restricted K/V positions.

### Preprocessing ownership

Shape-preserving inherited Q/K/V transforms such as Untwisting RoPE must execute exactly once on the complete post-RoPE physical sequence **before** VDN performs any grouped gather. Flow preserves the generic `attention_preprocess_v1` chain on its partitioned override; Sol consumes that chain and republishes it under the VDN preprocessing key; VDN then applies it before constructing restricted query/K/V groups.

A partitioned path must fail its source-contract gate if Flow, Sol, and VDN disagree about that preprocessing key or if the inherited transform can be applied twice, dropped, or moved after the gather.

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

This design must not revive the deprecated `mixed_grid_low_suffix` API or its node. The replacement uses a new versioned, narrowly scoped contract for partitioned progressive video domains.

The contract must carry at minimum:

- domain identity and owner generation;
- query domain (`global/nonvideo`, `prefix_target`, `suffix_source`);
- target/source spatial shape and temporal extent;
- immutable physical query-position map inside any VDN-restricted K/V domain;
- per-domain physical key-measure scale or exact additive log-measure bias;
- a versioned Sol request ABI identifying single-union key-measure execution;
- bounded structural descriptor/cache identity that excludes per-request map contents and runtime bias values;
- a fail-closed path for unsupported geometry/backend combinations.

The SM120 output+LSE primitive remains a useful arithmetic oracle and diagnostic primitive, but production sparse routing is not required to decompose the permitted K/V union into independently thresholded sparse calls.

No compatibility adapter may silently reinterpret the deprecated Mixed-Grid contract as this new one.

## Bounded implementation sequence

### Phase A — structural prototype

- keep v0.3.5 behavior unchanged by default;
- add a separate partitioned scheduler/runtime contract rather than reusing a deprecated `exact_prefix_mode`;
- implement partition metadata and fail-closed validation before any sampler lifetime begins;
- preserve all current exact-mask and audio-overlap invariants;
- add unit tests proving that unsupported preflight conditions return to the conservative target-grid fallback without partially executing a second numerical path.

### Phase B — arithmetic and transport oracles

Before any media claim:

- compare partition-LSE dense attention against explicit dense weighted-union attention on synthetic tiny domains;
- compare the production single-union additive-bias dense reference against the same partition-LSE oracle;
- include unequal spatial carrier density;
- verify BF16 production accumulation uses FP32 where required;
- prove the Flow `attention_preprocess_v1` chain is consumed exactly once by Sol and published under the exact preprocessing key VDN consumes.

### Phase C — real Sol/VDN backend

- extend the rectangular Sol backend with one bounded single-union request carrying additive key-measure bias plus mapped-neighbor metadata;
- keep the partition-LSE primitive as the dense arithmetic oracle rather than using independent sparse thresholds in production;
- extend VDN provider ownership only where its restricted-domain semantics require physical query-position transport;
- apply inherited preprocessing on the complete physical sequence before VDN gathers;
- preserve native global/anchor fallbacks and mapped-neighbor v4 local routing;
- keep specialization keys structural: runtime descriptor values, request digests, positions, and measure values must not create unbounded kernel specializations.

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
- no duplicated or dropped inherited attention preprocessing;
- no unsupported backend fallback in the matched production run.

The current VDN candidate intentionally disables its geometry-dependent learned linear complement for the heterogeneous stage because the released branch assumes one fixed `tokens_per_frame`. That omission is explicit and remains a promotion gate until either a variable-grid linear oracle/runtime is implemented or a separately documented production decision replaces that requirement. It must not be silently treated as equivalent to full released VDN.

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

The replacement is a new partitioned exact-prefix progressive backend contract: explicit target-prefix/source-suffix physical domains, dense partition-LSE arithmetic as the semantic oracle, and a production single-union Sol execution that preserves one sparse routing threshold while carrying the exact physical key-measure bias and VDN-owned query-position mapping. It is not a scheduler tweak and not a reactivation of deprecated Mixed-Grid.
