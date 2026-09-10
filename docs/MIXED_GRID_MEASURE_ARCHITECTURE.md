# Mixed-Grid attention measure and physical decode architecture

Status: implementation specification; design only. CUDA performance and decoded-media acceptance of the proposed operator are release gates, not completed results. Coordination: [Flow design PR #29](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/29).

## 1. Decisions and evidence

Adopt source-carrier-normalized **per-key spatial measure**, retaining every query, key and value. Provide an exact dense implementation and accelerated implementations that force every block containing non-unit measure through exact attention. Keep unit-measure regions on each provider's existing sparse/tail algorithm. Move temporal decode-context ownership to Continuum, after the final latent processor and before video decoding. These are independent changes.

“Exact” in the accelerated formulation means token-level evaluation rather than a pooled block; it does not remove floating-point or INT8 quantization error. Sparse Sol-Attn remains an approximation to dense attention.

| Finding | Classification and scope |
| --- | --- |
| Mixed prefix and suffix have different spatial row densities, but current uncorrected attention counts rows equally | Confirmed in Flow/native source. For 1064 versus 540 rows/frame, equal-logit prefix mass is 1.97037 times suffix mass. Actual attention mass also depends on learned logits. |
| Native RoPE positions alone correct that density imbalance | False: coordinates change scores, not the counting measure. Flow already preserves native coordinates and global temporal phase. |
| All-Q/reduced-KV removes the observed framing defect | Production evidence recorded in merged Flow #24 and Sol #2; not independently rerendered here. This supports the measure hypothesis but does not uniquely prove it: subsampling also changes content, routing and hidden-state evolution. |
| Target prefix-replacement mismatch exists | Recorded algebraic closure in Flow's representation experiment; it was insufficient to remove the framing jump. |
| Source-space warp is a satisfactory repair | Rejected by the recorded finite-horizon experiments. Remains retired; do not restore it. |
| Spectrum #106 caused the framing defect | No supporting evidence. Preserve its accepted schedule and calibration/history proof. No scheduler redesign in this work. |
| Independent decoding can differ despite exact latent overlap | Confirmed by native window semantics, Flow's existing source oracle, and upstream Continuum Issue #19/WORKLOG evidence: 5/5 exact latent boundaries had nonexact RGB overlap; reported visible-join MAE increase 10.657–21.899%. These are upstream-reported results, not new measurements. |
| VDN is the framing root cause | Unproven; its external mode already disables the geometry-dependent complement and retains the learned softmax gate. Preserve it. |

The attention correction cannot be called a universal perceptual cure. H3 was trained on discrete uniform layouts; a heterogeneous hidden stream is an inference extension. It is coherent as quadrature over a common frame domain, but finite-resolution feature statistics, aspect changes, later block evolution and low-to-high transfer remain empirical concerns. No evidence justifies abandoning the representation before testing the all-row formulation.

The existing `docs/mixed-grid-seam-repair.md` still contains pre-acceptance language. Merged PR #24/#2 contain later acceptance evidence. Do not infer current PR state from that document.

## 2. Operator and alternatives

Let N_s be source spatial rows/frame, N_p protected-prefix rows/frame, and P the packed row interval occupied by target-video protected-prefix keys. Define:

```
w_j = N_s / N_p   if j in P
      1           otherwise
b_j = ln(w_j)
z_ij = scale * dot(q_i, k_j) + existing_additive_mask_ij + b_j
y_i = sum_j exp(z_ij - m_i) v_j / sum_j exp(z_ij - m_i)
m_i = max_j z_ij
```

Apply the bias after score scaling. Never multiply K or V by w as a substitute; that changes similarity or omits denominator weighting. Query weights are unnecessary. All query rows, including text/audio/reference and protected-prefix queries, use the same key measure. Restricting the correction to suffix queries is invalid because prefix hidden states feed subsequent blocks.

This normalization deliberately preserves the native low carrier's video-versus-conditioning mass: source video keys and every pre-video packed key remain weight one. Giving every frame unit total mass would also change video versus text/audio/reference balance and is outside scope. Native temporal spans are not additional temporal weights; changing them would alter uniform H3 semantics.

Native `_frame_grid` has area-normalized rectangular coordinates: the product of spatial support extents is constant, so relative spatial cell area is N_s/N_p. Preserve the actual native coordinates. Different aspect ratios can change support shape even at equal area; measure normalization does not fix crop/FOV mismatch. New metadata records both grids and native geometry policy. Test anisotropic and rounded grids explicitly; never relabel them as coordinate-equivalent.

Identical-key subdivision is an invariant: splitting a key into r identical keys with weight w/r preserves dense output. This is the reason to prefer explicit measure to row-count heuristics. It does not prove invariance when the finer keys contain different information.

### Accelerated release formulation: weighted keys are exact

For provider block width B, force all K blocks intersecting P into the exact set for **every** query block, including ragged and mixed boundary blocks. Preserve existing conditioning sinks; if the kernel accepts one sink interval, use its smallest containing interval with the new prefix interval. This may force additional unit-weight keys exact. Do not expand query sinks merely because key sinks expand.

For exact keys E and remaining blocks C:

```
numerator_i   = sum(j in E) exp(z_ij) v_j
              + sum(c in C) exp(a_ic) Vsum_c
denominator_i = sum(j in E) exp(z_ij)
              + sum(c in C) exp(a_ic) n_c
```

Every c in C has only unit-weight live keys. `a_ic` is the provider's established approximate block score; Sana's and kitchen's tail evaluation need not be numerically identical. Accumulate both branches using a shared stable running maximum/log-sum-exp. Never normalize branches independently and average outputs. Exact blocks must be removed from pooled/token-candidate contributions to avoid double counting.

**Selection decision:** weighting cannot be added only to final token logits if weighted keys may remain pooled. In the selected design, force-exact membership makes this safe: no weighted key participates in the approximate tail or optional token candidate pool. Retain existing tau statistics and non-sink selection initially; kitchen top-k continues excluding sinks from its optional budget. This is a defined approximation policy, not a claim that unweighted proxy rankings optimize weighted approximation error. No new weighted top-k/tau algorithm is required for this release.

For future weighted pooling, use M_c=sum(w), Kbar_c=sum(w*K)/M_c and Vsum_c=sum(w*V), with denominator M_c. Routing must then account for the weighted block contribution (including log M_c where comparing mass), and handle variance/correlation approximations explicitly. Such pooling is not exact for nonconstant K. It is a separate optimization with new numerical identity and media gates, not an implementer shortcut.

| Family | Mathematical/detail consequences | Cost, compatibility and decision |
| --- | --- | --- |
| All-row weighted dense | Exact defined discrete operator; retains all prefix detail and query evolution | O(H T² D) work; tiled/streaming softmax avoids materialized T² storage. Required reference and compatible non-Sol route. |
| All-row weighted exact-prefix plus existing tail | Correct measure in exact prefix; original approximation only in unit-weight region; no prefix K/V loss | Additional forced-prefix work roughly O(H T P D), with unchanged QKV domain. Chosen accelerated architecture for Sana/core BSA. CUDA quality/performance pending. |
| Current deterministic representatives | Low-density quadrature approximation; discards unsampled prefix keys/values each block, though all Q survive. Later prefix evolution changes; aliasing possible | Rectangular Sol path already media-validated and cheaper. Mapping can cease to be injective with anisotropic ratios. Retain frozen legacy comparator, not canonical exact operator. |
| Area-aware K/V pooling | Preserves averages but softmax of averaged K is not average of exponentials; loses fine detail. Averaging post-RoPE K mixes phases | More preprocessing and new mapping/routing; arbitrary ratios require overlap-area construction. Not selected without accuracy evidence. |
| Separate prefix/suffix attention | Exact if each domain returns numerator/denominator or output plus LSE and results are combined by LSE weights; all other packed segments must also participate | Same dense arithmetic, more launches/ABI work. Existing output-only calls cannot be averaged. Allowed dense streaming implementation strategy, not required cross-repo API. |
| Uniform low-density hidden stream plus separate high-res conditioning | Changes conditioning topology, gates and checkpoint semantics; loses current exact-prefix hidden-state role | Likely needs model-level validation/training and new contracts. No current evidence favors it; outside this release. |

All-row weighting generalizes algebraically to arbitrary positive grids, including anisotropic scaling; it requires no integer grid ratio or representative mapping. Preserve Flow's existing patch-safe/no-axis-increase constraints. Equal grids produce unit measure and native behavior. Malformed geometry or unsupported masks must fail before execution, not silently drop weights.

## 3. Mandatory contract and execution ownership

The following are proposed new APIs, not claims about current code.

### Producer payload

Flow publishes `attention_measure_v1` in each actual mixed block's copied transformer options, independent of seam controls:

```json
{
  "api": 1,
  "operator": "key_log_measure",
  "normalization": "h3_native_source_carrier_v1",
  "topology": "mixed_grid_low_suffix",
  "q_rows": 56029,
  "kv_rows": 56029,
  "video_start": 11941,
  "temporal": 70,
  "prefix_t": 12,
  "source_grid": [20, 27],
  "prefix_grid": [28, 38],
  "segments": [
    {"start": 0, "stop": 11941, "mass_num": 1, "mass_den": 1},
    {"start": 11941, "stop": 24709, "mass_num": 540, "mass_den": 1064},
    {"start": 24709, "stop": 56029, "mass_num": 1, "mass_den": 1}
  ],
  "coordinate_policy": "minimax_h3_native_frame_grid_v1"
}
```

The example illustrates schema only; compute all offsets from the current layout. Do not hardcode it. Segments form a sorted, nonoverlapping, complete partition. Integers must exclude booleans; masses are positive rationals normalized for semantic identity. Reject nonfinite derived biases, invalid counts, gaps, overlaps, stale geometry, duplicated ownership or disagreement with actual block layout. Only the target-video prefix is reweighted; references ahead of target video are unchanged.

The generic API represents key measure, not a gather algorithm or VDN mode. H3-specific geometry validation belongs to Flow/core H3 adapter. Generic attention providers validate row partition and measure semantics. Cross-check VDN API 2 when present; **VDN is not required** for measure support. Keep direct `args.layout` and `transformer_options.minimax_h3_layout` identical, preserving Flow #26.

### Capability and route binding

Use a proposed `attention_measure_capabilities_v1` mapping of provider identities to callable capability objects. Each object supplies `prepare(request, execution_context) -> BoundMeasurePlan`. The execution context contains block owner, actual layout, dtype/device, head dimension, mask class, preprocessing chain and numerical route settings. The bound plan contains:

- ABI/operator version, supported implementation profile (`dense_exact_v1` or `weighted_exact_blocks_v1`);
- immutable semantic digest of normalized payload and numerical policy;
- concrete callable/patch owner and generation token, block coverage and preprocessing owner;
- resolved exact K-block ranges, immutable O(T) bias buffer or equivalent segment representation;
- verification/receipt hook bound to that implementation, not a user-set boolean.

Resolve structural ownership before the sampler begins; finish shape-dependent binding before the first mixed block. Predict all blocks, dense warmup/min-token/sigma routes, probes and supported fallback paths. Detect wrappers that replace the bound callable. Do not prove support by class name, version string, attribute presence alone, or by registering a provider that execution can bypass.

Core is the eventual owner of generic capability validation and native dense measure handling. Flow may ship a narrowly scoped dense adapter and contract validator during the coordinated compatibility window, then delegate to core's identical ABI. Never monkeypatch global `PackedLayout` or global attention classes. Sol-H3 implements a provider behind the generic API. VDN forwards it to its softmax owner and retains gate/projection ownership.

Unknown selected providers or unsupported compositions fail with topology, block and missing capability before an H3 call. An explicitly selected dense compliant route is allowed; silently falling back to ordinary Sage/SDPA or unweighted BSA is not. OOM/cancellation is propagated; no hidden retry through a different numerical operator. Dense adapter must preserve existing masks by adding bias, including correctly handling boolean mask semantics. Reject mask forms it cannot faithfully compose.

### Receipts and Spectrum identity

Keep `attention_backend_history_v1` and `attention_backend_receipts_v1` as existing Spectrum interfaces. Add measure evidence to each actual provider receipt and its acceptance predicate rather than creating a disconnected evidence system. A receipt contains `(call_token, block_index, owner_generation, semantic_digest, implementation_profile, numerical_route, q_rows, kv_rows, exact_range_digest, preprocess_digest, completed)`.

An actual emits exactly one successful receipt per bound block after the provider invocation returns. The outer Flow verifier rejects missing, duplicate, stale or wrong-owner receipts before accepting the model output. CUDA errors discovered on synchronization invalidate the call; host receipts attest dispatch/ownership, not measured CUDA numerical accuracy. No per-block synchronization in production. Failed calls discard their receipt collection.

Spectrum receives stable numerical identity containing layout/measure digest, provider/source identity, patch generation, mask and conditioning identity, preprocessing chain, dtype/device, approximation settings and calibration ownership policy. Telemetry counters, display labels, call tokens and receipt spelling are excluded from **numerical** identity. A changed measure/operator must invalidate history; a new spelling for the same audited route must not.

Do not install a generic history policy in a way that bypasses Spectrum's current core-BSA audit: `_preflight` chooses registered policies before the special BSA audit. The first implementation extends Spectrum's reviewed BSA audit/Flow identity/receipt tables for the new core source and measure digest, retaining `_ColdSuccessorProof` and the existing route names. If migrating to core-owned generic history later, first port the full cold-to-primed proof transactionally; that is not required here.

Preserve exact adjacent cold-to-primed tensor-owner proof, all non-route/non-pool identity, deferred bootstrap entitlement, rollback state and accepted 18L/14A/4F behavior in the applicable core-BSA production fixture. Never fake receipts on forecast steps, create calibration during a forecast, mark skipped calibration refreshed, or force actual NFE merely to consume measure metadata. Forecasts use the existing accepted history proof; invalid attention capabilities cannot be repaired by Spectrum's actual-only fallback.

Bias/cache lifetime is run/model-owner scoped and bounded by layout/device/dtype/digest. Never cache activations, retain user tensors globally, key only by row count, or mutate incoming option dictionaries. Reentrant runs use separate tokens/receipt lists. Stage exit restores outer metadata in `finally`; high-grid/native stages see no mixed measure. Cancellation clears owned transient state without deleting another model's pool. BSA's existing kmean/vscale tensor owners and in-place update semantics are preserved.

## 4. Provider implementation details

### Dense native H3

Add a capability-aware path at native attention dispatch so unweighted backends cannot ignore the request. Use SDPA with broadcast additive bias when a validated memory-efficient backend supports it; otherwise explicitly choose a bounded streaming dense implementation with online max/numerator/denominator accumulation. Do not allocate `[B,H,T,T]` merely to expand a key bias. Return every row in native order and preserve output projection and conditioning. Dense warmup for accelerated providers must use the same all-row weighted operator and full-domain preprocessing exactly once.

### Sana Sol-H3 SM120

Likely files: `sol_h3/mixed_measure.py`, `runtime.py`, vendor `interface.py`, `preprocess.py`, `sm120/mainloop.py` and launch/cache plumbing. Existing SM120 routing already ORs an exact K-sink interval into the route mask and removes those blocks from the approximate branch. Extend its exact-score path with natural-log key bias, converted exactly once to the kernel's base-2/scaled-score representation **before running-max and exponentiation**. The current helper stores raw-dot-product maxima; either add `b/scale` in those units with validated positive scale or refactor weighted mode consistently to log2 logits. Prefer the separate weighted specialization using log2 logits; preserve no-bias compilation/numerics.

Carry the bias pointer/segment arguments through cached descriptors, specialization keys, capacity padding and launch validation. Mask padded keys after bias; never read bias past logical K rows. Preserve Q/K/V length validation, native Q positions, exact-block tracing and full-domain Untwist. New weighted mode does not call `reduce_kv`. Legacy mode keeps it unchanged. Unsupported SM90/SM100 kernels must not advertise weighted capability merely because SM120 has it; use explicitly bound dense support or reject.

### Core BSA and comfy-kitchen

Current kitchen ordinary `sol_attn` accepts exact-only `key_bias`; **current `sol_attn_chunked` does not**. The producer calls `quant_k_rows(..., nullptr, ...)`, and `sol_attn_core` has no bias argument. A Python-only forwarded kwarg is insufficient.

Required kitchen extension: add optional keyword-only `key_bias` to chunked API, normalize to contiguous natural-log bias, convert to kernel log2 units once, propagate through nanobind/C++ producer launch and `sol_attn_producer.cu` into existing per-key `ksb` bias storage. Audit existing centering compensation in `quant_k_rows`; preserve it and add the external bias rather than overwrite an existing term. Alternatively a dedicated post-producer bias-write kernel is acceptable if it preserves the same ABI/results and benchmarks favor it; do not route bias through kmean/vscale. Update bindings, fake/shape contracts where applicable, CUDA tests and HIP parity/capability declaration. Retain callable producer bootstrap and chunk alignment validation.

Validate that **every nonzero-bias key's block is exact-covered**, on ordinary and chunked paths. O(T) validation can be done at bind time for immutable metadata; avoid a per-block device sync. Bias buffers must stay alive through queued kernels. Zero/no-bias calls retain current behavior. Weighted blocks must never reach token augmentation candidates; current route excludes sinks, and tests must lock this in. No change to QKV quantization calibration or source-owned pool shape is necessary: K centering is a rowwise score shift, and measure lives in logits, not activation statistics.

Core `nodes_sparse_attention.py`: bind measure before route selection; extend K sinks and pass bias on both ordinary override and H3 chunked paths. Dense reasons must invoke compliant weighted dense attention. Keep VSA tile planning disabled/rejected for Mixed-Grid; it assumes uniform geometry and is not the VDN API-2 dense branch. Pool identities need the measure/provider policy for history safety, while the actual kmean/vscale objects and cold/primed lifecycle remain unchanged. Preserve allocation lifetime handling (`model_prefetch.pause_malloc_graph`) added upstream for long-lived sparse pools.

Upstream ComfyUI and kitchen changes are required for **native chunked BSA support**, but the Sol/dense path can be developed first. Do not emulate chunked BSA by materializing full QKV silently; that loses its memory contract. Existing core routes that would bypass VDN's learned gate are ineligible: compose through the audited VDN softmax-provider bridge, or explicitly reject the combination until that route is bound. Being able to run an attention kernel is not proof of VDN gate ownership.

### VDN, Untwist and DiffAid

`vdn_h3/hybrid.py` retains `vdn_h3_external_sequence_v1`, API 2, `dense_gate_no_linear`, and learned gate/output projection. Propagate measure and actual mixed layout through its softmax call for branch-enabled and base-attention blocks; no geometry-dependent local or linear complement in mixed mode. Gate coverage must be verified per block, not inferred from installed-node presence.

Untwist's existing `attention_preprocess_v1` chain executes once on full original mixed Q/K/V coordinates before reduction (legacy mode), weighting/routing or quantization where required by its contract. For chunked BSA, preserve the already audited preprocessing bridge; reject additional full-domain preprocessors that cannot be expressed by it. Bias does not authorize duplicating preprocessing at warmup/fallback. DiffAid block replacements remain chained in their current order and receive actual mixed layout. Flow must wrap existing blocks rather than replace them wholesale.

## 5. Node and workflow migration

New Mixed-Grid configuration schema 2 publishes measure unconditionally. New controls separate `prefix_representation_reconcile` from existing `suffix_dc_bridge`. Representation reconciliation defaults enabled for the validated learned-transfer Mixed-Grid path; it is separately testable and must not run on an unsupported transfer method. Preserve the accepted learned-upscaler/default work in Flow #28 after re-fetching its current state; this design does not change source scale or upscaler defaults.

Retain old node identifiers/input positions and old positional widget ordering. Append named schema/version fields; use an explicit graph migration for LiteGraph and a documented API JSON migration. Never infer migration solely from widget count, append a middle widget, or let a Python default turn an absent legacy field into a new operator silently.

| Serialized input | Load/execution policy |
| --- | --- |
| Schema 2 | Weighted measure mandatory; independent representation and DC controls. No measure-off switch. |
| Legacy `suffix_geometric_bridge=true` | Preserve as named `legacy_representative_v1` compatibility profile with its existing Sol consumer/history semantics and representation reconciliation enabled. Report that profile explicitly; it is an approximate legacy operator, not all-row weighting. Provide explicit upgrade to schema 2. |
| Legacy false or absent | Load the graph unchanged, but require explicit upgrade before executing heterogeneous unweighted attention. Explain the missing measure contract and offer migration to schema 2; do not silently re-enable two old repairs. Migration sets representation reconciliation false to preserve that independent choice, DC unchanged. |
| Conflicting legacy and explicit schema-2 fields | Schema-2 explicit fields are authoritative; migration removes/deprecates legacy field in the saved copy and reports the mapping. Malformed schema values fail. |

For the bounded compatibility window, Flow publishes the existing legacy measure contract only for the legacy profile, and the new generic contract only for weighted mode; never both. The legacy profile still requires a verified actual consumer and receipts, so it cannot silently run square uncorrected attention. New bundled workflows use schema 2 after acceptance. Keep the frozen legacy comparator available for at least one announced migration release; retirement requires an explicit migration/release notice, not a hidden behavioral change.

`suffix_dc_bridge` retains its current one-token mean arithmetic. Representation reconciliation adds only the zero-spatial-mean component of authoritative-minus-learned last-prefix residual to the first suffix token. Both remain Flow-owned optional transfer operations; neither can modify the authoritative prefix or suffix token 1+. Source warp remains no-op and is removed from new UI wording. No decode correction is exposed as an attention option.

## 6. Continuum decode ownership and exact context

### Native temporal contract

Current `MiniMaxH3VideoVAE`: temporal ratio 4, clip length 17, token drop 3, stride 5, latent overlap 2, per-piece pretrim 3 frames, decoded overlap 5 frames. Windows start at 0,5,10,... and consume up to seven latent tokens. ViT decoder attention sees the whole window. “Two overlap tokens” is not the required future lookahead.

For a left length L=5a+2 and shared prefix P=5b+2, right origin is L-P, a multiple of five. The next missing window starts at L-2 and consumes `[L-2,L+5)`. It therefore needs exactly **five real future latent tokens** beyond the left group. Appending right `[P:P+5]` supplies it. This yields 17 extra decode-only frames; retain only the original group's frames. For overlap P=2,7,12 the discarded right prefix is respectively 5,22,39 decoded frames.

This is equivalence to native **windowed** decoding of the same logical latent timeline, not decoding all latents through one giant ViT window. With identical decoder, precision, spatial tiling, window lattice and deterministic kernels, the same retained windows/blends can match. Bit-exact RGB is not guaranteed across different devices, batch shapes, adaptive tiling/OOM fallbacks or decoder implementations; compare within a fixed execution configuration and bound numerical error against repeatability.

### Canonical execution

Continuum owns a reusable non-UI `prepare_video_decode_views` module and an integrated video-decode entry point. It accepts final physical groups after all upscale/refine processing, the original assembly plan and decoder descriptor. It validates native split video or joint `[video,audio]` input, extracts video only without changing audio, and builds a separate immutable decode-view plan. Reuse/cooperate with Flow #25's joint-AV extraction contract; do not change the sampler's valid joint output representation.

1. Use `decode_groups` before logical `chunks`; preserve Terminal Merge groups atomically.
2. Validate total/trim/net counts, geometry, finite latents, overlap length, revision lineage and temporal origin. Never infer exactness from a denoise-mask label alone.
3. For each exact boundary, verify shared latent values after final processing. Same dtype/shape and exact values are required; do not cast to manufacture equality. Different device placement may be normalized by copying decode-only views without changing dtype or sources.
4. Decode the left group with the five real future tokens; assemble `[trim_frames:total_frames]`. The left group owns the shared decoded overlap, now with future context. The right group discards its original overlap and owns only its retained suffix. No RGB crossfade is added; only the native VAE blend operates.
5. Keep accepted CPU latents, masks, state/session and generation plans immutable. Allocate/stream one decode view at a time, then release it. Optional optimized window reuse may eliminate duplicate overlap work only after proving identical assembly.

Proposed decode-view plan fields: `api=1`, `decoder_semantics=minimax_h3_temporal_5_2_v1`, source physical-group/revision IDs, original plan digest, input shape/dtype, global latent origin, borrowed ranges with source IDs, original retained frame interval, expected extended frame count and status. The decode receipt records descriptor, view digest and actual output shape. This is decode/cache identity only; it must not enter Spectrum or sampling identity.

Integrated decoding discovers temporal semantics from the actual native VAE or an explicit equivalent decoder capability. TensorRT/TAE/other decoders are not assumed to share native windows. Legacy external IMAGE assembly remains decoder-neutral and cannot retroactively repair missing context. Its old valid-input behavior stays available, labeled as not attesting continuous-context equivalence. A requested exact-context path with an unknown decoder capability fails specifically for missing semantics, not an arbitrary model allowlist.

### Edges, finalization and cache invalidation

- First group uses native start behavior and no invented past tokens. Single group is unchanged.
- Final group receives native terminal padding only; never borrow from another scene/run or repeat future tokens at an internal boundary.
- If a short group cannot provide five new tokens, gather forward across exact, same-geometry physical groups, skipping duplicated prefixes. If fewer than five real future tokens exist before the true end, coalesce the bounded terminal suffix into one decode-only group and use native end padding, with an explicit frame map. Never pad at a boundary that has future data. One-token/image-special cases use the decoder's dedicated image semantics; they cannot be treated as a `5k+2` video overlap.
- Nonexact overlaps (Guide, independently refined groups), changed spatial geometry or inconsistent lattice define separate decode timelines. Preserve separate decoding with explicit `nonexact_boundary` status; do not replace accepted values to force equivalence. Malformed plans are errors. Exact-context claims apply only to validated contiguous runs.
- In review mode a currently terminal group's last overlap frames are provisional. Once a successor is accepted, regenerate the predecessor's decode view/cache from the final latents; never regenerate its sampling. Cache identity must include borrowed successor revision/content identity. Reroll, branch/resume and terminal merge invalidate affected neighbor decodes.
- Audio LATENT identity, audio decoder, trimming/sample rounding and seam policy are untouched. Plus Native Masked's exact 39-frame/65-audio-step requirement is independent; do not replace it with upstream's rounded 22-frame transport. Keep each repository's existing continuation semantics.
- Preserve exact-total-duration and final-anchor policies. Assemble from original physical total/trim/net counts first, then perform existing duration enforcement. Decode-only extensions never increase target duration or alter audio cut points.

A post-RGB assembler alone cannot make this automatic: it no longer has the missing latent context. Bundled workflows must route final physical latents through the integrated Continuum decoder, or the Continuum prepare-context adapter immediately before external Core Video VAE Decode. Preserve Core node display titles in examples. Once migrated, Flow's `H3ContinuumDecodeContext` becomes a deprecated delegating adapter; its old ID remains loadable. Prevent double extension using view-plan identity and reject already-extended tensors presented as original groups. Remove the Flow-owned implementation only after supported Continuum releases contain the shared contract.

Likely files differ by repository: Plus has `masked_continuation.py` and `v3/assembly.py`; current upstream has `v3/masked_continuation.py`, `v3/assembly_v35.py` and a V3.8 public facade. Port the non-UI decode invariant rather than transplanting node schemas or overriding protected upstream Assembly/Run Storage contracts.

## 7. Companion order and compatibility

| Repository | Mandatory work and dependency |
| --- | --- |
| Flow | Contract fixtures, copied actual layout, capability/receipt enforcement, schema migration and separated transfer controls in `mixed_grid.py`, `runtime.py`, `target_sparse_node.py`, config and workflows. Activate new defaults last. Host specification, not every implementation. |
| comfy-kitchen | Chunked exact-key bias end-to-end producer/binding support; sink coverage validation; no-bias parity; CUDA/HIP capability reporting. Prerequisite for compliant core BSA. |
| ComfyUI core | Generic dense measure adapter/dispatch and BSA binding/dense-route compliance in native H3/attention dispatch and `nodes_sparse_attention.py`. Preserve layout and wrapper APIs. Depends on kitchen for chunked weighted mode. |
| Sol-H3 | Generic consumer and all-row weighted SM120 specialization, dense warmup parity and legacy profile separation. Can develop alongside kitchen/core against shared fixtures. |
| VDN-H3-Plus | Softmax capability/receipt forwarding and learned-gate composition tests; API 2 unchanged. No learned weights or normal-grid changes. |
| Spectrum | Extend existing source-gated BSA/Flow audits and Sol receipt identity for new operator; preserve cold/primed proof and entitlement. Depends on concrete provider implementation/source, before default activation. |
| Continuum upstream and Plus | Reusable decode-context/view/cache ownership plus integrated path and workflow migration, adapted to each live facade. Independent of attention release. Coordinate with Flow #25 and preserve upstream contracts/manifests. |

| Configuration | New weighted mode |
| --- | --- |
| Native H3 dense, no VDN/Sol | Supported through explicitly compliant dense provider; all rows, no extra NFE. |
| Sana Sol-H3 SM120 | Supported only after weighted kernel and CUDA/media gates; native and external mixed ownership audited. |
| Core BSA | Current build unsupported for weighted Mixed-Grid; supported after kitchen/core/Spectrum companions. Ordinary BSA remains unchanged. |
| VDN API 2 + Sol/dense/BSA | Supported only if the bound route preserves its gate and excludes local/linear complement in mixed mode. |
| Spectrum | Same applicable schedule/history; no receipt fabrication on forecasts. Unknown history remains actual-only only when the attention operator itself is valid. |
| Untwist | Supported through existing full-domain preprocessing proof; unsupported opaque transforms rejected before mixed execution. |
| DiffAid | Existing replacement chain preserved and tested; no global patch replacement. |
| Continuum Native Masked | Sampling untouched; independent decode-context support after final physical latent processing. |
| Non-SM120/no Sol | Correct explicit dense route, or another tested weighted provider; never silent unweighted fallback. |
| VSA uniform tiles, arbitrary Q-dependent mask in accelerated mixed mode, opaque provider | Unsupported until separately implemented/validated. Dense supports only mask forms it can compose exactly. |

Implementation sequence:

1. Open companion draft PRs against current live bases, with shared contract fixtures and explicit dependencies. No implementation in this design PR.
2. Implement dense reference and producer schema behind explicit schema-2 activation; keep existing production behavior until migration/release gates.
3. Implement kitchen/core BSA and Sol paths in parallel development tracks; then VDN forwarding and Spectrum audits against their concrete heads. Do not enable weighted BSA without the Spectrum companion in the accepted accelerated stack.
4. Validate coordinated CUDA + media + schedule matrix. Release providers first, then VDN/Spectrum integration, then Flow schema-2 defaults/workflows. If only Sol/dense is ready, Flow must clearly reject weighted BSA; do not advertise full-stack acceptance prematurely.
5. Implement Continuum decode companions independently; reconcile #25, validate saved-latent decode-only A/B, then migrate workflows and delegate Flow helper.

Use existing suitable open implementation PRs when live scope matches, otherwise new companion PRs. Do not reopen merged Flow #24/#26, Sol #2 or Spectrum #106. Preserve independent Flow #25/#28 and any live Sol #1 work rather than rewriting them for this design. Work on mirror/checkpoint branches; preserve established one-implementation-commit topology of existing PRs when consolidating. Keep all final handoffs as open PRs. Do not merge/release on unit results alone.

Rollback is a versioned operator/profile rollback at a new run boundary, never mid-run history reuse. Preserve accepted legacy matched stacks and workflows, and revert coordinated activation if new media/performance gates fail. Do not use `suffix_geometric_bridge=false` as a rollback into silently unweighted heterogeneous attention. Decode rollback affects decode views only; original accepted latents remain available for re-decoding.

## 8. Validation and explicit release gates

### Contract and operator tests

- Dense FP64 oracle: identical-key subdivision invariance; equal-grid unit weights; exact source/prefix total mass; non-video preservation; non-square grids; ragged blocks; scale/bias units; finite/malformed/boolean counts; masked/all-masked behavior matching native attention conventions.
- Compare joint LSE accumulation to dense output; show representative/pooling differences on nonconstant keys rather than asserting false equivalence. Verify every Q output and next-block prefix hidden state participate.
- Exact-prefix accelerated oracle: biased blocks always exact for all Q, no tail/candidate double counting, partially covered block and conditioning-sink union, top-k budget and tau extremes, token augmentation 0 and enabled.
- Cross-repo fixtures execute real producer/provider/VDN/Spectrum ownership chains; wrong source/layout, absent/duplicate receipts, provider replacement, cancelled calls, concurrent runs, cache reuse, low/probe/high transitions and dense fallback. Do not test only duplicated dictionaries.
- Native mask/noise/conditioning and authoritative prefix bit-exactness; representation/DC components independently toggled; no source warp; original suffix beyond first token unchanged by transfer repairs.

### CUDA/kernel and schedule gates

Test packaged Sana SM120 on the production RTX PRO 6000 Blackwell path and kitchen chunked CUDA separately. Test FP16/BF16 modes actually advertised, head dim 128, representative production lengths, prompt length changes, capacity padding, nonmultiple-of-64 boundaries, warmup and cold/primed calls. Compare weighted all-exact kernel output to weighted SDPA, then sparse output to a matching weighted sparse oracle; record absolute/relative/RMS error and cosine separately. Use baseline repeatability and existing kernel tolerances; tightening or relaxing acceptance thresholds requires recorded justification, never a post-hoc silent threshold change.

Run memory/race sanitization for changed indexing/bias plumbing. Measure peak VRAM, kernel/producer/attention latency, and full hot sampler/end-to-end timings with identical settings. No-bias native path must retain established numerical/performance behavior. Weighted mode must fit the production memory budget; a consistent >5% hot sampler regression versus accepted representative mode requires explicit performance review before default promotion (proposed release criterion, not measured performance).

Reproduce the applicable accepted core-BSA schedule: 18 logical, 14 actual, 4 forecasts; low 8A/2F, high 4A/2F, probes 2A. Verify exact required per-block receipts, no fallback, no added H3 call, no fake pool refresh, same cold/primed ownership and rollback behavior. Do not apply this count universally to other samplers/Sol configurations: historical Sol #2 has a separate 13A/5F accepted fixture. Compare each to its own matched schedule.

### Decode-only and media gates

Use fixed saved final physical latents to compare old independent decode, context-consistent decode and a native continuous-timeline oracle; **no resampling** for this gate. Cover P=2/7/12, short groups, one group, terminal merge, true terminal padding, nonexact refined overlaps, joint AV, decoder capability mismatch, review/reroll neighbor cache invalidation, RAM/disk assembly and both exact-duration policies. Existing Flow source oracle must pass against fetched current native VAE; extend it for edge cases. RGB equality outside changed boundary support is required in deterministic native tests. CUDA comparisons use the same decoder/window/tiling policy and repeatability-based tolerance. Video latents, audio LATENT/PCM and original plan/accepted state remain exact.

For generation quality, compare frozen legacy representatives and new weighting with decoder context held fixed; separately compare decoder context off/on on identical saved latents. Use the same model, seed, prompt, references, dimensions, sampling settings and transfer controls. Include the known framing reproduction plus at least three seeds spanning static framing and fast motion, with isotropic and anisotropic grids. Inspect complete videos around every join and delayed regions, not only a seam metric. Reject return of shrink/top-edge reveal, delayed scale pulse, identity/detail loss, motion/audio regression or authoritative-prefix violation. Record registration scale/translation, frame residuals and wall times as supporting evidence; human full-media review remains required.

The measure correction and decode change must each pass their own gate before a combined run establishes cross-repo acceptance. Neither mathematical tests nor CI proves perceptual correctness.

## 9. Audited source and residual risk

Snapshots fetched for this design (paths are navigation aids; re-fetch before implementation):

| Repository | Commit |
| --- | --- |
| xmarre/MiniMax-H3-Flow-Aligned-Regenerate | `0c0a87107f3927cc9e3f87a9ada62932c2860438` |
| xmarre/ComfyUI-Sol-H3 | `f82ff2693be37dbad3438a30eb389d77136c0276` |
| xmarre/ComfyUI-Spectrum-MiniMax-H3 | `455bd357cb45637c8e852f7f448dc57b52de94f8` |
| xmarre/ComfyUI-VDN-H3-Plus | `76b31323f9e09019b435237dcd8bad1e05476ce1` |
| xmarre/ComfyUI-H3-Continuum-Plus | `a5b8943844594545301b20d01af5d9e3fa38ae29` |
| ukr8b3g-cmyk/ComfyUI-H3-Continuum | `f41e6937476df85d5cbc4198f8a5322081b09506` |
| Comfy-Org/ComfyUI | `1f641fd9337f0ec4d635a28415a8d25a8d15f753` |
| Comfy-Org/comfy-kitchen | `21003fa97bf3b180393446d729ae630ceb6c2a52` |

Primary evidence: [Flow #24](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/24), [Flow #26](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/26), [Sol #2](https://github.com/xmarre/ComfyUI-Sol-H3/pull/2), [Spectrum #106](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/pull/106), [kitchen #117](https://github.com/Comfy-Org/comfy-kitchen/pull/117), [kitchen token routing #156](https://github.com/Comfy-Org/comfy-kitchen/pull/156), and [Continuum WORKLOG](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum/blob/f41e6937476df85d5cbc4198f8a5322081b09506/WORKLOG.md) section “Issue19”.

Source audit covered Flow mixed/layout/transfer/decode code and existing oracle; Sol runtime, representative consumer, SM120 route/mainloop and preprocessing; VDN hybrid external/gate path; Spectrum backend history, BSA audit and cold-successor recovery; core H3 coordinates/VAE and BSA producer; kitchen eager/direct/chunked/producer/route/exact paths; both Continuum assembly implementations and masked continuation, upstream boundary diagnostics/history. Core's recent sparse changes include allocation-lifetime fixes (#16148/#16180) and node UI harmonization (#16154); kitchen added token routing (#156). Preserve these live changes.

Small NumPy checks performed during design verified subdivision invariance and shared exact-prefix/pooled-tail normalization to maximum absolute error 1.11e-16, plus 464 native lattice/context arithmetic cases. They are algebraic checks only. The fetched native PyTorch decode oracle could not be executed in this environment because PyTorch is absent. No CUDA/kernel or new decoded-media result is claimed. Existing GPU evidence is attributed to its source PR/worklog.

Residual risks are empirical all-row model behavior, forced-prefix cost, kernel bias unit/index correctness, receipt integration with source-gated Spectrum, and decoder cache invalidation on review/branch changes. They have concrete acceptance gates above. The architecture does not claim to solve Continuum's distinct cumulative appearance drift or arbitrary nonexact refined joins.
