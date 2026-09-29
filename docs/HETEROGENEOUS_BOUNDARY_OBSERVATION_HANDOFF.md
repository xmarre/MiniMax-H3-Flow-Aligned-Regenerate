# Heterogeneous boundary observation: qualification handoff

Status: observation implementation qualified by local source tests;
matched SM120 intervention and rendered-media gates pending. No destination-
stencil numerical policy or production fix is implemented.

The authoritative specification is
[the design at b97ed34d0db253f26e5c7a391c3fb1358c3c3684](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/b97ed34d0db253f26e5c7a391c3fb1358c3c3684/docs/HETEROGENEOUS_EXACT_PREFIX_BOUNDARY_IMPLEMENTATION_DESIGN.md).
Its hardware conditional remains in force: a mathematical ordering difference
does not establish that cross-grid taps cause the rendered artifact.

## Refreshed source baseline

| Component | Ref |
|---|---|
| Flow main | `a6249b8343becc1458a4555d2bb25cd523983c90` |
| Flow #89 | `fa8d65b84e0666ce467b838544d92c801394a002` |
| VDN Plus main | `b78e94d0365ffc5059924a048af707e565d0380e` |
| VDN Plus #33 | `da3627f85d494bdf4213251deb3ba2a94b8a2f36` |
| Sol main / v0.1.6 | `bef9b300275a89290ca2d53eafff79af06f5e0ef` |
| Sol source-contract pin | `93b3e03f2b7b579aaf55fa0f87f55083b259e25c` |
| Continuum Plus main | `e870875b1a29968d39d72ba304e9f2b05544c015` |
| Rejected Continuum #35 | `28624d2577a62b6b48a04f3897e7b03a12697eb7` |
| Rejected Continuum #36, actuator disabled | `9ea1e75b2ddf8a3cb2a30e26c369aa63c0a32aea` |
| Learned upscaler | `620165a311de9b28a36260219fb5cd370a304e3c` |
| Core current main | `9d80841aa1990305cc7a280c5ea505f317efcc2d` |
| Flow Core oracle / native-mask oracle | `1af040bf022569d7a890241c8dd79b296cda483f` / `421a1c245c682c04d4325ba365f40c834c66f5b0` |
| VDN Core oracle | `6c53f8c9a06d95f3d847009ceaae55c624169247` |
| OpenVDN oracle | `b8cb28fbfca0266d1c7742a9f25ab8b58191de97` |

All requested branches/pins were fetched. Flow/VDN review threads were resolved
at retrieval. Sol's source-contract pin differs from main only in release
metadata. These are fetched source refs, not proof of installed Patcher trees.
The installed Core log abbreviates `34b50ec9` on `patcher/stack`; that is not the
public Core oracle/main and its complete identity remains unavailable.

Both repositories have remote `checkpoint/heterogeneous-observation-start-20260930`
refs at their protected PR heads. Development uses separate
`mirror/heterogeneous-observation-20260930` branches. Pre-review checkpoints
preserve the first observation implementation. Sol and Continuum source branches
remain untouched.

## Original evidence recovered

The original bytes were recovered using the design's materialized paths and
verified against its hashes:

| Original file | SHA256 |
|---|---|
| `metrics_00717_.json` | `8abe3148210934d0765b36064108cccfd73e1af08bd313628d30c602ec575add` |
| `Pasted text(20260929-203040).txt` | `11ac6cfedde9c97ddda976f16fdd8fb46b8cd7c8edc7c59fea125679961c6e52` |

00717 uses prefix12, temporal62, target latent48×48 and source34×34. Its
upper45 pre-high pair0 is approximately (−0.021,+0.010) target cell, while
pair1 is (−3.898,−3.809). This window is not clean. Whole-workflow counters for
two physical groups are 14 actual H3 evaluations (low8/probe2/high4),
18 logical calls, four forecasts and six sampler/four history boundaries.
One continuation remains three sampler lifetimes/two history boundaries.
The original MP4, original complete workflow JSON, full overlay graph and
checkpoint hashes were not available to this implementation environment.

## Implementation and explicit deviations

1. Milestone one adds an observation ABI, not the proposed numerical-policy
   ABI. Flow owns a default-off CPU sink; VDN exposes actual raw/filtered/mapped/
   activated features and A/B/alpha norms. No conditional arithmetic is enabled.
   The evidence needed to select it is still missing.
2. A temporary stacked **diagnostic** Flow draft provides Patcher access while
   #89 stays at its one-commit baseline. This is not a replacement production
   PR. The design requires hardware acceptance before consolidation into #89;
   publishing an unqualified numerical change there would violate that gate.
   After qualification, checkpoint and consolidate approved changes into the
   existing #89 above its then-current main as one clean commit.
3. The VDN observer is a separate draft stacked on #33. #33's batching commit
   and base remain intact. A destination-stencil implementation can extend that
   draft only after the intervention supports it. No stencil capability or
   digest is advertised prematurely.
4. The current-Core native mixed-grid test fixture now accepts Core's
   `attention` keyword. Production source was not changed for that API mismatch.
   The invariant tested remains authoritative prefix conditioning of the suffix.
5. Observation is bounded to two complete heads in the first actual low-stage
   convolution block, the first mixed boundary and temporal radius ≤4. This
   supplies complete L2 channels for those heads; it does not establish all-head/
   all-layer equivalence. Existing stage observers remain in use. Wider kernels
   and missing paired observer support fail explicitly when observation is
   selected. These limits affect evidence completeness, not default arithmetic.

Tests cover observation neutrality, dtype/branch/anchor coverage, CPU lifetime,
budget enforcement, capability rejection, stage owner cleanup, option cloning
and the paired source-contract ABI. A missing selected low-stage observation
cannot silently become a successful diagnostic sample.
Final review corrected receipt metadata so bounds and physical measure scales
use the same optionally anchor-trimmed frame domain as captured features. The
untrimmed runtime scales are recorded separately; production weighting is
unchanged. The four suppression/anchor combinations verify this metadata.

## Supplied paper check

The supplied DMD2, Sol-Attn, Sol-engine, VSA, Spectrum and Video DeltaNet PDFs
were inspected. Video DeltaNet section 2.3 confirms the released K/V depthwise
5×5 spatial and five-tap temporal filters, SiLU and Q/K L2, with separately
calibrated branches. Appendix A.5 makes checkpoint architecture metadata part
of the release contract. Sol-Attn routing and Spectrum feature history remain
separate mechanisms. These papers provide no new evidence accepting a
heterogeneous destination stencil or overturning the rejected output repairs.
The design and its SM120/media gates therefore remain unchanged.

## Patcher dependency order

The implementation PR descriptions record exact final diagnostic SHAs. Use
those SHAs and the following dependency order, refreshing all overlays before
starting ComfyUI:

1. VDN Plus main, #33, then [VDN #34](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/34)
   at `cae13fb5e8d71b93ee3134b629d23fec7c819c5b`.
2. Flow main, existing #89, then [Flow #93](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/93), its stacked boundary-observation draft.
3. Sol v0.1.6/main at the ref above; no diagnostic Sol overlay.
4. Continuum Plus main, then #36 solely for its retained diagnostics and disabled
   rigid actuator. Do not add #35. Verify loaded #36 source at the ref above.
5. Learned upscaler at the ref above, and the remaining already-selected
   Spectrum/DiffAid/RefDelta/Untwist/LoRA/DoRA providers at frozen installed refs.

The diagnostic draft bases are the corresponding existing PR branches, not
main. Applying only a stacked draft's delta without its parent is invalid.
Record Patcher's effective tree and ordered overlay graph after refresh; the
public PR sequence alone does not prove which code is loaded. Do not update
Core or any unrelated provider between comparison arms.

## Matched workflow and observation

Recover the original workflow and reference inputs before claiming a matched
00717 comparison. The metrics/log do not reconstruct every prompt, seed,
adapter strength and input. Freeze that workflow and its effective checkpoint
and source manifest. Disable decoded geometry mutation; with #36 use
`Video Seam = Analyze Only` to preserve raw video while keeping diagnostics.
Keep the original audio policy in both arms and record any indirect audio
output differences independently.

Use node type `H3PartitionedExactPrefixDiagnosticHandoff`, displayed as
**MiniMax H3 Partitioned Exact-Prefix Handoff**, with its diagnostic selectors. Verified
00717 selectors are:

| Selector | Both arms |
|---|---|
| `prefix_transformer_context` | `exact_target_partitioned` |
| `audio_position_domain` | `source_carrier` |
| `audio_guided_overlap_mode` | `sampler_mask_exact_timestep` |
| `audio_guided_overlap_ticks` | `4` |
| `provider_boundary_stabilization` | `soft_support_v1` |
| target/source latent grid | `48×48 / 34×34` |
| exact prefix / T | `12 / 62` |

Keep all remaining controls, the learned full-sequence provider, existing
one-token bridge, Sol tau/routing, Spectrum, seeds/noise and LoRA/DoRA settings
unchanged. Do not select a source-uniform shadow path. Run:

* A: `vdn_linear_diagnostic=normal`.
* B: `vdn_linear_diagnostic=suppress_cross_grid_temporal_taps`.

For each observed arm, set an absolute writable directory before the ordinary
ComfyUI launcher starts, for example:

```bash
export H3_FLOW_BOUNDARY_WITNESS_DIR=/home/toor/ComfyUI/output/boundary-ab/A
```

Use a separate directory for B. This is a new default-off observer switch,
not a stencil selection or a dense-attention switch. It creates one paired
`.pt`/`.json` artifact for each observed continuation low lifetime and publishes
`partitioned_boundary_witness` in Flow metrics. The `.json` contains loaded
module paths/hashes, actual precision settings, frame/head indices and tensor
SHA256. Requested observation requires the paired VDN draft before sampling.

Verify actual suppression counters and
`partitioned_vdn_linear_diagnostic_verified`; a widget label is insufficient.
Do not promote B as a production solution. If B changes the defect materially
while retaining background continuity, qualify the destination-stencil C with
the numerical-policy/history contract and independent scalar oracle specified
in the design. If it does not, stop that promotion path and use the prescribed
conditional bypass/dense/provider discriminators with documented attribution.
The old Sol dense environment switch does not exist on these refs.

## Source and checkpoint provenance

The read-only helper hashes installed tracked source bytes, full HEAD/tree,
status, workflow and explicitly named checkpoints; safetensors headers retain
every tensor shape/dtype and metadata. Run it separately from timed inference:

```bash
python /home/toor/ComfyUI/custom_nodes/minimax-h3-flow-aligned-regenerate/tools/capture_boundary_provenance.py \
  --comfy-root /home/toor/ComfyUI \
  --workflow /absolute/path/frozen-workflow-A.json \
  --patcher-state /absolute/path/patcher-overlay-export.json \
  --checkpoint /absolute/path/actual-model.safetensors \
  --checkpoint /absolute/path/actual-vdn.safetensors \
  --checkpoint /absolute/path/actual-upscaler.safetensors \
  --witness-receipt /absolute/path/boundary-witness-ID.json \
  --output /absolute/path/provenance-A.json
```

The Flow directory above is the path reported by the original log; verify the
current loaded path. Replace input/output placeholders with actual paths. Repeat
`--checkpoint` for every effective model, BF16 affine source, encoder, VAE,
LoRA/DoRA and adapter checkpoint. Include their effective strengths and VDN
configuration in the frozen workflow/Patcher export. The separate helper's
`collector_process_precision` is not the running model's precision; use the
actual witness's `precision_at_actual_call` and runtime log for that.

The helper deliberately leaves `installed_provenance_complete=false` until
the overlay graph, all loaded modules/checkpoints/settings and reference/noise
identities have been checked. It does not reconstruct unavailable Patcher
state. Missing runtime hashes must remain missing evidence.

Analyze each tensor witness with VDN's `tools/analyze_boundary_witness.py` as
documented in VDN's `docs/BOUNDARY_FEATURE_WITNESS.md`. Preserve the original
observed receipt and report the CPU replay as additional diagnostic work.

## Runtime, media and costs

Save each existing Core decoder IMAGE output before Continuum assembly and the
assembled movie after it. Fan out the already-decoded frames to saving nodes;
do not add another VAE decode. Preserve raw physical-group frames including the
discarded prefix context so frame indices and trim39 can be independently
checked. Store the native fps and embedded frozen workflow with each movie.

First run the square 17→24 patch-grid comparison. Acceptance requires two
continuation boundaries, then a non-square case and Spectrum enabled/disabled
history validation. If the original graph has only one continuation, its
original two-group reproduction remains the first comparison; additional
three-group validation is separately recorded experimental work. Do not
pretend a new third-group prompt was part of the original run.

Inspect the first retained frame at local39, at least 24–40 subsequent frames
and the complete retained segment at normal speed. Assemble annotated A/B
side-by-side media using fixed top-edge/cabinet/curtain landmarks. Assess upper,
central and lower correspondences/flow/scale/shear over multiple pairs and 4×4
regions against ordinary pre-boundary camera motion. Reject a displaced jump,
freeze, border exposure, expansion or compensation drift even when first-pair
phase error improves. Exact-main background continuity must remain preserved.
Final candidate acceptance requires user inspection of these rendered movies.

Per run, the implementation adds zero H3 NFE, provider calls, VAE calls,
sampler lifetimes or history boundaries. It adds default-off bounded GPU
indexing/norm reductions, blocking CPU copies, hashes and file writes. The
32 MiB bound covers retained CPU tensor payload, not total process/serialization
RSS or transient device workspace. Diagnostic timing spans include intervening
actual block computation and synchronization; they are not a pure production
overhead measurement. Compare cold/warm, observer-disabled primed timing and
actual allocated/reserved VRAM separately. Offline witness replay and full-file
checkpoint hashing are counted diagnostic costs. No SM120 cost or speedup is
measured in this environment.

## Executed validation and open gates

CPU environment: Torch2.14.0+cpu, no CUDA. Results:

| Check | Result |
|---|---|
| Flow full tests with pinned Flow Core | 612 passed; later added stage-copy regression passes in targeted checks |
| VDN full tests with pinned Core/OpenVDN | 224 passed, including official linear/hybrid oracle; final receipt correction: 34 affected tests passed |
| Sol partitioned/mapped/history source tests | 38 passed after installing kernel import dependencies; no kernel execution |
| Flow native mixed-grid/decode with current Core | 39 passed after fixture compatibility change |
| Flow native mixed-grid/decode with pinned Core | 39 passed |
| Native/sibling source-contract scripts | passed at their documented source pins |
| Partitioned Flow/Sol/VDN contracts | passed, including bounded witness transport and unchanged provider-v4 geometry |
| VDN current-Core node/compiler smoke | passed |
| Flow Ruff/format; VDN CI-selected Ruff; compileall | passed |

Remaining gates: effective installed provenance, actual SM120 feature witness,
matched A/B and conditional C, all-selected weighted-dense GPU arithmetic,
two-boundary/non-square/history runtime validation, cold/warm performance/VRAM,
raw and assembled rendered media, and final user inspection. Audio remains
independently unresolved. No first-pair metric, source test or green CI can
substitute for these gates.

Rollback uses Patcher's paired diagnostic-overlay removal to the frozen #89 /
#33 baseline, retaining exact-main semantics. Verify the effective source
hashes again. Sol/Continuum branches stay unchanged. The destination-stencil
policy, production consolidation into #89 and a release remain unqualified.
