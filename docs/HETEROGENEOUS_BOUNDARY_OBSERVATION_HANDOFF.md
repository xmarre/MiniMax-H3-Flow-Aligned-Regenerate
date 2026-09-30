# Heterogeneous boundary observation: qualification handoff

Status: matched SM120 A/B/C plus complete partitioned learned-linear bypass
discrimination is complete. Candidate C changed the internal pre-high trajectory
but **00722 still rendered the frame shift**. 00724 then bypassed the complete
VDN learned-linear complement and **the visible frame shift still remained**.
No VDN production fix is promoted.

00726 successfully executed the direct same-grid target control. The user still
observed a slight frame shift, but reported that it was **substantially less
prominent**. This is partial rendered evidence that the low-resolution ->
high-resolution stage transition is materially involved, not evidence that it
is the sole cause. The run also exposed a separate Flow clean-state ownership
bug: a source-residual handoff was later inverted as though it had used the
deterministic Gaussian-noise contract, producing a synthetic clean mismatch and
a nonzero first-suffix DC mutation even though the same-grid provider and frame
gauge both reported identity. Flow #93 now preserves the actual clean
postprocess tensor and refuses that invalid inverse. A same-grid rerun after
this correction is the next qualification gate. No production fix is promoted.

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
| VDN Plus #34 observer | `cae13fb5e8d71b93ee3134b629d23fec7c819c5b` |
| VDN Plus #35 diagnostic head (candidate-C retained; equal-grid contract added) | `109e75c3fb9f97815504d41226dd4c853cd434c0` |
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
Earlier installed evidence abbreviated Core `34b50ec9` on `patcher/stack`.
The 00726 hardware log instead reports ComfyUI `v0.38.0-12-g0d48b6032` on
`patcher/stack` with Torch 2.10.0+cu130. This runtime change is a comparison
confound relative to 00724 and must not be silently attributed to the same-grid
selector.

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

1. Milestone one remains the bounded observation ABI. Flow owns a default-off
   CPU sink; VDN #34 exposes actual raw/filtered/mapped/activated features and
   A/B/alpha norms. Observation itself is output-neutral.
2. 00719 established the actual-checkpoint operator mismatch. 00721 then ran the
   prescribed `suppress_cross_grid_temporal_taps` B arm on SM120 and materially
   changed the inherited pre-high boundary. That satisfies the design's gate to
   implement candidate C, but not its production/media acceptance gate.
3. VDN candidate C is isolated in **VDN #35**, stacked on observer #34, which is
   stacked on #33. #33 and #34 remain intact. Flow #93 carries only the paired
   selector/contract/capability/receipt logic required to invoke and verify #35.
   Flow #89 remains untouched at its one-clean-commit production-candidate
   baseline.
4. Candidate C adds independently versioned numerical leaf
   `h3_flow_partitioned_vdn_temporal_carrier_v1` API 1. Absence remains
   `native_grid_then_map_v1`; the opt-in arm is
   `destination_grid_stencil_v1`. The numerical digest binds the Flow semantic
   plan digest, diagnostic mode, actual checkpoint short-conv specification and
   physical mapping/precision policy.
5. For cross-grid temporal taps only, VDN #35 maps the **raw projected** feature
   to the receiving H3 lattice with the existing FP32 bilinear/border mapper,
   restores feature dtype, applies the trained depthwise 5x5 spatial stencil on
   that destination grid, then preserves the existing temporal weight, SiLU,
   Q/K L2, A/B/alpha, recurrence and readout path. Same-grid arithmetic remains
   the existing native batched path. No authoritative residual rows, RoPE,
   sampler latents, output geometry or decoded pixels are resampled.
6. The candidate is fail-closed: it requires paired VDN carrier API1,
   `vdn_linear_diagnostic=normal`, exact partitioned transformer context and a
   compatible checkpoint short-conv declaration/weight geometry. Native/default
   execution does not depend on candidate capability and retains the previous
   provider-cache identity.
7. The current-Core native mixed-grid test fixture accepts Core's `attention`
   keyword only for compatibility testing; production model code is unchanged.
8. Observation remains bounded to two complete heads in the first actual
   low-stage convolution block, the first mixed boundary and temporal radius ≤4.
   These bounds affect evidence completeness, not default arithmetic.

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
separate mechanisms. These papers do not by themselves establish rendered causality or overturn the
rejected output repairs. The 00721 hardware intervention, rather than the paper
review, is what authorized implementation of candidate C. The design's
whole-window SM120/media acceptance gate remains unchanged.

## Patcher dependency order

The implementation PR descriptions record exact final diagnostic SHAs. Use
those SHAs and the following dependency order, refreshing all overlays before
starting ComfyUI:

1. VDN Plus main, #33, [VDN #34](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/34)
   at `cae13fb5e8d71b93ee3134b629d23fec7c819c5b`, then
   [VDN #35](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/35) at
   `312a036139ba7422e9122ea418e13ac7cf3ee23e`.
2. Flow main, existing #89, then [Flow #93](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/93), its stacked qualification/candidate-control draft.
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

For each observed arm, set the node's `capture_boundary_witness=true`.
This is a per-run control and does **not** require an environment variable or a
ComfyUI restart. Artifacts use unique filenames under
`ComfyUI/output/h3-flow-boundary-witness`; the receipt records the active
`vdn_linear_diagnostic` mode, so A and B remain attributable without separate
process launches. Set it independently for:

* A: `vdn_linear_diagnostic=normal`, `capture_boundary_witness=true`.
* B: `vdn_linear_diagnostic=suppress_cross_grid_temporal_taps`,
  `capture_boundary_witness=true`.

The historical `H3_FLOW_BOUNDARY_WITNESS_DIR` process variable remains only as
a headless/backward-compatible fallback when no explicit node selection is
present. An explicit node OFF overrides a stale environment value.

Witness capture is a default-off observer, not a stencil selection or a
dense-attention switch. It creates one paired `.pt`/`.json` artifact for each
observed continuation low lifetime and publishes `partitioned_boundary_witness`
in Flow metrics. The `.json` contains loaded module paths/hashes, actual
precision settings, frame/head indices and tensor SHA256. Requested observation
requires the paired VDN draft before sampling.

Verify actual suppression counters and
`partitioned_vdn_linear_diagnostic_verified`; a widget label is insufficient.

### Completed A/B hardware evidence

00719 is the witness-enabled A/normal arm. Its witness is actual low-stage block
0, heads 0/55, boundary inner frame 11, and contains the cross-grid K/V temporal
taps. Offline replay on that captured input shows a material
native-filter-then-map versus map-then-destination-filter difference, while
00718 and 00719 trajectory receipts are identical, establishing witness
neutrality.

00721 is the B arm. It verified 250 suppression calls, 3000 cross-grid taps and
1,297,500 destination rows with the same 14 actual H3 evaluations and existing
sampler/history topology. The upper45 exact-restored pre-high successor pair1
changed from A `(-3.89833,-3.80939)` target cells to B
`(-0.94051,-0.32042)`, about an 81.8% reduction in vector magnitude. The
cross-grid short-conv path is therefore causally material to the inherited
pre-high discontinuity.

B is not a fix. Its final upper45 window remains displaced and its aggregate
upper45 net displacement is worse than A. No rendered A/B movies were supplied
with 00721, so background-retention/media acceptance cannot be inferred from
metrics alone.

### Candidate C result: 00722 rejected by rendered-media gate

00722 executed candidate C exactly as intended on SM120:

* `vdn_linear_diagnostic=normal`
* `vdn_temporal_carrier_policy=destination_grid_stencil_v1`
* 250 verified destination-stencil calls
* 3,000 cross-grid taps
* 2,000 mapped carriers / 865,000 mapped carrier rows
* stable numerical digest
  `82fd2c133290aece625eb432cee2764be70d73a7f0e68b17bc5b3306f7025547`
  across low/probe and final verification
* unchanged 14 actual H3 NFE / 18 logical calls / six sampler invocations /
  four history boundaries.

The actual low-stage witness records block 0, heads 0/55, K/V short conv,
destination-grid policy and the same numerical digest with zero extra
H3/provider/VAE calls.

C materially improves the internal inherited pre-high transition:
upper45 successor pair1 changes from A/native
`(-3.89833,-3.80939)` to B/suppression
`(-0.94051,-0.32042)` and C
`(-0.70003,-0.07865)`. C therefore reproduces the causal B-arm effect without
deleting the learned cross-grid contribution.

That internal improvement is **not sufficient**. User inspection of the actual
00722 rendered output reports that the same frame shift is still visible. This
fails the design's explicit whole-window rendered-media criterion. The
destination-stencil path must not be tuned further or promoted from these
metrics. Preserve VDN #35 and the 00722 artifacts as rejected diagnostic
evidence only.

### 00724 result: complete learned-linear bypass also fails rendered acceptance

00724 is a valid bypass arm:

* `vdn_linear_diagnostic=bypass_partitioned_linear`
* `vdn_temporal_carrier_policy=native_grid_then_map_v1`
* 250 fail-closed verified bypass calls / 5,340,500 video rows
* unchanged 14 actual H3 NFE / 18 logical calls / six sampler invocations /
  four history boundaries
* source-carrier audio-position execution verified and the run completed.

The bypass strongly changes internal geometry but does not remove the rendered
defect. Upper45 `exact_restored_pre_high` successor pair1 becomes
`(+2.97523,-0.24417)`, i.e. it reverses direction relative to A/native rather
than simply shrinking. After target-high the upper45 net is
`(-0.37610,+0.28953)`. Despite that numerical change, user inspection reports
that the same visible frame shift remains. This rejects the complete partitioned
VDN learned-linear complement as a sufficient cause of the rendered artifact.

The more important localization is earlier in the handoff. In 00724 the
source-low upper45 successor pair1 is only about
`(-0.12913,+0.01193)` target-equivalent cells. Immediately after the learned
34x34 -> 48x48 clean-video transfer it is about
`(+3.08880,-0.35257)`, roughly 24x larger in vector magnitude. Exact-prefix
restoration then changes pair0 as intended but leaves the successor jump at
`(+2.97523,-0.24417)`. Similar source->learned amplification occurs in the
other matched arms despite their different VDN interventions. This makes the
low->high transfer the leading current suspect.

This is not yet proof that the learned checkpoint itself is the sole cause:
the remaining alternatives are (a) the learned 3D upscaler introduces the
spatial/temporal gauge change, or (b) the broader low->high resolution
handoff plus target-high response does so even with a simple spatial transfer.

### 00725 first same-grid attempt: aborted by stale strict-smaller contract guards

The first hardware attempt at the new same-grid control did **not** test the
hypothesis. The control reached continuation low-stage setup, but Flow's
`PartitionedExactPrefixPlan` still rejected equal source/target grids before
the first continuation transformer evaluation with:

`partitioned exact-prefix plan requires a strictly smaller source grid`.

That exposed an incomplete implementation: the scheduler/stage layer allowed
the explicit same-grid diagnostic, but the published partition contract and the
paired VDN parser still encoded the old heterogeneous-only invariant.

The fix is now coordinated across both repositories:

* Flow #93 authorizes equal source/target partition geometry only when the
  explicit same-grid control is active, publishes zero prefix measure bias and
  `heterogeneous_spatial_domains=false`, and routes the same-grid case through
  the full handoff plumbing instead of returning early on equal geometry.
* VDN #35 accepts equal-grid partition contracts, preserves the ordinary
  strict/non-expanding geometry checks, and canonicalizes the equal-grid
  contract with zero measure bias and
  `heterogeneous_spatial_domains=false`.

The failed attempt produced no rendered discriminator result and must not be
counted as evidence for or against the resolution-transition hypothesis.

### 00726 same-grid hardware result and clean-state ownership defect

00726 is the first successful execution of the direct same-grid control. Runtime
receipts prove that configured progressive source intent remained 34x34 while
continuation low/probe actually ran at 48x48, target-high remained 48x48, the
handoff operator was `diagnostic:same_grid_target_identity`, it executed once,
and the actual learned-checkpoint provider was not called. The complete
partitioned VDN learned-linear bypass also remained active.

Rendered-media inspection remains authoritative: the user reports that a
**slight frame shift still happens, but it is much less prominent**. Removing
the low->high spatial transition therefore materially changes the visible
defect, but does not eliminate it.

The pair is not perfectly controlled. 00726 recorded 15 actual H3 NFE
(low9/probe2/high4) and three Spectrum forecasts, whereas 00724 recorded 14
actual H3 NFE (low8/probe2/high4) and four forecasts. 00726 also ran a newer
ComfyUI/Patcher Core. Those differences prevent assigning the entire visible
improvement to resolution alone from this single comparison.

00726 additionally exposed a concrete Flow ownership bug. The handoff noise
policy was `source_residual_patch_refinement_v1`; at equal 48x48 geometry that
residual transport itself was identity, and frame-gauge registration returned
`identity/already_aligned`. Despite that, downstream splice logic reported
`splice_clean_source=inverse_recovered` and
`splice_recovery=inverse_conditional_renoise`: it reconstructed clean video
with deterministic Gaussian noise even though the conditional state had been
built from the carried source residual.

At the 00726 split sigma (~0.8780488), that incorrect inverse strongly amplifies
the residual-vs-Gaussian difference. The synthetic clean tensor then drove a
production one-token DC bridge: one suffix token was changed with delta RMS
~0.231223 and max absolute delta ~0.455920 even though the same-grid provider
prefix was already aligned with the authoritative prefix. Consequently the
00726 `learned_native` / `exact_restored_pre_high` diagnostics are
contaminated by this reconstruction error and are not reliable evidence of a
real identity-handoff spatial jump.

Flow #93 now fixes ownership at the source:

* the clean postprocess hook retains the exact clean tensor that actually feeds
  conditional re-noising;
* downstream splice/bridge logic uses that tensor directly;
* source-residual mode fails closed if that tensor is missing instead of
  falling back to Gaussian inverse recovery;
* deterministic inverse recovery remains available only for the independent
  deterministic-noise contract;
* regression tests prove that an identity same-grid handoff cannot manufacture
  a suffix DC correction from this mismatch.

The corrected Flow head is
`53e99780c3379e2d7606c2cf0de2b0b6eb1252dc`. CI run `36739778101`
completed successfully across source-contracts and Python 3.10/3.11/3.12/3.13.
Checkpoints preserve both the pre-fix 00726 state and the CI-green correction.

The next hardware run must repeat 00726's same-grid settings. It should report
`splice_clean_source=actual_clean_postprocess` and
`splice_recovery=actual_clean_postprocess_no_inverse`. With identity same-grid
transfer and identity frame gauge, the one-token DC delta should collapse to
zero; any nonzero DC correction in that exact condition is a fail-closed
ownership violation.

### Revised next discriminator: remove the spatial grid transition itself

The previous learned-vs-bicubic proposal is **not** the required test for the
current hypothesis. The user is specifically testing whether Flow Regenerate's
low-resolution -> high-resolution stage transition is what creates the frame
shift. Replacing one 34x34 -> 48x48 resize operator with another still retains
that transition and therefore cannot answer the question directly.

Flow #93 now exposes an appended diagnostic selector
`spatial_stage_control`:

* `progressive_low_to_high` — historical/default behavior. The configured
  reduced source geometry remains active.
* `same_grid_target_control` — ignores the reduced source geometry for actual
  low/probe execution and runs low/probe directly on the caller target grid.
  The original configured reduced source geometry is retained only for handoff
  split selection, so fixed and auto-computed handoff timing remain matched.
  At the handoff, an identity provider is used through the same handoff plumbing:
  no spatial resize and no learned-checkpoint invocation occur. Residual/noise
  transport, clean postprocess, exact-prefix restoration, audio state, guidance,
  Spectrum/history ownership and target-high sampling remain in place.

The control is fail-closed. It requires
`handoff_transfer_control=learned_3d` and
`vdn_temporal_carrier_policy=native_grid_then_map_v1`; combining it with the
bicubic transfer diagnostic or destination-grid stencil is rejected.

For the next matched hardware run keep the complete **00724** workflow and
every selector frozen and change only:

* `spatial_stage_control=same_grid_target_control`

In particular keep:

* `handoff_transfer_control=learned_3d`
* `vdn_linear_diagnostic=bypass_partitioned_linear`
* `vdn_temporal_carrier_policy=native_grid_then_map_v1`
* `capture_boundary_witness=false`
* the existing source scale/width/height values unchanged.

Do **not** manually set source scale to 1.0; the ordinary configuration rejects
that by design. The diagnostic selector overrides only the effective low/probe
spatial stage internally while preserving the configured source geometry as
provenance and for handoff split selection.

The run must emit `partitioned_stage_plan` with
`same_grid_control_active=true`, equal effective source/target H/W, and the
original `configured_progressive_source_hw`. It must also emit
`partitioned_handoff_transfer_control` with operator
`diagnostic:same_grid_target_identity`, exactly one spatial-control call and
zero actual learned-checkpoint provider calls.

Interpretation is direct:

* if the visible frame shift disappears/materially collapses, the
  low-resolution -> high-resolution stage transition is causally implicated;
* if it remains with comparable character, the artifact is not caused simply by
  changing spatial resolution between low/probe and high, and the next
  discriminator must move elsewhere.

The earlier `bicubic_same_source_control` remains available as archived
diagnostic machinery but is **not** requested for this hypothesis.

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

Completed gates now include the actual SM120 feature witness, matched A/B
suppression discriminator, 00722 candidate-C execution, and 00724 complete
partitioned learned-linear bypass. Both C and bypass fail the rendered-media
gate: the visible frame shift persists. The next open gate is the direct
`same_grid_target_control`, which removes the low->high spatial resolution
transition while preserving the handoff split and downstream target-high stage.
Same-domain weighted-dense attention is intentionally deferred pending that
result because the transfer boundary is now directly implicated by repeated
source->target amplification. Audio remains independently unresolved. No first-pair metric,
source test or green CI can substitute for rendered whole-window acceptance.

Candidate C is retained only as rejected diagnostic evidence in VDN #35. For
subsequent discriminators keep #35 installed but return the Flow selector to
`native_grid_then_map_v1` so the candidate arithmetic is inert and source
topology remains frozen. Full rollback can remove #35 and the #34/#93 diagnostic
overlays to the frozen #89/#33 baseline. Sol/Continuum stay unchanged.
Production consolidation into #89 and any release remain unqualified.
