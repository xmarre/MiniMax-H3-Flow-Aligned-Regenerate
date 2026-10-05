# Exact-overlap successor continuity

> Status: historical. The coupled four-token contract described below is retired.
> Production now emits `partitioned_exact_overlap_dc_only_v5` within
> `dense_drift_handoff_plus_one_token_dc_v6`: one-token, per-channel DC support
> (`PARTITIONED_EXACT_OVERLAP_PRODUCTION_WEIGHTS == (1.0,)`). Multi-token
> production weights are rejected.

A learned 3D transfer produces a coherent target-grid prefix and suffix. Exact-prefix continuation replaces the learned prefix with the caller-owned authoritative prefix. The last overlapping token therefore measures a representation residual that must be reconciled before target-high sampling.

Let `D = E_p - L_p`, where `E_p` is the authoritative last prefix token and `L_p` is the learned last prefix token. Decompose `D = S + M`, where `M` is the per-channel spatial mean and `S` has zero spatial mean.

## Shared temporal support

The composed exact-overlap bridge uses one weight sequence for both components:

```text
weights = (1.0, 0.75, 0.5, 0.25)
C_j = weights[j] * (S + M)
```

The first generated suffix token receives the full residual. Its transition from the authoritative prefix equals the learned provider's original prefix-to-suffix transition. Every following transition, including the return to the untouched suffix, adds only `-0.25 * D`.

Using four-token support for `S` and one-token support for `M` violates that successor bound. The first suffix-to-suffix transition then adds `-0.25 * S - M`. A DC-only residual drops its entire correction at once. Preserving the first transition or checking only zero-mean corrections does not detect this defect.

The scheduler passes the same weights to the representation and DC primitives. The composed helper has no independent DC-support argument. The standalone DC bridge retains its existing one-token default for callers outside the composed exact-overlap path.

A spatially constant offset in latent channels is not a verified output-brightness-only control. This decomposition provides an algebraic continuity contract; it does not establish that independently changing one component preserves decoded geometry or improves tone.

## State and ownership

The clean correction maps onto the existing conditional state as:

```text
state_after = state_before + (1 - sigma) * (clean_after - clean_before)
```

Only supported suffix rows are modified. The original noise realization, authoritative prefix, and suffix outside the support remain intact. The correction restores the bridge arithmetic at commit `e27b301d`; it preserves target-high video-overlap ownership and exact final prefix restoration.

The retired v4 production receipt identified:

- `partitioned_exact_overlap_coupled_successor_taper_v4`;
- structural and DC support of four tokens;
- `dc_temporal_weights=[1.0, 0.75, 0.5, 0.25]`;
- `suffix_dc_bridge_corrected_tokens=4`, first weight `1.0`, last weight `0.25`.

## Validation and limits

The new parameterized regression checks DC-only, centered-only, and mixed residuals, sigma values `0.0` and `0.8`, and suffixes of four and six tokens. It verifies the full successor residual, conditional-state equivalence with the same noise, input immutability, and prefix/support ownership. Eight DC-containing cases fail on the preceding one-token-DC implementation; the four centered-only cases pass.

The arithmetic correction passed 72 targeted tests and the local suite: 678 passed, 13 optional native-source tests skipped. A bounded comparison with the preceding four-token implementation produces bitwise-identical state, clean tensors, and primitive receipts in 18 cases spanning FP32, BF16, FP16, three sigma values, and one-/four-token support.

Receipt validation recognizes the coupled policy in rigid-veto, guidance-only rejection, and hardware-invalidated shadow arms. It checks the actual clean-state source, structural/DC weights, support width, endpoints, and matching transaction/transfer receipts. Pure-DC and inactive/no-op receipts retain their distinct component status. Legacy one-token and shadow-only policies keep their existing validation restrictions. Eighteen new receipt cases exercise these paths and reject mismatched support, missing transfer evidence, or a false no-op; 143 affected tests pass after the validator synchronization. Repository CI separately exercises the complete suite, native-source contracts, Python 3.10–3.13, packaging, and isolated-wheel loading.

These results establish the restored arithmetic and software contracts. Decoded boundary alignment and continuation tone require rendered validation with fixed conditioning, references, seed, sampler, and stage geometry. Whole-chunk overcooking remains an unresolved empirical qualification.

## Failed rendered boundary qualification: 00958

The rendered report for 00958 identifies a returning frame shift/zoom. This fails boundary acceptance despite the restored coupled bridge, active four-token target-high video overlap, and exact final caller prefix. Exact-prefix identity and first-transition arithmetic do not establish continuity of the generated decoder window.

The third measured latent transition changes from vertical motion of approximately `-0.232` cells in the upper-45% ROI and `+0.124` cells over the full frame before high refinement to `-0.900` and `+0.939` respectively in the first actual high H3 prediction, before Flow correction. The same prediction after Flow measures approximately `-0.900` and `+0.939`. This locates the change in these motion statistics before the direct Flow correction and before high-stage forecast calls. It does not establish that the wrapped model, attention adapters, low/probe history, handoff state, or high inpaint conditioning individually caused the rendered defect. Opposing ROI motion does not justify a global translation or zoom correction.

Decode Context explicitly supplied five real future latents. All 22 sampled plan-aligned duplicate-overlap frames, including the five-frame right-context tail, have zero sampled RGB difference RMS. These measurements use images sampled to a 192-pixel long side; they are not a full-resolution byte comparison or a guarantee of generated-suffix quality. Video Seam Auto kept the native boundary and replaced zero frames. Decoded affine scale estimates have low confidence and do not determine the reported zoom magnitude. The pre-patch luma standard deviation rises by about 6% during the first four retained frames, but composition and motion can also change that statistic.

The recorded prompts, geometry, strengths and handoff innovation seed match 00947, but the actual carried-prefix tensor hash differs. Conditioning descriptor hashes do not hash embedded conditioning tensors. These runs therefore do not provide a matched source-state A/B, and they cannot attribute the returning defect to the later read-only receipts or test-only overlay compatibility change.

The high-stage prediction trace now observes up to five generated tokens, covering every generated token in the first native decoder window for a phase-aligned boundary. `high_boundary_prediction_v2` passes the actual available width to the trajectory measurement; retaining its four-step default would still omit the fifth transition. The call bound stays at 16, and the trace modifies no sampler operands or outputs and adds no H3/VAE calls. A regression case places all motion and all before/after Flow change in the fifth token; a different sixth token remains outside the observation. Short suffixes retain their actual available width. The production bridge support, masks, strengths and steps are unchanged. This is observation coverage, not a rendered boundary fix.

The fifth-token regression fails against the preceding trace; the two short-suffix cases already pass. All 48 affected trace, boundary-ownership and guidance tests pass after extending the observation, with Ruff and formatting checks passing.

A controlled high-overlap comparison must change only the existing `video_guided_overlap_tokens` control while retaining the coupled bridge and the same accepted prefix/handoff state. Setting that control to zero tests the released-prefix/final-restore interaction; it is an experimental control, not an established correction, and can bring back the earlier boundary hitch. Perceptual geometry attribution still requires the existing rendered boundary clip or a matched rendered comparison.
