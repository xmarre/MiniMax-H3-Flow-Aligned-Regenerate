# Exact-overlap successor continuity

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

The production receipt identifies:

- `partitioned_exact_overlap_coupled_successor_taper_v4`;
- structural and DC support of four tokens;
- `dc_temporal_weights=[1.0, 0.75, 0.5, 0.25]`;
- `suffix_dc_bridge_corrected_tokens=4`, first weight `1.0`, last weight `0.25`.

## Validation and limits

The new parameterized regression checks DC-only, centered-only, and mixed residuals, sigma values `0.0` and `0.8`, and suffixes of four and six tokens. It verifies the full successor residual, conditional-state equivalence with the same noise, input immutability, and prefix/support ownership. Eight DC-containing cases fail on the preceding one-token-DC implementation; the four centered-only cases pass.

The arithmetic correction passed 72 targeted tests and the local suite: 678 passed, 13 optional native-source tests skipped. A bounded comparison with the preceding four-token implementation produces bitwise-identical state, clean tensors, and primitive receipts in 18 cases spanning FP32, BF16, FP16, three sigma values, and one-/four-token support.

Receipt validation recognizes the coupled policy in rigid-veto, guidance-only rejection, and hardware-invalidated shadow arms. It checks the actual clean-state source, structural/DC weights, support width, endpoints, and matching transaction/transfer receipts. Pure-DC and inactive/no-op receipts retain their distinct component status. Legacy one-token and shadow-only policies keep their existing validation restrictions. Sixteen new receipt cases exercise these paths and reject mismatched support or a false no-op; 141 affected tests pass after the validator synchronization. Repository CI separately exercises the complete suite, native-source contracts, Python 3.10–3.13, packaging, and isolated-wheel loading.

These results establish the restored arithmetic and software contracts. Decoded boundary alignment and continuation tone require rendered validation with fixed conditioning, references, seed, sampler, and stage geometry. Whole-chunk overcooking remains an unresolved empirical qualification.
