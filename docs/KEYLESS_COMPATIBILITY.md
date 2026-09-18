# MiniMax H3 Keyless compatibility

Flow-Aligned-Regenerate treats MiniMax H3 Keyless as a distinct model architecture rather than as a QKV checkpoint with a missing key tensor.

## Model admission

A model that advertises `minimax_h3_keyless_contract_v1` is accepted only when the public contract and live module topology agree with `h3_keyless_core50_v1`:

- 50 main transformer blocks use packed `qv_proj` weights with shape `[14336, 5376]`;
- the two token-refiner blocks remain native QKV;
- routing is derived from projected V through its own RMSNorm/RoPE policy;
- retrieval uses the raw projected V view;
- main-block attention exposes no compatibility `qkv_proj` alias.

The outer Comfy model must still derive from the native `MiniMaxH3` base model. Explicit but malformed Keyless advertisements fail closed instead of being treated as ordinary H3.

## Production Target Input and trajectory paths

The released Progressive Target Input path is model-opaque at the Flow layer. Flow controls sampler geometry, trajectory capture, handoff state and exact-mask behavior; it does not manufacture K tensors or rewrite attention internals.

For exact protected continuation, the released conservative Target Input path stays on the target grid for one sampler lifetime. That path remains the correctness fallback when a more specialized attention composition cannot prove Keyless semantics.

Flow trajectory capture stores model outputs rather than Q/K/V internals. This structural property does not establish decoded-media parity between a distilled Keyless checkpoint and its QKV teacher.

## Row-domain rule

Keyless sparse/restricted attention has one non-negotiable ownership rule: select the physical V rows once, then derive routing from those exact selected rows with the same aligned routing positions and key measure.

A compatibility layer must not independently select a pseudo-key domain and a V domain. A backend that only understands ordinary K/V reduction must fail closed rather than interpret raw V as K.

## Experimental target-sparse path

Flow's own target-sparse transformation selects hidden rows before the attention projections. At that boundary it does not create an independent K domain: Q, V and the value-derived routing view are all produced from the same reduced hidden sequence.

That structural fact does not authorize a QKV-only external VDN or Sol owner. Any installed attention owner must still advertise and implement the Keyless value-domain contract before this combination can be treated as supported.

## Deprecated Mixed-Grid compatibility

The deprecated Mixed-Grid node is retained only for serialized-workflow compatibility.

Its ordinary heterogeneous hidden sequence is not reinterpreted by Flow as a K/V operation. The optional legacy attention-measure repair is different: it reduces projected K/V rows to compensate for target-prefix carrier density. That K/V-only post-projection contract cannot express the Keyless rule above, so Flow rejects `attention_measure=True` on a validated Keyless model.

No new Keyless production path should be built on this deprecated repair.

## Partitioned exact-prefix continuation

The partitioned exact-prefix candidate uses VDN external-sequence API 4 and a Sol single-union request. Its current contract is explicitly QKV-shaped: VDN gathers projected K/V domains, transports physical key measure, and retains a learned linear complement that consumes raw Q/K/V.

Canonical Keyless requires the corresponding restricted domain to be expressed as selected V plus aligned routing-position/measure ownership. The current Keyless Sol reference bridge also rejects explicit row-domain execution until that provider is reviewed.

Therefore a validated Keyless model fails the partitioned preflight before any low/probe/high sampler lifetime begins. Flow then uses the released exact target-grid fallback. This preserves correctness and avoids a mid-sampler numerical-path switch.

Partitioned Keyless acceleration can be enabled only after the active Sol/VDN stack provides a reviewed value-domain provider that:

1. accepts Q + selected V + routing specification rather than Q/K/V;
2. derives routing after the single authoritative V selection;
3. carries aligned value-domain positions and additive key measure;
4. defines a Keyless-specific replacement for any learned branch that depends on teacher K;
5. publishes distinct backend/history/calibration identity so QKV evidence cannot qualify Keyless execution.

## Validation boundary

Unit and source-contract tests can establish admission, fail-closed routing and metadata ownership. They do not establish BF16 distillation quality, INT8 ConvRot quality, CUDA kernel parity, Spectrum forecast quality, reference/audio behavior, or decoded-media parity. Those remain empirical release gates for the Keyless model and its attention providers.
