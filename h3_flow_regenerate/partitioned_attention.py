"""Attention dispatch for physical partitioned Q/K/V domains.

Sol requests retain their request-owned sparse backend. Other models use the
selected ComfyUI attention backend with the partition's additive key measure.
VDN supplies already-preprocessed, already-gathered tensors to the same call.
"""

from __future__ import annotations

import math

PARTITIONED_ATTENTION_PROVIDER_KEY = "vdn_partitioned_attention_provider_v1"
SOL_RUNTIME_KEY = "sol_h3_runtime_v1"
# Spectrum's generic numerical-attention history contract.
ATTENTION_BACKEND_HISTORY_KEY = "attention_backend_history_v1"
ATTENTION_BACKEND_RECEIPTS_KEY = "attention_backend_receipts_v1"
PARTITIONED_NATIVE_HISTORY_NAME = "h3_flow_partitioned_native_attention"
PARTITIONED_NATIVE_HISTORY_POLICY = "h3_flow_partitioned_native_attention_history_v1"


def sol_attention_selected(options) -> bool:
    metadata = options.get(SOL_RUNTIME_KEY)
    return isinstance(metadata, dict) and metadata.get("backend") == "sol"


def native_partitioned_attention(
    q,
    k,
    v,
    *,
    options,
    terminal,
    scale,
    prefix_k_range,
    prefix_log_key_measure,
    force_dense=False,
    query_position_map=None,
):
    """Apply the selected backend to one exact, possibly rectangular K/V union."""
    if (
        any(t.ndim != 3 for t in (q, k, v))
        or k.shape != v.shape
        or q.shape[1:] != k.shape[1:]
        or q.shape[0] <= 0
        or k.shape[0] <= 0
        or q.shape[1] <= 0
        or q.shape[2] <= 0
        or any(t.dtype != q.dtype or t.device != q.device for t in (k, v))
        or not q.is_floating_point()
    ):
        raise RuntimeError("partitioned attention requires matching floating Q [Tq,H,D] and KV [Tkv,H,D]")
    if not math.isfinite(float(scale)) or float(scale) != q.shape[-1] ** -0.5:
        raise RuntimeError("partitioned attention requires the native H3 attention scale")
    measure = float(prefix_log_key_measure)
    if not math.isfinite(measure):
        raise RuntimeError("partitioned attention key measure must be finite")
    if prefix_k_range is not None:
        if (
            not isinstance(prefix_k_range, tuple)
            or len(prefix_k_range) != 2
            or any(type(i) is not int for i in prefix_k_range)
            or not 0 <= prefix_k_range[0] < prefix_k_range[1] <= k.shape[0]
        ):
            raise RuntimeError("partitioned attention key-measure range is outside its gathered K/V union")
    elif measure != 0.0:
        raise RuntimeError("partitioned attention non-unit key measure requires a K/V range")

    # A broadcast key mask uses O(Tkv) storage. Rectangular/mapped local queries
    # cannot use a square sparse kernel's implicit diagonal. A zero mask also
    # requests Core BSA's existing dense route for these calls and exact queries.
    masked = force_dense or query_position_map is not None or q.shape[0] != k.shape[0]
    masked = masked or (prefix_k_range is not None and measure != 0.0)
    mask = None
    if masked:
        mask = q.new_zeros((1, 1, 1, k.shape[0]))
        if prefix_k_range is not None:
            mask[..., prefix_k_range[0] : prefix_k_range[1]] = measure

    from comfy.ldm.modules.attention import optimized_attention

    # QKV preprocessing has already run on the full physical sequence. Bypass
    # the outer attention hook to avoid applying it again or recursing into Flow.
    local = dict(options)
    local.pop("optimized_attention_override", None)
    kwargs = dict(
        mask=mask, skip_reshape=True, transformer_options=local, scale=float(scale), _inside_attn_wrapper=True
    )
    tensors = tuple(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    if terminal is None:
        out = optimized_attention(*tensors, int(q.shape[1]), **kwargs)
    else:
        out = terminal(optimized_attention, *tensors, int(q.shape[1]), **kwargs)
    expected = (1, q.shape[0], q.shape[1] * q.shape[2])
    if out.shape != expected or out.dtype != q.dtype or out.device != q.device:
        raise RuntimeError("partitioned attention backend returned incompatible shape/dtype/device")
    return out.reshape_as(q)


class PartitionedNativeHistoryPolicy:
    """Spectrum history identity for one non-Sol partitioned sampler stage.

    Every attention subcall of the stage goes through Flow's partitioned provider:
    the outer attention override is bypassed and Core BSA's block producer is
    removed, so the selected backend only sees stateless dense or sparse calls.
    The numerical route is therefore fixed by the stage geometry and diagnostics.
    """

    __slots__ = ("_runtime",)

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def identity(self) -> tuple:
        runtime = self._runtime
        plan = runtime.plan
        band = runtime.target_band
        domain = runtime.target_band_domain
        return (
            PARTITIONED_NATIVE_HISTORY_POLICY,
            int(plan.prefix_t),
            int(plan.temporal),
            (int(plan.source_h), int(plan.source_w)),
            tuple(int(value) for value in plan.target_hw),
            None if band is None else (int(band.protected_t), int(band.band_t), bool(band.same_grid_control)),
            None if domain is None else str(domain.policy),
            runtime.attention_head_t,
            str(runtime.vdn_linear_diagnostic),
            str(runtime.softmax_diagnostic),
            str(runtime.vdn_temporal_carrier_policy),
            str(runtime.prefix_transformer_context),
            str(runtime.audio_position_domain),
        )

    def __call__(self, *, layout, options, model):
        from .partitioned_stage import PARTITIONED_STAGE_KEY

        if options.get(PARTITIONED_STAGE_KEY) is not self._runtime or sol_attention_selected(options):
            return None
        return (
            *self.identity(),
            repr(getattr(layout, "signature", None)),
            getattr(layout, "seq_len", None),
            tuple(getattr(layout, "segments", ())),
            str(getattr(model, "dtype", None)),
        )

    def receipt(self) -> tuple:
        return (PARTITIONED_NATIVE_HISTORY_POLICY, self.identity())

    def accept_receipts(self, receipts) -> bool:
        expected = self.receipt()
        return bool(receipts) and all(item == expected for item in receipts)


def _record_history_receipt(options) -> None:
    from .partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStageStreamView

    sink = options.get(ATTENTION_BACKEND_RECEIPTS_KEY)
    policy = (options.get(ATTENTION_BACKEND_HISTORY_KEY) or {}).get(PARTITIONED_NATIVE_HISTORY_NAME)
    if not isinstance(sink, list) or not isinstance(policy, PartitionedNativeHistoryPolicy):
        return
    stage = options.get(PARTITIONED_STAGE_KEY)
    # Domain-uniform streams publish a read-only per-stream view of the stage.
    if isinstance(stage, PartitionedStageStreamView):
        stage = object.__getattribute__(stage, "_owner")
    if stage is not policy._runtime:
        sink.append(("h3_flow_partitioned_native_attention_foreign_stage",))
        return
    receipt = policy.receipt()
    if not sink or sink[-1] != receipt:
        sink.append(receipt)


def call_partitioned_attention(q, k, v, *, terminal, metrics, **kwargs):
    options = kwargs["transformer_options"]
    if sol_attention_selected(options):
        from sol_h3.partitioned_request import partitioned_request_attention

        result = partitioned_request_attention(q, k, v, **kwargs)
        metrics.increment("partitioned_sol_kernel_calls")
        return result
    result = native_partitioned_attention(
        q,
        k,
        v,
        options=options,
        terminal=terminal,
        scale=kwargs["scale"],
        prefix_k_range=kwargs["prefix_k_range"],
        prefix_log_key_measure=kwargs["prefix_log_key_measure"],
        force_dense=kwargs.get("force_dense", False),
        query_position_map=kwargs.get("query_position_map"),
    )
    metrics.increment("partitioned_native_attention_calls")
    _record_history_receipt(options)
    return result


def partitioned_block_extra(extra):
    """Keep Core BSA's block producer from bypassing partitioned attention.

    Its producer projects and attends internally without the additive mask or
    VDN's learned branch. The normal block attention reaches the selected BSA
    override, which retains its native masked/rectangular fallback behavior.
    Other block and attention replacements keep their existing execution.
    """
    original = extra["original_block"]

    def call(args):
        attention = args.get("attention")
        if getattr(
            attention, "__module__", None
        ) == "comfy_extras.nodes_sparse_attention" and "make_h3_block_patch.<locals>.attention" in getattr(
            attention, "__qualname__", ""
        ):
            args = dict(args)
            args.pop("attention")
        return original(args)

    return {**extra, "original_block": call}
