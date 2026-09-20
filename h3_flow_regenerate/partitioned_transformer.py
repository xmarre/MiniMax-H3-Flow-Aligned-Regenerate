"""Partitioned exact-prefix transformer runtime.

This module owns the new production-shaped heterogeneous attention contract.  It
does not consume or publish the retired Mixed-Grid contract: target-grid exact
prefix rows and source-grid generated suffix rows are carried as explicit physical
domains, and the exact physical key-measure correction is transported directly
to the Sol/VDN partitioned backend.
"""

from __future__ import annotations

import copy

import torch

from .partitioned_diagnostics import (
    PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY,
    PartitionedAudioModelTimestepContext,
    normalize_vdn_linear_diagnostic,
)
from .partitioned_prefix import (
    PARTITIONED_PREFIX_KEY,
    PARTITIONED_PREFIX_TOPOLOGY,
    PartitionedExactPrefixPlan,
    validate_partitioned_contract,
)
from .partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    PartitionedStagePlan,
    PartitionedStageRuntime,
    partitioned_carrier_layout,
    partitioned_mod_segments,
    partitioned_positions_for_runtime,
)

PARTITIONED_WRAPPER_KEY = "h3_flow_regenerate.partitioned_exact_prefix.v1"
PARTITIONED_BLOCK_INDEX_KEY = "h3_flow_partitioned_block_index_v1"
VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
VDN_PARTITIONED_SEQUENCE_API = 4
VDN_PARTITIONED_SEQUENCE_MODE = "partitioned_attention_variable_grid_linear"
_PREPROCESS_ATTR = "attention_preprocess_v1"
_DEPRECATED_MIXED_GRID_KEYS = (
    "h3_flow_mixed_grid_v1",
    "h3_flow_mixed_grid_attention_measure_v1",
)


def _vdn_external_contract(plan: PartitionedExactPrefixPlan) -> dict:
    flow = plan.to_contract()
    return {
        "api": VDN_PARTITIONED_SEQUENCE_API,
        "mode": VDN_PARTITIONED_SEQUENCE_MODE,
        "topology": PARTITIONED_PREFIX_TOPOLOGY,
        "sequence_rows": plan.sequence_rows,
        "video_start": plan.video_start,
        "temporal": plan.temporal,
        "prefix_t": plan.prefix_t,
        "source_rows_per_frame": plan.source_rows,
        "target_rows_per_frame": plan.target_rows,
        "flow_semantic_digest": flow["semantic_digest"],
    }


def _preprocess_chain(previous):
    transforms = []
    provider = previous
    seen = set()
    while provider is not None and hasattr(provider, _PREPROCESS_ATTR):
        if id(provider) in seen:
            raise RuntimeError("partitioned attention inherited a cyclic preprocessing chain")
        seen.add(id(provider))
        entry = getattr(provider, _PREPROCESS_ATTR)
        if not isinstance(entry, tuple) or len(entry) != 2 or not callable(entry[0]):
            raise RuntimeError("partitioned attention inherited malformed preprocessing metadata")
        transform, provider = entry
        transforms.append(transform)
    return tuple(transforms), provider


def _call_provider(provider, original, q, k, v, heads, mask, kw):
    if provider is None:
        return original(q, k, v, heads, mask=mask, **kw)
    return provider(original, q, k, v, heads, mask=mask, **kw)


def _provider_name(provider):
    """Mirror the Sol history-v1 provider implementation naming contract."""
    if provider is None:
        return "comfy.default"
    module = getattr(provider, "__module__", "<unknown>")
    qualname = getattr(provider, "__qualname__", type(provider).__name__)
    return f"{module}.{qualname}"


def _inherited_provider_state(previous):
    """Resolve the numerical identity that Sol history-v1 already considers stable.

    Generic attention_preprocess_v1 wrappers are allowed to be rebuilt around
    the same terminal dense provider. Sol deliberately identifies those transforms
    by implementation name and the terminal provider by name + object identity.
    Partitioned Flow must use that same boundary when deciding whether its own
    visible provider object may remain stable; raw outer-wrapper identity is too
    strict and caused the 00500/00503/00506/00507 provider transitions.
    """
    transforms, terminal = _preprocess_chain(previous)
    identity = (
        tuple(_provider_name(transform) for transform in transforms),
        _provider_name(terminal),
        id(terminal),
    )
    return identity, transforms, terminal


def _runtime_provider_state(runtime: PartitionedStageRuntime):
    transforms = runtime.attention_provider_transforms
    terminal = runtime.attention_provider_terminal
    identity = runtime.attention_provider_identity
    if not isinstance(transforms, tuple) or not all(callable(item) for item in transforms):
        raise RuntimeError("partitioned exact-prefix inherited preprocess state is malformed")
    if terminal is not None and not callable(terminal):
        raise RuntimeError("partitioned exact-prefix inherited terminal provider is malformed")
    if not isinstance(identity, tuple) or len(identity) != 3:
        raise RuntimeError("partitioned exact-prefix inherited provider identity is malformed")
    return transforms, terminal


def make_partitioned_attention_override(runtime: PartitionedStageRuntime, metrics):
    """Create one partitioned provider for one inherited numerical identity.

    The callable itself stays stable while an equivalent generic preprocess wrapper
    is reconstructed by an outer model-function wrapper. The runtime owner is
    rebound to the current call's preprocess functions before H3 executes, so
    dynamic per-call configuration is never frozen into the cached provider.
    """

    def apply_preprocess(q, k, v, heads, **kw):
        transforms, _terminal = _runtime_provider_state(runtime)
        for transform in transforms:
            q, k, v = transform(q, k, v, heads=heads, **kw)
        return q, k, v

    def partition_leaf(original, q, k, v, heads, mask=None, **kw):
        _transforms, terminal = _runtime_provider_state(runtime)
        options = kw.get("transformer_options") or {}
        raw_contract = options.get(PARTITIONED_PREFIX_KEY)
        if raw_contract is None:
            return _call_provider(terminal, original, q, k, v, heads, mask, kw)
        if any(key in options for key in _DEPRECATED_MIXED_GRID_KEYS):
            raise RuntimeError("partitioned exact-prefix refuses deprecated Mixed-Grid attention contracts")
        if mask is not None or not kw.get("skip_reshape") or kw.get("skip_output_reshape"):
            raise RuntimeError("partitioned exact-prefix attention requires unmasked skip_reshape H3 attention")
        if any(t.ndim != 4 for t in (q, k, v)) or q.shape != k.shape or q.shape != v.shape:
            raise RuntimeError("partitioned exact-prefix attention requires square BHTD input before VDN gathering")
        if q.shape[0] != 1 or q.shape[1] != int(heads) or q.shape[-1] != 128:
            raise RuntimeError("partitioned exact-prefix attention received unsupported H3 head geometry")
        plan = validate_partitioned_contract(raw_contract, sequence_rows=int(q.shape[2]))
        block_index = options.get(PARTITIONED_BLOCK_INDEX_KEY)
        if type(block_index) is not int or block_index < 0:
            raise RuntimeError("partitioned exact-prefix attention is missing its block identity")

        from sol_h3.partitioned_request import partitioned_request_attention

        q_thd = q[0].transpose(0, 1)
        k_thd = k[0].transpose(0, 1)
        v_thd = v[0].transpose(0, 1)
        output = partitioned_request_attention(
            q_thd,
            k_thd,
            v_thd,
            transformer_options=options,
            block_index=block_index,
            kind="full",
            scale=q.shape[-1] ** -0.5,
            sink_rows=plan.video_start,
            prefix_k_range=plan.prefix_range,
            prefix_log_key_measure=plan.prefix_log_key_measure,
            semantic_digest=str(raw_contract["semantic_digest"]),
        )
        metrics.increment("partitioned_attention_calls")
        metrics.increment("partitioned_sol_kernel_calls")
        metrics.increment("partitioned_requested_q_rows", int(q.shape[2]))
        metrics.increment("partitioned_kernel_q_rows", int(q.shape[2]))
        metrics.increment("partitioned_kv_rows", int(k.shape[2]))
        return output.reshape(q.shape[0], q.shape[2], -1)

    transforms, _terminal = _runtime_provider_state(runtime)
    if transforms:

        def override(original, q, k, v, heads, mask=None, **kw):
            q, k, v = apply_preprocess(q, k, v, heads, **kw)
            return partition_leaf(original, q, k, v, heads, mask=mask, **kw)

        override.attention_preprocess_v1 = (apply_preprocess, partition_leaf)
    else:
        override = partition_leaf

    override._h3_flow_partitioned_attention_override = True
    override._h3_flow_partitioned_previous = None
    override._h3_flow_partitioned_provider_identity = runtime.attention_provider_identity
    return override


def _stage_partitioned_attention_override(runtime: PartitionedStageRuntime, previous, metrics):
    """Return a stable partitioned provider across equivalent outer wrappers.

    Sol history-v1 treats generic preprocessing wrappers as part of a semantic
    chain: transform implementation names are significant, while the terminal
    dense provider keeps strict object identity. Mirror that exact distinction
    here. This preserves conservative transitions for genuine provider changes
    without turning an equivalent per-call Untwist wrapper reconstruction into a
    new partitioned numerical backend.
    """
    if not isinstance(runtime, PartitionedStageRuntime):
        raise RuntimeError("partitioned exact-prefix stage runtime owner is malformed")
    cache = runtime.attention_provider_cache
    if not isinstance(cache, dict):
        raise RuntimeError("partitioned exact-prefix attention provider cache is malformed")

    identity, transforms, terminal = _inherited_provider_state(previous)
    previous_identity = runtime.attention_provider_identity
    runtime.attention_provider_transforms = transforms
    runtime.attention_provider_terminal = terminal
    runtime.attention_provider_identity = identity

    if previous_identity is not None and previous_identity != identity:
        metrics.increment("partitioned_attention_inherited_provider_transitions")

    cached = cache.get(identity)
    if cached is not None:
        if not callable(cached) or not getattr(cached, "_h3_flow_partitioned_attention_override", False):
            raise RuntimeError("partitioned exact-prefix attention provider cache entry is malformed")
        if getattr(cached, "_h3_flow_partitioned_previous", None) is not previous:
            metrics.increment("partitioned_attention_equivalent_provider_rebindings")
        cached._h3_flow_partitioned_previous = previous
        cached._h3_flow_partitioned_provider_identity = identity
        metrics.increment("partitioned_attention_provider_reuses")
        return cached

    override = make_partitioned_attention_override(runtime, metrics)
    override._h3_flow_partitioned_previous = previous
    override._h3_flow_partitioned_provider_identity = identity
    cache[identity] = override
    metrics.increment("partitioned_attention_provider_creations")
    return override


def _partitioned_transformer_options(
    options,
    partitioned_layout,
    partition_contract,
    metrics,
    *,
    runtime: PartitionedStageRuntime,
    block_index,
):
    block_options = dict(options)
    if any(key in block_options for key in _DEPRECATED_MIXED_GRID_KEYS):
        raise RuntimeError("partitioned exact-prefix found a deprecated Mixed-Grid contract")
    block_options["minimax_h3_layout"] = partitioned_layout
    block_options[PARTITIONED_PREFIX_KEY] = partition_contract
    block_options[PARTITIONED_BLOCK_INDEX_KEY] = int(block_index)
    linear_mode = normalize_vdn_linear_diagnostic(runtime.vdn_linear_diagnostic)
    existing_linear_mode = block_options.get(PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY)
    if existing_linear_mode is not None and existing_linear_mode != linear_mode:
        raise RuntimeError("partitioned exact-prefix VDN linear diagnostic transport drifted")
    block_options[PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY] = linear_mode
    expected_vdn = _vdn_external_contract(
        validate_partitioned_contract(
            partition_contract,
            sequence_rows=int(partitioned_layout.seq_len),
        )
    )
    existing_vdn = block_options.get(VDN_EXTERNAL_SEQUENCE_KEY)
    if existing_vdn is not None and existing_vdn != expected_vdn:
        raise RuntimeError("partitioned exact-prefix found an already-owned VDN external sequence")
    block_options[VDN_EXTERNAL_SEQUENCE_KEY] = expected_vdn
    metrics.increment("partitioned_contract_publications")
    return block_options


def _audio_model_timestep_kwargs(options, kwargs):
    """Override only MiniMax-H3's inner audio timestep labels, never sampler ownership."""

    context = options.get(PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY)
    if context is None:
        return kwargs
    if not isinstance(context, PartitionedAudioModelTimestepContext):
        raise RuntimeError("partitioned audio model-timestep context is malformed")
    exact_audio_mask = kwargs.get("audio_denoise_mask")
    if not torch.is_tensor(exact_audio_mask):
        raise RuntimeError("model-timestep-only audio guidance requires the exact native audio denoise mask")
    context_audio_mask = context.audio_mask
    if not torch.is_tensor(context_audio_mask) or tuple(context_audio_mask.shape) != tuple(exact_audio_mask.shape):
        raise RuntimeError("model-timestep-only audio guidance mask geometry drifted")
    local = dict(kwargs)
    local["audio_denoise_mask"] = context_audio_mask.to(
        device=exact_audio_mask.device,
        dtype=exact_audio_mask.dtype,
    )
    context.record_call()
    return local


def partitioned_diffusion_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    """Expose exact target-prefix and low-grid suffix as explicit physical domains."""
    options = transformer_options or {}
    kwargs = _audio_model_timestep_kwargs(options, kwargs)
    runtime = options.get(PARTITIONED_STAGE_KEY)
    if runtime is None:
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    if any(key in options for key in _DEPRECATED_MIXED_GRID_KEYS):
        raise RuntimeError("partitioned exact-prefix refuses deprecated Mixed-Grid stage state")
    if not isinstance(runtime, PartitionedStageRuntime):
        raise RuntimeError("partitioned exact-prefix stage contract must be a runtime owner object")

    plan = runtime.plan
    metrics = runtime.metrics
    inner = executor.class_obj
    if not isinstance(plan, PartitionedStagePlan) or metrics is None or len(inner.blocks) == 0:
        raise RuntimeError("partitioned exact-prefix requires a valid stage plan and metrics owner")
    prefix_context = str(runtime.prefix_transformer_context)
    if prefix_context not in (
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    ):
        raise RuntimeError(f"unsupported partitioned prefix transformer context {prefix_context!r}")
    attention_override = options.get("optimized_attention_override")
    if getattr(attention_override, "_h3_flow_attention_override", False):
        raise RuntimeError("partitioned exact-prefix does not support uniform-grid Flow Attention Lab overrides")
    if tuple(x[0].shape) != (1, 24, plan.temporal, plan.source_h, plan.source_w):
        raise RuntimeError("stale partitioned exact-prefix plan does not match sampler geometry")
    if tuple(plan.prefix_noise.shape) != tuple(plan.prefix.shape):
        raise RuntimeError("partitioned exact-prefix plan is missing protected-prefix sampler noise")

    if prefix_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE:
        # Diagnostic only: keep Core's complete low/probe hidden sequence on the
        # native source grid. This intentionally does not publish Flow's
        # heterogeneous partition contract, so VDN and Sol execute their ordinary
        # uniform-grid paths. The outer exact mask still owns caller-visible carried
        # prefix values; learned handoff/high-stage behavior is unchanged.
        metrics.increment("partitioned_source_carrier_uniform_transformer_calls")
        metrics.increment("partitioned_source_carrier_uniform_prefix_frames", int(plan.prefix_t))
        metrics.event(
            "partitioned_prefix_transformer_context",
            mode=prefix_context,
            temporal=int(plan.temporal),
            prefix_t=int(plan.prefix_t),
            source_hw=(int(plan.source_h), int(plan.source_w)),
            target_hw=tuple(map(int, plan.target_hw)),
            heterogeneous_partition_contract_published=False,
            exact_target_prefix_injected_into_transformer=False,
            native_source_carrier_preserved=True,
            external_exact_prefix_owner_unchanged=True,
            diagnostic_only=True,
        )
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    import comfy.ldm.minimax.model as native

    payload = dict(minimax_payload or {})
    layout = partitioned_carrier_layout(
        native,
        plan,
        context.shape[1],
        x[1].shape[-1],
        payload,
    )
    payload["layout"] = layout
    video_start, video_end, _ = layout.segments[-1]
    carrier_prefix_rows = plan.prefix_t * plan.source_rows
    positions, position_policy = partitioned_positions_for_runtime(native, runtime, layout)
    partitioned_layout = copy.copy(layout)
    partitioned_layout.position_ids = positions
    partitioned_layout.seq_len = video_start + plan.partitioned_rows
    partitioned_layout.segments = [
        *layout.segments[:-1],
        (video_start, partitioned_layout.seq_len, "video"),
    ]
    partitioned_layout.signature = (
        PARTITIONED_PREFIX_KEY,
        *layout.signature,
        plan.prefix_t,
        *plan.target_hw,
    )
    if position_policy is not None:
        partitioned_layout.signature = (*partitioned_layout.signature, position_policy.signature)
    keep = layout.img_pos < video_start
    partitioned_layout.img_pos = torch.cat(
        (layout.img_pos[keep], torch.arange(video_start, partitioned_layout.seq_len))
    )
    partitioned_layout.img_update = torch.cat(
        (layout.img_update[keep], torch.ones(plan.partitioned_rows, dtype=torch.bool))
    )
    partition_plan = PartitionedExactPrefixPlan(
        video_start=int(video_start),
        temporal=int(plan.temporal),
        prefix_t=int(plan.prefix_t),
        source_grid_h=int(plan.source_grid[0]),
        source_grid_w=int(plan.source_grid[1]),
        target_grid_h=int(plan.target_grid[0]),
        target_grid_w=int(plan.target_grid[1]),
    )
    partition_contract = partition_plan.to_contract()
    if partition_plan.sequence_rows != int(partitioned_layout.seq_len):
        raise RuntimeError("partitioned exact-prefix plan does not match transformed sequence rows")

    local = dict(options)
    local["optimized_attention_override"] = _stage_partitioned_attention_override(
        runtime,
        local.get("optimized_attention_override"),
        metrics,
    )
    patches = dict(local.get("patches_replace") or {})
    blocks = dict(patches.get("dit") or {})
    patches["dit"] = blocks
    local["patches_replace"] = patches
    cached = {}

    def wrap(layer, previous):
        def call(args, extra):
            img = args["img"]
            if layer == 0:
                # Preserve native MiniMax-H3 visual-conditioning semantics for
                # the authoritative target-grid prefix.  The low-grid carrier
                # prefix is never presented to the transformer as exact context.
                aug = float(native.VISUAL_COND_TIMESTEP)
                prefix = plan.prefix.to(device=img.device, dtype=torch.float32)
                prefix_noise = plan.prefix_noise.to(device=img.device, dtype=torch.float32)
                prefix_rows = native.patchify_video(aug * prefix + (1.0 - aug) * prefix_noise)
                prefix_embed = inner.video_patch_proj(prefix_rows).to(img)
                img = torch.cat((img[:video_start], prefix_embed, img[video_start + carrier_prefix_rows :]))
                metrics.increment("partitioned_transformer_calls")
                metrics.event(
                    "partitioned_exact_prefix_transformer",
                    native_sequence_rows=int(layout.seq_len),
                    partitioned_sequence_rows=int(partitioned_layout.seq_len),
                    video_start=int(video_start),
                    temporal=int(plan.temporal),
                    prefix_t=int(plan.prefix_t),
                    source_rows_per_frame=int(plan.source_rows),
                    target_rows_per_frame=int(plan.target_rows),
                    prefix_log_key_measure=float(partition_plan.prefix_log_key_measure),
                    semantic_digest=partition_contract["semantic_digest"],
                    prefix_exact_latent_resized=False,
                    prefix_native_inpaint_augmentation=True,
                    prefix_visual_cond_timestep=aug,
                    prefix_target_grid_rope=True,
                    suffix_source_grid_rope=True,
                    low_suffix_real_latent=True,
                    vdn_external_sequence_api=VDN_PARTITIONED_SEQUENCE_API,
                    sol_single_union=True,
                    deprecated_mixed_grid_contract_active=False,
                    audio_position_domain=str(runtime.audio_position_domain),
                    audio_position_policy_active=(position_policy is not None),
                    audio_position_policy_signature=(position_policy.signature if position_policy is not None else None),
                    audio_position_target_w=(position_policy.target_audio_w if position_policy is not None else None),
                    audio_position_source_w=(position_policy.source_audio_w if position_policy is not None else None),
                    audio_position_temporal_digest=(position_policy.temporal_digest if position_policy is not None else None),
                    audio_position_non_audio_digest=(position_policy.non_audio_digest if position_policy is not None else None),
                    prefix_rope_position_digest=(
                        position_policy.prefix_rope_position_digest if position_policy is not None else None
                    ),
                    suffix_rope_position_digest=(
                        position_policy.suffix_rope_position_digest if position_policy is not None else None
                    ),
                    position_digest=(position_policy.position_digest if position_policy is not None else None),
                )
                if runtime.audio_position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE:
                    metrics.increment("partitioned_audio_position_source_carrier_block0_calls")
            if len(img) != partitioned_layout.seq_len:
                raise RuntimeError("partitioned exact-prefix transformer row count mismatch")
            if "rope" not in cached:
                cached["rope"] = native.rope_rotation_table(
                    inner.rope_freqs(positions, img.device),
                    img.dtype,
                )
            forwarded = dict(args)
            forwarded.update(
                img=img,
                layout=partitioned_layout,
                rope_freqs=cached["rope"],
                mod_segments=partitioned_mod_segments(
                    args["mod_segments"],
                    plan,
                    video_start,
                    video_end,
                ),
            )
            forwarded["transformer_options"] = _partitioned_transformer_options(
                args["transformer_options"],
                partitioned_layout,
                partition_contract,
                metrics,
                runtime=runtime,
                block_index=layer,
            )
            output = previous(forwarded, extra) if previous else extra["original_block"](forwarded)
            result = output["img"]
            if result.shape != img.shape:
                raise RuntimeError("partitioned exact-prefix transformer returned incompatible hidden state")
            if layer == len(inner.blocks) - 1:
                # Prefix transformer outputs are context-only.  Native low-grid
                # unpatchify receives only genuine low-grid suffix carrier rows.
                result = torch.cat(
                    (
                        result[:video_start],
                        result.new_zeros((carrier_prefix_rows, result.shape[1])),
                        result[video_start + plan.prefix_rows :],
                    )
                )
                output = {**output, "img": result}
            return output

        return call

    for layer in range(len(inner.blocks)):
        blocks[("double_block", layer)] = wrap(layer, blocks.get(("double_block", layer)))
    output = executor(
        x,
        timestep,
        context,
        local,
        minimax_payload=payload,
        **kwargs,
    )
    video = output[0].clone()
    video[:, :, : plan.prefix_t] = 0
    return [video, output[1]]


__all__ = [
    "PARTITIONED_BLOCK_INDEX_KEY",
    "PARTITIONED_WRAPPER_KEY",
    "VDN_PARTITIONED_SEQUENCE_API",
    "VDN_PARTITIONED_SEQUENCE_MODE",
    "make_partitioned_attention_override",
    "partitioned_diffusion_wrapper",
]
