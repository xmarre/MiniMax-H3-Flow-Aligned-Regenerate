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

from .partitioned_prefix import (
    PARTITIONED_PREFIX_KEY,
    PARTITIONED_PREFIX_TOPOLOGY,
    PartitionedExactPrefixPlan,
    validate_partitioned_contract,
)
from .partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    PartitionedStagePlan,
    partitioned_carrier_layout,
    partitioned_mod_segments,
    partitioned_positions,
)

PARTITIONED_WRAPPER_KEY = "h3_flow_regenerate.partitioned_exact_prefix.v1"
PARTITIONED_BLOCK_INDEX_KEY = "h3_flow_partitioned_block_index_v1"
PARTITIONED_ATTENTION_CACHE_KEY = "h3_flow_partitioned_attention_override_cache_v1"
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


def make_partitioned_attention_override(previous, metrics):
    """Create the no-VDN full-sequence partitioned attention owner.

    VDN's partition-aware forward calls Sol directly after its full-sequence
    preprocessing and grouped gathers.  This override owns the same heterogeneous
    sequence when VDN is absent and exposes preprocessing-chain metadata that VDN
    consumes before those gathers.
    """
    transforms, terminal = _preprocess_chain(previous)

    def apply_preprocess(q, k, v, heads, **kw):
        for transform in transforms:
            q, k, v = transform(q, k, v, heads=heads, **kw)
        return q, k, v

    def partition_leaf(original, q, k, v, heads, mask=None, **kw):
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

    if transforms:

        def override(original, q, k, v, heads, mask=None, **kw):
            q, k, v = apply_preprocess(q, k, v, heads, **kw)
            return partition_leaf(original, q, k, v, heads, mask=mask, **kw)

        override.attention_preprocess_v1 = (apply_preprocess, partition_leaf)
    else:
        override = partition_leaf

    override._h3_flow_partitioned_attention_override = True
    override._h3_flow_partitioned_previous = previous
    return override


def _stage_partitioned_attention_override(stage: dict, previous, metrics):
    """Return one stable partitioned provider for one sampler-stage lifetime.

    Sol history-v1 deliberately includes dense-provider object identity. Rebuilding
    the otherwise identical partition leaf for every H3 evaluation therefore
    creates a false numerical-backend transition and forces Spectrum history back
    to actual execution. The stage dictionary is owned by one low/probe sampler
    context, so caching here keeps identity stable only while that lifetime is
    active. A real inherited-provider change still creates a distinct owner.
    """
    cache = stage.get(PARTITIONED_ATTENTION_CACHE_KEY)
    if cache is None:
        cache = {}
        stage[PARTITIONED_ATTENTION_CACHE_KEY] = cache
    if not isinstance(cache, dict):
        raise RuntimeError("partitioned exact-prefix attention provider cache is malformed")

    key = id(previous)
    cached = cache.get(key)
    if cached is not None:
        if not isinstance(cached, tuple) or len(cached) != 2:
            raise RuntimeError("partitioned exact-prefix attention provider cache entry is malformed")
        cached_previous, cached_override = cached
        if cached_previous is previous and callable(cached_override):
            metrics.increment("partitioned_attention_provider_reuses")
            return cached_override
        cache.pop(key, None)

    override = make_partitioned_attention_override(previous, metrics)
    cache[key] = (previous, override)
    metrics.increment("partitioned_attention_provider_creations")
    return override


def _partitioned_transformer_options(options, partitioned_layout, partition_contract, metrics, *, block_index):
    block_options = dict(options)
    if any(key in block_options for key in _DEPRECATED_MIXED_GRID_KEYS):
        raise RuntimeError("partitioned exact-prefix found a deprecated Mixed-Grid contract")
    block_options["minimax_h3_layout"] = partitioned_layout
    block_options[PARTITIONED_PREFIX_KEY] = partition_contract
    block_options[PARTITIONED_BLOCK_INDEX_KEY] = int(block_index)
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
    stage = options.get(PARTITIONED_STAGE_KEY)
    if stage is None:
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
    if not isinstance(stage, dict):
        raise RuntimeError("partitioned exact-prefix stage contract must be a dictionary")

    import comfy.ldm.minimax.model as native

    plan = stage.get("plan")
    metrics = stage.get("metrics")
    inner = executor.class_obj
    if not isinstance(plan, PartitionedStagePlan) or metrics is None or len(inner.blocks) == 0:
        raise RuntimeError("partitioned exact-prefix requires a valid stage plan and metrics owner")
    attention_override = options.get("optimized_attention_override")
    if getattr(attention_override, "_h3_flow_attention_override", False):
        raise RuntimeError("partitioned exact-prefix does not support uniform-grid Flow Attention Lab overrides")
    if tuple(x[0].shape) != (1, 24, plan.temporal, plan.source_h, plan.source_w):
        raise RuntimeError("stale partitioned exact-prefix plan does not match sampler geometry")
    if tuple(plan.prefix_noise.shape) != tuple(plan.prefix.shape):
        raise RuntimeError("partitioned exact-prefix plan is missing protected-prefix sampler noise")

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
    positions = partitioned_positions(native, plan, layout)
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
        stage,
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
                )
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
    "PARTITIONED_ATTENTION_CACHE_KEY",
    "PARTITIONED_BLOCK_INDEX_KEY",
    "PARTITIONED_WRAPPER_KEY",
    "VDN_PARTITIONED_SEQUENCE_API",
    "VDN_PARTITIONED_SEQUENCE_MODE",
    "make_partitioned_attention_override",
    "partitioned_diffusion_wrapper",
]
