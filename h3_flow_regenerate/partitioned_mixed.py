"""Partitioned exact-prefix transformer runtime.

The sampler-facing low/probe/high scaffold is intentionally reused from the
retired Mixed-Grid experiment while this implementation is isolated on a new
experimental node.  Numerical attention is not Mixed-Grid: every target-prefix
and source-suffix row is retained, explicit physical key measures are published,
and Sol evaluates two rectangular K/V domains whose normalized results are
combined from kernel LSE.
"""
from __future__ import annotations

import copy

import torch

from .mixed_grid import (
    MIXED_GRID_KEY,
    MIXED_GRID_MEASURE_KEY,
    MixedGridPlan,
    carrier_layout,
    mixed_mod_segments,
    mixed_positions,
)
from .partitioned_prefix import (
    PARTITIONED_PREFIX_KEY,
    PARTITIONED_PREFIX_TOPOLOGY,
    PartitionedExactPrefixPlan,
    validate_partitioned_contract,
)

PARTITIONED_WRAPPER_KEY = "h3_flow_regenerate.partitioned_exact_prefix.v1"
VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
VDN_PARTITIONED_SEQUENCE_API = 3
VDN_PARTITIONED_SEQUENCE_MODE = "partitioned_attention_no_linear"
_PREPROCESS_ATTR = "attention_preprocess_v1"


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
    """Create an override that becomes active only under the partition contract."""
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
        if mask is not None or not kw.get("skip_reshape") or kw.get("skip_output_reshape"):
            raise RuntimeError("partitioned exact-prefix attention requires unmasked skip_reshape H3 attention")
        if any(t.ndim != 4 for t in (q, k, v)) or q.shape != k.shape or q.shape != v.shape:
            raise RuntimeError("partitioned exact-prefix attention requires square BHTD input before K/V split")
        if q.shape[0] != 1 or q.shape[1] != int(heads) or q.shape[-1] != 128:
            raise RuntimeError("partitioned exact-prefix attention received unsupported H3 head geometry")
        plan = validate_partitioned_contract(raw_contract, sequence_rows=int(q.shape[2]))
        prefix_start, prefix_end = plan.prefix_range
        if prefix_start <= 0 or prefix_end >= q.shape[2]:
            raise RuntimeError("partitioned exact-prefix ranges do not contain both nonvideo and suffix rows")

        # Q keeps every heterogeneous physical row. K/V are split into the dense
        # target-prefix domain and the measure-1 union of nonvideo + low-grid
        # generated suffix. This needs two rectangular Sol calls per H3 block,
        # not one call per query-domain/key-domain Cartesian product.
        other_k = torch.cat((k[:, :, :prefix_start], k[:, :, prefix_end:]), dim=2)
        other_v = torch.cat((v[:, :, :prefix_start], v[:, :, prefix_end:]), dim=2)
        prefix_k = k[:, :, prefix_start:prefix_end]
        prefix_v = v[:, :, prefix_start:prefix_end]

        from sol_h3.partitioned import partitioned_sm120_attention

        output, _lse = partitioned_sm120_attention(
            q.transpose(1, 2),
            [other_k.transpose(1, 2), prefix_k.transpose(1, 2)],
            [other_v.transpose(1, 2), prefix_v.transpose(1, 2)],
            [0.0, plan.prefix_log_key_measure],
            sink_ranges=[(0, prefix_start), (0, 0)],
        )
        metrics.increment("partitioned_attention_calls")
        metrics.increment("partitioned_sol_kernel_calls", 2)
        metrics.increment("partitioned_requested_q_rows", int(q.shape[2]))
        metrics.increment("partitioned_kernel_q_rows", 2 * int(q.shape[2]))
        metrics.increment("partitioned_kv_rows", int(q.shape[2]))
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


def _partitioned_transformer_options(options, mixed_layout, partition_contract, metrics):
    block_options = dict(options)
    block_options["minimax_h3_layout"] = mixed_layout
    block_options[PARTITIONED_PREFIX_KEY] = partition_contract
    existing_vdn = block_options.get(VDN_EXTERNAL_SEQUENCE_KEY)
    expected_vdn = _vdn_external_contract(
        validate_partitioned_contract(
            partition_contract,
            sequence_rows=int(mixed_layout.seq_len),
        )
    )
    if existing_vdn is not None and existing_vdn != expected_vdn:
        raise RuntimeError("partitioned exact-prefix found an already-owned VDN external sequence")
    block_options[VDN_EXTERNAL_SEQUENCE_KEY] = expected_vdn
    block_options.pop(MIXED_GRID_MEASURE_KEY, None)
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
    contract = options.get(MIXED_GRID_KEY)
    if contract is None:
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    import comfy.ldm.minimax.model as native

    plan = contract["plan"]
    metrics = contract["metrics"]
    inner = executor.class_obj
    if not isinstance(plan, MixedGridPlan) or len(inner.blocks) == 0:
        raise RuntimeError("partitioned exact-prefix requires a valid progressive plan")
    attention_override = options.get("optimized_attention_override")
    if getattr(attention_override, "_h3_flow_attention_override", False):
        raise RuntimeError(
            "partitioned exact-prefix does not support uniform-grid Flow Attention Lab overrides"
        )
    if tuple(x[0].shape) != (1, 24, plan.temporal, plan.source_h, plan.source_w):
        raise RuntimeError("stale partitioned exact-prefix plan does not match sampler geometry")
    if plan.prefix_noise is None or tuple(plan.prefix_noise.shape) != tuple(plan.prefix.shape):
        raise RuntimeError("partitioned exact-prefix plan is missing protected-prefix sampler noise")

    payload = dict(minimax_payload or {})
    layout = carrier_layout(
        native,
        plan,
        context.shape[1],
        x[1].shape[-1],
        payload,
    )
    payload["layout"] = layout
    va, vb, _ = layout.segments[-1]
    old_prefix = plan.prefix_t * plan.source_rows
    positions = mixed_positions(native, plan, layout)
    mixed_layout = copy.copy(layout)
    mixed_layout.position_ids = positions
    mixed_layout.seq_len = va + plan.mixed_rows
    mixed_layout.segments = [
        *layout.segments[:-1],
        (va, mixed_layout.seq_len, "video"),
    ]
    mixed_layout.signature = (
        PARTITIONED_PREFIX_KEY,
        *layout.signature,
        plan.prefix_t,
        *plan.target_hw,
    )
    keep = layout.img_pos < va
    mixed_layout.img_pos = torch.cat(
        (layout.img_pos[keep], torch.arange(va, mixed_layout.seq_len))
    )
    mixed_layout.img_update = torch.cat(
        (layout.img_update[keep], torch.ones(plan.mixed_rows, dtype=torch.bool))
    )
    partition_plan = PartitionedExactPrefixPlan(
        video_start=int(va),
        temporal=int(plan.temporal),
        prefix_t=int(plan.prefix_t),
        source_grid_h=int(plan.source_grid[0]),
        source_grid_w=int(plan.source_grid[1]),
        target_grid_h=int(plan.target_grid[0]),
        target_grid_w=int(plan.target_grid[1]),
    )
    partition_contract = partition_plan.to_contract()
    if partition_plan.sequence_rows != int(mixed_layout.seq_len):
        raise RuntimeError("partitioned exact-prefix plan does not match transformed sequence rows")

    local = dict(options)
    local["optimized_attention_override"] = make_partitioned_attention_override(
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
                # the authoritative target-grid prefix. The low-grid carrier
                # prefix is never presented to the transformer as exact context.
                aug = float(native.VISUAL_COND_TIMESTEP)
                prefix = plan.prefix.to(device=img.device, dtype=torch.float32)
                prefix_noise = plan.prefix_noise.to(
                    device=img.device,
                    dtype=torch.float32,
                )
                prefix_rows = native.patchify_video(
                    aug * prefix + (1.0 - aug) * prefix_noise
                )
                prefix_embed = inner.video_patch_proj(prefix_rows).to(img)
                img = torch.cat(
                    (img[:va], prefix_embed, img[va + old_prefix :])
                )
                metrics.increment("partitioned_transformer_calls")
                metrics.event(
                    "partitioned_exact_prefix_transformer",
                    native_sequence_rows=int(layout.seq_len),
                    partitioned_sequence_rows=int(mixed_layout.seq_len),
                    video_start=int(va),
                    temporal=int(plan.temporal),
                    prefix_t=int(plan.prefix_t),
                    source_rows_per_frame=int(plan.source_rows),
                    target_rows_per_frame=int(plan.target_rows),
                    prefix_log_key_measure=float(
                        partition_plan.prefix_log_key_measure
                    ),
                    semantic_digest=partition_contract["semantic_digest"],
                    prefix_exact_latent_resized=False,
                    prefix_target_grid_rope=True,
                    suffix_source_grid_rope=True,
                    low_suffix_real_latent=True,
                    vdn_external_sequence_api=VDN_PARTITIONED_SEQUENCE_API,
                    sol_rectangular_kv_partitions=2,
                )
            if len(img) != mixed_layout.seq_len:
                raise RuntimeError("partitioned exact-prefix transformer row count mismatch")
            if "rope" not in cached:
                cached["rope"] = native.rope_rotation_table(
                    inner.rope_freqs(positions, img.device),
                    img.dtype,
                )
            forwarded = dict(args)
            forwarded.update(
                img=img,
                layout=mixed_layout,
                rope_freqs=cached["rope"],
                mod_segments=mixed_mod_segments(
                    args["mod_segments"],
                    plan,
                    va,
                    vb,
                ),
            )
            forwarded["transformer_options"] = _partitioned_transformer_options(
                args["transformer_options"],
                mixed_layout,
                partition_contract,
                metrics,
            )
            output = (
                previous(forwarded, extra)
                if previous
                else extra["original_block"](forwarded)
            )
            result = output["img"]
            if result.shape != img.shape:
                raise RuntimeError("partitioned exact-prefix transformer returned incompatible hidden state")
            if layer == len(inner.blocks) - 1:
                # Prefix transformer outputs are context-only. Native low-grid
                # unpatchify receives only genuine low-grid suffix carrier rows.
                result = torch.cat(
                    (
                        result[:va],
                        result.new_zeros((old_prefix, result.shape[1])),
                        result[va + plan.prefix_rows :],
                    )
                )
                output = {**output, "img": result}
            return output

        return call

    for layer in range(len(inner.blocks)):
        blocks[("double_block", layer)] = wrap(
            layer,
            blocks.get(("double_block", layer)),
        )
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
    "PARTITIONED_WRAPPER_KEY",
    "VDN_PARTITIONED_SEQUENCE_API",
    "VDN_PARTITIONED_SEQUENCE_MODE",
    "make_partitioned_attention_override",
    "partitioned_diffusion_wrapper",
]
