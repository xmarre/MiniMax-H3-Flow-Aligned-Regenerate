"""Mixed target-prefix/source-suffix transformer boundary.

The sampler and native output projection use a regular low-grid carrier. Only
the transformer sees the larger exact prefix. No generated hidden rows are
interpolated: suffix rows enter and leave unchanged in native source order.

When the experimental attention-measure repair is enabled, Flow also publishes
an independent backend contract describing how the denser protected-prefix K/V
rows can be stratified down to the source-grid density. Q rows remain untouched,
so the authoritative target-grid prefix representation and all generated suffix
rows keep their original hidden-state topology. The existing VDN external
sequence API-2 contract is deliberately unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .geometry import unpack_streams

MIXED_GRID_KEY = "h3_flow_mixed_grid_v1"
MIXED_WRAPPER_KEY = "h3_flow_regenerate.mixed_grid.v1"
MIXED_GRID_MEASURE_KEY = "h3_flow_mixed_grid_attention_measure_v1"
MIXED_GRID_MEASURE_MODE = "prefix_kv_stratified_subsample"


@dataclass(frozen=True, slots=True)
class MixedGridPlan:
    prefix: torch.Tensor
    temporal: int
    source_h: int
    source_w: int
    prefix_noise: torch.Tensor | None = None
    attention_measure: bool = False

    @property
    def prefix_t(self):
        return int(self.prefix.shape[2])

    @property
    def target_hw(self):
        return tuple(map(int, self.prefix.shape[-2:]))

    @property
    def source_grid(self):
        return self.source_h // 2, self.source_w // 2

    @property
    def target_grid(self):
        return self.target_hw[0] // 2, self.target_hw[1] // 2

    @property
    def source_rows(self):
        return self.source_h * self.source_w // 4

    @property
    def target_rows(self):
        return self.target_hw[0] * self.target_hw[1] // 4

    @property
    def prefix_rows(self):
        return self.prefix_t * self.target_rows

    @property
    def mixed_rows(self):
        return self.prefix_rows + (self.temporal - self.prefix_t) * self.source_rows


def build_mixed_grid_plan(
    mask,
    shapes,
    internal_latent,
    sampler_noise,
    *,
    source_h,
    source_w,
    attention_measure=False,
):
    video_mask, _ = unpack_streams(mask, shapes)
    video, _ = unpack_streams(internal_latent, shapes)
    video_noise, _ = unpack_streams(sampler_noise, shapes)
    if video.ndim != 5 or video.shape[:2] != (1, 24):
        raise ValueError("mixed-grid continuation requires batch-one native H3 video")
    if tuple(video_noise.shape) != tuple(video.shape):
        raise ValueError("mixed-grid sampler noise does not match target video geometry")
    if not torch.isfinite(video_mask).all():
        raise ValueError("mixed-grid video mask must be finite")
    if not torch.isfinite(video_noise).all():
        raise ValueError("mixed-grid sampler noise must be finite")
    frames = video_mask.permute(2, 0, 1, 3, 4).reshape(video.shape[2], -1)
    protected = (frames == 0).all(1)
    generated = (frames == 1).all(1)
    prefix_t = int(protected.sum().item())
    if not 0 < prefix_t < video.shape[2]:
        raise ValueError("mixed-grid continuation requires a nonempty protected prefix and generated suffix")
    if not bool(protected[:prefix_t].all() and generated[prefix_t:].all()):
        raise ValueError("mixed-grid continuation requires contiguous whole-frame zero-prefix/one-suffix protection")
    target_h, target_w = map(int, video.shape[-2:])
    if any(n < 2 or n % 2 for n in (source_h, source_w, target_h, target_w)):
        raise ValueError("mixed-grid spatial axes must be positive and patch-safe")
    if source_h > target_h or source_w > target_w or (source_h, source_w) == (target_h, target_w):
        raise ValueError("mixed-grid source must reduce at least one axis without increasing another")
    return MixedGridPlan(
        video[:, :, :prefix_t].detach().clone(),
        int(video.shape[2]),
        source_h,
        source_w,
        video_noise[:, :, :prefix_t].detach().clone(),
        bool(attention_measure),
    )


def mixed_attention_measure_contract(plan: MixedGridPlan, *, video_start: int, sequence_rows: int):
    """Describe an optional uniform-measure K/V domain without changing VDN API 2.

    The mixed hidden stream has target-grid prefix frames and source-grid suffix
    frames. Equal softmax weight per row therefore gives every protected prefix
    frame ``target_rows/source_rows`` times as much spatial measure as a suffix
    frame. A compatible attention backend can remove that representation-density
    artifact by retaining all Q rows while selecting one target-prefix K/V row
    nearest each source-grid spatial coordinate. The suffix K/V domain is already
    at source density and is copied unchanged.
    """
    if not plan.attention_measure:
        return None
    if type(video_start) is not int or type(sequence_rows) is not int:
        raise TypeError("mixed-grid attention-measure row counts must be integers")
    if video_start <= 0 or sequence_rows != video_start + plan.mixed_rows:
        raise ValueError("mixed-grid attention-measure rows do not match the mixed sequence")
    source_grid_h, source_grid_w = plan.source_grid
    prefix_grid_h, prefix_grid_w = plan.target_grid
    expected_kv_rows = video_start + plan.temporal * plan.source_rows
    return {
        "api": 1,
        "mode": MIXED_GRID_MEASURE_MODE,
        "video_start": video_start,
        "sequence_rows": sequence_rows,
        "temporal": plan.temporal,
        "prefix_t": plan.prefix_t,
        "source_grid_h": source_grid_h,
        "source_grid_w": source_grid_w,
        "prefix_grid_h": prefix_grid_h,
        "prefix_grid_w": prefix_grid_w,
        "source_rows_per_frame": plan.source_rows,
        "prefix_rows_per_frame": plan.target_rows,
        "expected_kv_rows": expected_kv_rows,
        "exact_prefix_queries_preserved": True,
        "suffix_kv_unchanged": True,
    }


def carrier_layout(native, plan, text_len, audio_t, payload):
    """Keep native target conditioning/audio positions; give target video a low grid."""
    layout = native.PackedLayout(
        text_len, plan.temporal, *plan.target_hw, audio_t, keyframes=payload.get("keyframes"), refs=payload.get("refs")
    )
    va, _, _ = layout.segments[-1]
    origin = float(layout.position_ids[va, 0])
    frame, _ = native._frame_grid(plan.source_h, plan.source_w)
    positions = native._video_grid(plan.temporal, frame, origin)
    layout.position_ids = torch.cat((layout.position_ids[:va], positions))
    layout.seq_len = va + len(positions)
    layout.segments = [*layout.segments[:-1], (va, layout.seq_len, "video")]
    keep = layout.img_pos < va
    layout.img_pos = torch.cat((layout.img_pos[keep], torch.arange(va, layout.seq_len)))
    layout.img_update = torch.cat((layout.img_update[keep], torch.ones(len(positions), dtype=torch.bool)))
    layout.signature = (text_len, plan.temporal, plan.source_h, plan.source_w, audio_t)
    return layout


def mixed_positions(native, plan, layout):
    va, _, _ = layout.segments[-1]
    origin = float(layout.position_ids[va, 0])
    frame, _ = native._frame_grid(*plan.target_hw)
    prefix = native._video_grid(plan.temporal, frame, origin)[: plan.prefix_rows]
    # Slice the globally constructed low timeline; never restart k % 5.
    suffix = layout.position_ids[va + plan.prefix_t * plan.source_rows :]
    return torch.cat((layout.position_ids[:va], prefix, suffix))


def mixed_mod_segments(segments, plan, va, vb):
    result = []
    for start, stop, row in segments:
        if stop <= va:
            result.append((start, stop, row))
        elif (start, stop) == (va, vb):
            if not torch.is_tensor(row) or row.ndim != 1 or row.numel() != vb - va:
                raise RuntimeError("mixed-grid prefix requires native per-video-row timestep indices")
            old_prefix = plan.prefix_t * plan.source_rows
            if not bool((row[:old_prefix] == row[0]).all()):
                raise RuntimeError("mixed-grid protected prefix has inconsistent timestep labels")
            rows = torch.cat((row[:1].expand(plan.prefix_rows), row[old_prefix:]))
            result.append((va, va + plan.mixed_rows, rows))
        else:
            raise RuntimeError("mixed-grid requires target video to be the final native segment")
    return result


def mixed_diffusion_wrapper(executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
    options = transformer_options or {}
    contract = options.get(MIXED_GRID_KEY)
    if contract is None:
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    import comfy.ldm.minimax.model as native

    plan = contract["plan"]
    metrics = contract["metrics"]
    inner = executor.class_obj
    if not isinstance(plan, MixedGridPlan) or len(inner.blocks) == 0:
        raise RuntimeError("mixed-grid requires a valid plan and native transformer blocks")
    attention_override = options.get("optimized_attention_override")
    if getattr(attention_override, "_h3_flow_attention_override", False):
        raise RuntimeError("mixed-grid continuation does not support uniform-grid Flow Attention Lab overrides")
    if tuple(x[0].shape) != (1, 24, plan.temporal, plan.source_h, plan.source_w):
        raise RuntimeError("stale mixed-grid plan does not match sampler geometry")
    if plan.prefix_noise is None or tuple(plan.prefix_noise.shape) != tuple(plan.prefix.shape):
        raise RuntimeError("mixed-grid plan is missing the native protected-prefix sampler noise")
    payload = dict(minimax_payload or {})
    layout = carrier_layout(native, plan, context.shape[1], x[1].shape[-1], payload)
    payload["layout"] = layout
    va, vb, _ = layout.segments[-1]
    old_prefix = plan.prefix_t * plan.source_rows
    positions = mixed_positions(native, plan, layout)
    mixed_layout = copy.copy(layout)
    mixed_layout.position_ids = positions
    mixed_layout.seq_len = va + plan.mixed_rows
    mixed_layout.segments = [*layout.segments[:-1], (va, mixed_layout.seq_len, "video")]
    # Other providers must not interpret this as uniform target video geometry.
    mixed_layout.signature = ("h3_flow_mixed_grid_v1", *layout.signature, plan.prefix_t, *plan.target_hw)
    keep = layout.img_pos < va
    mixed_layout.img_pos = torch.cat((layout.img_pos[keep], torch.arange(va, mixed_layout.seq_len)))
    mixed_layout.img_update = torch.cat((layout.img_update[keep], torch.ones(plan.mixed_rows, dtype=torch.bool)))
    measure_contract = mixed_attention_measure_contract(
        plan,
        video_start=va,
        sequence_rows=mixed_layout.seq_len,
    )
    local = dict(options)
    patches = dict(local.get("patches_replace") or {})
    blocks = dict(patches.get("dit") or {})
    patches["dit"] = blocks
    local["patches_replace"] = patches
    cached = {}

    def wrap(layer, previous):
        def call(args, extra):
            img = args["img"]
            if layer == 0:
                # Match native MiniMaxH3.scale_latent_inpaint exactly for the
                # protected target-grid context: the authoritative clean prefix
                # stays at target geometry, but the model sees H3's 0.999
                # visual-conditioning augmentation with the caller's original
                # target-grid sampler noise. The low carrier prefix never becomes
                # conditioning.
                aug = float(native.VISUAL_COND_TIMESTEP)
                prefix = plan.prefix.to(device=img.device, dtype=torch.float32)
                prefix_noise = plan.prefix_noise.to(device=img.device, dtype=torch.float32)
                prefix_rows = native.patchify_video(aug * prefix + (1.0 - aug) * prefix_noise)
                prefix_embed = inner.video_patch_proj(prefix_rows).to(img)
                img = torch.cat((img[:va], prefix_embed, img[va + old_prefix :]))
                metrics.increment("mixed_grid_actual_calls")
                metrics.event(
                    "mixed_grid_transformer",
                    mixed_video_rows=plan.mixed_rows,
                    prefix_video_rows=plan.prefix_rows,
                    suffix_video_rows=(plan.temporal - plan.prefix_t) * plan.source_rows,
                    native_sequence_rows=layout.seq_len,
                    mixed_sequence_rows=mixed_layout.seq_len,
                    vdn_external_sequence_mode="dense_gate_no_linear",
                    vdn_external_sequence_api=2,
                    attention_measure_requested=bool(plan.attention_measure),
                    attention_measure_mode=(MIXED_GRID_MEASURE_MODE if measure_contract is not None else "disabled"),
                    attention_measure_prefix_kv_rows_before=plan.prefix_rows,
                    attention_measure_prefix_kv_rows_after=plan.prefix_t * plan.source_rows,
                    attention_measure_kv_rows_before=mixed_layout.seq_len,
                    attention_measure_kv_rows_after=(
                        layout.seq_len if measure_contract is not None else mixed_layout.seq_len
                    ),
                    attention_measure_prefix_density_ratio=plan.target_rows / plan.source_rows,
                    prefix_exact_latent_resized=False,
                    prefix_native_inpaint_augmentation=True,
                    prefix_visual_cond_timestep=aug,
                    low_suffix_real_latent=True,
                )
            if len(img) != mixed_layout.seq_len:
                raise RuntimeError("mixed-grid transformer row count mismatch")
            if "rope" not in cached:
                cached["rope"] = native.rope_rotation_table(inner.rope_freqs(positions, img.device), img.dtype)
            forwarded = dict(args)
            forwarded.update(
                img=img,
                layout=mixed_layout,
                rope_freqs=cached["rope"],
                mod_segments=mixed_mod_segments(args["mod_segments"], plan, va, vb),
            )
            block_options = dict(args["transformer_options"])
            if "vdn_h3_external_sequence_v1" in block_options:
                raise RuntimeError("mixed-grid found an already-owned VDN external sequence")
            block_options["vdn_h3_external_sequence_v1"] = {
                "api": 2,
                "mode": "dense_gate_no_linear",
                "topology": "mixed_grid_low_suffix",
                "native_sequence_rows": layout.seq_len,
                "sequence_rows": mixed_layout.seq_len,
                "video_start": va,
                "temporal": plan.temporal,
                "prefix_t": plan.prefix_t,
                "source_rows_per_frame": plan.source_rows,
                "prefix_rows_per_frame": plan.target_rows,
            }
            if measure_contract is not None:
                if MIXED_GRID_MEASURE_KEY in block_options:
                    raise RuntimeError("mixed-grid found an already-owned attention-measure contract")
                block_options[MIXED_GRID_MEASURE_KEY] = measure_contract
            forwarded["transformer_options"] = block_options
            output = previous(forwarded, extra) if previous else extra["original_block"](forwarded)
            result = output["img"]
            if result.shape != img.shape:
                raise RuntimeError("mixed-grid transformer returned incompatible hidden state")
            if layer == len(inner.blocks) - 1:
                # Discard exact-prefix transformer output. The low carrier is
                # masked; only genuine suffix hidden rows reach native unpatchify.
                result = torch.cat(
                    (result[:va], result.new_zeros((old_prefix, result.shape[1])), result[va + plan.prefix_rows :])
                )
                output = {**output, "img": result}
            return output

        return call

    for layer in range(len(inner.blocks)):
        blocks[("double_block", layer)] = wrap(layer, blocks.get(("double_block", layer)))
    output = executor(x, timestep, context, local, minimax_payload=payload, **kwargs)
    video = output[0].clone()
    video[:, :, : plan.prefix_t] = 0
    return [video, output[1]]
