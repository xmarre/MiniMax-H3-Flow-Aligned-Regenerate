"""MiniMax-H3 VAE tile diagnostics.

The production decoder uses independent spatial ViT tiles. This module keeps
experiments call-scoped: it can vary tile geometry, or keep Core's native tile
geometry while remapping only the spatial rotary-position coordinates onto the
full latent canvas. All temporary VAE mutations are restored in ``finally``.
"""

from __future__ import annotations

import math
import threading
import types

import torch

_DECODE_LOCK = threading.Lock()
_LOG = logging.getLogger(__name__)
_MISSING = object()


def _h3_video_vae_model(vae: object):
    model = getattr(vae, "first_stage_model", None)
    required = ("tiling", "tile_size", "tile_overlap_min", "vae_ratio", "split_tiles")
    if model is None or not all(hasattr(model, name) for name in required):
        raise ValueError("expected a ComfyUI MiniMax-H3 video VAE")
    if model.__class__.__name__ != "MiniMaxH3VideoVAE":
        raise ValueError(
            "MiniMax-H3 VAE diagnostics are restricted to ComfyUI "
            f"MiniMaxH3VideoVAE, got {model.__class__.__name__}"
        )
    return model


def _validate_samples(samples: dict[str, object]) -> torch.Tensor:
    if not isinstance(samples, dict) or not torch.is_tensor(samples.get("samples")):
        raise ValueError("samples must be a LATENT dictionary containing a tensor")
    latent = samples["samples"]
    if latent.ndim != 5 or int(latent.shape[1]) != 24:
        raise ValueError("MiniMax-H3 video latent must be [B,24,T,H,W]")
    return latent


def _validate_profile(tile_size: int, tile_overlap: int, vae_ratio: int) -> tuple[int, int]:
    tile_size = int(tile_size)
    tile_overlap = int(tile_overlap)
    vae_ratio = int(vae_ratio)
    if vae_ratio <= 0:
        raise ValueError("MiniMax-H3 VAE spatial ratio must be positive")
    if tile_size < 256 or tile_size > 512 or tile_size % vae_ratio:
        raise ValueError(
            f"tile_size must be 256..512 and divisible by the VAE ratio ({vae_ratio})"
        )
    if tile_overlap < 64 or tile_overlap >= tile_size or tile_overlap % vae_ratio:
        raise ValueError(
            f"tile_overlap must be >=64, < tile_size, and divisible by {vae_ratio}"
        )
    return tile_size, tile_overlap


def _tile_boundaries(model: object, height: int, width: int) -> tuple[list[int], list[int]]:
    y_idx, _y_len, _y_overlap = model.split_tiles(int(height))
    x_idx, _x_len, _x_overlap = model.split_tiles(int(width))
    return [int(x) for x in x_idx[1:]], [int(y) for y in y_idx[1:]]


def _edge_ratio(
    images: torch.Tensor,
    *,
    axis: int,
    position: int,
    radius: int = 12,
) -> float:
    # Never cast/copy the complete decoded video for diagnostics: that can be
    # multiple GiB at Continuum resolutions. Promote only adjacent pixel lines.
    rgb = images[..., :3]
    if axis == 2:
        length = int(rgb.shape[2])
        if not 1 <= position < length:
            return math.nan
        seam = (rgb[:, :, position].float() - rgb[:, :, position - 1].float()).abs().mean()
        candidates = []
        for x in range(max(1, position - radius), min(length, position + radius + 1)):
            if x == position:
                continue
            candidates.append(
                (rgb[:, :, x].float() - rgb[:, :, x - 1].float()).abs().mean()
            )
    elif axis == 1:
        length = int(rgb.shape[1])
        if not 1 <= position < length:
            return math.nan
        seam = (rgb[:, position].float() - rgb[:, position - 1].float()).abs().mean()
        candidates = []
        for y in range(max(1, position - radius), min(length, position + radius + 1)):
            if y == position:
                continue
            candidates.append(
                (rgb[:, y].float() - rgb[:, y - 1].float()).abs().mean()
            )
    else:
        raise ValueError("tile seam axis must be image H or W")
    if not candidates:
        return math.nan
    baseline = torch.stack(candidates).median()
    denominator = float(baseline.item())
    return float(seam.item()) / denominator if denominator > 0.0 else math.inf


def _format_seam_report(
    images: torch.Tensor,
    x_seams: list[int],
    y_seams: list[int],
) -> str:
    x = [(pos, _edge_ratio(images, axis=2, position=pos)) for pos in x_seams]
    y = [(pos, _edge_ratio(images, axis=1, position=pos)) for pos in y_seams]

    def fmt(items):
        return (
            ",".join(
                f"{pos}:{ratio:.3f}x" if math.isfinite(ratio) else f"{pos}:inf"
                for pos, ratio in items
            )
            or "none"
        )

    return f"x_seams=[{fmt(x)}] y_seams=[{fmt(y)}]"


def _flatten_core_video_output(images: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(images) or images.ndim not in (4, 5):
        raise RuntimeError(
            "MiniMax-H3 VAE returned unexpected decoded shape "
            f"{getattr(images, 'shape', None)}"
        )
    if images.ndim == 5:
        # Match Core VAEDecode: video VAEs may return [B,T,H,W,C], while IMAGE
        # consumers receive a single leading frame/batch dimension.
        images = images.reshape(
            -1,
            images.shape[-3],
            images.shape[-2],
            images.shape[-1],
        )
    return images


def _emit_report(report: str) -> None:
    # The STRING output is useful in workflows, but a diagnostic must remain
    # observable when that output is not wired.
    print(f"[MiniMax-H3 VAE diagnostic] {report}")


def _restore_instance_attribute(obj: object, name: str, previous: object) -> None:
    if previous is _MISSING:
        with contextlib.suppress(AttributeError):
            delattr(obj, name)
    else:
        setattr(obj, name, previous)


def _remap_spatial_position_ids(
    img_ids: torch.Tensor,
    *,
    image_token_count: int,
    local_hw: tuple[int, int],
    full_hw: tuple[int, int],
    spatial_offsets: list[tuple[int, int]],
) -> torch.Tensor:
    """Map tile-local Y/X IDs to the full latent canvas, preserving T/suffix IDs."""
    if img_ids.ndim != 3 or int(img_ids.shape[-1]) != 3:
        raise RuntimeError("MiniMax-H3 decoder position IDs must be [B,S,3]")

    batch = int(img_ids.shape[0])
    if len(spatial_offsets) != batch:
        raise RuntimeError(
            "MiniMax-H3 global-position diagnostic batch/offset count mismatch"
        )

    local_h, local_w = map(int, local_hw)
    full_h, full_w = map(int, full_hw)
    if min(local_h, local_w, full_h, full_w) <= 0:
        raise RuntimeError("MiniMax-H3 decoder spatial dimensions must be positive")
    spatial_tokens = local_h * local_w
    if image_token_count <= 0 or image_token_count % spatial_tokens:
        raise RuntimeError("MiniMax-H3 decoder image-token geometry is inconsistent")
    if int(img_ids.shape[1]) < image_token_count:
        raise RuntimeError("MiniMax-H3 decoder position IDs are shorter than image tokens")

    local_t = image_token_count // spatial_tokens
    adjusted = img_ids.clone()
    image_ids = adjusted[:, :image_token_count].reshape(
        batch,
        local_t,
        local_h,
        local_w,
        3,
    )

    offsets = torch.tensor(
        spatial_offsets,
        dtype=torch.float32,
        device=img_ids.device,
    )
    y = torch.arange(local_h, dtype=torch.float32, device=img_ids.device) + 0.5
    x = torch.arange(local_w, dtype=torch.float32, device=img_ids.device) + 0.5
    y = 2.0 * ((y[None, :] + offsets[:, 0:1]) / float(full_h)) - 1.0
    x = 2.0 * ((x[None, :] + offsets[:, 1:2]) / float(full_w)) - 1.0

    image_ids[..., 1] = y[:, None, :, None].to(image_ids.dtype)
    image_ids[..., 2] = x[:, None, None, :].to(image_ids.dtype)
    return adjusted



def _globalize_spatial_token_ids(
    img_ids: torch.Tensor,
    *,
    tile_shape: tuple[int, int, int],
    full_hw: tuple[int, int],
    origins: list[tuple[int, int]],
) -> torch.Tensor:
    """Replace tile-local Y/X RoPE coordinates with full-frame coordinates."""
    if img_ids.ndim != 3 or int(img_ids.shape[-1]) != 3:
        raise RuntimeError("MiniMax-H3 decoder token ids must be [B,S,3]")
    tile_t, tile_h, tile_w = map(int, tile_shape)
    full_h, full_w = map(int, full_hw)
    if min(tile_t, tile_h, tile_w, full_h, full_w) <= 0:
        raise RuntimeError("MiniMax-H3 global-position dimensions must be positive")
    if int(img_ids.shape[0]) != len(origins):
        raise RuntimeError("MiniMax-H3 global-position batch/origin count mismatch")
    patch_tokens = tile_t * tile_h * tile_w
    if int(img_ids.shape[1]) < patch_tokens:
        raise RuntimeError("MiniMax-H3 global-position token count is shorter than the tile volume")

    out = img_ids.clone()
    t = (torch.arange(0.5, tile_t, device=img_ids.device, dtype=img_ids.dtype) / tile_t) * 2.0 - 1.0
    for batch_index, (origin_y, origin_x) in enumerate(origins):
        origin_y, origin_x = int(origin_y), int(origin_x)
        if origin_y < 0 or origin_x < 0 or origin_y + tile_h > full_h or origin_x + tile_w > full_w:
            raise RuntimeError(
                "MiniMax-H3 global-position tile origin is outside the full latent grid: "
                f"origin={(origin_y, origin_x)} tile={(tile_h, tile_w)} full={(full_h, full_w)}"
            )
        y = (
            (origin_y + torch.arange(0.5, tile_h, device=img_ids.device, dtype=img_ids.dtype))
            / full_h
        ) * 2.0 - 1.0
        x = (
            (origin_x + torch.arange(0.5, tile_w, device=img_ids.device, dtype=img_ids.dtype))
            / full_w
        ) * 2.0 - 1.0
        coords = torch.stack(torch.meshgrid(t, y, x, indexing="ij"), dim=-1).reshape(-1, 3)
        out[batch_index, :patch_tokens] = coords
    return out


def _install_instance_method(obj: object, name: str, fn):
    namespace = getattr(obj, "__dict__", {})
    had_instance_value = name in namespace
    previous_instance_value = namespace.get(name)
    setattr(obj, name, types.MethodType(fn, obj))

    def restore() -> None:
        if had_instance_value:
            setattr(obj, name, previous_instance_value)
        else:
            try:
                delattr(obj, name)
            except AttributeError:
                pass

    return restore


@contextlib.contextmanager
def _global_spatial_position_patch(model: object):
    """Use full-frame spatial RoPE coordinates while retaining Core tile geometry.

    Core decodes every spatial tile independently and the ViT decoder constructs
    normalized Y/X token coordinates from the tile's local H/W. This bounded
    diagnostic keeps the exact Core split/blend path but makes overlapping
    physical locations receive identical full-frame Y/X coordinates.
    """
    decoder = getattr(model, "decoder", None)
    pos_embed = getattr(decoder, "pos_embed", None)
    if decoder is None or pos_embed is None:
        raise RuntimeError("MiniMax-H3 global-position diagnostic requires the Core ViT decoder")
    ratio = int(model.vae_ratio)
    if ratio <= 0:
        raise RuntimeError("MiniMax-H3 global-position diagnostic requires a positive VAE ratio")

    original_tiled_decode = model.tiled_decode
    original_pos_forward = pos_embed.forward
    context: dict[str, object] = {
        "full_hw": None,
        "y_starts": None,
        "row": 0,
        "origins": None,
        "tile_shape": None,
    }

    def tiled_decode_with_global_context(self, z):
        height = int(z.shape[-2]) * ratio
        width = int(z.shape[-1]) * ratio
        y_idx, _y_len, _y_overlap = self.split_tiles(height)
        context["full_hw"] = (int(z.shape[-2]), int(z.shape[-1]))
        context["y_starts"] = [int(value) // ratio for value in y_idx]
        context["row"] = 0
        context["origins"] = None
        context["tile_shape"] = None
        try:
            return original_tiled_decode(z)
        finally:
            context["origins"] = None
            context["tile_shape"] = None
            context["y_starts"] = None

    def decode_tile_row_with_global_context(self, z_row, x_idx, x_len):
        # Match Core's bounded batch policy so this diagnostic changes only
        # decoder spatial coordinates, not tile grouping or memory behavior.
        import comfy.model_management as model_management

        y_starts = context.get("y_starts")
        full_hw = context.get("full_hw")
        if not isinstance(y_starts, list) or full_hw is None:
            raise RuntimeError("MiniMax-H3 global-position row context was not initialized")
        row = int(context["row"])
        if row >= len(y_starts):
            raise RuntimeError("MiniMax-H3 global-position row index exceeded the Core tile plan")
        context["row"] = row + 1
        origin_y = int(y_starts[row])

        free = model_management.get_free_memory(z_row.device)
        source_batch = int(z_row.shape[0])
        batch = int(max(1, min(4, free // (128 * 2**20 * source_batch))))
        slices = [
            z_row[..., j_pos // ratio : (j_pos + j_len) // ratio]
            for j_pos, j_len in zip(x_idx, x_len)
        ]
        for k in range(0, len(slices), batch):
            group = slices[k : k + batch]
            group_starts = x_idx[k : k + len(group)]
            origins: list[tuple[int, int]] = []
            for j_pos in group_starts:
                origins.extend([(origin_y, int(j_pos) // ratio)] * source_batch)
            merged = torch.cat(group)
            context["origins"] = origins
            context["tile_shape"] = (
                int(merged.shape[-3]),
                int(merged.shape[-2]),
                int(merged.shape[-1]),
            )
            try:
                decoded = self._decode_pixels(merged)
            finally:
                context["origins"] = None
                context["tile_shape"] = None
            yield from decoded.chunk(len(group))

    def pos_forward_with_global_spatial_ids(self, img_ids):
        origins = context.get("origins")
        tile_shape = context.get("tile_shape")
        full_hw = context.get("full_hw")
        if origins is None or tile_shape is None or full_hw is None:
            return original_pos_forward(img_ids)
        global_ids = _globalize_spatial_token_ids(
            img_ids,
            tile_shape=tile_shape,
            full_hw=full_hw,
            origins=origins,
        )
        return original_pos_forward(global_ids)

    restorers = [
        _install_instance_method(model, "tiled_decode", tiled_decode_with_global_context),
        _install_instance_method(model, "_decode_tile_row", decode_tile_row_with_global_context),
        _install_instance_method(pos_embed, "forward", pos_forward_with_global_spatial_ids),
    ]
    try:
        yield
    finally:
        for restore in reversed(restorers):
            restore()

def decode_minimax_h3_large_tile(
    vae: object,
    samples: dict[str, object],
    *,
    tile_size: int = 320,
    tile_overlap: int = 128,
    global_spatial_positions: bool = False,
) -> tuple[torch.Tensor, str]:
    """Decode with a changed spatial tile profile; retained as a tile-size diagnostic."""
    latent = _validate_samples(samples)
    model = _h3_video_vae_model(vae)
    tile_size, tile_overlap = _validate_profile(
        tile_size,
        tile_overlap,
        int(model.vae_ratio),
    )
    output_height = int(latent.shape[-2]) * int(model.vae_ratio)
    output_width = int(latent.shape[-1]) * int(model.vae_ratio)

    with _DECODE_LOCK:
        original = (
            bool(model.tiling),
            int(model.tile_size),
            int(model.tile_overlap_min),
        )
        native_x_seams, native_y_seams = _tile_boundaries(
            model,
            output_height,
            output_width,
        )
        try:
            model.tiling = True
            model.tile_size = tile_size
            model.tile_overlap_min = tile_overlap
            x_seams, y_seams = _tile_boundaries(model, output_height, output_width)
            position_context = (
                _global_spatial_position_patch(model)
                if global_spatial_positions
                else contextlib.nullcontext()
            )
            with position_context:
                images = vae.decode(latent)
        finally:
            model.tiling, model.tile_size, model.tile_overlap_min = original

    images = _flatten_core_video_output(images)
    report = (
        "mode=tile_size; "
        f"tile={tile_size}px overlap>={tile_overlap}px "
        f"output={output_width}x{output_height} "
        f"tiles={(len(x_seams) + 1)}x{(len(y_seams) + 1)}; active_"
        + _format_seam_report(images, x_seams, y_seams)
        + "; original_profile_locations_"
        + _format_seam_report(images, native_x_seams, native_y_seams)
        + f"; restored_profile={original[1]}/{original[2]}"
    )
    _emit_report(report)
    return images, report


def decode_minimax_h3_global_spatial_position(
    vae: object,
    samples: dict[str, object],
) -> tuple[torch.Tensor, str]:
    """Decode at Core's 256/64 tile geometry with full-canvas spatial RoPE IDs.

    Only Y/X position coordinates change. Tile extents, overlap/blending, temporal
    chunking, per-tile attention domains, weights, and the VAE wrapper stay on the
    normal Core path. This is a causal diagnostic for tile-local position resets.
    """
    latent = _validate_samples(samples)
    model = _h3_video_vae_model(vae)
    for name in ("_decode_tile_row", "_decode_pixels", "decoder"):
        if not hasattr(model, name):
            raise RuntimeError(
                f"MiniMax-H3 global-position diagnostic requires model.{name}"
            )
    decoder = model.decoder
    if not hasattr(decoder, "pos_embed") or not hasattr(decoder, "num_register_tokens"):
        raise RuntimeError(
            "MiniMax-H3 global-position diagnostic requires the Core ViT3D decoder"
        )

    ratio = int(model.vae_ratio)
    tile_size, tile_overlap = _validate_profile(256, 64, ratio)
    full_hw = (int(latent.shape[-2]), int(latent.shape[-1]))
    output_height = full_hw[0] * ratio
    output_width = full_hw[1] * ratio

    with _DECODE_LOCK:
        original_profile = (
            bool(model.tiling),
            int(model.tile_size),
            int(model.tile_overlap_min),
        )
        previous_row = getattr(model, "__dict__", {}).get("_decode_tile_row", _MISSING)
        previous_pixels = getattr(model, "__dict__", {}).get("_decode_pixels", _MISSING)
        previous_pos_forward = getattr(decoder.pos_embed, "__dict__", {}).get(
            "forward",
            _MISSING,
        )
        original_row = model._decode_tile_row
        original_pixels = model._decode_pixels
        original_pos_forward = decoder.pos_embed.forward
        state: dict[str, object] = {
            "row_calls": 0,
            "position_context": None,
        }

        try:
            model.tiling = True
            model.tile_size = tile_size
            model.tile_overlap_min = tile_overlap
            y_idx, _y_len, _y_overlap = model.split_tiles(output_height)
            x_idx, _x_len, _x_overlap = model.split_tiles(output_width)
            if any(v % ratio for v in [*y_idx, *x_idx]):
                raise RuntimeError(
                    "MiniMax-H3 tile starts must align to the latent spatial ratio"
                )

            def patched_pos_forward(_pos_self, img_ids):
                context = state["position_context"]
                if context is None:
                    return original_pos_forward(img_ids)
                remapped = _remap_spatial_position_ids(
                    img_ids,
                    image_token_count=int(context["image_token_count"]),
                    local_hw=context["local_hw"],
                    full_hw=full_hw,
                    spatial_offsets=context["spatial_offsets"],
                )
                return original_pos_forward(remapped)

            def patched_decode_pixels(_model_self, z):
                row_context = state.get("row_context")
                if row_context is None:
                    return original_pixels(z)
                base_batch = int(row_context["base_batch"])
                total_batch = int(z.shape[0])
                if base_batch <= 0 or total_batch % base_batch:
                    raise RuntimeError(
                        "MiniMax-H3 tile decoder batch cannot be mapped to tile offsets"
                    )
                tile_count = total_batch // base_batch
                start = int(row_context["next_x"])
                stop = start + tile_count
                x_offsets = row_context["x_offsets"][start:stop]
                if len(x_offsets) != tile_count:
                    raise RuntimeError(
                        "MiniMax-H3 tile decoder consumed an unexpected number of columns"
                    )
                row_context["next_x"] = stop
                y_offset = int(row_context["y_offset"])
                offsets = [
                    (y_offset, int(x_offset))
                    for x_offset in x_offsets
                    for _ in range(base_batch)
                ]
                state["position_context"] = {
                    "image_token_count": int(z.shape[-3] * z.shape[-2] * z.shape[-1]),
                    "local_hw": (int(z.shape[-2]), int(z.shape[-1])),
                    "spatial_offsets": offsets,
                }
                try:
                    return original_pixels(z)
                finally:
                    state["position_context"] = None

            def patched_decode_tile_row(_model_self, z_row, row_x_idx, row_x_len):
                row_number = int(state["row_calls"]) % len(y_idx)
                state["row_calls"] = int(state["row_calls"]) + 1
                if list(map(int, row_x_idx)) != list(map(int, x_idx)):
                    raise RuntimeError(
                        "MiniMax-H3 tile-column geometry changed during diagnostic decode"
                    )
                row_context = {
                    "base_batch": int(z_row.shape[0]),
                    "next_x": 0,
                    "x_offsets": [int(value) // ratio for value in row_x_idx],
                    "y_offset": int(y_idx[row_number]) // ratio,
                }
                state["row_context"] = row_context
                try:
                    yield from original_row(z_row, row_x_idx, row_x_len)
                    if int(row_context["next_x"]) != len(row_context["x_offsets"]):
                        raise RuntimeError(
                            "MiniMax-H3 tile decoder did not consume every tile column"
                        )
                finally:
                    state["row_context"] = None
                    state["position_context"] = None

            decoder.pos_embed.forward = types.MethodType(
                patched_pos_forward,
                decoder.pos_embed,
            )
            model._decode_pixels = types.MethodType(patched_decode_pixels, model)
            model._decode_tile_row = types.MethodType(patched_decode_tile_row, model)
            images = vae.decode(latent)
        finally:
            _restore_instance_attribute(
                decoder.pos_embed,
                "forward",
                previous_pos_forward,
            )
            _restore_instance_attribute(model, "_decode_pixels", previous_pixels)
            _restore_instance_attribute(model, "_decode_tile_row", previous_row)
            model.tiling, model.tile_size, model.tile_overlap_min = original_profile

    images = _flatten_core_video_output(images)
    x_seams = [int(value) for value in x_idx[1:]]
    y_seams = [int(value) for value in y_idx[1:]]
    report = (
        "mode=global_spatial_rope; "
        "tile=256px overlap>=64px "
        f"output={output_width}x{output_height} "
        f"tiles={(len(x_seams) + 1)}x{(len(y_seams) + 1)}; "
        "temporal_ids=core_local; spatial_ids=full_canvas; "
        + _format_seam_report(images, x_seams, y_seams)
        + f"; restored_profile={original_profile[1]}/{original_profile[2]}"
    )
    _emit_report(report)
    return images, report


class H3MiniMaxVAEDecodeLargeTile:
    CATEGORY = "MiniMax H3/flow regenerate/experimental"
    DESCRIPTION = (
        "Diagnostic only: decode MiniMax-H3 video latents with a changed spatial "
        "tile size/overlap. This does not alter decoder positional semantics. "
        "The original VAE profile is restored after every call."
    )
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": (
                    "INT",
                    {"default": 320, "min": 256, "max": 512, "step": 16},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 128, "min": 64, "max": 256, "step": 16},
                ),
            }
        }

    def decode(self, samples, vae, tile_size, tile_overlap):
        return decode_minimax_h3_large_tile(
            vae,
            samples,
            tile_size=int(tile_size),
            tile_overlap=int(tile_overlap),
        )


class H3MiniMaxVAEDecodeGlobalPositionDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate/experimental"
    DESCRIPTION = (
        "Causal decoder diagnostic: preserve Core's native 256/64 tile geometry "
        "and independent tile attention, but give each tile full-canvas spatial "
        "RoPE coordinates. All method/profile overrides are call-scoped."
    )
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }

    def decode(self, samples, vae):
        return decode_minimax_h3_global_spatial_position(vae, samples)


class H3MiniMaxVAEDecodeGlobalPosition:
    CATEGORY = "MiniMax H3/flow regenerate/experimental"
    DESCRIPTION = (
        "Diagnostic decode that preserves the VAE's current spatial tile size, overlap, "
        "split, blending, and batching, but gives every tile full-frame spatial RoPE "
        "coordinates. Use this to isolate tile-local position reset from independent "
        "tile attention/context as the source of rectangular/checkerboard artifacts."
    )
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }

    def decode(self, samples, vae):
        model = _h3_video_vae_model(vae)
        return decode_minimax_h3_large_tile(
            vae,
            samples,
            tile_size=int(model.tile_size),
            tile_overlap=int(model.tile_overlap_min),
            global_spatial_positions=True,
        )


NODE_CLASS_MAPPINGS = {
    "H3MiniMaxVAEDecodeLargeTile": H3MiniMaxVAEDecodeLargeTile,
    "H3MiniMaxVAEDecodeGlobalPositionDiagnostic": (
        H3MiniMaxVAEDecodeGlobalPositionDiagnostic
    ),
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MiniMaxVAEDecodeLargeTile": "MiniMax H3 VAE Decode — Tile Size [Diagnostic]",
    "H3MiniMaxVAEDecodeGlobalPositionDiagnostic": (
        "MiniMax H3 VAE Decode — Global Spatial Position [Diagnostic]"
    ),
}
