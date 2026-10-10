"""Opt-in native VAE construction of a carried uniform-source video prefix."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .decode_context import _frames, video_latent_fingerprint
from .local_boundary_audit import _decode_owned_pixels
from .prefix_projection_audit import _encode_owned_pixels, resize_rgb
from .source_prefix_carry import SOURCE_PREFIX_CARRY_KEY, SourcePrefixCarry

SOURCE_PREFIX_PROJECTION_KEY = "h3_flow_source_prefix_projection_v1"
SOURCE_PREFIX_PROJECTION_OPTIONS = ("latent_bicubic", "vae_rgb_roundtrip", "native_source_carry")
SOURCE_PREFIX_ROUNDTRIP_POLICY = "native_vae_rgb_roundtrip_v1"


@dataclass(frozen=True)
class SourcePrefixProjection:
    # ModelPatcher copies containers, preserving this model-local VAE owner.
    vae: object


def validate_native_video_vae(vae):
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    model = getattr(vae, "first_stage_model", None)
    if not isinstance(model, MiniMaxH3VideoVAE) or (
        model.vae_ratio,
        model.vae_ratio_t,
        model.clip_length,
        model.token_drop,
        model.token_overlap,
    ) != (16, 4, 17, 3, 2):
        raise ValueError("VAE source-prefix projection requires the native MiniMax H3 video encoder/decoder")


def configure_source_prefix_projection(model, mode, vae, spatial_stage_control):
    from .partitioned_diagnostics import PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCES

    if mode not in SOURCE_PREFIX_PROJECTION_OPTIONS:
        raise ValueError(f"unsupported source_prefix_projection: {mode!r}")
    options = dict(model.model_options)
    options.pop(SOURCE_PREFIX_CARRY_KEY, None)
    if mode == "latent_bicubic":
        options.pop(SOURCE_PREFIX_PROJECTION_KEY, None)
    else:
        if spatial_stage_control not in PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCES:
            raise ValueError(f"{mode} requires a progressive_uniform_source spatial stage")
        if mode == "native_source_carry":
            options.pop(SOURCE_PREFIX_PROJECTION_KEY, None)
            options[SOURCE_PREFIX_CARRY_KEY] = SourcePrefixCarry()
            model.model_options = options
            return
        if vae is None:
            raise ValueError("Connect the generation's native video VAE to video_vae for vae_rgb_roundtrip")
        validate_native_video_vae(vae)
        options[SOURCE_PREFIX_PROJECTION_KEY] = SourcePrefixProjection(vae)
    model.model_options = options


def project_source_prefix(vae, prefix, source_h, source_w):
    """One decode and one encode, using the diagnostic's identical RGB contract.

    Input is the authoritative model-internal prefix. Output is caller/VAE
    domain for the source sampler's latent_image. No generated tokens or audio
    enter this function; no projection is cached across chunks or invocations.
    """
    from comfy.latent_formats import MiniMaxH3Video

    validate_native_video_vae(vae)
    if (
        prefix.ndim != 5
        or tuple(prefix.shape[:2]) != (1, 24)
        or not prefix.is_floating_point()
        or not bool(torch.isfinite(prefix).all())
        or any(n % 2 for n in prefix.shape[-2:])
        or any(type(n) is not int or n <= 0 or n % 2 for n in (source_h, source_w))
        or source_h > prefix.shape[-2]
        or source_w > prefix.shape[-1]
        or (source_h, source_w) == tuple(prefix.shape[-2:])
    ):
        raise ValueError("VAE source-prefix projection requires native [1,24,T,H,W] on a larger target grid")
    frames = _frames(int(prefix.shape[2]))
    projection_started = time.perf_counter()
    before_prefix = video_latent_fingerprint(prefix)
    device = torch.device(getattr(vae, "device", "cpu"))

    def memory_sample():
        if device.type != "cuda":
            return None
        torch.cuda.synchronize(device)
        return {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }

    timings, memory = {}, {}

    def timed(name, call):
        before = memory_sample()
        started = time.perf_counter()
        value = call()
        after = memory_sample()
        timings[name] = (time.perf_counter() - started) * 1000
        memory[name] = {"before": before, "after": after}
        return value

    pixels = timed(
        "target_prefix_decode",
        lambda: _decode_owned_pixels(vae, prefix.detach(), MiniMaxH3Video().process_out, frames),
    )[0].cpu()
    reference, clipped = resize_rgb(pixels, source_h * 16, source_w * 16)
    del pixels
    expected = (1, 24, prefix.shape[2], source_h, source_w)
    projected = timed("source_rgb_encode", lambda value=reference: _encode_owned_pixels(vae, value, expected))
    del reference
    if video_latent_fingerprint(prefix)["sha256_float32"] != before_prefix["sha256_float32"]:
        raise RuntimeError("VAE source-prefix projection changed the authoritative prefix")
    receipt = {
        "policy": SOURCE_PREFIX_ROUNDTRIP_POLICY,
        "mode": "vae_rgb_roundtrip",
        "prefix_t": int(prefix.shape[2]),
        "decoded_frames": frames,
        "authoritative_prefix": before_prefix,
        "projected_prefix": video_latent_fingerprint(projected),
        "rgb_resize_clipped_components": clipped,
        "rgb_resize": "half_pixel_bicubic_antialias_clamp_0_1_no_temporal_resample",
        "timings_ms": timings,
        "elapsed_ms": (time.perf_counter() - projection_started) * 1000,
        "cuda_memory": memory,
        "cuda_peak_scope": "cumulative process peak; global peak counters were not reset",
        "vae_class": type(vae.first_stage_model).__name__,
        "vae_dtype": str(getattr(vae, "vae_dtype", "unknown")),
        "vae_device": str(device),
        "extra_vae_decode_calls": 1,
        "extra_vae_encode_calls": 1,
        "extra_h3_nfe": 0,
        "scope": "protected_prefix_only_before_low_sampling",
        "authoritative_prefix_preserved_bitwise": True,
    }
    return projected, receipt
