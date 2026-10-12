"""Paired VAE-only source-prefix projection experiment from verified witnesses.

No sampler state is consumed or returned. All comparisons use the same RGB
canvas and frame labels, including explicit native decoder-context controls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .geometry import resize_spatial_5d
from .local_boundary_audit import (
    _AUDIT_MODEL_OWNERS,
    _decode_owned_pixels,
    _resident_admit,
    normalize_bundle_path,
    read_verified_operand,
    replay_plan,
)
from .partitioned_diagnostics import PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCES
from .stage_static_roi_audit import PROFILES, parse_static_rois

LOG = logging.getLogger(__name__)
POLICY = "h3_source_prefix_vae_rgb_roundtrip_ab_v1"


def load_prefix_operands(bundle_path, join_frame):
    """Verify only the operands needed for projection and end-context controls."""
    path = normalize_bundle_path(bundle_path, platform=os.name, wsl_distro=os.environ.get("WSL_DISTRO_NAME"))
    directory = Path(path).expanduser().resolve()
    if directory.name == "manifest.json":
        directory = directory.parent
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Select an existing boundary bundle containing {manifest_path.name}: {directory}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    metadata = manifest.get("metadata", {})
    if (
        manifest.get("schema") != 1
        or manifest.get("kind") != "h3_flow_native_boundary_decoder_window_evidence"
        or metadata.get("policy") != "native_boundary_decoder_window_evidence_v1"
        or metadata.get("domain") != "model_internal_clean_except_sampler_input_and_mask"
        or metadata.get("spatial_stage_control") not in PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCES
        or metadata.get("source_prefix_projection_policy") != "half_pixel_latent_v1"
        or metadata.get("process_latent_out_required_before_vae") is not True
        or metadata.get("first_high_actual") is not True
        or metadata.get("provider_clean_provenance") != "actual_learned_provider_uniform_source"
        or metadata.get("full_video_snapshots") is not True
        or metadata.get("full_video_temporal_start_t") != 0
        or metadata.get("low_probe_native_carrier_decodable") is not True
    ):
        raise ValueError("prefix projection A/B requires a verified uniform-source half_pixel_latent_v1 bundle")
    plan = replay_plan(metadata, join_frame)
    hashes = {}
    entries = manifest["tensor_bytes"]
    prefix = read_verified_operand(directory, entries, "authoritative_prefix_full", hashes)
    grid = metadata.get("source_probe_clean_grid")
    if (
        prefix.shape[2] != plan["prefix_t"]
        or any(n % 2 for n in prefix.shape[-2:])
        or not isinstance(grid, list)
        or len(grid) != 2
        or any(type(n) is not int or n <= 0 or n % 2 for n in grid)
        or any(a > b for a, b in zip(grid, prefix.shape[-2:], strict=True))
        or tuple(grid) == tuple(prefix.shape[-2:])
    ):
        raise ValueError("invalid native source/target prefix geometry")
    prefix_t = plan["prefix_t"]
    source = read_verified_operand(directory, entries, "source_probe_clean_full", hashes)
    if tuple(source.shape) != (1, 24, plan["temporal"], *grid):
        raise ValueError("saved source trajectory shape differs from metadata")
    legacy = resize_spatial_5d(prefix, *grid, mode="bicubic")
    if not torch.allclose(source[:, :, :prefix_t], legacy, atol=1e-4, rtol=1e-5):
        raise ValueError("saved source prefix differs from production latent-bicubic projection")
    projection_delta = source[:, :, :prefix_t] - legacy
    projection_error = {
        "rms": float(projection_delta.square().mean().sqrt()),
        "abs_max": float(projection_delta.abs().max()),
    }
    # Decode the actual production-device projection bytes, after verifying
    # their contract; CPU interpolation can differ by float32 roundoff.
    legacy = source[:, :, :prefix_t].clone()
    source_context = source[:, :, : prefix_t + 5].clone()
    del source
    final = read_verified_operand(directory, entries, "final_post_high_internal_clean_full", hashes)
    if tuple(final.shape) != (1, 24, plan["temporal"], *prefix.shape[-2:]) or not torch.equal(
        final[:, :, :prefix_t].contiguous().view(torch.int32), prefix.contiguous().view(torch.int32)
    ):
        raise ValueError("saved final trajectory changed the authoritative prefix or its geometry")
    target_context = final[:, :, : prefix_t + 5].clone()
    del final
    mask = read_verified_operand(directory, entries, "initial_high_video_mask_full", hashes)
    if (
        tuple(mask.shape) != (1, 24, plan["temporal"], *prefix.shape[-2:])
        or not bool((mask[:, :, :prefix_t] == 0).all())
        or not bool((mask[:, :, prefix_t:] == 1).all())
    ):
        raise ValueError("saved high mask violates protected-prefix ownership")
    return (
        prefix,
        legacy,
        target_context,
        source_context,
        {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "operand_sha256": hashes,
            "prefix_t": prefix_t,
            "decoded_frames": metadata["window"]["decoded_trim_frames"],
            "source_latent_hw": grid,
            "target_latent_hw": list(prefix.shape[-2:]),
            "source_projection_policy": metadata["source_prefix_projection_policy"],
            "saved_projection_check_tolerance": {"atol": 1e-4, "rtol": 1e-5},
            "saved_projection_check_error": projection_error,
            "baseline_latents": "verified_saved_production_projection_bytes",
        },
    )


def resize_rgb(frames, height, width):
    """Spatial-only half-pixel, antialiased RGB resize; never change frame order."""
    out = F.interpolate(
        frames.movedim(-1, 1).float(), size=(height, width), mode="bicubic", align_corners=False, antialias=True
    ).movedim(1, -1)
    # Bicubic can overshoot [0,1]; both the reference and encoder see the same clipping.
    clipped = int(((out < 0) | (out > 1)).sum().item())
    return out.clamp_(0, 1).contiguous(), clipped


def _encode_owned_pixels(vae, frames, expected_shape):
    """Use Core's native RGB/latent normalization exactly once, without eviction."""
    if not all(hasattr(vae, name) for name in ("patcher", "device", "vae_dtype", "output_device", "process_input")):
        with torch.inference_mode():
            encoded = vae.encode(frames.clone())
    else:
        import comfy.model_management as management

        shape = (1, 3, len(frames), *frames.shape[1:3])
        resident = any(model is vae.patcher for model in management.loaded_models()) and (
            vae.patcher.current_loaded_device() == vae.device
        )
        _resident_admit(vae, 0 if resident else vae.memory_used_encode(shape, vae.vae_dtype))
        try:
            with management.cuda_device_context(vae.device), torch.inference_mode():
                # H3 streams 17-frame clips from CPU inside encode_temporal.
                pixels = frames.movedim(-1, 0).unsqueeze(0).clone()
                pixels = vae.process_input(pixels).to(dtype=vae.vae_dtype)
                encoded = vae.first_stage_model.encode(pixels, device=vae.device)
                encoded = encoded.to(device=vae.output_device, dtype=vae.vae_output_dtype(), copy=True)
        except getattr(management, "OOM_EXCEPTION", torch.cuda.OutOfMemoryError):
            raise RuntimeError(
                "Prefix Projection A/B ran out of encode memory. Resident models were preserved; "
                "the audit did not retry through model-unloading encode."
            ) from None
    if tuple(encoded.shape) != tuple(expected_shape) or not bool(torch.isfinite(encoded).all()):
        raise ValueError("native H3 encode changed temporal ownership, source dimensions or finite latent values")
    return encoded


def _luma(rgb):
    return rgb @ rgb.new_tensor([0.2126, 0.7152, 0.0722])


def _gradients(luma):
    x = (luma[:, :-2, 2:] + 2 * luma[:, 1:-1, 2:] + luma[:, 2:, 2:]) - (
        luma[:, :-2, :-2] + 2 * luma[:, 1:-1, :-2] + luma[:, 2:, :-2]
    )
    y = (luma[:, 2:, :-2] + 2 * luma[:, 2:, 1:-1] + luma[:, 2:, 2:]) - (
        luma[:, :-2, :-2] + 2 * luma[:, :-2, 1:-1] + luma[:, :-2, 2:]
    )
    return torch.stack((x, y), dim=1) / 8


def paired_rgb_metrics(reference, candidate, labels, rois):
    """Unregistered same-grid errors; blur cannot improve detail-error metrics."""
    if reference.shape != candidate.shape or len(labels) != len(reference) or reference.shape[-1] != 3:
        raise ValueError("paired projection metrics require matching RGB canvas and temporal labels")
    output = {}
    regions = {"full_frame": (0, 0, 1, 1), **rois}
    h, w = reference.shape[1:3]
    for name, (x0, y0, x1, y1) in regions.items():
        bounds = [round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)]
        ax, ay, bx, by = bounds
        if min(bx - ax, by - ay) < 11:
            raise ValueError(f"projection ROI {name!r} must be at least 11 pixels wide and high")
        a, b = reference[:, ay:by, ax:bx], candidate[:, ay:by, ax:bx]
        delta = b - a
        la, lb = _luma(a), _luma(b)
        dl = lb - la
        ga, gb = _gradients(la), _gradients(lb)
        dims = (1, 2)
        values = {
            "reference_rgb_mean": a.mean((1, 2)),
            "candidate_rgb_mean": b.mean((1, 2)),
            "rgb_mean_error": delta.mean((1, 2)),
            "rgb_rmse": delta.square().mean((1, 2, 3)).sqrt(),
            "reference_luma_mean": la.mean(dims),
            "candidate_luma_mean": lb.mean(dims),
            "luma_bias": dl.mean(dims),
            "luma_rmse": dl.square().mean(dims).sqrt(),
            "reference_luma_std": la.std(dims, correction=0),
            "candidate_luma_std": lb.std(dims, correction=0),
            "reference_sobel_rms": ga.square().sum(1).mean(dims).sqrt(),
            "candidate_sobel_rms": gb.square().sum(1).mean(dims).sqrt(),
            "sobel_error_rms": (gb - ga).square().sum(1).mean(dims).sqrt(),
        }
        for size in (3, 9):
            pad = size // 2
            ha = la[:, pad:-pad, pad:-pad] - F.avg_pool2d(la[:, None], size, stride=1)[:, 0]
            hb = lb[:, pad:-pad, pad:-pad] - F.avg_pool2d(lb[:, None], size, stride=1)[:, 0]
            values[f"reference_highpass_{size}_rms"] = ha.square().mean(dims).sqrt()
            values[f"candidate_highpass_{size}_rms"] = hb.square().mean(dims).sqrt()
            values[f"highpass_{size}_error_rms"] = (hb - ha).square().mean(dims).sqrt()
        # SSIM uses fixed [0,1] constants and unregistered 11px local windows.
        means = [F.avg_pool2d(v[:, None], 11, stride=1)[:, 0] for v in (la, lb)]
        ma, mb = means
        va = (F.avg_pool2d(la[:, None].square(), 11, stride=1)[:, 0] - ma.square()).clamp_min(0)
        vb = (F.avg_pool2d(lb[:, None].square(), 11, stride=1)[:, 0] - mb.square()).clamp_min(0)
        cov = F.avg_pool2d((la * lb)[:, None], 11, stride=1)[:, 0] - ma * mb
        values["luma_ssim"] = (
            (2 * ma * mb + 0.01**2)
            * (2 * cov + 0.03**2)
            / ((ma.square() + mb.square() + 0.01**2) * (va + vb + 0.03**2))
        ).mean(dims)
        temporal = (delta[1:] - delta[:-1]).square().mean((1, 2, 3)).sqrt()
        frames = []
        for i, label in enumerate(labels):
            row = {"frame": label, **{key: value[i].tolist() for key, value in values.items()}}
            row["temporal_rgb_error_increment_rmse"] = None if i == 0 else float(temporal[i - 1])
            frames.append(row)
        error_names = [key for key in values if key.endswith(("rmse", "error_rms"))]
        summary = {key: float(values[key].mean()) for key in error_names}
        summary.update(
            luma_bias_mean=float(values["luma_bias"].mean()),
            luma_abs_bias_mean=float(values["luma_bias"].abs().mean()),
            luma_ssim_mean=float(values["luma_ssim"].mean()),
            temporal_rgb_error_increment_rmse_mean=float(temporal.mean()) if len(temporal) else None,
            last_five_rgb_rmse_mean=float(values["rgb_rmse"][-5:].mean()),
            last_five_luma_abs_bias_mean=float(values["luma_bias"][-5:].abs().mean()),
        )
        output[name] = {"bounds_xyxy_px": bounds, "summary": summary, "frames": frames}
    return output


def _same_time_geometry(reference, candidate, labels, rois):
    """Match reference→candidate at each same-time frame, never across time."""
    from .feature_background_tracking import track_background_features

    rows = []
    for index, label in enumerate(labels):
        fit = track_background_features(
            torch.stack((reference[index], candidate[index])), [0, 1], join_frame=1, rois=rois
        )
        row = fit.get("frame_trajectories", {}).get("1", {"status": fit["status"]})
        rows.append({"frame": label, **row})
    return {"method": "same_time_forward_backward_LK_RANSAC_similarity", "frames": rows}


def audit_prefix_projection(vae, bundle_path, join_frame, process_out, *, rois, feature_tracking=True):
    prefix, legacy, target_context, source_context, identity = load_prefix_operands(bundle_path, join_frame)
    original_prefix = prefix.contiguous().view(torch.int32).clone()
    frames = identity["decoded_frames"]
    height, width = (n * 16 for n in identity["source_latent_hw"])
    labels = list(range(join_frame - frames, join_frame))
    if feature_tracking and (not rois or min(height, width) < 80):
        raise ValueError("same-time geometry needs named ROIs and at least an 80px canvas")
    timings = {}
    memory = {}
    device = getattr(vae, "device", None)
    cuda = device is not None and torch.device(device).type == "cuda"

    def memory_sample():
        if not cuda:
            return None
        torch.cuda.synchronize(device)
        return {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }

    def timed(name, call):
        LOG.info("Prefix Projection A/B: %s", name)
        before = memory_sample()
        started = time.perf_counter()
        result = call()
        after = memory_sample()
        timings[name] = (time.perf_counter() - started) * 1000
        memory[name] = {"before": before, "after": after}
        return result

    # The reference uses precisely the bytes being projected, including native
    # terminal padding. Context controls below quantify that padding's effect.
    target = timed("target_prefix_decode", lambda: _decode_owned_pixels(vae, prefix, process_out, frames))[0]
    reference, clipped = resize_rgb(target.cpu(), height, width)
    del target
    context_frames = frames + 17
    target_pixels = timed(
        "target_future_context_decode",
        lambda value=target_context: _decode_owned_pixels(vae, value, process_out, context_frames),
    )[0, :frames].cpu()
    target_physical, _ = resize_rgb(target_pixels, height, width)
    del target_pixels, target_context
    context_control = paired_rgb_metrics(reference, target_physical, labels, rois)
    a = timed("latent_bicubic_decode", lambda: _decode_owned_pixels(vae, legacy, process_out, frames))[0].cpu()
    source_pixels = timed(
        "source_future_context_decode",
        lambda value=source_context: _decode_owned_pixels(vae, value, process_out, context_frames),
    )[0, :frames].cpu()
    source_control = paired_rgb_metrics(a, source_pixels, labels, rois)
    source_context_against_target = paired_rgb_metrics(target_physical, source_pixels, labels, rois)
    del source_pixels, source_context
    encoded = timed("source_rgb_encode", lambda: _encode_owned_pixels(vae, reference, legacy.shape))
    # encode returns VAE-domain normalized latents, so do NOT process_out again.
    b = timed("roundtrip_decode", lambda value=encoded: _decode_owned_pixels(vae, value, lambda x: x, frames))[0].cpu()
    del encoded
    results = {}
    context_results = {}
    for name, value in (("latent_bicubic", a), ("vae_rgb_roundtrip", b)):
        results[name] = {"regions": paired_rgb_metrics(reference, value, labels, rois)}
        context_results[name] = {"regions": paired_rgb_metrics(target_physical, value, labels, rois)}
        if feature_tracking:
            results[name]["same_time_geometry"] = _same_time_geometry(reference, value, labels, rois)
    del target_physical
    delta = {}
    for region in results["latent_bicubic"]["regions"]:
        sa = results["latent_bicubic"]["regions"][region]["summary"]
        sb = results["vae_rgb_roundtrip"]["regions"][region]["summary"]
        delta[region] = {key: sb[key] - sa[key] for key in sa if sa[key] is not None and sb[key] is not None}
    if not torch.equal(prefix.contiguous().view(torch.int32), original_prefix):
        raise RuntimeError("projection audit mutated its authoritative prefix")
    report = {
        "schema": 1,
        "policy": POLICY,
        "identity": identity,
        "vae": {
            "class": type(getattr(vae, "first_stage_model", vae)).__name__,
            "dtype": str(getattr(vae, "vae_dtype", "unknown")),
            "device": str(getattr(vae, "device", "unknown")),
            "checkpoint_identity": "connected_workflow_VAE; record_loader_checkpoint_with_report",
            "encode_normalization": (
                "Core process_input once; native encoder returns normalized mean, no posterior sampling"
            ),
        },
        "chunk_join_frame": join_frame,
        "frame_labels": labels,
        "common_rgb_canvas_hw": [height, width],
        "rgb_resize": "half_pixel_bicubic_antialias_clamp_0_1_no_temporal_resample",
        "rgb_resize_clipped_components": clipped,
        "reference": "native isolated authoritative prefix decode spatially resized in RGB",
        "comparisons": results,
        "comparisons_against_target_with_saved_future": context_results,
        "roundtrip_minus_bicubic_summary": delta,
        "decoder_context_controls": {
            "following_saved_tokens": 5,
            "target_prefix_isolated_vs_saved_future": context_control,
            "source_prefix_isolated_vs_saved_future": source_control,
            "source_with_saved_future_vs_target_with_saved_future": source_context_against_target,
            "note": (
                "Context controls use saved generated suffix, not future ground truth. End-frame "
                "discrepancies confound acceptance."
            ),
        },
        "timings_ms": timings,
        "cuda_memory": memory,
        "cuda_peak_scope": "cumulative process peak; global peak counters were not reset",
        "output_rgb_bytes": sum(value.numel() * value.element_size() for value in (reference, a, b)),
        "extra_vae_decode_calls": 5,
        "extra_vae_encode_calls": 1,
        "extra_h3_nfe": 0,
        "production_sampling_rerun": False,
        "production_output_modified": False,
        "authoritative_prefix_preserved_bitwise": True,
        "audio_masks_sigmas_sampler_state": "never supplied to VAE; no production objects mutated",
        "candidate_integration": "not_enabled; GPU paired reconstruction and visual acceptance required",
        "limitations": [
            "RGB/luma are finalized encoded-display values, not linear-light radiometry.",
            "Same-time RGB error has no alignment/photometric correction; spatial-frequency amplitude alone "
            "is not accuracy.",
            "SSIM, Sobel and box-highpass bands are proxies; inspect original-scale outputs for changed "
            "details or pose.",
            "Insufficient LK/RANSAC confidence is indeterminate, not zero displacement. Global similarity "
            "misses local/nonrigid changes.",
            "Prefix-only decode and encode use native terminal padding; future-context controls must be "
            "reviewed at the last frames.",
            "VAE-only success does not establish reduced-grid continuation geometry, speech quality or speed.",
        ],
    }
    return report, reference, a, b


class H3FlowPrefixProjectionAudit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE",),
                "bundle_path": (
                    "STRING",
                    {"default": "", "tooltip": "Existing uniform-source boundary bundle directory or manifest.json."},
                ),
                "chunk_join_frame": ("INT", {"default": 175, "min": 0, "max": 10000000}),
                "static_roi_profile": (list(PROFILES), {"default": "01784_room"}),
                "feature_tracking_enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {"static_roi_json": ("STRING", {"default": "", "multiline": True})},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("rgb_reference", "latent_bicubic", "vae_rgb_roundtrip", "numerical_report")
    FUNCTION = "audit"
    CATEGORY = "MiniMax H3/diagnostics"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Paired VAE-only source-prefix A/B from an existing uniform-source witness. "
        "Connect the native H3 video VAE. Five decodes and one encode; no H3 rerun or production change. "
        "Outputs aligned RGB clips and saves paired numerical metrics plus terminal-context controls."
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        _AUDIT_MODEL_OWNERS.retain()
        return float("nan")

    def audit(
        self,
        video_vae,
        bundle_path,
        chunk_join_frame,
        static_roi_profile="01784_room",
        feature_tracking_enabled=True,
        static_roi_json="",
    ):
        _AUDIT_MODEL_OWNERS.retain()
        import folder_paths
        from comfy.latent_formats import MiniMaxH3Video
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        if not isinstance(video_vae.first_stage_model, MiniMaxH3VideoVAE):
            raise ValueError("Prefix Projection A/B requires the native MiniMax H3 video encoder/decoder")
        model = video_vae.first_stage_model
        if (model.vae_ratio, model.vae_ratio_t, model.clip_length, model.token_drop, model.token_overlap) != (
            16,
            4,
            17,
            3,
            2,
        ):
            raise ValueError("connected H3 VAE differs from the native 17k+5 temporal contract")
        rois = parse_static_rois(static_roi_profile, static_roi_json)
        report, reference, a, b = audit_prefix_projection(
            video_vae,
            bundle_path,
            chunk_join_frame,
            MiniMaxH3Video().process_out,
            rois=rois,
            feature_tracking=feature_tracking_enabled,
        )
        text = json.dumps(report, indent=2, allow_nan=False)
        directory = Path(folder_paths.get_output_directory()) / "h3_flow_regenerate" / "prefix_projection_audits"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="prefix-projection-", suffix=".json", dir=directory, delete=False
        ) as stream:
            stream.write(text + "\n")
        return {
            "ui": {"text": [text, f"Numerical report: {Path(stream.name).name}"]},
            "result": (reference, a, b, text),
        }


NODE_CLASS_MAPPINGS = {"H3FlowPrefixProjectionAudit": H3FlowPrefixProjectionAudit}
NODE_DISPLAY_NAME_MAPPINGS = {"H3FlowPrefixProjectionAudit": "MiniMax H3 Prefix Projection A/B"}
