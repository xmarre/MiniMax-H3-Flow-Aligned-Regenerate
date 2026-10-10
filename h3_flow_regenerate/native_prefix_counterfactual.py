"""Read-only 01795 source-prefix counterfactual through the production native H3 VAE.

Both arms use the *same saved generated source suffix*. The intervention replaces
only the protected source-grid prefix; it never invokes a denoiser or a sampler.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch

from .decode_context import video_latent_fingerprint
from .feature_background_tracking import track_background_features
from .geometry import resize_spatial_5d
from .local_boundary_audit import (
    _AUDIT_MODEL_OWNERS,
    _decode_owned_pixels,
    geometry_comparison,
    load_replay_operands,
    normalize_bundle_path,
    read_verified_operand,
)
from .stage_static_roi_audit import (
    PROFILES,
    compare_same_frame_prefix_rois,
    measure_stage_static_rois,
    parse_static_rois,
)

POLICY = "h3_native_source_prefix_decoder_counterfactual_v1"
CAPTURE_01795_MANIFEST_SHA256 = "80a082e53f5a0c5ab433cc39701da21f7da4429b3c8f4754361e7a509d47bd56"
CAPTURE_01795_PREFIX_SHA256 = "de374402b384b34645af023e9978c6666ef0b8d82cd621bc284bc6438dd91977"
_LUMA = (0.2126, 0.7152, 0.0722)


def make_prefix_counterfactual(source, target_prefix, prefix_t):
    """Construct A/B on independent CPU copies; verify the entire suffix bitwise."""
    if (
        source.ndim != 5
        or target_prefix.ndim != 5
        or tuple(source.shape[:2]) != (1, 24)
        or tuple(target_prefix.shape[:2]) != (1, 24)
        or source.dtype != torch.float32
        or target_prefix.dtype != torch.float32
        or source.device.type != "cpu"
        or target_prefix.device.type != "cpu"
        or type(prefix_t) is not int
        or prefix_t < 2
        or (prefix_t - 2) % 5
        or target_prefix.shape[2] != prefix_t
        or source.shape[2] <= prefix_t
        or (source.shape[2] - 2) % 5
        or any(n % 2 for n in (*source.shape[-2:], *target_prefix.shape[-2:]))
        or any(a > b for a, b in zip(source.shape[-2:], target_prefix.shape[-2:], strict=True))
        or not bool(torch.isfinite(source).all())
        or not bool(torch.isfinite(target_prefix).all())
    ):
        raise ValueError("counterfactual requires finite native H3 source and authoritative target-clean operands")
    a, b = source.clone(), source.clone()
    projected = resize_spatial_5d(target_prefix, *source.shape[-2:], mode="bicubic")
    if projected.shape != b[:, :, :prefix_t].shape or not bool(torch.isfinite(projected).all()):
        raise ValueError("bicubic authoritative prefix projection has invalid geometry or values")
    b[:, :, :prefix_t] = projected
    def bits(t):
        return t.contiguous().view(torch.int32)
    if not torch.equal(bits(a[:, :, prefix_t:]), bits(b[:, :, prefix_t:])):
        raise RuntimeError("generated source suffix changed during prefix counterfactual")
    if not torch.equal(bits(a), bits(source)) or not torch.equal(bits(target_prefix), bits(target_prefix.clone())):
        raise RuntimeError("counterfactual changed a captured operand")
    return a, b


def _native_pixels(vae, clean, process_out, origin, labels):
    tokens = int(clean.shape[2])
    total_frames = ((tokens - 2) // 5) * 17 + 5
    indices = [frame - origin for frame in labels]
    if not indices or min(indices) < 0 or max(indices) >= total_frames:
        raise ValueError("requested pixel times fall outside the native decoder context")
    started = time.perf_counter()
    decoded = _decode_owned_pixels(vae, clean, process_out, total_frames)
    # Hold just the measured timeline on CPU. The full native decode is released
    # before starting the next VAE call, avoiding simultaneous full RGB outputs.
    pixels = decoded[0, indices].detach().float().cpu().clone()
    del decoded
    return pixels, (time.perf_counter() - started) * 1000


def _temporal_increment(a, b, labels):
    delta = b - a
    luma = delta @ delta.new_tensor(_LUMA)
    rows = {}
    for i, frame in enumerate(labels):
        rows[str(frame)] = {
            "rgb_rmse": float(delta[i].square().mean().sqrt()),
            "luma_mean_change": float(luma[i].mean()),
            "temporal_increment_change_rms": (
                None if i == 0 else float((delta[i] - delta[i - 1]).square().mean().sqrt())
            ),
        }
    return rows


def _variant_pixels(pixels, labels, join_frame, rois, feature_tracking):
    result = {
        "static_background_rois": measure_stage_static_rois(
            pixels, labels, join_frame=join_frame, rois=rois
        ),
        "adjacent_frame_geometry": geometry_comparison(pixels[:-1], pixels[1:], labels[1:]),
        "adjacent_frame_geometry_upper45": geometry_comparison(
            pixels[:-1, :round(pixels.shape[1] * .45)],
            pixels[1:, :round(pixels.shape[1] * .45)],
            labels[1:],
        ),
    }
    if feature_tracking:
        result["feature_tracking_from_f174"] = track_background_features(
            pixels, labels, join_frame=join_frame, rois=rois
        )
        # A separate, within-suffix anchor avoids mistaking an intervention to
        # the decoded f174 reference for an improvement in later-frame geometry.
        suffix_anchor = join_frame + 3
        if suffix_anchor in labels:
            result["feature_tracking_from_f178"] = track_background_features(
                pixels, labels, join_frame=suffix_anchor + 1, rois=rois
            )
    return result


def audit_native_prefix_counterfactual(
    vae,
    bundle_path,
    join_frame,
    process_out,
    *,
    static_roi_profile="01784_room",
    static_roi_json="",
    feature_tracking_enabled=True,
    expected_manifest_sha256=None,
):
    """Compare the exact captured native source against prefix-only intervention.

    Full native decoding is used for both arms. An independent short-window
    baseline reproduces the existing Local Boundary Audit operand/window.
    Neither a full/cropped pixel match nor corrected expansion is assumed.
    """
    if type(join_frame) is not int or join_frame < 1:
        raise ValueError("a positive assembled chunk_join_frame is required")
    if not isinstance(feature_tracking_enabled, bool):
        raise TypeError("feature_tracking_enabled must be boolean")
    rois = parse_static_rois(static_roi_profile, static_roi_json)
    if not rois:
        raise ValueError("counterfactual requires named static background ROIs")
    native = getattr(vae, "first_stage_model", None)
    expected = {"tokens_chunk_size": 5, "token_overlap": 2, "frame_pre_padding": 3, "clip_length": 17}
    if native is None or any(getattr(native, key, None) != value for key, value in expected.items()):
        raise ValueError("connect the exact native MiniMax H3 production video VAE")
    plan, stages, identity = load_replay_operands(bundle_path, join_frame, include_source=True)
    if (
        plan["head_t"] != plan["prefix_t"]
        or identity["source_prefix_projection_policy"] != "native_source_carry_v1"
        or (expected_manifest_sha256 is not None
            and identity["manifest_sha256"] != expected_manifest_sha256)
    ):
        raise ValueError("counterfactual requires a hash-matched uniform native_source_carry_v1 bundle")

    import os
    directory = Path(normalize_bundle_path(
        bundle_path, platform=os.name, wsl_distro=os.environ.get("WSL_DISTRO_NAME")
    )).expanduser().resolve()
    if directory.name == "manifest.json":
        directory = directory.parent
    raw_manifest = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != identity["manifest_sha256"]:
        raise ValueError("bundle manifest changed after provenance validation")
    manifest = json.loads(raw_manifest)
    if manifest["metadata"].get("domain") != "model_internal_clean_except_sampler_input_and_mask":
        raise ValueError("source counterfactual requires model-internal clean latent domain")
    hashes = {}
    source = read_verified_operand(directory, manifest["tensor_bytes"], "source_probe_clean_full", hashes)
    target = read_verified_operand(directory, manifest["tensor_bytes"], "authoritative_prefix_full", hashes)
    if hashes["source_probe_clean_full"] != identity["operand_sha256"]["source_probe_clean_full"]:
        raise ValueError("saved source operand changed after provenance validation")
    if hashes["authoritative_prefix_full"] != identity["operand_sha256"]["authoritative_prefix_full"]:
        raise ValueError("saved authoritative prefix changed after provenance validation")
    if tuple(source.shape) != (1, 24, plan["temporal"], *manifest["metadata"]["source_probe_clean_grid"]):
        raise ValueError("captured source timeline or spatial shape changed")
    if not torch.equal(
        source[:, :, plan["token_start"]:plan["token_stop"]].contiguous().view(torch.int32),
        stages["source_grid"].contiguous().view(torch.int32),
    ):
        raise ValueError("source replay baseline does not match verified Local Boundary Audit bytes")
    prefix_t = plan["prefix_t"]
    if (
        target.shape[2] != prefix_t
        or (expected_manifest_sha256 == CAPTURE_01795_MANIFEST_SHA256
            and video_latent_fingerprint(target)["sha256_float32"] != CAPTURE_01795_PREFIX_SHA256)
    ):
        raise ValueError("authoritative target prefix does not match the selected capture")
    original_source = video_latent_fingerprint(source)
    original_target = video_latent_fingerprint(target)
    a, b = make_prefix_counterfactual(source, target, prefix_t)
    del stages
    suffix_bitwise = torch.equal(
        a[:, :, prefix_t:].contiguous().view(torch.int32),
        b[:, :, prefix_t:].contiguous().view(torch.int32),
    )
    if not suffix_bitwise:
        raise RuntimeError("source suffix differs before VAE decoding")
    changed_prefix_elements = int(
        (a[:, :, :prefix_t].contiguous().view(torch.int32)
         != b[:, :, :prefix_t].contiguous().view(torch.int32)).sum().item()
    )
    start, end = plan["measured_local_frames"]
    labels = list(range(plan["decoded_origin_frame"] + start - 1, plan["decoded_origin_frame"] + end))
    full_origin = join_frame - plan["prefix_t"] // 5 * 17 - 5
    # Use the same native temporal decoding and pre-clamp overlap stitching for
    # both full-length arms. Do not reassemble standalone seven-token windows.
    baseline, t_a = _native_pixels(vae, a, process_out, full_origin, labels)
    counterfactual, t_b = _native_pixels(vae, b, process_out, full_origin, labels)
    cropped, t_crop = _native_pixels(
        vae, source[:, :, plan["token_start"]:plan["token_stop"]],
        process_out, plan["decoded_origin_frame"], labels
    )
    if (
        video_latent_fingerprint(source) != original_source
        or video_latent_fingerprint(target) != original_target
        or not torch.equal(a[:, :, prefix_t:].contiguous().view(torch.int32),
                            b[:, :, prefix_t:].contiguous().view(torch.int32))
    ):
        raise RuntimeError("captured source, target or generated suffix mutated during decoding")
    full_rois = {**rois, "upper45_full": (0.0, 0.0, 1.0, 0.45)}
    same_frame = compare_same_frame_prefix_rois(baseline, counterfactual, labels, rois=full_rois)
    cropped_control = compare_same_frame_prefix_rois(cropped, baseline, labels, rois=full_rois)
    results = {
        "policy": POLICY, "schema": 1,
        "identity": {
            **identity, "source_latent_fingerprint": original_source,
            "authoritative_prefix_fingerprint": original_target,
            "source_shape": list(source.shape), "target_prefix_shape": list(target.shape),
            "source_model_domain": "model_internal_clean",
            "target_model_domain": "model_internal_clean",
            "projection": "source_grid_half_pixel_bicubic_align_corners_false",
            "projection_method": "resize_spatial_5d(mode=bicubic)",
            "native_source_carry_was_original_prefix": True,
        },
        "plan": plan,
        "join_frame": join_frame,
        "frame_labels": labels,
        "frame_174_is_variant_dependent": True,
        "post_join_tracking_anchor": join_frame + 3,
        "generated_suffix_bitwise_identical": suffix_bitwise,
        "generated_suffix_tokens": [prefix_t, int(source.shape[2])],
        "replacement_tokens": [0, prefix_t],
        "changed_prefix_float32_elements": changed_prefix_elements,
        "source_prefix_equal_to_projection": changed_prefix_elements == 0,
        "capture_inputs_unchanged": True,
        "baseline_matches_saved_source_window_bytes": True,
        "full_vs_cropped_baseline": {
            "note": "Full and cropped VAE temporal contexts may disagree; never assume pixel identity",
            "same_frame": cropped_control,
        },
        "baseline": _variant_pixels(baseline, labels, join_frame, full_rois, feature_tracking_enabled),
        "counterfactual": _variant_pixels(counterfactual, labels, join_frame, full_rois, feature_tracking_enabled),
        "same_frame_A_to_B": same_frame,
        "same_frame_geometry": geometry_comparison(baseline, counterfactual, labels),
        "same_frame_geometry_upper45": geometry_comparison(
            baseline[:, :round(baseline.shape[1] * .45)],
            counterfactual[:, :round(counterfactual.shape[1] * .45)],
            labels,
        ),
        "temporal_increment_delta": _temporal_increment(baseline, counterfactual, labels),
        "vae": {
            "class": f"{type(native).__module__}.{type(native).__qualname__}",
            "dtype": str(getattr(vae, "vae_dtype", "unknown")),
            "device": str(getattr(vae, "device", "unknown")),
            "weights": "connected production VAE; record the checkpoint name separately",
            "full_native_decode_calls": 2, "cropped_native_decode_controls": 1,
            "extra_vae_calls": 3, "extra_vae_encode_calls": 0, "extra_h3_nfe": 0,
            "decode_elapsed_ms": {"baseline_full": t_a, "counterfactual_full": t_b, "baseline_cropped": t_crop},
        },
        "production_sampling_rerun": False, "production_output_modified": False,
        "decoder_only": True, "audio_state_supplied": False,
        "rendered_acceptance": False,
        "limitations": [
            "Replacing the prefix changes the f174 anchor; f174-relative scale is confounded.",
            "Same generated latent suffix does not imply equal decoded pixel suffix under temporal VAE context.",
            "The f178 tracking anchor is inside the suffix but can still depend on decoder context.",
            "Scale estimates describe matched static-image features and are not camera ground truth.",
            "Similarity, Sobel and photometric changes are diagnostic, not proof of rendering improvement.",
            "A VAE-only counterfactual cannot demonstrate correction of the source sampler trajectory.",
        ],
    }
    return results


class H3NativePrefixCounterfactual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE",),
                "bundle_path": ("STRING", {"default": ""}),
                "chunk_join_frame": ("INT", {"default": 175, "min": 1, "max": 10000000}),
            },
            "optional": {
                "static_roi_profile": (list(PROFILES), {"default": "01784_room"}),
                "static_roi_json": ("STRING", {"default": "", "multiline": True}),
                "feature_tracking_enabled": ("BOOLEAN", {"default": True}),
                "expected_manifest_sha256": ("STRING", {
                    "default": CAPTURE_01795_MANIFEST_SHA256,
                    "tooltip": "Exact 01795 capture hash. Clear to explicitly inspect another native carry capture.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("numerical_report",)
    FUNCTION = "audit"
    CATEGORY = "MiniMax H3/diagnostics"
    OUTPUT_NODE = True
    DESCRIPTION = "Decoder-only paired prefix intervention using the saved native source suffix; no sampling."

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        _AUDIT_MODEL_OWNERS.retain()
        return float("nan")

    def audit(
        self, video_vae, bundle_path, chunk_join_frame, static_roi_profile="01784_room",
        static_roi_json="", feature_tracking_enabled=True,
        expected_manifest_sha256=CAPTURE_01795_MANIFEST_SHA256,
    ):
        _AUDIT_MODEL_OWNERS.retain()
        import folder_paths
        from comfy.latent_formats import MiniMaxH3Video

        report = audit_native_prefix_counterfactual(
            video_vae, bundle_path, chunk_join_frame, MiniMaxH3Video().process_out,
            static_roi_profile=static_roi_profile, static_roi_json=static_roi_json,
            feature_tracking_enabled=feature_tracking_enabled,
            expected_manifest_sha256=expected_manifest_sha256 or None,
        )
        output = json.dumps(report, indent=2, allow_nan=False)
        directory = Path(folder_paths.get_output_directory()) / "h3_flow_regenerate" / "boundary_audits"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="native-prefix-counterfactual-",
            suffix=".json", dir=directory, delete=False
        ) as stream:
            stream.write(output + "\n")
        return {"ui": {"text": [output, f"Numerical report: {Path(stream.name).name}"]}, "result": (output,)}


NODE_CLASS_MAPPINGS = {"H3NativePrefixCounterfactual": H3NativePrefixCounterfactual}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3NativePrefixCounterfactual": "MiniMax H3 Native Prefix Counterfactual"
}
