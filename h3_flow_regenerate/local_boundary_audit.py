"""Offline target-band replay through the connected native H3 VAE.

Only scalar measurements leave this node. Binary operands and decoded pixels
stay local; no sampler, production latent, or decoder configuration is modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from .geometry import resize_spatial_5d_h3_patch_lattice
from .transfer_lattice import measure_paired_prefix_affine

LOG = logging.getLogger(__name__)
STAGES = {
    "provider": "provider_native_clean_full",
    "pre_high": "pre_high_exact_restored_full",
    "first_high_before_flow": "first_high_before_flow_full",
    "first_high_after_flow": "first_high_after_flow_full",
    "final": "final_post_high_internal_clean_full",
}
SCOPES = ("stage_continuity", "transfer_and_decoder_context")


def replay_plan(metadata, join_frame):
    window = metadata["window"]
    prefix, temporal = window["prefix_t"], window["temporal"]
    band = metadata.get("target_band_tokens")
    head = metadata.get("target_band_transfer_start_t")
    if (
        type(prefix) is not int
        or type(temporal) is not int
        or type(band) is not int
        or band <= 0
        or type(head) is not int
        or head != prefix + band
        or prefix < 2
        or (prefix - 2) % 5
        or (temporal - 2) % 5
        or type(join_frame) is not int
        or join_frame < 0
    ):
        raise ValueError("local boundary audit requires native target-band timing")
    trim = 17 * ((prefix - 2) // 5) + 5
    if window.get("decoded_trim_frames") != trim:
        raise ValueError("saved prefix trim differs from native H3 timing")
    # Include the preceding prefix window and both contexts near the band/tail
    # edge. All starts retain the native five-token phase.
    start = max(0, (prefix // 5 - 1) * 5)
    band_window_start = max(0, (head // 5 - 1) * 5)
    stop = max(start + 12, band_window_start + 12)
    if stop > temporal:
        raise ValueError("saved timeline lacks the complete following decoder window")
    decoded_frames = (stop - start - 2) // 5 * 17 + 5
    origin = join_frame - trim + 17 * (start // 5)
    first = max(6 if start else 1, join_frame - 4 - origin)
    return {
        "prefix_t": prefix,
        "head_t": head,
        "temporal": temporal,
        "token_start": start,
        "token_stop": stop,
        "decoded_frames": decoded_frames,
        "decoded_origin_frame": origin,
        "shared_tokens": [[k, k + 2] for k in range(start + 5, stop - 2, 5)],
        "temporal_blend_local_frames": [
            [17 * (k - start) // 5, 17 * (k - start) // 5 + 5] for k in range(start + 5, stop - 2, 5)
        ],
        "measured_local_frames": [first, 17 * (band_window_start - start) // 5 + 26],
        "omitted_preceding_blend_local_frames": [0, 5] if start else [],
        "join_frame": join_frame,
    }


def normalize_bundle_path(bundle_path, *, platform, wsl_distro):
    """Translate only a verified local WSL share; preserve native path spelling."""
    path = os.fspath(bundle_path).strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in ('"', "'"):
        path = path[1:-1]
    if not path:
        raise ValueError("select the boundary bundle directory or its manifest.json")
    if platform != "posix":
        return path
    if path.startswith(("\\\\", "//")):
        parts = path.replace("\\", "/")[2:].split("/")
        if parts[0].casefold() in ("wsl.localhost", "wsl$"):
            if len(parts) < 3 or not parts[1]:
                raise ValueError("incomplete WSL path; use the Linux bundle path, starting with /")
            if not wsl_distro:
                raise ValueError(
                    "cannot verify this WSL share: WSL_DISTRO_NAME is unavailable; "
                    "use the Linux bundle path, starting with /"
                )
            if parts[1].casefold() != wsl_distro.casefold():
                raise ValueError(
                    f"WSL path names distribution {parts[1]!r}, but ComfyUI runs in {wsl_distro!r}; "
                    "use a bundle accessible to this distribution"
                )
            return "/" + "/".join(parts[2:])
        # A double-leading slash is also a valid POSIX spelling. Reject actual
        # backslash UNC paths, but leave unrelated POSIX // paths unchanged.
        if path.startswith("\\\\"):
            raise ValueError("Windows network paths are not local Linux paths; use the Linux bundle path")
    if re.match(r"^[A-Za-z]:", path):
        raise ValueError("Windows drive paths are not Linux paths; use the mounted Linux path, such as /mnt/c/...")
    return path


def load_replay_operands(bundle_path, join_frame, *, include_source=False):
    path = normalize_bundle_path(bundle_path, platform=os.name, wsl_distro=os.environ.get("WSL_DISTRO_NAME"))
    directory = Path(path).expanduser().resolve()
    if directory.name == "manifest.json":
        directory = directory.parent
    elif directory.is_file():
        raise ValueError("select the boundary bundle directory or its manifest.json")
    manifest_path = directory / "manifest.json"
    try:
        raw_manifest = manifest_path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Boundary bundle manifest not found: {manifest_path}. "
            "Select an existing exported bundle directory containing manifest.json "
            "and its .bin operands, or select manifest.json itself. "
            "On Linux, use /home/... or a WSL share for the running distribution."
        ) from exc
    manifest = json.loads(raw_manifest)
    metadata = manifest.get("metadata", {})
    if (
        manifest.get("schema") != 1
        or manifest.get("kind") != "h3_flow_native_boundary_decoder_window_evidence"
        or metadata.get("policy") != "native_boundary_decoder_window_evidence_v1"
        or metadata.get("first_high_actual") is not True
        or metadata.get("provider_clean_provenance") != "actual_learned_provider_before_target_band_splice"
        or metadata.get("decoder_comparison_prefix") != "replace_with_authoritative_prefix_bytes"
        or metadata.get("process_latent_out_required_before_vae") is not True
        or metadata.get("full_video_snapshots") is not True
        or metadata.get("full_video_temporal_start_t") != 0
        or metadata.get("low_probe_native_carrier_decodable") is not False
    ):
        raise ValueError("local audit requires a complete target-band boundary bundle")
    plan = replay_plan(metadata, join_frame)
    entries = manifest["tensor_bytes"]
    hashes = {}

    def read(name):
        entry = entries[name]
        shape = entry["shape"]
        if (
            entry.get("dtype") != "torch.float32"
            or entry.get("byte_order") != "native_torch_contiguous"
            or len(shape) != 5
            or shape[:2] != [1, 24]
            or any(type(n) is not int or n <= 0 for n in shape)
            or entry.get("nbytes") != math.prod(shape) * 4
        ):
            raise ValueError(f"invalid saved operand geometry: {name}")
        path = (directory / entry["file"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("saved tensor path leaves the selected bundle")
        if path.stat().st_size != entry["nbytes"]:
            raise ValueError(f"saved operand byte count differs: {name}")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"saved operand hash differs: {name}")
        value = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"saved operand is not finite: {name}")
        hashes[name] = digest
        return value

    prefix = read("authoritative_prefix_full")
    if prefix.shape[2] != plan["prefix_t"]:
        raise ValueError("authoritative prefix length differs from saved timing")
    shape = (1, 24, plan["temporal"], *prefix.shape[-2:])
    mask = read("initial_high_video_mask_full")
    if tuple(mask.shape) != shape or not bool((mask[:, :, : plan["prefix_t"]] == 0).all()):
        raise ValueError("saved high mask does not protect the authoritative prefix")
    if not bool((mask[:, :, plan["prefix_t"] :] == 1).all()):
        raise ValueError("saved high mask must leave every generated token editable")
    del mask
    stages = {}
    crop = slice(plan["token_start"], plan["token_stop"])
    for stage, name in STAGES.items():
        value = read(name)
        if tuple(value.shape) != shape:
            raise ValueError(f"saved stage geometry differs: {name}")
        if stage in ("pre_high", "final") and not torch.equal(
            value[:, :, : plan["prefix_t"]].contiguous().view(torch.int32),
            prefix.contiguous().view(torch.int32),
        ):
            raise ValueError(f"saved stage changed the authoritative prefix: {name}")
        # Restore only carried context for a fair decoded stage comparison.
        # The native band and transferred tail remain each stage's saved bytes.
        value[:, :, : plan["prefix_t"]] = prefix
        stages[stage] = value[:, :, crop].clone()
        del value
    native = read("low_probe_native_carrier_clean_full")
    if tuple(native.shape) != shape:
        raise ValueError("native low carrier geometry differs")
    lo = max(plan["prefix_t"], plan["token_start"])
    hi = min(plan["head_t"], plan["token_stop"])
    if not torch.equal(
        native[:, :, lo:hi].contiguous().view(torch.int32),
        stages["pre_high"][:, :, lo - plan["token_start"] : hi - plan["token_start"]].contiguous().view(torch.int32),
    ):
        raise ValueError("pre-high band differs from the native low/probe prediction")
    if include_source:
        source_grid = metadata.get("source_probe_clean_grid")
        if (
            not isinstance(source_grid, list)
            or len(source_grid) != 2
            or any(type(n) is not int or n <= 0 or n % 2 for n in source_grid)
            or any(a > b for a, b in zip(source_grid, prefix.shape[-2:], strict=True))
            or tuple(source_grid) == tuple(prefix.shape[-2:])
        ):
            raise ValueError("extended audit requires the saved uniform reduced-grid handoff view")
        source = read("source_probe_clean_full")
        if tuple(source.shape) != (*shape[:3], *source_grid):
            raise ValueError("saved reduced-grid handoff geometry differs from its metadata")
        h, w = source_grid
        head = plan["head_t"]
        if not torch.equal(
            source[:, :, head:].contiguous().view(torch.int32),
            native[:, :, head:, :h, :w].contiguous().view(torch.int32),
        ):
            raise ValueError("reduced-grid handoff tail differs from native low/probe storage")
        native_head = native[:, :, :head].clone()
        native_head[:, :, : plan["prefix_t"]] = prefix
        expected_head = resize_spatial_5d_h3_patch_lattice(native_head, h, w)
        # Saved projection ran on the production device. CPU reconstruction can
        # differ by float32 interpolation roundoff, so this check is numerical.
        if not torch.allclose(source[:, :, :head], expected_head, atol=1e-4, rtol=1e-5):
            raise ValueError("reduced-grid handoff head differs from the projected native head")
        stages = {"source_grid": source[:, :, crop].clone(), **stages}
    return plan, stages, {"manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(), "operand_sha256": hashes}


def geometry_comparison(reference, candidate, frame_labels):
    """Diagnostic same-pixel-time affine fit, including scale/shear gradients."""
    h, w = reference.shape[1:3]
    width = min(w, 192)
    height = round(h * width / w)
    if min(height, width) < 16:
        return {"status": "unsupported", "reason": "region_too_small"}
    a = F.interpolate(reference.permute(0, 3, 1, 2), (height, width), mode="area")
    b = F.interpolate(candidate.permute(0, 3, 1, 2), (height, width), mode="area")
    sx, sy = w / width, h / height
    fitted_frames = []
    textured = (a.var((-1, -2), correction=0).mean(1) > 1e-12) & (b.var((-1, -2), correction=0).mean(1) > 1e-12)
    for start in range(0, len(frame_labels), 4):
        stop = min(start + 4, len(frame_labels))
        indices = [i for i in range(start, stop) if bool(textured[i])]
        for i in range(start, stop):
            if not bool(textured[i]):
                fitted_frames.append({"frame": frame_labels[i], "status": "insufficient_texture"})
        if not indices:
            continue
        exact = a[indices].permute(1, 0, 2, 3)[None]
        learned = b[indices].permute(1, 0, 2, 3)[None]
        fit = measure_paired_prefix_affine(learned, exact, prefix_t=len(indices), frames=len(indices))
        for row in fit.get("frames", []):
            dx, dy = row["center_dx_dy_cells"]
            g = row["affine_displacement_gradients"]
            fitted_frames.append(
                {
                    "frame": frame_labels[indices[row["frame"]]],
                    "status": "estimated",
                    "center_dx_dy_pixels": [dx * sx, dy * sy],
                    "displacement_gradients": [[g[0][0], g[0][1] * sx / sy], [g[1][0] * sy / sx, g[1][1]]],
                    "zero_huber": row["zero_huber"],
                    "affine_huber": row["affine_huber"],
                    "normalized_tile_rms_3x3": row["normalized_tile_rms_3x3"],
                }
            )
    return {
        "status": "estimated" if bool(textured.any()) else "insufficient_texture",
        "analysis_hw": [height, width],
        "sign_convention": "sample candidate at (x-dx,y-dy) to compare with reference",
        "fit_is_diagnostic_not_causal_proof": True,
        "frames": sorted(fitted_frames, key=lambda row: row["frame"]),
    }


def _decode_owned_pixels(vae, latent, process_out, expected_frames):
    with torch.inference_mode():
        decoded = vae.decode(process_out(latent.clone()))
    expected_shape = (1, expected_frames, latent.shape[-2] * 16, latent.shape[-1] * 16, 3)
    if tuple(decoded.shape) != expected_shape or not bool(torch.isfinite(decoded).all()):
        raise ValueError("connected VAE returned unexpected native temporal-window pixels")
    if not bool(((decoded >= 0) & (decoded <= 1)).all()):
        raise ValueError("connected VAE must return finalized RGB pixels in [0,1]")
    return decoded


def measure_window_context(vae, latent, process_out, plan):
    """Compare identical pixel times from two standalone seven-token contexts.

    These are finalized/clamped standalone pixels. Native production blending
    happens before clamping, so these outputs must not reassemble the video.
    """
    overlap = []
    window_start = max(0, (plan["head_t"] // 5 - 1) * 5)
    first_offset = window_start - plan["token_start"]
    for offset, begin in ((first_offset, 17), (first_offset + 5, 0)):
        decoded = _decode_owned_pixels(vae, latent[:, :, offset : offset + 7], process_out, 22)
        overlap.append(decoded[0, begin : begin + 5].detach().float().cpu().clone())
        del decoded
    left, right = overlap
    origin = plan["decoded_origin_frame"] + first_offset // 5 * 17
    labels = list(range(origin + 17, origin + 22))
    delta = right - left
    luma = delta @ torch.tensor([0.2126, 0.7152, 0.0722])
    upper_h = round(left.shape[1] * 0.45)
    return {
        "frame_labels": labels,
        "token_contexts": [[window_start, window_start + 7], [window_start + 5, window_start + 12]],
        "domain": "finalized_standalone_window_rgb",
        "comparison": "right_window_vs_left_window_at_identical_pixel_times",
        "production_blends_before_pixel_clamp": True,
        "standalone_pixels_used_to_reassemble_output": False,
        "rgb_difference_rms": delta.square().mean((1, 2, 3)).sqrt().tolist(),
        "luma_mean_change": luma.mean((1, 2)).tolist(),
        "geometry": geometry_comparison(left, right, labels),
        "geometry_upper45": geometry_comparison(left[:, :upper_h], right[:, :upper_h], labels),
    }


def audit_local_boundary(vae, bundle_path, join_frame, process_out, *, scope="stage_continuity"):
    if scope not in SCOPES:
        raise ValueError(f"unsupported local boundary audit scope: {scope!r}")
    extended = scope == "transfer_and_decoder_context"
    native = getattr(vae, "first_stage_model", None)
    expected = {"tokens_chunk_size": 5, "token_overlap": 2, "frame_pre_padding": 3, "clip_length": 17}
    if native is None or any(getattr(native, key, None) != value for key, value in expected.items()):
        raise ValueError("connect the native MiniMax H3 video VAE used for production decoding")
    plan, stages, identity = load_replay_operands(bundle_path, join_frame, include_source=extended)
    begin, end = plan["measured_local_frames"]
    labels = list(range(plan["decoded_origin_frame"] + begin, plan["decoded_origin_frame"] + end))
    report = {
        "policy": "local_target_band_native_window_audit_v2",
        "scope": scope,
        "fps": 24,
        **identity,
        "plan": plan,
        "frames_before_join_are_discarded_chunk_context": True,
        "temporal_blend_reproduced_for_measured_frames": True,
        "connected_production_vae_used": True,
        "decoded_pixels_saved": False,
        "production_sampling_rerun": False,
        "production_output_modified": False,
        "rendered_acceptance": False,
        "extra_vae_calls": 0,
        "vae_class": f"{type(native).__module__}.{type(native).__qualname__}",
        "vae_dtype": str(getattr(vae, "vae_dtype", "unknown")),
        "torch_version": str(torch.__version__),
        "stages": {},
        "comparisons": {},
    }
    pixels = {}
    for name, latent in stages.items():
        LOG.info("H3 local boundary audit: decoding %s through the native temporal windows", name)
        decoded = _decode_owned_pixels(vae, latent, process_out, plan["decoded_frames"])
        pixels[name] = decoded[0, begin - 1 : end].detach().float().cpu().clone()
        report["extra_vae_calls"] += 1
        del decoded
        frames = pixels[name]
        luma = frames @ torch.tensor([0.2126, 0.7152, 0.0722])
        report["stages"][name] = {
            "frame_labels": labels,
            "decoded_pixel_hw": list(frames.shape[1:3]),
            "geometry_units": "pixels_on_this_stage_native_decoder_grid",
            "luma_mean": luma[1:].mean((1, 2)).tolist(),
            "luma_std": luma[1:].std((1, 2), correction=0).tolist(),
            "adjacent_rgb_difference_rms": (frames[1:] - frames[:-1]).square().mean((1, 2, 3)).sqrt().tolist(),
            "adjacent_luma_mean_change": (luma[1:] - luma[:-1]).mean((1, 2)).tolist(),
        }
        if name == "source_grid":
            report["stages"][name]["state_role"] = "uniform_reduced_view_with_projected_target_grid_head"
            report["stages"][name]["native_reduced_grid_generation_for_head"] = False
        if extended:
            LOG.info("H3 local boundary audit: comparing decoder contexts of %s", name)
            report["stages"][name]["window_context"] = measure_window_context(vae, latent, process_out, plan)
            report["extra_vae_calls"] += 2
    pairs = [
        ("provider", "pre_high"),
        ("pre_high", "first_high_before_flow"),
        ("first_high_before_flow", "first_high_after_flow"),
        ("first_high_after_flow", "final"),
    ]
    for left, right in pairs:
        LOG.info("H3 local boundary audit: comparing %s to %s", left, right)
        a, b = pixels[left][1:], pixels[right][1:]
        a_previous, b_previous = pixels[left][:-1], pixels[right][:-1]
        report["comparisons"][f"{left}_to_{right}"] = {
            "frame_labels": labels,
            "rgb_difference_rms": (b - a).square().mean((1, 2, 3)).sqrt().tolist(),
            # Difference between temporal increments, not a speech/scene-cut or
            # defect classifier. It distinguishes a stable per-stage change from
            # one that develops between these two adjacent pixel times.
            "temporal_increment_change_rms": (
                ((b - b_previous) - (a - a_previous)).square().mean((1, 2, 3)).sqrt().tolist()
            ),
            "geometry": geometry_comparison(a, b, labels),
            "geometry_upper45": geometry_comparison(
                a[:, : round(a.shape[1] * 0.45)], b[:, : round(b.shape[1] * 0.45)], labels
            ),
        }
    for name, frames in pixels.items():
        LOG.info("H3 local boundary audit: measuring temporal continuity of %s", name)
        report["stages"][name]["adjacent_frame_geometry"] = geometry_comparison(frames[:-1], frames[1:], labels)
        upper = frames[:, : round(frames.shape[1] * 0.45)]
        report["stages"][name]["adjacent_frame_geometry_upper45"] = geometry_comparison(upper[:-1], upper[1:], labels)
    # Preserve the existing final-stage field for report consumers.
    report["final_adjacent_frame_geometry"] = report["stages"]["final"]["adjacent_frame_geometry"]
    return report


class H3FlowLocalBoundaryAudit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE",),
                "bundle_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Bundle directory or manifest.json. Use a Linux path in WSL, "
                            "or a WSL UNC path for the running distribution."
                        ),
                    },
                ),
                "chunk_join_frame": ("INT", {"default": 0, "min": 0, "max": 10000000}),
            },
            "optional": {
                "audit_scope": (
                    list(SCOPES),
                    {
                        "default": "stage_continuity",
                        "tooltip": (
                            "Stage continuity uses five VAE calls. Transfer and decoder context uses eighteen: "
                            "adds the saved reduced-grid view and same-time comparisons of both decoder windows."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("numerical_report",)
    FUNCTION = "audit"
    CATEGORY = "MiniMax H3/diagnostics"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Replay saved target-band operands across native VAE windows spanning both joins. "
        "Use the production video VAE and the boundary bundle directory. "
        "Linux paths and verified local WSL UNC paths are accepted. "
        "Set the assembled chunk-join frame, or zero for relative frame labels. "
        "Saves numerical JSON only; images and latent tensors remain local."
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def audit(self, video_vae, bundle_path, chunk_join_frame, audit_scope="stage_continuity"):
        import folder_paths
        from comfy.latent_formats import MiniMaxH3Video

        report = audit_local_boundary(
            video_vae, bundle_path, chunk_join_frame, MiniMaxH3Video().process_out, scope=audit_scope
        )
        text = json.dumps(report, indent=2, allow_nan=False)
        directory = Path(folder_paths.get_output_directory()) / "h3_flow_regenerate" / "boundary_audits"
        directory.mkdir(parents=True, exist_ok=True)
        # Create a fresh report even when the selected bundle was audited before.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="boundary-", suffix=".json", dir=directory, delete=False
        ) as stream:
            stream.write(text + "\n")
        return {"ui": {"text": [text, f"Numerical report: {Path(stream.name).name}"]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {"H3FlowLocalBoundaryAudit": H3FlowLocalBoundaryAudit}
NODE_DISPLAY_NAME_MAPPINGS = {"H3FlowLocalBoundaryAudit": "MiniMax H3 Local Boundary Audit"}
