"""Offline continuation replay through the resident native H3 VAE.

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

from .geometry import resize_spatial_5d, resize_spatial_5d_h3_patch_lattice
from .partitioned_diagnostics import PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCES
from .stage_static_roi_audit import (
    PROFILES as STATIC_ROI_PROFILES,
)
from .stage_static_roi_audit import (
    compare_same_frame_stage_rois,
    measure_stage_static_rois,
    parse_static_rois,
)
from .transfer_lattice import H3_TRANSFER_LATTICE, measure_paired_prefix_affine

LOG = logging.getLogger(__name__)
STAGES = {
    "provider": "provider_native_clean_full",
    "pre_high": "pre_high_exact_restored_full",
    "first_high_before_flow": "first_high_before_flow_full",
    "first_high_after_flow": "first_high_after_flow_full",
    "final": "final_post_high_internal_clean_full",
}
SCOPES = ("stage_continuity", "transfer_and_decoder_context", "high_prediction_tone")
DETAIL_REGIONS = ("off", "upper_left")


class _AuditModelOwners:
    """Keep weakly registered patchers alive across diagnostic cache pruning.

    Core evaluates IS_CHANGED before pruning the preceding workflow's owners.
    Its public cache-provider lifecycle releases our replacement owners after
    the next non-audit prompt. Explicit unload and memory admission still work.
    This provider stores no cached outputs and does not change Core functions.
    """

    def __init__(self):
        self.models = ()
        self.claimed = False
        self.registered = False

    def retain(self):
        import comfy.model_management as model_management
        from comfy_execution.cache_provider import register_cache_provider

        # Take new owners before dropping the preceding audit's owners.
        self.models = tuple(model_management.loaded_models())
        self.claimed = True
        if not self.registered:
            register_cache_provider(self)
            self.registered = True

    def on_prompt_start(self, prompt_id):
        self.claimed = False

    def on_prompt_end(self, prompt_id):
        if not self.claimed:
            self.models = ()

    def should_cache(self, context, value=None):
        return False

    async def on_lookup(self, context):
        return None

    async def on_store(self, context, value):
        pass


_AUDIT_MODEL_OWNERS = _AuditModelOwners()


def replay_plan(metadata, join_frame):
    window = metadata["window"]
    prefix, temporal = window["prefix_t"], window["temporal"]
    band = metadata.get("target_band_tokens")
    head = metadata.get("target_band_transfer_start_t")
    uniform_source = metadata.get("spatial_stage_control") in PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCES
    if (
        type(prefix) is not int
        or type(temporal) is not int
        or type(band) is not int
        or (band != 0 if uniform_source else band <= 0)
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
    # With one generated source trajectory there is no band edge. Retain the
    # same four-token observation extent beyond the protected prefix instead.
    measured_head = prefix + 4 if uniform_source else head
    band_window_start = max(0, (measured_head // 5 - 1) * 5)
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


def load_replay_operands(
    bundle_path,
    join_frame,
    *,
    include_source=False,
    include_high_predictions=False,
    include_full_final=False,
):
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
    uniform_source = metadata.get("spatial_stage_control") in PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCES
    expected_provenance = (
        "actual_learned_provider_uniform_source"
        if uniform_source
        else "actual_learned_provider_before_target_band_splice"
    )
    if (
        manifest.get("schema") != 1
        or manifest.get("kind") != "h3_flow_native_boundary_decoder_window_evidence"
        or metadata.get("policy") != "native_boundary_decoder_window_evidence_v1"
        or metadata.get("first_high_actual") is not True
        or metadata.get("provider_clean_provenance") != expected_provenance
        or metadata.get("decoder_comparison_prefix") != "replace_with_authoritative_prefix_bytes"
        or metadata.get("process_latent_out_required_before_vae") is not True
        or metadata.get("full_video_snapshots") is not True
        or metadata.get("full_video_temporal_start_t") != 0
        or metadata.get("low_probe_native_carrier_decodable") is not uniform_source
    ):
        raise ValueError("local audit requires a complete target-band boundary bundle")
    plan = replay_plan(metadata, join_frame)
    entries = manifest["tensor_bytes"]
    hashes = {}

    def read(name):
        return read_verified_operand(directory, entries, name, hashes)

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
    full_final = None
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
        if stage == "final" and include_full_final:
            full_final = value.clone()
        stages[stage] = value[:, :, crop].clone()
        del value
    native = None
    if not uniform_source:
        native = read("low_probe_native_carrier_clean_full")
        if tuple(native.shape) != shape:
            raise ValueError("native low carrier geometry differs")
        lo = max(plan["prefix_t"], plan["token_start"])
        hi = min(plan["head_t"], plan["token_stop"])
        if not torch.equal(
            native[:, :, lo:hi].contiguous().view(torch.int32),
            stages["pre_high"][:, :, lo - plan["token_start"] : hi - plan["token_start"]]
            .contiguous()
            .view(torch.int32),
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
        if not uniform_source and not torch.equal(
            source[:, :, head:].contiguous().view(torch.int32),
            native[:, :, head:, :h, :w].contiguous().view(torch.int32),
        ):
            raise ValueError("reduced-grid handoff tail differs from native low/probe storage")
        native_head = prefix.clone() if uniform_source else native[:, :, :head].clone()
        native_head[:, :, : plan["prefix_t"]] = prefix
        # Older bundles used the physical projection even for uniform clips.
        # Never reinterpret those bytes under the corrected image-grid contract.
        projection_policy = metadata.get("source_prefix_projection_policy", H3_TRANSFER_LATTICE)
        if projection_policy == "half_pixel_latent_v1" and uniform_source:
            expected_head = resize_spatial_5d(native_head, h, w, mode="bicubic")
        elif projection_policy == "native_vae_rgb_roundtrip_v1" and uniform_source:
            from .decode_context import video_latent_fingerprint

            receipt = metadata.get("source_prefix_projection", {})
            actual = video_latent_fingerprint(source[:, :, :head])
            target = video_latent_fingerprint(prefix)
            if (
                receipt.get("policy") != projection_policy
                or receipt.get("prefix_t") != head
                or receipt.get("projected_prefix", {}).get("shape") != actual["shape"]
                or receipt.get("projected_prefix", {}).get("sha256_float32") != actual["sha256_float32"]
                or receipt.get("authoritative_prefix", {}).get("sha256_float32") != target["sha256_float32"]
            ):
                raise ValueError("saved VAE source prefix differs from its projection receipt")
            # A lossy native encode cannot be replayed by interpolation. The
            # hash-verified source operand is authoritative for this experiment.
            expected_head = source[:, :, :head]
        elif projection_policy == H3_TRANSFER_LATTICE:
            expected_head = resize_spatial_5d_h3_patch_lattice(native_head, h, w)
        else:
            raise ValueError("unsupported saved source prefix projection policy")
        # Saved projection ran on the production device. CPU reconstruction can
        # differ by float32 interpolation roundoff, so this check is numerical.
        if not torch.allclose(source[:, :, :head], expected_head, atol=1e-4, rtol=1e-5):
            raise ValueError("reduced-grid handoff head differs from the projected native head")
        stages = {"source_grid": source[:, :, crop].clone(), **stages}
    identity = {"manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(), "operand_sha256": hashes}
    if include_high_predictions:
        trace = metadata["window"].get("high_prediction_trace")
        if not isinstance(trace, dict) or trace.get("policy") != "bounded_high_prediction_windows_v1":
            raise ValueError("this bundle predates high-call capture; select a new capture_boundary_witness bundle")
        calls = trace.get("calls")
        omitted = trace.get("omitted_call_indices")
        if (
            trace.get("max_calls") != 16
            or not isinstance(calls, list)
            or not calls
            or len(calls) > 16
            or not isinstance(omitted, list)
            or any(type(i) is not int or not 0 <= i < 16 for i in omitted)
            or len(set(omitted)) != len(omitted)
        ):
            raise ValueError("invalid saved high prediction trace")
        window = metadata["window"]
        start, stop = plan["prefix_t"] - 2, plan["prefix_t"] + 5
        if (window.get("window_start_t"), window.get("window_stop_t"), window.get("window_tokens")) != (start, stop, 7):
            raise ValueError("saved high prediction windows differ from native boundary timing")
        indices = []
        records = []
        for call in calls:
            index = call.get("call_index") if isinstance(call, dict) else None
            sigma = call.get("sigma") if isinstance(call, dict) else None
            if (
                type(index) is not int
                or not 0 <= index < 16
                or index in omitted
                or (indices and index <= indices[-1])
                or type(sigma) not in (int, float)
                or not math.isfinite(sigma)
                or not 0 < sigma <= 1
                or type(call.get("actual")) is not bool
            ):
                raise ValueError("invalid high prediction call provenance")
            record = dict(call)
            for point in ("before_flow", "after_flow"):
                expected = (
                    f"first_high_{point}"
                    if index == metadata.get("first_high_call_index")
                    else f"high_prediction_{index:02d}_{point}"
                )
                if call.get(point) != expected:
                    raise ValueError("saved high prediction operand ownership differs")
                value = read(expected)
                if tuple(value.shape) != (1, 24, 7, *prefix.shape[-2:]):
                    raise ValueError("saved high prediction window geometry differs")
                value[:, :, :2] = prefix[:, :, start : plan["prefix_t"]]
                key = f"high_call_{index:02d}_{point}"
                stages[key] = value
                record["window_" + point] = key
            indices.append(index)
            records.append(record)
        if indices[0] != metadata.get("first_high_call_index") or calls[0]["actual"] is not True:
            raise ValueError("saved high prediction trace does not begin with the first actual call")
        identity["high_prediction_calls"] = records
        identity["omitted_high_call_indices"] = omitted
    if include_full_final:
        stages["_full_final_decode_validation"] = full_final
    return plan, stages, identity


def read_verified_operand(directory, entries, name, hashes):
    """Read one immutable witness operand; shared by narrowly scoped replays."""
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


def _resident_admit(vae, workspace):
    """Admit diagnostic VAE work without evicting unrelated loaded models.

    ``VAE.decode`` and even ``load_models_gpu(memory_required=0)`` can unload
    unrelated models to reserve workspace. A VAE already on its decode device is
    used as it is; partially loaded weights stream through ComfyUI's per-layer
    casting. A VAE that is not on its device is loaded only when free memory
    already covers its weights and the decode, so admission evicts nothing.
    """
    import comfy.model_management as model_management

    patcher = vae.patcher
    resident = any(model is patcher for model in model_management.loaded_models()) and (
        patcher.current_loaded_device() == vae.device
    )
    if not resident:
        missing = int(patcher.model_size())
        if patcher.current_loaded_device() == vae.device:
            missing = max(0, missing - int(patcher.loaded_size()))
        workspace = int(workspace)
        reserve = max(
            int(model_management.minimum_inference_memory()),
            workspace + int(model_management.extra_reserved_memory()),
        )
        # Core reserves 110% of missing device weights during admission, not
        # just their raw size. Off-device weights count as entirely missing.
        required = math.ceil(missing * 1.1) + reserve
        free = int(model_management.get_free_memory(vae.device))
        if free < required:
            raise RuntimeError(
                "Local Boundary Audit requires the connected VAE to be resident, or enough free memory to load it "
                f"without unloading other models ({required / 2**30:.2f} GiB needed, "
                f"{free / 2**30:.2f} GiB free)."
            )
        # Free memory already covers the weights and the workspace, so ComfyUI's
        # admission finds nothing to unload.
        model_management.load_models_gpu(
            [patcher], memory_required=workspace, force_full_load=bool(vae.disable_offload)
        )


def _resident_decode(vae, samples):
    """Reuse the connected VAE with the diagnostic residency contract."""
    import comfy.model_management as model_management

    # Resident/partially loaded VAEs need no admission or workspace estimate.
    resident = any(model is vae.patcher for model in model_management.loaded_models()) and (
        vae.patcher.current_loaded_device() == vae.device
    )
    _resident_admit(vae, 0 if resident else vae.memory_used_decode(samples.shape, vae.vae_dtype))
    try:
        with model_management.cuda_device_context(vae.device), torch.inference_mode():
            pixels = vae.first_stage_model.decode(samples.to(device=vae.device, dtype=vae.vae_dtype))
            pixels = pixels.to(device=vae.output_device, dtype=vae.vae_output_dtype(), copy=True)
            vae.process_output(pixels)
        return pixels.movedim(1, -1)
    except getattr(model_management, "OOM_EXCEPTION", torch.cuda.OutOfMemoryError):
        pass
    raise RuntimeError(
        "Local Boundary Audit ran out of decode memory. Resident models were preserved; "
        "the audit did not retry through the model-unloading decode path."
    ) from None


def _decode_owned_pixels(vae, latent, process_out, expected_frames):
    samples = process_out(latent.clone())
    if all(hasattr(vae, name) for name in ("patcher", "device", "vae_dtype", "output_device", "process_output")):
        decoded = _resident_decode(vae, samples)
    else:
        with torch.inference_mode():
            decoded = vae.decode(samples)
    expected_shape = (1, expected_frames, latent.shape[-2] * 16, latent.shape[-1] * 16, 3)
    if tuple(decoded.shape) != expected_shape or not bool(torch.isfinite(decoded).all()):
        raise ValueError("connected VAE returned unexpected native temporal-window pixels")
    if not bool(((decoded >= 0) & (decoded <= 1)).all()):
        raise ValueError("connected VAE must return finalized RGB pixels in [0,1]")
    return decoded


def _detail_crop(pixels, region):
    if region == "off":
        return None
    if region != "upper_left":
        raise ValueError(f"unsupported local boundary detail region: {region!r}")
    h, w = pixels.shape[1:3]
    stop_y, stop_x = max(1, round(h * 0.45)), max(1, round(w / 3))
    return pixels[:, :stop_y, :stop_x], {
        "region": region,
        "bounds_xyxy": [0, 0, stop_x, stop_y],
        "decoded_canvas_hw": [h, w],
        "geometry_units": "pixels_on_this_stage_native_decoder_grid_at_region_center",
    }


def _detail_pair(reference, candidate, labels, region, *, previous=None):
    selected = _detail_crop(reference, region)
    if selected is None:
        return None
    a, metadata = selected
    b = _detail_crop(candidate, region)[0]
    delta = b - a
    luma = delta @ torch.tensor([0.2126, 0.7152, 0.0722])
    result = {
        **metadata,
        "frame_labels": labels,
        "rgb_difference_rms": delta.square().mean((1, 2, 3)).sqrt().tolist(),
        "rgb_mean_change": delta.mean((1, 2)).tolist(),
        "luma_mean_change": luma.mean((1, 2)).tolist(),
        "geometry": geometry_comparison(a, b, labels),
    }
    if previous is not None:
        a_previous = _detail_crop(previous[0], region)[0]
        b_previous = _detail_crop(previous[1], region)[0]
        result["temporal_increment_change_rms"] = (
            ((b - b_previous) - (a - a_previous)).square().mean((1, 2, 3)).sqrt().tolist()
        )
    return result


def measure_window_context(vae, latent, process_out, plan, *, detail_region="off"):
    """Compare identical pixel times from two standalone seven-token contexts.

    These are finalized/clamped standalone pixels. Native production blending
    happens before clamping, so these outputs must not reassemble the video.
    """
    if detail_region not in DETAIL_REGIONS:
        raise ValueError(f"unsupported local boundary detail region: {detail_region!r}")
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
    result = {
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
    if detail_region != "off":
        result["detail_region"] = _detail_pair(left, right, labels, detail_region)
    return result


def audit_local_boundary(
    vae,
    bundle_path,
    join_frame,
    process_out,
    *,
    scope="stage_continuity",
    detail_region="off",
    static_roi_profile="off",
    static_roi_json="",
    validate_full_video_decoder=False,
    feature_tracking_enabled=False,
):
    if not isinstance(feature_tracking_enabled, bool):
        raise TypeError("feature tracking enabled must be boolean")
    if not isinstance(validate_full_video_decoder, bool):
        raise TypeError("full-video decode validation must be boolean")
    if scope not in SCOPES:
        raise ValueError(f"unsupported local boundary audit scope: {scope!r}")
    if detail_region not in DETAIL_REGIONS:
        raise ValueError(f"unsupported local boundary detail region: {detail_region!r}")
    static_rois = parse_static_rois(static_roi_profile, static_roi_json)
    if feature_tracking_enabled and not static_rois:
        raise ValueError("feature tracking requires static_roi_profile != off")
    if validate_full_video_decoder and scope == "high_prediction_tone":
        raise ValueError("full-video decode validation requires stage_continuity or transfer_and_decoder_context")
    if static_rois and scope == "high_prediction_tone":
        raise ValueError("static ROI stage measurements require stage_continuity or transfer_and_decoder_context scope")
    extended = scope == "transfer_and_decoder_context"
    native = getattr(vae, "first_stage_model", None)
    expected = {"tokens_chunk_size": 5, "token_overlap": 2, "frame_pre_padding": 3, "clip_length": 17}
    if native is None or any(getattr(native, key, None) != value for key, value in expected.items()):
        raise ValueError("connect the native MiniMax H3 video VAE used for production decoding")
    plan, stages, identity = load_replay_operands(
        bundle_path,
        join_frame,
        include_source=extended,
        include_high_predictions=scope == "high_prediction_tone",
        include_full_final=validate_full_video_decoder,
    )
    full_final = stages.pop("_full_final_decode_validation", None)
    if scope == "high_prediction_tone":
        high_report = _audit_high_prediction_tone(vae, process_out, plan, stages, identity, detail_region)
        high_report["static_roi_profile"] = static_roi_profile
        high_report["static_roi_measurement_enabled"] = False
        return high_report
    begin, end = plan["measured_local_frames"]
    labels = list(range(plan["decoded_origin_frame"] + begin, plan["decoded_origin_frame"] + end))
    report = {
        "policy": "local_target_band_native_window_audit_v2",
        "scope": scope,
        "detail_region": detail_region,
        "static_roi_profile": static_roi_profile,
        "static_roi_measurement_enabled": bool(static_rois),
        "static_roi_bounds_xyxy": {name: list(rect) for name, rect in static_rois.items()},
        "full_video_decoder_comparison_requested": validate_full_video_decoder,
        "feature_tracking_enabled": feature_tracking_enabled,
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
        if static_rois:
            report["stages"][name]["static_background_rois"] = measure_stage_static_rois(
                frames,
                [labels[0] - 1, *labels],
                join_frame=join_frame,
                rois=static_rois,
            )
        if detail_region != "off":
            selected, metadata = _detail_crop(frames, detail_region)
            selected_luma = selected @ torch.tensor([0.2126, 0.7152, 0.0722])
            report["stages"][name]["detail_region"] = {
                **metadata,
                "frame_labels": labels,
                "rgb_mean": selected[1:].mean((1, 2)).tolist(),
                "luma_mean": selected_luma[1:].mean((1, 2)).tolist(),
                "luma_std": selected_luma[1:].std((1, 2), correction=0).tolist(),
                "adjacent_rgb_difference_rms": (selected[1:] - selected[:-1]).square().mean((1, 2, 3)).sqrt().tolist(),
                "adjacent_luma_mean_change": (selected_luma[1:] - selected_luma[:-1]).mean((1, 2)).tolist(),
                "adjacent_frame_geometry": geometry_comparison(selected[:-1], selected[1:], labels),
            }
        if feature_tracking_enabled:
            from .feature_background_tracking import track_background_features

            report["stages"][name]["tracked_background_features"] = track_background_features(
                frames, [labels[0] - 1, *labels], join_frame=join_frame, rois=static_rois
            )
        if name == "source_grid":
            report["stages"][name]["state_role"] = "uniform_reduced_view_with_projected_target_grid_head"
            report["stages"][name]["native_reduced_grid_generation_for_head"] = False
        if extended:
            LOG.info("H3 local boundary audit: comparing decoder contexts of %s", name)
            report["stages"][name]["window_context"] = measure_window_context(
                vae, latent, process_out, plan, detail_region=detail_region
            )
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
        if detail_region != "off":
            report["comparisons"][f"{left}_to_{right}"]["detail_region"] = _detail_pair(
                a, b, labels, detail_region, previous=(a_previous, b_previous)
            )
    for name, frames in pixels.items():
        LOG.info("H3 local boundary audit: measuring temporal continuity of %s", name)
        report["stages"][name]["adjacent_frame_geometry"] = geometry_comparison(frames[:-1], frames[1:], labels)
        upper = frames[:, : round(frames.shape[1] * 0.45)]
        report["stages"][name]["adjacent_frame_geometry_upper45"] = geometry_comparison(upper[:-1], upper[1:], labels)
    if static_rois:
        report["static_background_same_frame_stage_pairs"] = {}
        roi_pairs = ([("source_grid", "provider")] if "source_grid" in pixels else []) + pairs
        for left, right in roi_pairs:
            if left not in pixels or right not in pixels:
                continue
            report["static_background_same_frame_stage_pairs"][f"{left}_to_{right}"] = compare_same_frame_stage_rois(
                pixels[left],
                pixels[right],
                [labels[0] - 1, *labels],
                join_frame=join_frame,
                rois=static_rois,
            )
    if validate_full_video_decoder:
        report["full_video_decoder_context_validation"] = _audit_full_video_decoder_context(
            vae,
            full_final,
            process_out,
            plan,
            pixels["final"],
            [labels[0] - 1, *labels],
            rois=static_rois,
        )
        report["extra_vae_calls"] += 1
    # Preserve the existing final-stage field for report consumers.
    report["final_adjacent_frame_geometry"] = report["stages"]["final"]["adjacent_frame_geometry"]
    return report


def _audit_full_video_decoder_context(vae, full_latent, process_out, plan, cropped_pixels, labels, *, rois):
    """Compare full native decode with the identical saved-state cropped replay.

    Decodes the *complete* final target-grid latent (potentially expensive),
    then takes exactly the matching global pixel times. Native VAE temporal
    context, origin and blend behaviour are therefore part of the comparison.
    Returns numerical data only; no rendered frames, blobs or latent tensors.
    """
    if full_latent is None or tuple(full_latent.shape[:3]) != (1, 24, int(plan["temporal"])):
        raise ValueError("full decoded context requires saved full final target-grid latents")
    total_tokens = int(full_latent.shape[2])
    if (total_tokens - 2) % 5:
        raise ValueError("full decoded context requires a native 5-token temporal stride")
    total_frames = (total_tokens - 2) // 5 * 17 + 5
    decoded = _decode_owned_pixels(vae, full_latent, process_out, total_frames)
    # Chunk two starts with already retained context. In native H3 timing,
    # these are the prefix decoded frames trimmed from the new chunk.
    trim = 17 * ((int(plan["prefix_t"]) - 2) // 5) + 5
    global_origin = int(plan["join_frame"]) - trim
    first = int(labels[0]) - global_origin
    last = int(labels[-1]) - global_origin + 1
    if first < 0 or last > total_frames or last - first != len(cropped_pixels):
        raise ValueError("full and cropped decoder frame/time ownership differs")
    full_pixels = decoded[0, first:last].detach().float().cpu().clone()
    del decoded
    if full_pixels.shape != cropped_pixels.shape or not torch.isfinite(full_pixels).all():
        raise ValueError("full native decoder returned mismatched comparison pixels")
    delta = full_pixels - cropped_pixels
    per_frame_rms = delta.square().mean(dim=(1, 2, 3)).sqrt()
    luma_weights = delta.new_tensor([0.2126, 0.7152, 0.0722])
    luma_change = (delta @ luma_weights).mean(dim=(1, 2))
    result = {
        "policy": "h3_native_full_vs_crop_decoder_window_v1",
        "full_latent_tokens": total_tokens,
        "full_decoded_frames": total_frames,
        "full_decoded_global_origin": global_origin,
        "cropped_replay_global_origin": int(plan["decoded_origin_frame"]),
        "frame_labels": list(labels),
        "same_saved_final_clean_state": True,
        "same_connected_native_video_vae": True,
        "generated_frames_altered": False,
        "production_output_modified": False,
        "extra_vae_calls": 1,
        "extra_h3_nfe": 0,
        "per_frame_rgb_difference_rms": per_frame_rms.tolist(),
        "per_frame_luma_mean_change": luma_change.tolist(),
        "geometry": geometry_comparison(cropped_pixels, full_pixels, labels),
    }
    if rois:
        result["static_roi_same_frame"] = compare_same_frame_stage_rois(
            cropped_pixels, full_pixels, labels, join_frame=plan["join_frame"], rois=rois
        )
        result["full_decoder_static_rois"] = measure_stage_static_rois(
            full_pixels, labels, join_frame=plan["join_frame"], rois=rois
        )
    return result


def _audit_high_prediction_tone(vae, process_out, plan, stages, identity, detail_region):
    # Local frames 5..16 are unblended output of this native seven-token
    # decoder window. Frame 5 is the predecessor of the first measured pair.
    # Exclude the join pair itself and the next window's overlap.
    labels = list(range(plan["join_frame"] + 1, plan["join_frame"] + 12))
    report = {
        "policy": "native_high_prediction_tone_audit_v1",
        "scope": "high_prediction_tone",
        "fps": 24,
        "detail_region": detail_region,
        **identity,
        "frame_labels": labels,
        "sampling_rerun": False,
        "production_output_modified": False,
        "rendered_acceptance": False,
        "extra_vae_calls": 0,
        "native_unblended_window_only": True,
        "model_forecast_solver_causality_established": False,
        "calls": [],
    }
    previous_after = None
    for call in identity["high_prediction_calls"]:
        decoded = {}
        measurements = {}
        for point in ("before_flow", "after_flow"):
            latent = stages[call["window_" + point]]
            pixels = _decode_owned_pixels(vae, latent, process_out, 22)[0, 5:17].detach().float().cpu().clone()
            report["extra_vae_calls"] += 1
            decoded[point] = pixels
            regions = {"full": pixels}
            if detail_region != "off":
                regions[detail_region] = _detail_crop(pixels, detail_region)[0]
            measurements[point] = {}
            for name, region in regions.items():
                luma = region @ torch.tensor([0.2126, 0.7152, 0.0722])
                measurements[point][name] = {
                    "luma_mean": luma[1:].mean((1, 2)).tolist(),
                    "luma_std": luma[1:].std((1, 2), correction=0).tolist(),
                    "adjacent_luma_mean_change": (luma[1:] - luma[:-1]).mean((1, 2)).tolist(),
                }
        row = {
            "call_index": call["call_index"],
            "sigma": call["sigma"],
            "actual": call["actual"],
            "prediction": measurements,
            "immediate_flow_change": _tone_pixel_change(decoded["before_flow"], decoded["after_flow"], detail_region),
        }
        if previous_after is not None:
            previous_index, previous_pixels = previous_after
            row["previous_captured_call_index"] = previous_index
            row["between_calls_change"] = _tone_pixel_change(previous_pixels, decoded["before_flow"], detail_region)
        previous_after = call["call_index"], decoded["after_flow"]
        report["calls"].append(row)
    return report


def _tone_pixel_change(before, after, detail_region):
    regions = {"full": (before, after)}
    if detail_region != "off":
        regions[detail_region] = (_detail_crop(before, detail_region)[0], _detail_crop(after, detail_region)[0])
    result = {}
    for name, (a, b) in regions.items():
        luma_delta = (b - a) @ torch.tensor([0.2126, 0.7152, 0.0722])
        result[name] = {
            "same_frame_luma_change": luma_delta[1:].mean((1, 2)).tolist(),
            "adjacent_luma_increment_change": (luma_delta[1:] - luma_delta[:-1]).mean((1, 2)).tolist(),
            "same_frame_rgb_change_rms": (b[1:] - a[1:]).square().mean((1, 2, 3)).sqrt().tolist(),
        }
    return result


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
                            "adds the saved reduced-grid view and same-time comparisons of both decoder windows. "
                            "High prediction tone uses two native-window decodes per captured high call, with "
                            "actual/forecast provenance and paired before/after Flow measurements."
                        ),
                    },
                ),
                "detail_region": (
                    list(DETAIL_REGIONS),
                    {
                        "default": "off",
                        "tooltip": (
                            "Upper left adds measurements over the left third of the upper 45% of each decoded canvas. "
                            "Uses the same decoded pixels without extra VAE calls. "
                            "A region can still contain subject motion."
                        ),
                    },
                ),
                "static_roi_profile": (
                    list(STATIC_ROI_PROFILES),
                    {
                        "default": "off",
                        "tooltip": (
                            "Read-only static-background sharpness, image shift and small zoom by decoded stage. "
                            "01784_room uses fixed bookshelf/picture/curtain/wall fractions; custom uses JSON. "
                            "For source-grid replay choose Transfer and decoder context."
                        ),
                    },
                ),
                "feature_tracking_enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Optional OpenCV forward/backward feature tracking on final decoded replay. "
                            "Measures cumulative background movement using RANSAC; read-only and slower."
                        ),
                    },
                ),
                "validate_full_video_decoder": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Extra expensive VAE decode of the full saved target-grid clean video. "
                            "Numerically compares its join pixels with the shorter native replay; "
                            "may require substantial extra VRAM and CPU RAM. Never changes generation."
                        ),
                    },
                ),
                "static_roi_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            'Custom: {"books":[0,.42,.13,.60],"curtain":[.83,.04,.99,.36]}. '
                            "Fractional XYXY coordinates, at least two ROIs."
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
        "Replay saved continuation operands across native VAE windows at the protected-prefix join. "
        "Supports target-band and uniform-source bundles. Use the resident production video VAE "
        "and the boundary bundle directory. Preserves loaded model owners through prompt cleanup; "
        "insufficient decode memory stops the audit. "
        "Linux paths and verified local WSL UNC paths are accepted. "
        "Set the assembled chunk-join frame, or zero for relative frame labels. "
        "Saves numerical JSON only; images and latent tensors remain local."
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
        audit_scope="stage_continuity",
        detail_region="off",
        static_roi_profile="off",
        static_roi_json="",
        validate_full_video_decoder=False,
        feature_tracking_enabled=False,
    ):
        # Also cover execution with intermediate caching disabled (no IS_CHANGED).
        _AUDIT_MODEL_OWNERS.retain()
        import folder_paths
        from comfy.latent_formats import MiniMaxH3Video

        report = audit_local_boundary(
            video_vae,
            bundle_path,
            chunk_join_frame,
            MiniMaxH3Video().process_out,
            scope=audit_scope,
            detail_region=detail_region,
            static_roi_profile=static_roi_profile,
            static_roi_json=static_roi_json,
            validate_full_video_decoder=validate_full_video_decoder,
            feature_tracking_enabled=feature_tracking_enabled,
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
