"""Supply real right context to H3's existing overlapping temporal VAE decoder.

These LATENTs are decode-only views of accepted chunks. The original assembly
plan deliberately retains its original frame counts and trims the added future
frames. No sampler input, saved chunk, or audio tensor is changed.
"""

from __future__ import annotations

import hashlib
import logging

import torch

LOG = logging.getLogger(__name__)
_CYCLE = 5
_CYCLE_FRAMES = 17
_PREFIX_REMAINDER = 2


def video_latent_fingerprint(video: torch.Tensor) -> dict:
    """Dtype-independent identity of a caller-domain video latent.

    Values are hashed as contiguous CPU float32 so the same latent compares equal
    whether it is held as float32 or was stored in a wider type on another device.
    """
    values = video.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "sha256_float32": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
        "shape": list(values.shape),
        "dtype": str(video.dtype).removeprefix("torch."),
    }


def _frames(tokens: int) -> int:
    if tokens < 2 or tokens % _CYCLE != _PREFIX_REMAINDER:
        raise ValueError("Continuum decode context requires H3 video lengths of 5k+2 latents")
    return ((tokens - 2) // _CYCLE) * _CYCLE_FRAMES + 5


def prepare_decode_context(latents: list[dict], plan: dict) -> tuple[list[dict], str]:
    """Append one real decoder window stride at each bit-identical overlap.

    H3 decodes seven latents every five tokens, blending five pixel frames. A
    terminal chunk lacks the next window even when its protected overlap is exact.
    Appending five generated tokens makes that window available. Since independent
    chunk origins differ by complete five-token cycles, retained frames then use
    the same decoder windows as a continuously decoded latent timeline.

    Non-exact overlaps remain separate; inventing shared context for independently
    refined or Guide-mode chunks would change their decoder conditioning.
    """
    if not isinstance(plan, dict) or plan.get("magic") != "H3_CONTINUUM_ASSEMBLY_PLAN":
        raise ValueError("expected an H3 Continuum assembly plan")
    if plan.get("schema_version") != 1 or plan.get("fps") != 24:
        raise ValueError("unsupported H3 Continuum assembly plan schema or frame rate")
    groups = plan.get("decode_groups", plan.get("chunks"))
    if not isinstance(groups, list) or not groups or len(groups) != len(latents):
        raise ValueError("decode-context latent count must match physical assembly groups")

    videos = []
    for index, (latent, group) in enumerate(zip(latents, groups, strict=True)):
        video = latent.get("samples") if isinstance(latent, dict) else None
        if not torch.is_tensor(video) or video.ndim != 5 or tuple(video.shape[:2]) != (1, 24):
            raise ValueError(f"decode group {index + 1} requires native [1,24,T,H,W] video")
        if not video.is_floating_point():
            raise ValueError("H3 decode-context latents must be floating point")
        total = _frames(int(video.shape[2]))
        trim = int(group["trim_frames"])
        if total != int(group["total_frames"]) or total - trim != int(group["net_frames"]):
            raise ValueError(f"decode group {index + 1} latent duration differs from the assembly plan")
        if int(group["expected_video_latent_t"]) != video.shape[2] or trim < 0:
            raise ValueError(f"decode group {index + 1} has stale assembly metadata")
        videos.append(video)

    fingerprints = [video_latent_fingerprint(video) for video in videos]
    output = list(latents)
    reports = []
    joined = 0
    for index in range(len(videos) - 1):
        left, right = videos[index : index + 2]
        trim = int(groups[index + 1]["trim_frames"])
        if trim < 5 or (trim - 5) % _CYCLE_FRAMES:
            reports.append(f"boundary {index + 1}: unchanged (no native whole-cycle overlap)")
            continue
        prefix = ((trim - 5) // _CYCLE_FRAMES) * _CYCLE + 2
        if prefix > left.shape[2] or right.shape[2] - prefix < _CYCLE:
            reports.append(f"boundary {index + 1}: unchanged (insufficient overlap or generated context)")
            continue
        if left.shape[:2] != right.shape[:2] or left.shape[-2:] != right.shape[-2:]:
            reports.append(f"boundary {index + 1}: unchanged (spatial geometry differs)")
            continue
        if left.dtype != right.dtype or left.device != right.device:
            reports.append(f"boundary {index + 1}: unchanged (latent dtype/device differs)")
            continue
        if not torch.equal(left[:, :, -prefix:], right[:, :, :prefix]):
            reports.append(f"boundary {index + 1}: unchanged (protected overlap is not exact)")
            continue
        # New allocation: accepted CPU chunks remain immutable. Do not forward
        # stale noise masks or sampling metadata on an extended decode-only tensor.
        output[index] = {"samples": torch.cat((left, right[:, :, prefix : prefix + _CYCLE]), dim=2)}
        joined += 1
        reports.append(f"boundary {index + 1}: supplied 5 real future latents (17 decode-only frames)")
    report = (
        f"H3 Continuum decode context: {joined}/{max(0, len(videos) - 1)} exact boundaries. "
        "Use the original assembly plan; added frames are trimmed by Assemble.\n"
        + "\n".join(reports)
        + "\n"
        + "\n".join(
            f"input group {index + 1}: sha256_float32={fingerprint['sha256_float32']} "
            f"shape={fingerprint['shape']} dtype={fingerprint['dtype']}"
            for index, fingerprint in enumerate(fingerprints)
        )
    )
    LOG.info(
        "H3 Flow video decode-context receipt exact_boundaries=%d total_boundaries=%d "
        "right_context_latents=%d boundaries=%s",
        joined,
        max(0, len(videos) - 1),
        _CYCLE,
        reports,
    )
    for index, fingerprint in enumerate(fingerprints):
        LOG.info(
            "H3 Flow video decode-input fingerprint group=%d sha256_float32=%s shape=%s dtype=%s",
            index + 1,
            fingerprint["sha256_float32"],
            fingerprint["shape"],
            fingerprint["dtype"],
        )
    return output, report


class H3ContinuumDecodeContext:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "Place immediately before Video VAE Decode, after all sampling/refinement/upscaling. "
        "Supplies the next chunk's real decoder context at exact Continuum overlaps. "
        "Connect the unchanged assembly plan to Assemble; it trims the extra decode-only frames. "
        "Works with any progressive sampler. Output is for decoding only, not sampling or storage."
    )
    INPUT_IS_LIST = True
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("video_latents", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "prepare"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_latents": ("LATENT",), "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",)}}

    def prepare(self, video_latents, assembly_plan):
        if len(assembly_plan) != 1:
            raise ValueError("decode context requires one assembly plan for the video latent list")
        return prepare_decode_context(video_latents, assembly_plan[0])


def probe_image_geometry(images: torch.Tensor, *, join_frame: int, rois: dict) -> dict:
    """Static-background texture and geometry of an in-graph IMAGE timeline.

    Uses the Local Boundary Audit's ROI estimator on frames
    ``join_frame - 16 .. join_frame + 15`` of the given timeline, labelled by
    their index in it. The images are only read.
    """
    from .stage_static_roi_audit import measure_stage_static_rois

    if not torch.is_tensor(images) or images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("geometry probe expects an IMAGE tensor [frames,H,W,C]")
    first = max(0, int(join_frame) - 16)
    stop = min(int(images.shape[0]), int(join_frame) + 16)
    if not first < join_frame < stop:
        raise ValueError("geometry probe join frame is outside the image timeline")
    window = images[first:stop, ..., :3].detach().to(device="cpu", dtype=torch.float32)
    result = measure_stage_static_rois(
        window,
        list(range(first, stop)),
        join_frame=int(join_frame),
        rois=rois,
    )
    return {
        "policy": "h3_in_graph_image_geometry_probe_v1",
        "frames_in_timeline": int(images.shape[0]),
        "canvas_hw": [int(images.shape[1]), int(images.shape[2])],
        "measured_frames": [first, stop - 1],
        "images_modified": False,
        **result,
    }


class H3ContinuumImageGeometryProbe:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "Diagnostic. Measures static-background sharpness and position around a join on decoded or "
        "assembled IMAGE frames inside the graph, before video encoding, with the Local Boundary "
        "Audit's ROI estimator. Images pass through unchanged."
    )
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "probe"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "join_frame": ("INT", {"default": 175, "min": 1, "max": 100000}),
                "static_roi_profile": (["01784_room", "custom"], {"default": "01784_room"}),
                "static_roi_json": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def probe(self, images, join_frame, static_roi_profile, static_roi_json):
        import json

        from .stage_static_roi_audit import parse_static_rois

        rois = parse_static_rois(static_roi_profile, static_roi_json)
        report = json.dumps(
            probe_image_geometry(images, join_frame=int(join_frame), rois=rois),
            allow_nan=False,
        )
        LOG.info("H3 Flow image geometry probe %s", report)
        return images, report


NODE_CLASS_MAPPINGS = {
    "H3ContinuumDecodeContext": H3ContinuumDecodeContext,
    "H3ContinuumImageGeometryProbe": H3ContinuumImageGeometryProbe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumDecodeContext": "MiniMax H3 Continuum Decode Context",
    "H3ContinuumImageGeometryProbe": "MiniMax H3 Image Geometry Probe (diagnostic)",
}
