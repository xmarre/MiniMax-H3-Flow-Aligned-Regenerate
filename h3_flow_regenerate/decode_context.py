"""Supply real right context to H3's existing overlapping temporal VAE decoder.

These LATENTs are decode-only views of accepted chunks. The original assembly
plan deliberately retains its original frame counts and trims the added future
frames. No sampler input, saved chunk, or audio tensor is changed.
"""

from __future__ import annotations

from typing import Any

import torch

_CYCLE = 5
_CYCLE_FRAMES = 17
_PREFIX_REMAINDER = 2
_H3_VIDEO_CHANNELS = 24
_H3_AUDIO_CHANNELS = 32
_H3_AUDIO_CHANNELS_PER_SAMPLE = 2


def _frames(tokens: int) -> int:
    if tokens < 2 or tokens % _CYCLE != _PREFIX_REMAINDER:
        raise ValueError("Continuum decode context requires H3 video lengths of 5k+2 latents")
    return ((tokens - 2) // _CYCLE) * _CYCLE_FRAMES + 5


def _extract_decode_video(samples: Any, index: int) -> tuple[torch.Tensor, bool]:
    """Resolve split video or native joint H3 AV samples for decode-only use.

    Continuum itself exposes split video/audio LATENTs, while the integrated
    learned-upscale/refine node must rebuild native ``NestedTensor([video, audio])``
    samples for sampler 2. Decode Context is intentionally the adapter between
    those contracts: it accepts either representation and returns video-only
    decode views without mutating or rewriting the source AV state.
    """
    unbind = getattr(samples, "unbind", None)
    if bool(getattr(samples, "is_nested", False)) and callable(unbind):
        members = list(unbind())
        if len(members) != 2:
            raise ValueError(f"decode group {index + 1} native H3 AV samples require exactly [video, audio]")
        video, audio = members
        if (
            not torch.is_tensor(audio)
            or audio.ndim != 4
            or tuple(audio.shape[:3])
            != (
                1,
                _H3_AUDIO_CHANNELS,
                _H3_AUDIO_CHANNELS_PER_SAMPLE,
            )
        ):
            raise ValueError(f"decode group {index + 1} native H3 AV audio must be [1,32,2,T]")
        if not audio.is_floating_point():
            raise ValueError("H3 decode-context audio member must be floating point")
        joint_av = True
    else:
        video = samples
        joint_av = False

    if (
        not torch.is_tensor(video)
        or video.ndim != 5
        or tuple(video.shape[:2])
        != (
            1,
            _H3_VIDEO_CHANNELS,
        )
    ):
        raise ValueError(f"decode group {index + 1} requires native [1,24,T,H,W] video")
    if not video.is_floating_point():
        raise ValueError("H3 decode-context video latents must be floating point")
    return video, joint_av


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

    videos: list[torch.Tensor] = []
    decode_views: list[dict] = []
    joint_av_inputs = 0
    for index, (latent, group) in enumerate(zip(latents, groups, strict=True)):
        samples = latent.get("samples") if isinstance(latent, dict) else None
        video, joint_av = _extract_decode_video(samples, index)
        joint_av_inputs += int(joint_av)
        total = _frames(int(video.shape[2]))
        trim = int(group["trim_frames"])
        if total != int(group["total_frames"]) or total - trim != int(group["net_frames"]):
            raise ValueError(f"decode group {index + 1} latent duration differs from the assembly plan")
        if int(group["expected_video_latent_t"]) != video.shape[2] or trim < 0:
            raise ValueError(f"decode group {index + 1} has stale assembly metadata")
        videos.append(video)
        # Split Continuum video LATENTs keep their historical identity when no
        # extension is required. Joint AV sampler outputs cannot be sent to the
        # video VAE, so expose a minimal decode-only video view instead.
        decode_views.append({"samples": video} if joint_av else latent)

    output = list(decode_views)
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
        # New allocation: accepted chunks remain immutable. Do not forward stale
        # noise masks, joint AV wrappers, or sampling metadata on an extended
        # decode-only tensor.
        output[index] = {"samples": torch.cat((left, right[:, :, prefix : prefix + _CYCLE]), dim=2)}
        joined += 1
        reports.append(f"boundary {index + 1}: supplied 5 real future latents (17 decode-only frames)")
    representation = (
        f" Native joint AV inputs normalized for video decode: {joint_av_inputs}/{len(videos)}."
        if joint_av_inputs
        else ""
    )
    report = (
        f"H3 Continuum decode context: {joined}/{max(0, len(videos) - 1)} exact boundaries."
        f"{representation} Use the original assembly plan; added frames are trimmed by Assemble.\n" + "\n".join(reports)
    )
    return output, report


class H3ContinuumDecodeContext:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "Place immediately before Video VAE Decode, after all sampling/refinement/upscaling. "
        "Accepts either Continuum split video LATENTs or native joint H3 AV sampler output, "
        "normalizing the latter to a decode-only video view. Supplies the next chunk's real "
        "decoder context at exact Continuum overlaps. Connect the unchanged assembly plan to "
        "Assemble; it trims the extra decode-only frames. Output is for decoding only, not "
        "sampling or storage."
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


NODE_CLASS_MAPPINGS = {"H3ContinuumDecodeContext": H3ContinuumDecodeContext}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ContinuumDecodeContext": "MiniMax H3 Continuum Decode Context"}
