"""Supply real right context to H3's existing overlapping temporal VAE decoder.

These LATENTs are decode-only views of accepted chunks. The original assembly
plan deliberately retains its original frame counts and trims the added future
frames. No sampler input, saved chunk, or audio tensor is changed.
"""

from __future__ import annotations

import torch

_CYCLE = 5
_CYCLE_FRAMES = 17
_PREFIX_REMAINDER = 2


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
        "Use the original assembly plan; added frames are trimmed by Assemble.\n" + "\n".join(reports)
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


NODE_CLASS_MAPPINGS = {"H3ContinuumDecodeContext": H3ContinuumDecodeContext}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ContinuumDecodeContext": "MiniMax H3 Continuum Decode Context"}
