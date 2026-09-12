"""PR #32 diagnostic: give H3 audio decode real future latent context.

MiniMax-H3's BigVGAN audio decoder is non-causal: decoded samples near the end
of a latent tensor depend on right-hand latent context. Continuum normally
decodes each physical audio chunk independently, so the preceding chunk reaches
its join with decoder endpoint padding while the following chunk sees real
future latents. This module creates decode-only audio LATENT views that remove
that asymmetry at exact Native-Masked overlaps.

This is diagnostic-only. It never changes accepted sampler state, saved chunks,
video latents, the assembly plan, or H3 NFE. The original assembly plan remains
authoritative and trims/ignores all decode-only right context.
"""

from __future__ import annotations

import torch

_AUDIO_LATENT_FPS = 40
_VIDEO_FPS = 24
_AUDIO_CHANNELS = 32
_STEREO_CHANNELS = 2


def _exact_audio_prefix_steps(trim_frames: int) -> int | None:
    """Return the exact H3 audio-latent prefix for a 24-fps frame boundary."""

    numerator = int(trim_frames) * _AUDIO_LATENT_FPS
    if numerator < 0 or numerator % _VIDEO_FPS:
        return None
    return numerator // _VIDEO_FPS


def _validate_plan(plan: dict, latent_count: int) -> list[dict]:
    if not isinstance(plan, dict) or plan.get("magic") != "H3_CONTINUUM_ASSEMBLY_PLAN":
        raise ValueError("expected an H3 Continuum assembly plan")
    if plan.get("schema_version") != 1 or int(plan.get("fps", 0)) != _VIDEO_FPS:
        raise ValueError("unsupported H3 Continuum assembly plan schema or frame rate")
    groups = plan.get("decode_groups", plan.get("chunks"))
    if not isinstance(groups, list) or not groups or len(groups) != int(latent_count):
        raise ValueError("audio decode-context latent count must match physical assembly groups")
    return groups


def prepare_audio_decode_context(latents: list[dict], plan: dict) -> tuple[list[dict], str]:
    """Append the adjacent chunk's real generated audio suffix before VAE decode.

    The entire adjacent generated suffix is used deliberately for this causal
    diagnostic rather than guessing BigVGAN's finite right receptive field. The
    audio latent is tiny relative to H3 video state, and Assemble consumes only
    the original plan duration, so the extra decoded tail never enters output.

    Extension is authorized only when the overlap is on an exact 24-fps/40-Hz
    AV boundary and is bit-identical. Guide-mode/non-exact boundaries are left
    untouched rather than inventing decoder conditioning.
    """

    groups = _validate_plan(plan, len(latents))
    audios: list[torch.Tensor] = []
    for index, (latent, group) in enumerate(zip(latents, groups, strict=True)):
        audio = latent.get("samples") if isinstance(latent, dict) else None
        if (
            not torch.is_tensor(audio)
            or audio.ndim != 4
            or tuple(audio.shape[:3]) != (1, _AUDIO_CHANNELS, _STEREO_CHANNELS)
        ):
            raise ValueError(
                f"audio decode group {index + 1} requires native [1,32,2,T] audio"
            )
        if not audio.is_floating_point():
            raise ValueError("H3 audio decode-context latents must be floating point")
        expected_t = int(group.get("expected_audio_latent_t", 0))
        if expected_t != int(audio.shape[-1]):
            raise ValueError(
                f"audio decode group {index + 1} has stale assembly metadata: "
                f"T={int(audio.shape[-1])}, expected={expected_t}"
            )
        audios.append(audio)

    output = list(latents)
    reports: list[str] = []
    joined = 0
    for index in range(len(audios) - 1):
        left, right = audios[index : index + 2]
        trim_frames = int(groups[index + 1].get("trim_frames", -1))
        prefix = _exact_audio_prefix_steps(trim_frames)
        if prefix is None or prefix <= 0:
            reports.append(
                f"boundary {index + 1}: unchanged (no exact 24-fps/40-Hz audio boundary)"
            )
            continue
        if prefix > int(left.shape[-1]) or int(right.shape[-1]) <= prefix:
            reports.append(
                f"boundary {index + 1}: unchanged (insufficient exact overlap or future audio)"
            )
            continue
        if tuple(left.shape[:3]) != tuple(right.shape[:3]):
            reports.append(f"boundary {index + 1}: unchanged (audio geometry differs)")
            continue
        if left.dtype != right.dtype or left.device != right.device:
            reports.append(f"boundary {index + 1}: unchanged (audio dtype/device differs)")
            continue
        if not torch.equal(left[..., -prefix:], right[..., :prefix]):
            reports.append(
                f"boundary {index + 1}: unchanged (protected audio overlap is not exact)"
            )
            continue

        future = right[..., prefix:]
        if int(future.shape[-1]) <= 0:
            reports.append(f"boundary {index + 1}: unchanged (no generated future audio)")
            continue

        # Accepted chunks remain immutable. Expose a minimal decode-only LATENT
        # rather than forwarding stale masks or sampler metadata.
        output[index] = {"samples": torch.cat((left, future), dim=-1)}
        joined += 1
        reports.append(
            f"boundary {index + 1}: supplied {int(future.shape[-1])} real future audio "
            f"latents ({int(future.shape[-1]) / _AUDIO_LATENT_FPS:.3f}s) after exact "
            f"{prefix}-latent overlap"
        )

    report = (
        "PR #32 H3 Continuum audio decode context: "
        f"{joined}/{max(0, len(audios) - 1)} exact boundaries. "
        "This is a decode-only causal diagnostic; use the original assembly plan "
        "and Audio Seam = Off for the matched test.\n"
        + "\n".join(reports)
    )
    return output, report


class H3ContinuumAudioDecodeContextDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "PR #32 diagnostic. Place immediately before Audio VAE Decode on the "
        "Continuum audio_latents output. At exact Native-Masked boundaries it "
        "gives the preceding non-causal BigVGAN decode the next chunk's real "
        "future audio latents. Sampling, video, saved state and the assembly plan "
        "are untouched. Connect the unchanged assembly plan to Assemble and set "
        "Audio Seam to Off for the causal test."
    )
    INPUT_IS_LIST = True
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("audio_latents", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "prepare"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_latents": ("LATENT",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
            }
        }

    def prepare(self, audio_latents, assembly_plan):
        if len(assembly_plan) != 1:
            raise ValueError("audio decode context requires one assembly plan for the latent list")
        return prepare_audio_decode_context(audio_latents, assembly_plan[0])


NODE_CLASS_MAPPINGS = {
    "H3ContinuumAudioDecodeContextDiagnostic": H3ContinuumAudioDecodeContextDiagnostic,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumAudioDecodeContextDiagnostic": "MiniMax H3 Continuum Audio Decode Context (PR32 Diagnostic)",
}
