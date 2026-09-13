"""PR #32 one-node H3 Continuum audio boundary diagnostic pipeline.

The user-facing path is strictly serial:

    Continuum audio_latents + Audio VAE + assembly_plan
        -> Audio Boundary Pipeline
        -> phase_aligned_audio
        -> Continuum Assemble.audio

00417 removed the audible click while the global 40-Hz latent-phase correction
was active, but the user reported a mild audio-quality regression.  00418 keeps
the click removed but confirms that the broader quality problem remains while
the generated latent edge is still anomalous. Earlier controls had already
falsified independent Core normalization and full right-side decoder context as
sufficient causes of the click. This revision therefore keeps the phase
correction, leaves the final AUDIO path unchanged, and adds a sampling-grid
phase measurement around the exact carried audio prefix.

The original LATENT groups and assembly plan are used to prove exact protected
carry, measure the generated-latent transition, and distinguish the integer
40-Hz latent origin from the ideal 24-fps physical-window origin. No additional
H3 evaluation or VAE decode is introduced. The older decode-context/oracle nodes
remain available as isolated diagnostics, but their extended/shared-gain AUDIO
is no longer used by this integrated control.
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction

import torch

from .pr32_audio_decode_context import _exact_audio_prefix_steps, _validate_plan
from .pr32_audio_decode_oracle import _core_divisor, _decode_raw, _mean_scalar, _sample_rate
from .pr32_audio_phase_align import phase_align_decoded_audio

_LOGGER = logging.getLogger(__name__)
_AUDIO_LATENT_HZ = 40
_AUDIO_SAMPLE_RATE = 32000
_SAMPLES_PER_LATENT = _AUDIO_SAMPLE_RATE // _AUDIO_LATENT_HZ
_VIDEO_FPS = 24


def _rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(value.float().square())).item())


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def inspect_generated_audio_latent_joins(audio_latents: list[dict], assembly_plan: dict) -> str:
    """Measure the exact carried-prefix -> newly-generated audio-latent edge.

    Decoder future-context, shared normalization and global 40-Hz phase can all
    agree closely while a generated-latent transition remains. Compare the
    first newly generated right-group latent against nearby ordinary temporal
    latent deltas and report both where that edge lands relative to the 24-fps
    retained-video cut and the fractional 40-Hz phase inherited by the copied
    prefix at the physical target origin.

    It is observational only and touches neither accepted latents nor AUDIO.
    """

    groups = _validate_plan(assembly_plan, len(audio_latents))
    tensors: list[torch.Tensor] = []
    for index, latent in enumerate(audio_latents):
        value = latent.get("samples") if isinstance(latent, dict) else None
        if not torch.is_tensor(value) or value.ndim != 4 or tuple(value.shape[:3]) != (1, 32, 2):
            raise ValueError(f"audio latent join group {index + 1} requires native [1,32,2,T] audio")
        tensors.append(value)

    reports = ["PR #32 generated-audio latent-join diagnostic"]
    global_origin_latent = 0
    retained_frame_cursor = 0
    for index in range(len(tensors) - 1):
        left = tensors[index]
        right = tensors[index + 1]
        trim_frames = int(groups[index + 1].get("trim_frames", -1))
        prefix = _exact_audio_prefix_steps(trim_frames)
        retained_frame_cursor += int(groups[index].get("net_frames", 0))
        if prefix is None or prefix <= 0 or prefix >= int(right.shape[-1]):
            reports.append(f"boundary {index + 1}: unavailable (no exact protected audio prefix)")
            global_origin_latent += int(left.shape[-1])
            continue
        if prefix > int(left.shape[-1]) or not torch.equal(left[..., -prefix:], right[..., :prefix]):
            reports.append(f"boundary {index + 1}: unavailable (protected audio carry is not exact)")
            global_origin_latent += max(0, int(left.shape[-1]) - prefix)
            continue

        edge_delta = right[..., prefix] - left[..., -1]
        edge_rms = _rms(edge_delta)

        left_span = min(9, int(left.shape[-1]))
        left_tail = left[..., -left_span:]
        left_steps = left_tail[..., 1:] - left_tail[..., :-1]
        left_step_rms = torch.sqrt(torch.mean(left_steps.float().square(), dim=(0, 1, 2)))
        left_baseline = float(torch.median(left_step_rms).item()) if left_step_rms.numel() else math.nan

        right_stop = min(int(right.shape[-1]), prefix + 9)
        right_head = right[..., prefix:right_stop]
        right_steps = right_head[..., 1:] - right_head[..., :-1]
        right_step_rms = torch.sqrt(torch.mean(right_steps.float().square(), dim=(0, 1, 2)))
        right_baseline = float(torch.median(right_step_rms).item()) if right_step_rms.numel() else math.nan

        baselines = [value for value in (left_baseline, right_baseline) if math.isfinite(value) and value > 0.0]
        local_baseline = sum(baselines) / len(baselines) if baselines else math.nan
        ratio = edge_rms / local_baseline if math.isfinite(local_baseline) and local_baseline > 0.0 else math.nan

        join_global_latent = global_origin_latent + int(left.shape[-1])
        right_origin_latent = join_global_latent - prefix
        physical_start_frame = retained_frame_cursor - trim_frames
        ideal_start_tick = Fraction(physical_start_frame * _AUDIO_LATENT_HZ, _VIDEO_FPS)
        carry_phase_ticks = Fraction(right_origin_latent, 1) - ideal_start_tick
        carry_phase_ms = float(carry_phase_ticks) * 1000.0 / _AUDIO_LATENT_HZ
        ideal_cut_tick = Fraction(retained_frame_cursor * _AUDIO_LATENT_HZ, _VIDEO_FPS)
        join_phase_ticks = Fraction(join_global_latent, 1) - ideal_cut_tick

        join_sample = join_global_latent * _SAMPLES_PER_LATENT
        video_cut_sample = round(retained_frame_cursor / _VIDEO_FPS * _AUDIO_SAMPLE_RATE)
        offset_samples = join_sample - video_cut_sample
        offset_ms = offset_samples / _AUDIO_SAMPLE_RATE * 1000.0
        reports.append(
            f"boundary {index + 1}: exact_prefix={prefix} right_origin_latent={right_origin_latent} "
            f"physical_start_frame={physical_start_frame} ideal_start_tick={_fraction_text(ideal_start_tick)} "
            f"carry_phase_ticks={_fraction_text(carry_phase_ticks)} carry_phase_ms={carry_phase_ms:+.4f} "
            f"join_global_latent={join_global_latent} join_phase_ticks={_fraction_text(join_phase_ticks)} "
            f"join_sample={join_sample} video_cut_sample={video_cut_sample} "
            f"join_minus_video_cut={offset_samples:+d}s ({offset_ms:+.4f}ms) "
            f"latent_edge_rms={edge_rms:.8f} left_local_step_rms={left_baseline:.8f} "
            f"right_local_step_rms={right_baseline:.8f} edge_over_local={ratio:.6f}"
        )

        global_origin_latent = right_origin_latent

    return "\n".join(reports)


def decode_native_core_audio(
    audio_latents: list[dict],
    vae,
    assembly_plan: dict,
) -> tuple[list[dict], str]:
    """Decode only the original physical groups using Core-equivalent normalization.

    Unlike the earlier oracle, this control does not append future latents to a
    preceding group and does not derive a shared-stream gain. Each original
    group is decoded once, exactly as a Core VAEDecodeAudio branch would be,
    leaving phase alignment as the only waveform intervention before assembly.
    """

    groups = _validate_plan(assembly_plan, len(audio_latents))
    rate = _sample_rate(vae)
    output: list[dict] = []
    reports = [
        "PR #32 native per-group Core audio decode control",
        "decode_context_extension=false shared_gain=false; final-path intervention=global_latent_phase_only",
    ]
    for index, (latent, group) in enumerate(zip(audio_latents, groups, strict=True)):
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if not torch.is_tensor(samples):
            raise ValueError(f"native audio decode group {index + 1} requires a tensor LATENT")
        expected_t = int(group.get("expected_audio_latent_t", 0))
        if expected_t != int(samples.shape[-1]):
            raise ValueError(
                f"native audio decode group {index + 1} has stale assembly metadata: "
                f"T={int(samples.shape[-1])}, expected={expected_t}"
            )
        raw = _decode_raw(vae, latent, index)
        divisor = _core_divisor(raw)
        waveform = raw / divisor
        reports.append(
            f"group {index + 1}: latent_t={int(samples.shape[-1])} "
            f"raw_samples={int(raw.shape[-1])} core_divisor={_mean_scalar(divisor):.8f}"
        )
        output.append({"waveform": waveform, "sample_rate": rate})
    return output, "\n".join(reports)


def decode_and_phase_align_audio_boundary(
    audio_latents: list[dict],
    vae,
    assembly_plan: dict,
) -> tuple[list[dict], str]:
    """Native-decode once, phase-align AUDIO, and report the latent edge."""

    latent_join_report = inspect_generated_audio_latent_joins(audio_latents, assembly_plan)
    native_audio, decode_report = decode_native_core_audio(
        audio_latents,
        vae,
        assembly_plan,
    )
    phase_aligned_audio, phase_report = phase_align_decoded_audio(
        native_audio,
        audio_latents,
        assembly_plan,
    )
    report = "\n".join(
        (
            "PR #32 integrated decoded-audio boundary pipeline",
            latent_join_report,
            decode_report,
            phase_report,
            "FINAL AUDIO OUTPUT = native per-group Core decode after global latent-phase alignment. "
            "No future-context extension, shared gain, resampling or crossfade. "
            "Connect phase_aligned_audio directly to Continuum Assemble.audio; keep Audio Seam = Off.",
        )
    )
    return phase_aligned_audio, report


class H3ContinuumAudioBoundaryPipelineDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "PR #32 integrated audio boundary control. Connect the ORIGINAL Continuum audio_latents, "
        "the Audio VAE, and the unchanged assembly_plan here, then connect only phase_aligned_audio "
        "to Continuum Assemble.audio. It measures the exact generated-latent boundary and the "
        "24-fps/40-Hz sampling-grid phase, decodes each original physical group once with Core-equivalent "
        "normalization, then applies only the global 40-Hz latent-phase correction. No future-context "
        "extension, shared gain, H3 call, resampling, or crossfade is added."
    )
    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("phase_aligned_audio", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "decode_and_align"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_latents": ("LATENT",),
                "vae": ("VAE",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
            }
        }

    def decode_and_align(self, audio_latents, vae, assembly_plan):
        if len(vae) != 1:
            raise ValueError("integrated audio boundary diagnostic requires exactly one VAE")
        if len(assembly_plan) != 1:
            raise ValueError("integrated audio boundary diagnostic requires exactly one assembly plan")
        result, report = decode_and_phase_align_audio_boundary(
            audio_latents,
            vae[0],
            assembly_plan[0],
        )
        _LOGGER.warning(report)
        return result, report


NODE_CLASS_MAPPINGS = {
    "H3ContinuumAudioBoundaryPipelineDiagnostic": H3ContinuumAudioBoundaryPipelineDiagnostic,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumAudioBoundaryPipelineDiagnostic": ("MiniMax H3 Continuum Audio Boundary Pipeline (PR32 Diagnostic)"),
}
