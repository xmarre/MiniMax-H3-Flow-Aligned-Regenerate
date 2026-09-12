"""PR #32 diagnostic for Continuum's 24-fps / 40-Hz audio phase boundary.

H3 audio advances on an 800-sample latent grid (40 Hz), while Continuum places
physical chunk cuts on a 24-fps video timeline.  Exact protected overlaps such
as 39 video frames = 65 audio latents do not imply that every cumulative
retained-video boundary also lands on an integer audio latent.  When it does
not, slicing every independently decoded chunk only by ``trim_frames`` resets
the audio phase at the physical group origin.

This pass-through diagnostic derives each decode group's global audio-latent
origin from bit-identical protected carry and shifts only the decode-only AUDIO
container so Continuum's unchanged native frame-based trim selects the same
global 32-kHz phase.  It adds no H3 or VAE work, does not resample/crossfade,
and never touches sampler latents or the assembly plan.
"""

from __future__ import annotations

import logging

import torch

from .pr32_audio_decode_context import (
    _exact_audio_prefix_steps,
    _format_match,
    _frame_sample,
    _validate_plan,
)

_AUDIO_LATENT_HZ = 40
_COMPARE_SECONDS = 0.25
_LOGGER = logging.getLogger(__name__)


def _latent_tensor(latent: dict, index: int) -> torch.Tensor:
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not torch.is_tensor(samples) or samples.ndim != 4 or tuple(samples.shape[:3]) != (1, 32, 2):
        raise ValueError(f"audio phase group {index + 1} requires native [1,32,2,T] LATENT")
    if not samples.is_floating_point():
        raise ValueError("audio phase latents must be floating point")
    return samples


def _decoded_audio(audio: dict, index: int) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict):
        raise ValueError(f"audio phase group {index + 1} must be an AUDIO mapping")
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if not torch.is_tensor(waveform) or waveform.ndim != 3 or not waveform.is_floating_point():
        raise ValueError(f"audio phase group {index + 1} requires floating waveform [B,C,S]")
    if sample_rate <= 0 or sample_rate % _AUDIO_LATENT_HZ:
        raise ValueError("audio phase diagnostic requires a sample rate divisible by 40 Hz")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"audio phase group {index + 1} contains NaN or Inf")
    return waveform, sample_rate


def _shift_for_native_trim(waveform: torch.Tensor, delta_samples: int) -> torch.Tensor:
    """Make native trim N select original sample N-delta without resampling."""

    delta_samples = int(delta_samples)
    if delta_samples == 0:
        return waveform
    if delta_samples > 0:
        if int(waveform.shape[-1]) == 0:
            raise ValueError("cannot phase-shift an empty decoded waveform")
        prefix = waveform[..., :1].expand(*waveform.shape[:-1], delta_samples).clone()
        return torch.cat((prefix, waveform), dim=-1).contiguous()
    drop = -delta_samples
    if drop >= int(waveform.shape[-1]):
        raise ValueError("audio phase correction would consume the decoded waveform")
    return waveform[..., drop:].contiguous()


def phase_align_decoded_audio(
    audio: list[dict],
    audio_latents: list[dict],
    plan: dict,
) -> tuple[list[dict], str]:
    """Return decode-only AUDIO aligned to Continuum's cumulative video timeline."""

    if len(audio) != len(audio_latents):
        raise ValueError("decoded audio and audio latent group counts differ")
    groups = _validate_plan(plan, len(audio))
    latents = [_latent_tensor(item, index) for index, item in enumerate(audio_latents)]
    decoded = [_decoded_audio(item, index) for index, item in enumerate(audio)]
    rates = {rate for _waveform, rate in decoded}
    if len(rates) != 1:
        raise ValueError("audio sample rate changed between decoded groups")
    sample_rate = rates.pop()
    samples_per_latent = sample_rate // _AUDIO_LATENT_HZ

    # Global latent origins are authoritative only while every preceding
    # physical boundary proves exact protected carry.
    origins: list[int | None] = [0]
    boundary_reports: list[str] = []
    for index in range(1, len(latents)):
        previous_origin = origins[index - 1]
        trim_frames = int(groups[index].get("trim_frames", -1))
        prefix = _exact_audio_prefix_steps(trim_frames)
        left = latents[index - 1]
        right = latents[index]
        if (
            previous_origin is None
            or prefix is None
            or prefix <= 0
            or prefix > int(left.shape[-1])
            or prefix > int(right.shape[-1])
            or left.dtype != right.dtype
            or left.device != right.device
            or not torch.equal(left[..., -prefix:], right[..., :prefix])
        ):
            origins.append(None)
            boundary_reports.append(
                f"boundary {index}: phase alignment disabled (audio protected carry is not exact)"
            )
            continue
        origins.append(previous_origin + int(left.shape[-1]) - prefix)

    frame_cursor = 0
    output: list[dict] = []
    reports = [
        "PR #32 H3 Continuum audio latent-phase alignment (decode-only)",
        f"rate={sample_rate}Hz latent_hz={_AUDIO_LATENT_HZ} samples_per_latent={samples_per_latent}",
    ]
    reports.extend(boundary_reports)

    previous_original_segment: torch.Tensor | None = None
    previous_aligned_segment: torch.Tensor | None = None
    previous_waveform: torch.Tensor | None = None
    previous_phase_stop: int | None = None

    for index, ((waveform, rate), latent, group, origin_latents) in enumerate(
        zip(decoded, latents, groups, origins, strict=True)
    ):
        trim_frames = int(group.get("trim_frames", -1))
        net_frames = int(group.get("net_frames", -1))
        if trim_frames < 0 or net_frames < 0:
            raise ValueError(f"audio phase group {index + 1} has invalid frame geometry")
        expected_t = int(group.get("expected_audio_latent_t", 0))
        if expected_t != int(latent.shape[-1]):
            raise ValueError(
                f"audio phase group {index + 1} has stale assembly metadata: "
                f"T={int(latent.shape[-1])}, expected={expected_t}"
            )

        frame_stop = frame_cursor + net_frames
        native_trim = _frame_sample(trim_frames, rate)
        wanted = _frame_sample(frame_stop, rate) - _frame_sample(frame_cursor, rate)
        phase_trim: int | None = None
        delta = 0
        aligned = waveform
        reason = "native"
        if origin_latents is not None:
            desired_global_start = _frame_sample(frame_cursor, rate)
            phase_trim = desired_global_start - origin_latents * samples_per_latent
            if phase_trim < 0 or phase_trim > int(waveform.shape[-1]):
                raise ValueError(
                    f"audio phase group {index + 1} derived an out-of-range local phase offset {phase_trim}"
                )
            delta = native_trim - phase_trim
            aligned = _shift_for_native_trim(waveform, delta)
            reason = "global_latent_phase"

        original_segment = waveform[..., native_trim : native_trim + wanted]
        aligned_segment = aligned[..., native_trim : native_trim + wanted]
        if int(aligned_segment.shape[-1]) != wanted:
            raise ValueError(
                f"audio phase group {index + 1} still lacks retained samples after alignment: "
                f"{int(aligned_segment.shape[-1])} != {wanted}"
            )

        reports.append(
            f"group {index + 1}: origin_latent={origin_latents} trim={trim_frames}f "
            f"native_trim={native_trim}s phase_trim={phase_trim} phase_delta={delta:+d}s "
            f"({delta / rate * 1000.0:+.4f}ms) wanted={wanted}s "
            f"input_samples={int(waveform.shape[-1])} output_samples={int(aligned.shape[-1])} "
            f"reason={reason}"
        )

        if index > 0 and previous_original_segment is not None and previous_aligned_segment is not None:
            original_jump = float(
                torch.mean(torch.abs(previous_original_segment[..., -1] - original_segment[..., 0])).item()
            )
            aligned_jump = float(
                torch.mean(torch.abs(previous_aligned_segment[..., -1] - aligned_segment[..., 0])).item()
            )
            compare = max(8, round(rate * _COMPARE_SECONDS))
            if previous_waveform is not None and previous_phase_stop is not None:
                left_future_available = max(0, int(previous_waveform.shape[-1]) - previous_phase_stop)
                count = min(compare, left_future_available, int(aligned_segment.shape[-1]))
                if count > 0:
                    left_future = previous_waveform[..., previous_phase_stop : previous_phase_stop + count]
                    right_future = aligned_segment[..., :count]
                    match = _format_match("phase_future", left_future, right_future)
                else:
                    match = "phase_future=unavailable"
            else:
                match = "phase_future=unavailable"
            reports.append(
                f"boundary {index}: {match}; native_actual_jump={original_jump:.8f} "
                f"phase_actual_jump={aligned_jump:.8f}"
            )

        output.append({**audio[index], "waveform": aligned, "sample_rate": rate})
        previous_original_segment = original_segment
        previous_aligned_segment = aligned_segment
        previous_waveform = waveform
        previous_phase_stop = (phase_trim if phase_trim is not None else native_trim) + wanted
        frame_cursor = frame_stop

    return output, "\n".join(reports)


class H3ContinuumAudioPhaseAlignDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "PR #32 decode-only causal diagnostic for 24-fps video / 40-Hz H3 audio phase. "
        "Place after the self-contained Audio Decode Oracle and before Continuum Assemble. "
        "It derives each physical group's global audio-latent origin from exact protected carry, "
        "then shifts only the decoded AUDIO container so Assemble's unchanged trim selects the "
        "same 32-kHz phase. No resampling, crossfade, H3 call or VAE decode is added."
    )
    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("phase_aligned_audio", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "align"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "audio_latents": ("LATENT",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
            }
        }

    def align(self, audio, audio_latents, assembly_plan):
        if len(assembly_plan) != 1:
            raise ValueError("audio phase alignment requires exactly one assembly plan")
        result, report = phase_align_decoded_audio(audio, audio_latents, assembly_plan[0])
        _LOGGER.warning(report)
        return result, report


NODE_CLASS_MAPPINGS = {
    "H3ContinuumAudioPhaseAlignDiagnostic": H3ContinuumAudioPhaseAlignDiagnostic,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumAudioPhaseAlignDiagnostic": "MiniMax H3 Continuum Audio Phase Align (PR32 Diagnostic)",
}
