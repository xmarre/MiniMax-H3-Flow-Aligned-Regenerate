"""PR #32 one-node H3 Continuum audio boundary diagnostic pipeline.

This node intentionally collapses the decoded-audio oracle and latent-phase
alignment intervention into one executable graph node.  The user-facing path is
therefore strictly serial:

    Continuum audio_latents + Audio VAE + assembly_plan
        -> Audio Boundary Pipeline
        -> phase_aligned_audio
        -> Continuum Assemble.audio

The original LATENT groups and assembly plan are still used internally to prove
exact protected carry, derive the global 40-Hz audio-latent origin, and measure
the actual generated-latent transition where the exact carried prefix ends and
the newly generated suffix begins. They are not parallel audio branches and are
never emitted as competing AUDIO outputs.

No additional H3 evaluation or VAE decode is introduced: the oracle decodes each
physical group once, then phase alignment operates only on those decoded AUDIO
containers.  The existing standalone diagnostic nodes remain available for
isolated investigation, but they are not required for the matched PR #32 test.
"""

from __future__ import annotations

import logging
import math

import torch

from .pr32_audio_decode_context import _exact_audio_prefix_steps, _validate_plan
from .pr32_audio_decode_oracle import decode_audio_boundary_oracle
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


def inspect_generated_audio_latent_joins(audio_latents: list[dict], assembly_plan: dict) -> str:
    """Measure the exact carried-prefix -> newly-generated audio-latent edge.

    00412 showed that decoder future-context, shared normalization and global
    40-Hz phase can all agree essentially perfectly while the audible hiccup
    remains. That moves the leading boundary upstream: the first right-group
    latent *after* the exact protected overlap. This diagnostic compares that
    edge against nearby ordinary temporal latent deltas and reports where the
    edge lands relative to the 24-fps retained video cut.

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

        # The first generated right latent corresponds to the global position
        # immediately after the final left latent. Consecutive latent values are
        # not expected to be equal, so compare its delta against local temporal
        # deltas rather than against zero.
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
        join_sample = join_global_latent * _SAMPLES_PER_LATENT
        video_cut_sample = round(retained_frame_cursor / _VIDEO_FPS * _AUDIO_SAMPLE_RATE)
        offset_samples = join_sample - video_cut_sample
        offset_ms = offset_samples / _AUDIO_SAMPLE_RATE * 1000.0
        reports.append(
            f"boundary {index + 1}: exact_prefix={prefix} join_global_latent={join_global_latent} "
            f"join_sample={join_sample} video_cut_sample={video_cut_sample} "
            f"join_minus_video_cut={offset_samples:+d}s ({offset_ms:+.4f}ms) "
            f"latent_edge_rms={edge_rms:.8f} left_local_step_rms={left_baseline:.8f} "
            f"right_local_step_rms={right_baseline:.8f} edge_over_local={ratio:.6f}"
        )

        # The next group's global origin is the left origin plus the non-overlap
        # length. Exact carry makes this identity unambiguous.
        global_origin_latent += int(left.shape[-1]) - prefix

    return "\n".join(reports)


def decode_and_phase_align_audio_boundary(
    audio_latents: list[dict],
    vae,
    assembly_plan: dict,
) -> tuple[list[dict], str]:
    """Decode once, phase-align AUDIO, and report the generated latent edge."""

    latent_join_report = inspect_generated_audio_latent_joins(audio_latents, assembly_plan)
    _core_audio, shared_audio, oracle_report = decode_audio_boundary_oracle(
        audio_latents,
        vae,
        assembly_plan,
    )
    phase_aligned_audio, phase_report = phase_align_decoded_audio(
        shared_audio,
        audio_latents,
        assembly_plan,
    )
    report = "\n".join(
        (
            "PR #32 integrated decoded-audio boundary pipeline",
            latent_join_report,
            oracle_report,
            phase_report,
            "FINAL AUDIO OUTPUT = shared-gain decoded waveform after global latent-phase alignment. "
            "Connect phase_aligned_audio directly to Continuum Assemble.audio; keep Audio Seam = Off.",
        )
    )
    return phase_aligned_audio, report


class H3ContinuumAudioBoundaryPipelineDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "PR #32 integrated audio boundary diagnostic. Connect the ORIGINAL Continuum "
        "audio_latents, the Audio VAE, and the unchanged assembly_plan here, then connect "
        "only phase_aligned_audio to Continuum Assemble.audio. Internally this measures the "
        "exact carried-prefix/generated-suffix latent edge, performs the self-contained decode "
        "oracle, then applies global 40-Hz latent-phase alignment. It adds no H3 call, no extra "
        "VAE decode, no resampling, and no crossfade."
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
