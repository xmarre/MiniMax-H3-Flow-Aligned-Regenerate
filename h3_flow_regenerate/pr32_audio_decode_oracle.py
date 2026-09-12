"""PR #32 self-contained H3 Continuum audio boundary oracle.

This diagnostic replaces the separate latent-context -> Core VAEDecodeAudio ->
post-decode inspector chain with one node so the decisive waveform measurements
cannot be silently pruned from the executed ComfyUI graph.

It performs no H3 evaluations.  Each physical audio decode group is decoded
exactly once, matching Core VAEDecodeAudio's decode and normalization contract.
It also exposes a causal shared-stream-gain output that applies one normalization
divisor to all groups after deriving that divisor from the retained raw waveform
timeline.  This isolates independent per-chunk Core normalization from decoder
context/alignment without touching sampler state, video, masks, conditioning,
scheduler state, or the Continuum assembly plan.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from .pr32_audio_decode_context import (
    _format_match,
    _frame_sample,
    _validate_plan,
    prepare_audio_decode_context,
)

_AUDIO_CHANNELS = 32
_STEREO_CHANNELS = 2
_COMPARE_SECONDS = 0.25
_VIDEO_FPS = 24
_LOGGER = logging.getLogger(__name__)


def _native_audio_tensor(latent: dict, index: int) -> torch.Tensor:
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not torch.is_tensor(samples):
        raise ValueError(f"audio decode group {index + 1} requires a tensor LATENT")
    if samples.is_nested:
        samples = samples.unbind()[-1]
    if (
        samples.ndim != 4
        or tuple(samples.shape[:3]) != (1, _AUDIO_CHANNELS, _STEREO_CHANNELS)
    ):
        raise ValueError(
            f"audio decode group {index + 1} requires native [1,32,2,T] audio"
        )
    if not samples.is_floating_point():
        raise ValueError("H3 audio decode-oracle latents must be floating point")
    return samples


def _decode_raw(vae, latent: dict, index: int) -> torch.Tensor:
    samples = _native_audio_tensor(latent, index)
    audio = vae.decode(samples).movedim(-1, 1)
    if not torch.is_tensor(audio) or audio.ndim != 3:
        raise ValueError(
            f"audio VAE decode group {index + 1} returned unexpected shape "
            f"{tuple(audio.shape) if torch.is_tensor(audio) else type(audio).__name__}"
        )
    if not audio.is_floating_point():
        raise ValueError("decoded H3 audio waveform must be floating point")
    if not bool(torch.isfinite(audio).all().item()):
        raise ValueError(f"decoded audio group {index + 1} contains NaN or Inf")
    return audio


def _core_divisor(raw: torch.Tensor) -> torch.Tensor:
    divisor = torch.std(raw, dim=[1, 2], keepdim=True) * 5.0
    return torch.where(divisor < 1.0, torch.ones_like(divisor), divisor)


def _sample_rate(vae) -> int:
    rate = int(
        getattr(
            vae,
            "audio_sample_rate_output",
            getattr(vae, "audio_sample_rate", 44100),
        )
    )
    if rate <= 0:
        raise ValueError("audio VAE reported an invalid output sample rate")
    return rate


def _timeline_geometry(groups: list[dict], sample_rate: int) -> tuple[list[dict], int]:
    geometry: list[dict] = []
    frame_cursor = 0
    for index, group in enumerate(groups):
        trim_frames = int(group.get("trim_frames", -1))
        net_frames = int(group.get("net_frames", -1))
        if trim_frames < 0 or net_frames < 0:
            raise ValueError(f"audio decode group {index + 1} has invalid frame geometry")
        frame_stop = frame_cursor + net_frames
        trim_samples = _frame_sample(trim_frames, sample_rate)
        sample_start = _frame_sample(frame_cursor, sample_rate)
        sample_stop = _frame_sample(frame_stop, sample_rate)
        wanted = sample_stop - sample_start
        if wanted < 0:
            raise ValueError("decoded audio timeline interval is negative")
        geometry.append(
            {
                "trim_frames": trim_frames,
                "net_frames": net_frames,
                "frame_start": frame_cursor,
                "frame_stop": frame_stop,
                "trim_samples": trim_samples,
                "sample_start": sample_start,
                "sample_stop": sample_stop,
                "wanted": wanted,
            }
        )
        frame_cursor = frame_stop
    return geometry, frame_cursor


def _slice_consumed(waveform: torch.Tensor, trim_samples: int, wanted: int) -> torch.Tensor:
    result = waveform[..., trim_samples : trim_samples + wanted]
    missing = wanted - int(result.shape[-1])
    if missing > 0:
        if int(result.shape[-1]) > 0:
            result = torch.cat(
                (
                    result,
                    result[..., -1:].expand(*result.shape[:-1], missing).clone(),
                ),
                dim=-1,
            )
        else:
            result = F.pad(result, (0, missing))
    return result.contiguous()


def _jump(left: torch.Tensor, right: torch.Tensor) -> float:
    if int(left.shape[-1]) <= 0 or int(right.shape[-1]) <= 0:
        return math.nan
    return float(torch.mean(torch.abs(left[..., -1] - right[..., 0])).item())


def _mean_scalar(value: torch.Tensor) -> float:
    return float(value.detach().to(dtype=torch.float32, device="cpu").mean().item())


def _format_boundary_metrics(
    *,
    label: str,
    waveforms: list[torch.Tensor],
    segments: list[torch.Tensor],
    geometry: list[dict],
    sample_rate: int,
) -> list[str]:
    reports: list[str] = []
    compare_samples = max(8, round(sample_rate * _COMPARE_SECONDS))
    for boundary in range(len(waveforms) - 1):
        left = waveforms[boundary]
        right = waveforms[boundary + 1]
        left_segment = segments[boundary]
        right_segment = segments[boundary + 1]
        right_trim = int(geometry[boundary + 1]["trim_samples"])
        left_raw_stop = int(geometry[boundary]["trim_samples"]) + int(
            geometry[boundary]["wanted"]
        )

        protected = min(compare_samples, right_trim, int(left_segment.shape[-1]))
        if protected > 0:
            left_pre = left_segment[..., -protected:]
            right_pre = right[..., right_trim - protected : right_trim]
            pre_report = _format_match(f"{label}_precut", left_pre, right_pre)
        else:
            pre_report = f"{label}_precut=unavailable"

        left_future_available = max(0, int(left.shape[-1]) - left_raw_stop)
        right_future_available = max(0, int(right.shape[-1]) - right_trim)
        future_count = min(
            compare_samples, left_future_available, right_future_available
        )
        if future_count > 0:
            left_future = left[..., left_raw_stop : left_raw_stop + future_count]
            right_future = right[..., right_trim : right_trim + future_count]
            future_report = _format_match(
                f"{label}_future", left_future, right_future
            )
            continuous_left_jump = _jump(left_segment, left_future)
        else:
            future_report = f"{label}_future=unavailable"
            continuous_left_jump = math.nan

        actual_jump = _jump(left_segment, right_segment)
        reports.append(
            f"{label} boundary {boundary + 1}: {pre_report}; {future_report}; "
            f"{label}_actual_jump={actual_jump:.8f} "
            f"{label}_continuous_left_jump={continuous_left_jump:.8f}"
        )
    return reports


def decode_audio_boundary_oracle(
    latents: list[dict],
    vae,
    plan: dict,
) -> tuple[list[dict], list[dict], str]:
    """Decode once and return Core-equivalent and shared-stream-gain AUDIO lists.

    `core_audio` mirrors Core VAEDecodeAudio: each physical group receives its
    own global std-based divisor.

    `shared_gain_audio` uses the same raw VAE decodes but one divisor derived
    from the concatenated retained raw timeline.  It is a causal diagnostic for
    the independent-normalization hypothesis, not a production fix.
    """

    groups = _validate_plan(plan, len(latents))
    extended, context_report = prepare_audio_decode_context(latents, plan)
    rate = _sample_rate(vae)
    geometry, total_frames = _timeline_geometry(groups, rate)

    raw: list[torch.Tensor] = []
    core: list[torch.Tensor] = []
    core_divisors: list[torch.Tensor] = []
    raw_segments: list[torch.Tensor] = []

    for index, (latent, meta) in enumerate(zip(extended, geometry, strict=True)):
        waveform = _decode_raw(vae, latent, index)
        raw.append(waveform)
        divisor = _core_divisor(waveform)
        core_divisors.append(divisor)
        core.append(waveform / divisor)
        raw_segments.append(
            _slice_consumed(
                waveform,
                int(meta["trim_samples"]),
                int(meta["wanted"]),
            )
        )

    if any(tuple(item.shape[:-1]) != tuple(raw[0].shape[:-1]) for item in raw[1:]):
        raise ValueError("decoded audio batch/channel geometry changed between groups")
    if not raw_segments:
        raise RuntimeError("audio decode oracle received no groups")

    retained_raw = torch.cat(raw_segments, dim=-1)
    shared_divisor = _core_divisor(retained_raw)
    shared = [waveform / shared_divisor for waveform in raw]

    core_segments = [
        _slice_consumed(
            waveform,
            int(meta["trim_samples"]),
            int(meta["wanted"]),
        )
        for waveform, meta in zip(core, geometry, strict=True)
    ]
    shared_segments = [
        _slice_consumed(
            waveform,
            int(meta["trim_samples"]),
            int(meta["wanted"]),
        )
        for waveform, meta in zip(shared, geometry, strict=True)
    ]

    reports = [
        "PR #32 self-contained decoded-audio oracle",
        context_report,
        f"rate={rate}Hz total_retained_frames={total_frames} "
        f"shared_stream_divisor={_mean_scalar(shared_divisor):.8f}",
    ]
    for index, (waveform, divisor, meta) in enumerate(
        zip(raw, core_divisors, geometry, strict=True)
    ):
        exact_trim = (int(meta["trim_frames"]) * rate) % _VIDEO_FPS == 0
        raw_stop = int(meta["trim_samples"]) + int(meta["wanted"])
        extra = max(0, int(waveform.shape[-1]) - raw_stop)
        reports.append(
            f"group {index + 1}: raw_samples={int(waveform.shape[-1])} "
            f"trim={int(meta['trim_frames'])}f/{int(meta['trim_samples'])}s "
            f"exact_trim={exact_trim} consume={int(meta['wanted'])}s "
            f"raw_stop={raw_stop} extra_future={extra}s "
            f"core_divisor={_mean_scalar(divisor):.8f} "
            f"shared_divisor={_mean_scalar(shared_divisor):.8f}"
        )

    reports.extend(
        _format_boundary_metrics(
            label="raw",
            waveforms=[item.detach().to(device="cpu", dtype=torch.float32) for item in raw],
            segments=[item.detach().to(device="cpu", dtype=torch.float32) for item in raw_segments],
            geometry=geometry,
            sample_rate=rate,
        )
    )
    reports.extend(
        _format_boundary_metrics(
            label="core",
            waveforms=[item.detach().to(device="cpu", dtype=torch.float32) for item in core],
            segments=[item.detach().to(device="cpu", dtype=torch.float32) for item in core_segments],
            geometry=geometry,
            sample_rate=rate,
        )
    )
    reports.extend(
        _format_boundary_metrics(
            label="shared",
            waveforms=[item.detach().to(device="cpu", dtype=torch.float32) for item in shared],
            segments=[item.detach().to(device="cpu", dtype=torch.float32) for item in shared_segments],
            geometry=geometry,
            sample_rate=rate,
        )
    )
    reports.append(
        "core_audio reproduces independent Core VAEDecodeAudio normalization; "
        "shared_gain_audio is the causal shared-stream-gain intervention. "
        "Keep Continuum Audio Seam = Off."
    )

    core_audio = [
        {"waveform": waveform, "sample_rate": rate} for waveform in core
    ]
    shared_audio = [
        {"waveform": waveform, "sample_rate": rate} for waveform in shared
    ]
    return core_audio, shared_audio, "\n".join(reports)


class H3ContinuumAudioDecodeOracleDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "PR #32 self-contained diagnostic. Connect Continuum audio_latents, the "
        "Audio VAE and the unchanged assembly plan directly to this node. It "
        "replaces both the separate Audio Decode Context and Core VAE Decode "
        "nodes for the test, decodes each physical group once, logs raw/Core/"
        "shared-gain same-timeline boundary metrics, and outputs both the exact "
        "Core-normalized audio and a causal shared-stream-gain variant. Adds no "
        "H3 evaluations. Use Audio Seam = Off."
    )
    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("core_audio", "shared_gain_audio", "report")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_latents": ("LATENT",),
                "vae": ("VAE",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
            }
        }

    def decode(self, audio_latents, vae, assembly_plan):
        if len(vae) != 1:
            raise ValueError("audio decode oracle requires exactly one VAE")
        if len(assembly_plan) != 1:
            raise ValueError("audio decode oracle requires exactly one assembly plan")
        core_audio, shared_audio, report = decode_audio_boundary_oracle(
            audio_latents,
            vae[0],
            assembly_plan[0],
        )
        _LOGGER.warning(report)
        return core_audio, shared_audio, report


NODE_CLASS_MAPPINGS = {
    "H3ContinuumAudioDecodeOracleDiagnostic": H3ContinuumAudioDecodeOracleDiagnostic,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumAudioDecodeOracleDiagnostic": (
        "MiniMax H3 Continuum Audio Decode Oracle (PR32 Diagnostic)"
    ),
}
