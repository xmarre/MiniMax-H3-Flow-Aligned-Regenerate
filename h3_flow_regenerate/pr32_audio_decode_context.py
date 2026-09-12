"""PR #32 diagnostics for H3 Continuum audio chunk boundaries.

The latent-side diagnostic gives the preceding chunk real generated future
latents before Audio VAE Decode.  The decoded-audio diagnostic then measures
the exact waveform regions that Continuum will retain and join.  Both are
observational/diagnostic only with respect to accepted sampler state, saved
chunks, video, masks, conditioning, scheduler state, and the assembly plan.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

_AUDIO_LATENT_FPS = 40
_VIDEO_FPS = 24
_AUDIO_CHANNELS = 32
_STEREO_CHANNELS = 2
_H3_AUDIO_SAMPLE_RATE = 32000
_COMPARE_SECONDS = 0.25
_LOGGER = logging.getLogger(__name__)


def _exact_audio_prefix_steps(trim_frames: int) -> int | None:
    """Return the exact H3 audio-latent prefix for a 24-fps frame boundary."""

    numerator = int(trim_frames) * _AUDIO_LATENT_FPS
    if numerator < 0 or numerator % _VIDEO_FPS:
        return None
    return numerator // _VIDEO_FPS


def _validate_plan(plan: dict, item_count: int) -> list[dict]:
    if not isinstance(plan, dict) or plan.get("magic") != "H3_CONTINUUM_ASSEMBLY_PLAN":
        raise ValueError("expected an H3 Continuum assembly plan")
    if plan.get("schema_version") != 1 or int(plan.get("fps", 0)) != _VIDEO_FPS:
        raise ValueError("unsupported H3 Continuum assembly plan schema or frame rate")
    groups = plan.get("decode_groups", plan.get("chunks"))
    if not isinstance(groups, list) or not groups or len(groups) != int(item_count):
        raise ValueError("audio diagnostic item count must match physical assembly groups")
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
            raise ValueError(f"audio decode group {index + 1} requires native [1,32,2,T] audio")
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
            reports.append(f"boundary {index + 1}: unchanged (no exact 24-fps/40-Hz audio boundary)")
            continue
        if prefix > int(left.shape[-1]) or int(right.shape[-1]) <= prefix:
            reports.append(f"boundary {index + 1}: unchanged (insufficient exact overlap or future audio)")
            continue
        if tuple(left.shape[:3]) != tuple(right.shape[:3]):
            reports.append(f"boundary {index + 1}: unchanged (audio geometry differs)")
            continue
        if left.dtype != right.dtype or left.device != right.device:
            reports.append(f"boundary {index + 1}: unchanged (audio dtype/device differs)")
            continue
        if not torch.equal(left[..., -prefix:], right[..., :prefix]):
            reports.append(f"boundary {index + 1}: unchanged (protected audio overlap is not exact)")
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
        "and Audio Seam = Off for the matched test.\n" + "\n".join(reports)
    )
    return output, report


def _frame_sample(frame: int, sample_rate: int) -> int:
    """Mirror Continuum V3's cumulative frame->sample rounding."""

    return round(float(int(frame)) / _VIDEO_FPS * int(sample_rate))


def _decoded_audio(audio: dict, index: int) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict):
        raise ValueError(f"decoded audio group {index + 1} must be an AUDIO mapping")
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        raise ValueError(f"decoded audio group {index + 1} requires waveform [B,C,S]")
    if not waveform.is_floating_point():
        raise ValueError("decoded H3 audio waveform must be floating point")
    if sample_rate <= 0:
        raise ValueError(f"decoded audio group {index + 1} has invalid sample rate")
    value = waveform.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"decoded audio group {index + 1} contains NaN or Inf")
    return value, sample_rate


def _rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(value.square())).item())


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    count = min(int(a.numel()), int(b.numel()))
    if count < 8:
        return 0.0
    left = a.reshape(-1)[:count]
    right = b.reshape(-1)[:count]
    left = left - left.mean()
    right = right - right.mean()
    denom = torch.sqrt(torch.sum(left.square()) * torch.sum(right.square())).clamp_min(1e-12)
    value = float((torch.sum(left * right) / denom).item())
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0


def _match_metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float, float, float]:
    count = min(int(a.shape[-1]), int(b.shape[-1]))
    if count <= 0:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    left = a[..., :count]
    right = b[..., :count]
    delta = left - right
    rmse = _rms(delta)
    reference = max((_rms(left) + _rms(right)) * 0.5, 1e-12)
    relative_rmse = rmse / reference
    right_rms = _rms(right)
    gain_ratio = _rms(left) / max(right_rms, 1e-12)
    dc_delta = float((left.mean() - right.mean()).item())
    return rmse, relative_rmse, _correlation(left, right), gain_ratio, dc_delta


def _format_match(label: str, a: torch.Tensor, b: torch.Tensor) -> str:
    rmse, relative_rmse, corr, gain, dc = _match_metrics(a, b)
    if not math.isfinite(rmse):
        return f"{label}=unavailable"
    return (
        f"{label}_rmse={rmse:.8f} {label}_rel={relative_rmse:.6f} "
        f"{label}_corr={corr:.6f} {label}_gain={gain:.6f} {label}_dc={dc:+.8f}"
    )


def inspect_decoded_audio_boundaries(audio: list[dict], plan: dict) -> tuple[list[dict], str]:
    """Measure the waveform regions actually consumed by Continuum assembly.

    This mirrors V3's frame/sample arithmetic without changing the audio.  For
    each exact chunk boundary it compares two same-timeline regions:

    * the retained left pre-cut tail against the right protected-prefix tail;
    * the left decode-only future supplied by the latent diagnostic against the
      right chunk after its trim point.

    The second comparison is especially useful because both tensors represent
    the same generated future.  A mismatch therefore localizes the remaining
    defect before assembly (decoder context and/or per-decode normalization),
    while a close match moves the investigation to the trim/copy boundary.
    """

    groups = _validate_plan(plan, len(audio))
    decoded: list[torch.Tensor] = []
    sample_rate: int | None = None
    reports = ["PR #32 decoded-audio boundary localization (pass-through)"]
    segments: list[torch.Tensor] = []
    raw_stops: list[int] = []
    trim_samples_by_group: list[int] = []
    frame_cursor = 0

    for index, (item, group) in enumerate(zip(audio, groups, strict=True)):
        waveform, rate = _decoded_audio(item, index)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"decoded audio sample rate changed between groups: {sample_rate} -> {rate}")
        if index and tuple(waveform.shape[:-1]) != tuple(decoded[0].shape[:-1]):
            raise ValueError("decoded audio batch/channel geometry changed between groups")
        decoded.append(waveform)

        trim_frames = int(group.get("trim_frames", -1))
        net_frames = int(group.get("net_frames", -1))
        if trim_frames < 0 or net_frames < 0:
            raise ValueError(f"decoded audio group {index + 1} has invalid frame geometry")
        frame_stop = frame_cursor + net_frames
        trim_samples = _frame_sample(trim_frames, rate)
        sample_start = _frame_sample(frame_cursor, rate)
        sample_stop = _frame_sample(frame_stop, rate)
        wanted = sample_stop - sample_start
        if wanted < 0:
            raise ValueError("decoded audio timeline interval is negative")

        segment = waveform[..., trim_samples : trim_samples + wanted]
        missing = wanted - int(segment.shape[-1])
        if missing > 0:
            if int(segment.shape[-1]) > 0:
                segment = torch.cat(
                    (segment, segment[..., -1:].expand(*segment.shape[:-1], missing).clone()), dim=-1
                )
            else:
                segment = F.pad(segment, (0, missing))
        segment = segment.contiguous()
        segments.append(segment)
        raw_stop = trim_samples + wanted
        raw_stops.append(raw_stop)
        trim_samples_by_group.append(trim_samples)

        post_std = float(waveform.std().item()) if waveform.numel() > 1 else 0.0
        normalization_signature = abs(post_std - 0.2) <= 5e-4
        extra = max(0, int(waveform.shape[-1]) - raw_stop)
        exact_trim = (trim_frames * rate) % _VIDEO_FPS == 0
        reports.append(
            f"group {index + 1}: rate={rate} samples={int(waveform.shape[-1])} "
            f"trim={trim_frames}f/{trim_samples}s exact_trim={exact_trim} "
            f"consume={wanted}s raw_stop={raw_stop} extra_future={extra}s "
            f"post_std={post_std:.8f} core_norm_signature={normalization_signature}"
        )
        frame_cursor = frame_stop

    if sample_rate is None:
        raise RuntimeError("decoded-audio diagnostic received no groups")
    if sample_rate != _H3_AUDIO_SAMPLE_RATE:
        reports.append(
            f"warning: expected MiniMax-H3 output rate {_H3_AUDIO_SAMPLE_RATE}, observed {sample_rate}"
        )

    compare_samples = max(8, round(sample_rate * _COMPARE_SECONDS))
    for boundary in range(len(decoded) - 1):
        left = decoded[boundary]
        right = decoded[boundary + 1]
        left_segment = segments[boundary]
        right_segment = segments[boundary + 1]
        right_trim = trim_samples_by_group[boundary + 1]
        left_raw_stop = raw_stops[boundary]

        exact_trim = (int(groups[boundary + 1].get("trim_frames", 0)) * sample_rate) % _VIDEO_FPS == 0
        protected = min(compare_samples, right_trim, int(left_segment.shape[-1]))
        if protected > 0:
            left_pre = left_segment[..., -protected:]
            right_pre = right[..., right_trim - protected : right_trim]
            pre_report = _format_match("precut", left_pre, right_pre)
        else:
            pre_report = "precut=unavailable"

        left_future_available = max(0, int(left.shape[-1]) - left_raw_stop)
        right_future_available = max(0, int(right.shape[-1]) - right_trim)
        future_count = min(compare_samples, left_future_available, right_future_available)
        if future_count > 0:
            left_future = left[..., left_raw_stop : left_raw_stop + future_count]
            right_future = right[..., right_trim : right_trim + future_count]
            future_report = _format_match("future", left_future, right_future)
            continuous_left_jump = float(
                torch.mean(torch.abs(left_segment[..., -1] - left_future[..., 0])).item()
            )
        else:
            future_report = "future=unavailable"
            continuous_left_jump = math.nan

        if int(left_segment.shape[-1]) > 0 and int(right_segment.shape[-1]) > 0:
            actual_jump = float(
                torch.mean(torch.abs(left_segment[..., -1] - right_segment[..., 0])).item()
            )
        else:
            actual_jump = math.nan

        reports.append(
            f"boundary {boundary + 1}: exact_trim={exact_trim} right_trim={right_trim}s "
            f"left_extra={left_future_available}s {pre_report}; {future_report}; "
            f"actual_jump={actual_jump:.8f} continuous_left_jump={continuous_left_jump:.8f}"
        )

    report = "\n".join(reports)
    return list(audio), report


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
        result, report = prepare_audio_decode_context(audio_latents, assembly_plan[0])
        _LOGGER.warning(report)
        return result, report


class H3ContinuumDecodedAudioBoundaryDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "PR #32 pass-through diagnostic. Place immediately after Audio VAE Decode "
        "and before Continuum Assemble. It mirrors the assembly trim/sample "
        "arithmetic and compares same-timeline protected/future waveform regions, "
        "including the decode-only future supplied by the companion latent node. "
        "It changes no audio samples and adds no H3 or VAE evaluations."
    )
    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "inspect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
            }
        }

    def inspect(self, audio, assembly_plan):
        if len(assembly_plan) != 1:
            raise ValueError("decoded-audio boundary diagnostic requires one assembly plan for the audio list")
        result, report = inspect_decoded_audio_boundaries(audio, assembly_plan[0])
        _LOGGER.warning(report)
        return result, report


NODE_CLASS_MAPPINGS = {
    "H3ContinuumAudioDecodeContextDiagnostic": H3ContinuumAudioDecodeContextDiagnostic,
    "H3ContinuumDecodedAudioBoundaryDiagnostic": H3ContinuumDecodedAudioBoundaryDiagnostic,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumAudioDecodeContextDiagnostic": "MiniMax H3 Continuum Audio Decode Context (PR32 Diagnostic)",
    "H3ContinuumDecodedAudioBoundaryDiagnostic": "MiniMax H3 Continuum Decoded Audio Boundary (PR32 Diagnostic)",
}
