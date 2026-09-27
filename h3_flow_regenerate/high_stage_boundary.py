"""Bounded high-stage boundary ownership and model-output observations.

The exact-prefix continuation owns the caller prefix at the framework boundary.
When a handoff clean-reference is supplied, this module also keeps the first
generated clean-domain suffix values on that handoff trajectory while target-
high refinement takes over progressively. The correction is applied to existing
model predictions only; it adds no model, sampler, provider, or VAE work.
"""

from __future__ import annotations

import contextlib
import itertools
import time
from typing import Any

import torch

from .geometry import unpack_streams
from .seam_diagnostics import measure_translation_trajectory

HIGH_BOUNDARY_REFERENCE_POLICY = "handoff_clean_boundary_reference_v2"
HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS = (1.0, 0.75, 0.5, 0.25)

# The production AudioVAE is a finite-receptive-field BigVGAN. The pinned
# ComfyUI architecture needs fewer than 32 latent ticks of future context for
# any decoded sample. Hold the 500 ms (20 tick) seam window plus 32 ticks of
# future decoder context exactly on the low/probe clean trajectory, then release
# target-high ownership over another 32 ticks so the correction cannot create a
# delayed hard boundary at the end of its support.
HIGH_BOUNDARY_AUDIO_SEAM_TICKS = 20
HIGH_BOUNDARY_AUDIO_DECODER_CONTEXT_TICKS = 32
HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS = (
    HIGH_BOUNDARY_AUDIO_SEAM_TICKS + HIGH_BOUNDARY_AUDIO_DECODER_CONTEXT_TICKS
)
HIGH_BOUNDARY_AUDIO_RELEASE_TICKS = 32
HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS = HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS + HIGH_BOUNDARY_AUDIO_RELEASE_TICKS
HIGH_BOUNDARY_AUDIO_REFERENCE_WEIGHTS = (1.0,) * HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS + tuple(
    (HIGH_BOUNDARY_AUDIO_RELEASE_TICKS - offset) / (HIGH_BOUNDARY_AUDIO_RELEASE_TICKS + 1)
    for offset in range(HIGH_BOUNDARY_AUDIO_RELEASE_TICKS)
)


def _rms(value: torch.Tensor) -> float:
    return float(value.to(torch.float32).square().mean().sqrt().item())


def _exact_audio_prefix_ticks(mask_audio: torch.Tensor) -> int:
    """Return the canonical exact-prefix length for a native H3 audio mask."""

    if mask_audio.ndim != 4 or int(mask_audio.shape[2]) != 2:
        raise ValueError("high-boundary audio reference requires native BxCx2xT mask geometry")
    temporal_min = mask_audio.amin(dim=(0, 1, 2))
    temporal_max = mask_audio.amax(dim=(0, 1, 2))
    exact_zero = temporal_max <= 1e-8
    exact_one = temporal_min >= 1.0 - 1e-8
    temporal = int(mask_audio.shape[-1])
    prefix = 0
    while prefix < temporal and bool(exact_zero[prefix].item()):
        prefix += 1
    if prefix <= 0 or prefix >= temporal or not bool(exact_one[prefix:].all().item()):
        raise ValueError(
            "high-boundary audio reference requires a contiguous exact prefix followed by generated suffix"
        )
    return prefix


class HighStageBoundaryReferenceAnchor:
    """Blend a bounded high-stage clean prediction back to the handoff clean state."""

    def __init__(
        self,
        metrics,
        shapes,
        *,
        video_prefix_t: int,
        video_reference_suffix: torch.Tensor | None,
        audio_reference: torch.Tensor | None,
        exact_denoise_mask: torch.Tensor | None,
        video_weights: tuple[float, ...] = HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS,
        audio_weights: tuple[float, ...] = HIGH_BOUNDARY_AUDIO_REFERENCE_WEIGHTS,
    ):
        for name, weights in (("video", video_weights), ("audio", audio_weights)):
            if not weights or any(not 0.0 < float(weight) <= 1.0 for weight in weights):
                raise ValueError(f"high-boundary {name} reference weights must be non-empty values in (0, 1]")
            if any(float(left) < float(right) for left, right in itertools.pairwise(weights)):
                raise ValueError(f"high-boundary {name} reference weights must be monotonically non-increasing")
        self.metrics = metrics
        self.shapes = shapes
        self.video_prefix_t = int(video_prefix_t)
        self.video_weights = tuple(float(weight) for weight in video_weights)
        self.audio_weights = tuple(float(weight) for weight in audio_weights)
        self.calls = 0

        self.video_reference_suffix = None
        if video_reference_suffix is not None:
            if video_reference_suffix.ndim != 5:
                raise ValueError("high-boundary video reference must be native BxCxTxHxW")
            support = min(int(video_reference_suffix.shape[2]), len(self.video_weights))
            if support <= 0:
                raise ValueError("high-boundary video reference has no generated suffix support")
            self.video_reference_suffix = video_reference_suffix[:, :, :support].detach().clone()

        self.audio_prefix_ticks = 0
        self.audio_reference_suffix = None
        if audio_reference is not None:
            if exact_denoise_mask is None:
                raise ValueError("high-boundary audio reference requires the authoritative exact mask")
            if audio_reference.ndim != 4 or int(audio_reference.shape[2]) != 2:
                raise ValueError("high-boundary audio reference must be native BxCx2xT")
            mask_audio = unpack_streams(exact_denoise_mask, shapes)[1]
            if tuple(mask_audio.shape) != tuple(audio_reference.shape):
                raise ValueError("high-boundary audio reference/mask geometry drifted")
            prefix = _exact_audio_prefix_ticks(mask_audio)
            support = min(int(audio_reference.shape[-1]) - prefix, len(self.audio_weights))
            if support <= 0:
                raise ValueError("high-boundary audio reference has no generated suffix support")
            self.audio_prefix_ticks = prefix
            self.audio_reference_suffix = audio_reference[..., prefix : prefix + support].detach().clone()

    @property
    def active(self) -> bool:
        return self.video_reference_suffix is not None or self.audio_reference_suffix is not None

    @property
    def video_active(self) -> bool:
        return self.video_reference_suffix is not None

    @property
    def audio_active(self) -> bool:
        return self.audio_reference_suffix is not None

    def apply(self, packed: torch.Tensor, *, call_index: int, sigma: float, actual: bool) -> torch.Tensor:
        """Apply the bounded clean-reference blend to one existing high prediction."""

        if not self.active:
            return packed
        started = time.perf_counter()
        original_video, original_audio = unpack_streams(packed, self.shapes)
        result = packed.clone()
        video, audio = unpack_streams(result, self.shapes)
        fields: dict[str, Any] = {
            "policy": HIGH_BOUNDARY_REFERENCE_POLICY,
            "call_index": int(call_index),
            "sigma": float(sigma),
            "actual": bool(actual),
            "authoritative_prefix_modified": False,
            "suffix_outside_support_modified": False,
            "extra_h3_nfe": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
        }

        if self.video_reference_suffix is not None:
            support = int(self.video_reference_suffix.shape[2])
            stop = self.video_prefix_t + support
            if (
                self.video_prefix_t <= 0
                or stop > int(video.shape[2])
                or tuple(self.video_reference_suffix.shape[:2]) != tuple(video.shape[:2])
                or tuple(self.video_reference_suffix.shape[-2:]) != tuple(video.shape[-2:])
            ):
                raise RuntimeError("high-boundary video reference geometry drifted at model output")
            before = video[:, :, self.video_prefix_t : stop].detach().to(torch.float32).clone()
            reference = self.video_reference_suffix.to(device=video.device, dtype=video.dtype)
            weights = self.video_weights[:support]
            for offset, weight in enumerate(weights):
                target = reference[:, :, offset]
                if weight == 1.0:
                    video[:, :, self.video_prefix_t + offset] = target
                else:
                    current = video[:, :, self.video_prefix_t + offset].to(torch.float32)
                    blended = current + float(weight) * (target.to(torch.float32) - current)
                    video[:, :, self.video_prefix_t + offset] = blended.to(dtype=video.dtype)
            after = video[:, :, self.video_prefix_t : stop].detach().to(torch.float32)
            first_error = after[:, :, :1] - reference[:, :, :1].to(torch.float32)
            fields.update(
                video_applied=True,
                video_prefix_t=self.video_prefix_t,
                video_support_tokens=support,
                video_temporal_weights=list(weights),
                video_reference_domain="exact_restored_pre_high_clean",
                video_prediction_delta_rms=_rms(before - reference.to(torch.float32)),
                video_correction_rms=_rms(after - before),
                video_first_reference_error_rms=_rms(first_error),
            )
            if not torch.equal(video[:, :, : self.video_prefix_t], original_video[:, :, : self.video_prefix_t]):
                raise RuntimeError("high-boundary video reference modified the authoritative prefix")
            if not torch.equal(video[:, :, stop:], original_video[:, :, stop:]):
                raise RuntimeError("high-boundary video reference escaped its suffix support")
        else:
            fields.update(video_applied=False, video_support_tokens=0, video_temporal_weights=[])

        if self.audio_reference_suffix is not None:
            support = int(self.audio_reference_suffix.shape[-1])
            stop = self.audio_prefix_ticks + support
            if (
                self.audio_prefix_ticks <= 0
                or stop > int(audio.shape[-1])
                or tuple(self.audio_reference_suffix.shape[:-1]) != tuple(audio.shape[:-1])
            ):
                raise RuntimeError("high-boundary audio reference geometry drifted at model output")
            before = audio[..., self.audio_prefix_ticks : stop].detach().to(torch.float32).clone()
            reference = self.audio_reference_suffix.to(device=audio.device, dtype=audio.dtype)
            weights = self.audio_weights[:support]
            for offset, weight in enumerate(weights):
                target = reference[..., offset]
                if weight == 1.0:
                    audio[..., self.audio_prefix_ticks + offset] = target
                else:
                    current = audio[..., self.audio_prefix_ticks + offset].to(torch.float32)
                    blended = current + float(weight) * (target.to(torch.float32) - current)
                    audio[..., self.audio_prefix_ticks + offset] = blended.to(dtype=audio.dtype)
            after = audio[..., self.audio_prefix_ticks : stop].detach().to(torch.float32)
            first_error = after[..., :1] - reference[..., :1].to(torch.float32)
            fields.update(
                audio_applied=True,
                audio_prefix_ticks=self.audio_prefix_ticks,
                audio_support_ticks=support,
                audio_temporal_weights=list(weights),
                audio_reference_domain="low_probe_clean",
                audio_prediction_delta_rms=_rms(before - reference.to(torch.float32)),
                audio_correction_rms=_rms(after - before),
                audio_first_reference_error_rms=_rms(first_error),
            )
            if not torch.equal(audio[..., : self.audio_prefix_ticks], original_audio[..., : self.audio_prefix_ticks]):
                raise RuntimeError("high-boundary audio reference modified the authoritative prefix")
            if not torch.equal(audio[..., stop:], original_audio[..., stop:]):
                raise RuntimeError("high-boundary audio reference escaped its suffix support")
        else:
            fields.update(audio_applied=False, audio_support_ticks=0, audio_temporal_weights=[])

        self.calls += 1
        self.metrics.increment("high_boundary_reference_anchor_calls")
        self.metrics.event(
            "partitioned_high_boundary_reference_anchor",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            **fields,
        )
        return result


class HighStageBoundaryTrace:
    max_calls = 16
    suffix_tokens = 4

    def __init__(self, metrics, exact_prefix, shapes):
        self.metrics = metrics
        self.prefix_t = int(exact_prefix.shape[2])
        self.exact_tail = exact_prefix[:, :, -1:].detach().clone()
        self.shapes = shapes
        self.calls = 0
        self.previous_prediction = None

    def observe(self, packed, *, point, call_index, sigma, actual):
        if call_index >= self.max_calls:
            return
        started = time.perf_counter()
        video, _ = unpack_streams(packed, self.shapes)
        suffix = video[:, :, self.prefix_t : self.prefix_t + self.suffix_tokens].detach()
        # torch.cat owns this bounded witness; never modify a sampler operand.
        witness = torch.cat((self.exact_tail.to(suffix), suffix), dim=2)
        trajectories = {
            name: measure_translation_trajectory(witness, 1, backward_steps=0, roi_fraction=fraction)
            for name, fraction in (("upper45", 0.45), ("full", 1.0))
        }
        delta_rms = None
        if point == "after_flow" and self.previous_prediction is not None:
            delta_rms = float((suffix.float() - self.previous_prediction.float()).square().mean().sqrt().item())
        self.metrics.event(
            "partitioned_high_boundary_prediction",
            policy="high_boundary_prediction_v1",
            point=point,
            call_index=call_index,
            sigma=float(sigma),
            actual=bool(actual),
            domain="model_internal_predicted_clean",
            prefix_t=self.prefix_t,
            prefix_witness="authoritative_exact_tail_after_inpaint_restore",
            suffix_tokens=int(suffix.shape[2]),
            flow_suffix_delta_rms=delta_rms,
            trajectories=trajectories,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            extra_h3_nfe=0,
            extra_vae_calls=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
            diagnostic_only=True,
        )
        if point == "before_flow":
            self.previous_prediction = suffix.clone()
        elif point == "after_flow":
            self.previous_prediction = None


@contextlib.contextmanager
def high_boundary_contract(
    binding,
    exact_prefix,
    shapes,
    *,
    measure,
    video_reference_suffix: torch.Tensor | None = None,
    audio_reference: torch.Tensor | None = None,
    exact_denoise_mask: torch.Tensor | None = None,
):
    """Keep high-stage ownership, bounded correction, and evidence scoped to one lifetime."""

    if (
        binding.guidance_protected_prefix_t
        or binding.high_boundary_trace is not None
        or binding.high_boundary_anchor is not None
    ):
        raise RuntimeError("nested high-stage boundary ownership is unsupported")
    binding.guidance_protected_prefix_t = int(exact_prefix.shape[2])
    anchor = None
    try:
        if video_reference_suffix is not None or audio_reference is not None:
            anchor = HighStageBoundaryReferenceAnchor(
                binding.metrics,
                shapes,
                video_prefix_t=int(exact_prefix.shape[2]),
                video_reference_suffix=video_reference_suffix,
                audio_reference=audio_reference,
                exact_denoise_mask=exact_denoise_mask,
            )
            if anchor.active:
                binding.high_boundary_anchor = anchor
        if measure:
            binding.high_boundary_trace = HighStageBoundaryTrace(binding.metrics, exact_prefix, shapes)
        yield
    finally:
        trace = binding.high_boundary_trace
        active_anchor = binding.high_boundary_anchor
        binding.high_boundary_trace = None
        binding.high_boundary_anchor = None
        binding.guidance_protected_prefix_t = 0
        if trace is not None:
            binding.metrics.event(
                "partitioned_high_boundary_trace_complete",
                prediction_calls=trace.calls,
                observed_calls=min(trace.calls, trace.max_calls),
                truncated=trace.calls > trace.max_calls,
                max_calls=trace.max_calls,
            )
        if active_anchor is not None:
            binding.metrics.event(
                "partitioned_high_boundary_reference_complete",
                policy=HIGH_BOUNDARY_REFERENCE_POLICY,
                calls=active_anchor.calls,
                video_applied=active_anchor.video_active,
                audio_applied=active_anchor.audio_active,
                extra_h3_nfe=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
            )
