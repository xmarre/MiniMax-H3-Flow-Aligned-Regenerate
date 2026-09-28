"""Bounded reconciliation for exact-prefix target-high predictions.

ComfyUI evaluates MiniMax H3 before KSamplerX0Inpaint restores mask==0 output
values to the caller-owned latent.  A target-high prediction can therefore be
internally coherent across its predicted prefix/suffix boundary yet become
discontinuous when Comfy replaces only the prefix afterwards.

The video bridge below compensates exactly for that impending replacement.  For
each existing target-high prediction it measures the same-call difference
between the authoritative and predicted last-prefix token and transfers that
difference onto only the first generated suffix tokens with bounded monotonic
support.  It never replaces the suffix with a stored pre-high trajectory, so
target-high model and Flow evolution remain intact.

The optional audio reference remains a separate bounded clean-reference
experiment.  All work operates on existing predictions and adds no model,
sampler, provider, or VAE evaluation.
"""

from __future__ import annotations

import contextlib
import itertools
import time
from typing import Any

import torch

from .geometry import unpack_streams
from .seam_diagnostics import measure_translation_trajectory

HIGH_BOUNDARY_REFERENCE_POLICY = "high_stage_exact_prefix_reconciliation_v2"
HIGH_BOUNDARY_VIDEO_POLICY = "same_call_exact_prefix_replacement_bridge_v1"
HIGH_BOUNDARY_AUDIO_POLICY = "low_probe_clean_reference_v1"
HIGH_BOUNDARY_REFERENCE_WEIGHTS = (1.0, 0.75, 0.5, 0.25)


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
    """Reconcile one existing high prediction before Comfy restores its prefix."""

    def __init__(
        self,
        metrics,
        shapes,
        *,
        exact_video_prefix: torch.Tensor | None,
        audio_reference: torch.Tensor | None,
        exact_denoise_mask: torch.Tensor | None,
        weights: tuple[float, ...] = HIGH_BOUNDARY_REFERENCE_WEIGHTS,
    ):
        if not weights or any(not 0.0 < float(weight) <= 1.0 for weight in weights):
            raise ValueError("high-boundary reference weights must be non-empty values in (0, 1]")
        if any(float(left) < float(right) for left, right in itertools.pairwise(weights)):
            raise ValueError("high-boundary reference weights must be monotonically non-increasing")
        self.metrics = metrics
        self.shapes = shapes
        self.weights = tuple(float(weight) for weight in weights)
        self.calls = 0

        self.video_prefix_t = 0
        self.exact_video_tail = None
        self.video_support = 0
        if exact_video_prefix is not None:
            if exact_video_prefix.ndim != 5:
                raise ValueError("high-boundary exact video prefix must be native BxCxTxHxW")
            if not exact_video_prefix.is_floating_point():
                raise TypeError("high-boundary exact video prefix must be floating point")
            video_shape = tuple(shapes[0])
            prefix_t = int(exact_video_prefix.shape[2])
            if (
                prefix_t <= 0
                or prefix_t >= int(video_shape[2])
                or tuple(exact_video_prefix.shape[:2]) != tuple(video_shape[:2])
                or tuple(exact_video_prefix.shape[-2:]) != tuple(video_shape[-2:])
            ):
                raise ValueError("high-boundary exact video prefix geometry drifted")
            support = min(int(video_shape[2]) - prefix_t, len(self.weights))
            if support <= 0:
                raise ValueError("high-boundary exact video bridge has no generated suffix support")
            self.video_prefix_t = prefix_t
            self.video_support = support
            self.exact_video_tail = exact_video_prefix[:, :, -1].detach().clone()

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
            support = min(int(audio_reference.shape[-1]) - prefix, len(self.weights))
            if support <= 0:
                raise ValueError("high-boundary audio reference has no generated suffix support")
            self.audio_prefix_ticks = prefix
            self.audio_reference_suffix = audio_reference[..., prefix : prefix + support].detach().clone()

    @property
    def active(self) -> bool:
        return self.video_active or self.audio_active

    @property
    def video_active(self) -> bool:
        return self.exact_video_tail is not None

    @property
    def audio_active(self) -> bool:
        return self.audio_reference_suffix is not None

    def apply(self, packed: torch.Tensor, *, call_index: int, sigma: float, actual: bool) -> torch.Tensor:
        """Apply bounded post-Flow reconciliation to one existing prediction."""

        if not self.active:
            return packed
        started = time.perf_counter()
        original_video, original_audio = unpack_streams(packed, self.shapes)
        result = packed.clone()
        video, audio = unpack_streams(result, self.shapes)
        fields: dict[str, Any] = {
            "policy": HIGH_BOUNDARY_REFERENCE_POLICY,
            "video_policy": HIGH_BOUNDARY_VIDEO_POLICY if self.video_active else None,
            "audio_policy": HIGH_BOUNDARY_AUDIO_POLICY if self.audio_active else None,
            "call_index": int(call_index),
            "sigma": float(sigma),
            "actual": bool(actual),
            "application_point": "post_flow_pre_inpaint_restore",
            "authoritative_prefix_modified": False,
            "suffix_outside_support_modified": False,
            "extra_h3_nfe": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
        }

        if self.video_active:
            support = self.video_support
            stop = self.video_prefix_t + support
            exact_tail = self.exact_video_tail.to(device=video.device, dtype=torch.float32)
            predicted_tail = original_video[:, :, self.video_prefix_t - 1].to(torch.float32)
            before = original_video[:, :, self.video_prefix_t : stop].to(torch.float32)
            replacement_delta = exact_tail - predicted_tail
            native_edge = before[:, :, 0] - predicted_tail
            weights = self.weights[:support]
            for offset, weight in enumerate(weights):
                if weight == 0.0:
                    continue
                current = original_video[:, :, self.video_prefix_t + offset].to(torch.float32)
                video[:, :, self.video_prefix_t + offset] = (
                    current + float(weight) * replacement_delta
                ).to(dtype=video.dtype)
            after = video[:, :, self.video_prefix_t : stop].to(torch.float32)
            corrected_edge = after[:, :, 0] - exact_tail
            edge_error = corrected_edge - native_edge
            fields.update(
                video_applied=True,
                video_prefix_t=self.video_prefix_t,
                video_support_tokens=support,
                video_temporal_weights=list(weights),
                video_reference_domain="same_call_model_predicted_prefix_tail",
                video_prediction_prefix_replacement_rms=_rms(replacement_delta),
                video_prediction_prefix_replacement_abs_max=float(replacement_delta.abs().max().item()),
                video_native_edge_rms=_rms(native_edge),
                video_post_restore_edge_rms=_rms(corrected_edge),
                video_post_restore_edge_error_rms=_rms(edge_error),
                video_correction_rms=_rms(after - before),
            )
            if not torch.equal(video[:, :, : self.video_prefix_t], original_video[:, :, : self.video_prefix_t]):
                raise RuntimeError("high-boundary video bridge modified predicted prefix values")
            if not torch.equal(video[:, :, stop:], original_video[:, :, stop:]):
                raise RuntimeError("high-boundary video bridge escaped its bounded suffix support")
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
            before = original_audio[..., self.audio_prefix_ticks : stop].to(torch.float32)
            reference = self.audio_reference_suffix.to(device=audio.device, dtype=audio.dtype)
            weights = self.weights[:support]
            for offset, weight in enumerate(weights):
                target = reference[..., offset]
                if weight == 1.0:
                    audio[..., self.audio_prefix_ticks + offset] = target
                else:
                    current = original_audio[..., self.audio_prefix_ticks + offset].to(torch.float32)
                    blended = current + float(weight) * (target.to(torch.float32) - current)
                    audio[..., self.audio_prefix_ticks + offset] = blended.to(dtype=audio.dtype)
            after = audio[..., self.audio_prefix_ticks : stop].to(torch.float32)
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
                raise RuntimeError("high-boundary audio reference escaped its bounded suffix support")
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
        self.after_flow_prediction = None

    def observe(self, packed, *, point, call_index, sigma, actual):
        if call_index >= self.max_calls:
            return
        started = time.perf_counter()
        video, _ = unpack_streams(packed, self.shapes)
        suffix = video[:, :, self.prefix_t : self.prefix_t + self.suffix_tokens].detach()
        predicted_tail = video[:, :, self.prefix_t - 1 : self.prefix_t].detach()

        # The exact-restored witness is what Comfy will expose after
        # KSamplerX0Inpaint replaces mask==0 output rows.
        exact_witness = torch.cat((self.exact_tail.to(suffix), suffix), dim=2)
        trajectories = {
            name: measure_translation_trajectory(exact_witness, 1, backward_steps=0, roi_fraction=fraction)
            for name, fraction in (("upper45", 0.45), ("full", 1.0))
        }

        # Before the bridge, the raw predicted-prefix witness exposes whether the
        # H3 prediction itself is coherent and exact-prefix replacement is what
        # creates the apparent boundary.  It is diagnostic-only.
        predicted_prefix_trajectories = None
        if point != "after_bridge":
            predicted_witness = torch.cat((predicted_tail.to(suffix), suffix), dim=2)
            predicted_prefix_trajectories = {
                name: measure_translation_trajectory(
                    predicted_witness,
                    1,
                    backward_steps=0,
                    roi_fraction=fraction,
                )
                for name, fraction in (("upper45", 0.45), ("full", 1.0))
            }

        flow_delta_rms = None
        bridge_delta_rms = None
        if point == "before_flow":
            self.previous_prediction = suffix.clone()
            self.after_flow_prediction = None
        elif point == "after_flow":
            if self.previous_prediction is not None:
                flow_delta_rms = _rms(suffix.float() - self.previous_prediction.float())
            self.after_flow_prediction = suffix.clone()
        elif point == "after_bridge":
            if self.after_flow_prediction is not None:
                bridge_delta_rms = _rms(suffix.float() - self.after_flow_prediction.float())
            self.previous_prediction = None
            self.after_flow_prediction = None

        self.metrics.event(
            "partitioned_high_boundary_prediction",
            policy="high_boundary_prediction_v2",
            point=point,
            call_index=call_index,
            sigma=float(sigma),
            actual=bool(actual),
            domain="model_internal_predicted_clean",
            prefix_t=self.prefix_t,
            prefix_witness="authoritative_exact_tail_after_inpaint_restore",
            predicted_prefix_witness="same_call_model_predicted_tail",
            predicted_to_exact_prefix_rms=_rms(predicted_tail.float() - self.exact_tail.to(predicted_tail).float()),
            suffix_tokens=int(suffix.shape[2]),
            flow_suffix_delta_rms=flow_delta_rms,
            bridge_suffix_delta_rms=bridge_delta_rms,
            trajectories=trajectories,
            predicted_prefix_trajectories=predicted_prefix_trajectories,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            extra_h3_nfe=0,
            extra_vae_calls=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
            diagnostic_only=True,
        )


@contextlib.contextmanager
def high_boundary_contract(
    binding,
    exact_prefix,
    shapes,
    *,
    measure,
    video_exact_prefix_bridge: bool = False,
    audio_reference: torch.Tensor | None = None,
    exact_denoise_mask: torch.Tensor | None = None,
):
    """Keep high-stage ownership, bounded reconciliation, and evidence scoped to one lifetime."""

    if (
        binding.guidance_protected_prefix_t
        or binding.high_boundary_trace is not None
        or binding.high_boundary_anchor is not None
    ):
        raise RuntimeError("nested high-stage boundary ownership is unsupported")
    binding.guidance_protected_prefix_t = int(exact_prefix.shape[2])
    anchor = None
    try:
        if video_exact_prefix_bridge or audio_reference is not None:
            anchor = HighStageBoundaryReferenceAnchor(
                binding.metrics,
                shapes,
                exact_video_prefix=(exact_prefix if video_exact_prefix_bridge else None),
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
