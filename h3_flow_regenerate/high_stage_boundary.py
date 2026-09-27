"""Bounded observations of existing high-stage model predictions.

The witness uses authoritative prefix values because Comfy's inpaint wrapper
restores those values after predict wrappers return. No observed tensor is
returned to sampling, and no additional model evaluation is performed.
"""

from __future__ import annotations

import contextlib
import time

import torch

from .geometry import unpack_streams
from .seam_diagnostics import measure_translation_trajectory


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
        self.previous_prediction = suffix.clone() if point == "before_flow" else None


@contextlib.contextmanager
def high_boundary_contract(binding, exact_prefix, shapes, *, measure):
    """Keep ownership and optional evidence scoped to one high-stage lifetime."""
    if binding.guidance_protected_prefix_t or binding.high_boundary_trace is not None:
        raise RuntimeError("nested high-stage boundary ownership is unsupported")
    binding.guidance_protected_prefix_t = int(exact_prefix.shape[2])
    try:
        if measure:
            binding.high_boundary_trace = HighStageBoundaryTrace(binding.metrics, exact_prefix, shapes)
        yield
    finally:
        trace = binding.high_boundary_trace
        binding.high_boundary_trace = None
        binding.guidance_protected_prefix_t = 0
        if trace is not None:
            binding.metrics.event(
                "partitioned_high_boundary_trace_complete",
                prediction_calls=trace.calls,
                observed_calls=min(trace.calls, trace.max_calls),
                truncated=trace.calls > trace.max_calls,
                max_calls=trace.max_calls,
            )
