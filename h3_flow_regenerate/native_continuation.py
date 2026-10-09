"""One native target-grid sampler lifetime for carried-prefix continuation."""

from __future__ import annotations

import time

from .partitioned_diagnostics import (
    PARTITIONED_SPATIAL_STAGE_NATIVE_SINGLE,
    resolve_partitioned_uniform_source_detail_transport,
)
from .runtime import _begin_capture, _finish_capture, _flow_stage_contract


def run_native_continuation(
    executor,
    guider,
    binding,
    config,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask,
    callback,
    disable_pbar,
    seed,
    latent_shapes,
    exact_denoise_mask=None,
):
    """Keep native inpainting inputs and the entire original solver schedule.

    This is not the two-stage same-grid diagnostic: there is no clean probe,
    provider, re-noising, conditioning rebuild, refinement-stage intervention or
    solver/history restart inside a continuation chunk. Spectrum and native
    attention retain their own ordinary per-sampler lifetime policies.
    """
    transformer = guider.model_options["transformer_options"]
    if config.frame_gauge_repair or resolve_partitioned_uniform_source_detail_transport(transformer):
        raise ValueError("native_target_single_pass requires frame_gauge_repair and detail transport disabled")
    if len(latent_shapes) != 2 or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("native continuation requires a packed AV latent and a complete sigma schedule")
    if abs(float(sigmas[0]) - 1.0) > 1e-6 or abs(float(sigmas[-1])) > 1e-8:
        raise ValueError("native continuation requires the original full 1-to-0 sigma schedule")
    if binding.active_capture is not None:
        raise RuntimeError("nested native continuation capture is unsupported")

    started = time.perf_counter()
    event_start = len(binding.metrics.events)
    binding.metrics.event(
        "native_continuation_plan",
        policy=PARTITIONED_SPATIAL_STAGE_NATIVE_SINGLE,
        target_shape=tuple(latent_shapes[0]),
        schedule_steps=int(sigmas.numel() - 1),
        sampler_invocation_count=1,
        history_boundary_count=0,
        probe_model_calls=0,
        learned_provider_calls=0,
        prefix_projection_performed=False,
        handoff_renoise_performed=False,
        conditioning_rebuilt=False,
        exact_target_inputs_forwarded=True,
        sampler_mask_modified=denoise_mask is not exact_denoise_mask,
        progressive_guidance_applied=False,
        rendered_acceptance=False,
    )
    binding.metrics.increment("progressive_sampler_invocations")
    binding.metrics.increment("native_continuation_runs")
    error = None
    try:
        with _flow_stage_contract(guider, "native_continuation"):
            _begin_capture(binding, guider, sampler, sigmas, latent_shapes)
            result = executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        calls = [event for event in binding.metrics.events[event_start:] if event.kind == "model_call"]
        if not calls or not calls[0].fields.get("actual"):
            raise RuntimeError("native continuation must begin with an actual H3 model evaluation")
        binding.metrics.event(
            "native_continuation_complete",
            policy=PARTITIONED_SPATIAL_STAGE_NATIVE_SINGLE,
            sampler_invocation_count=1,
            history_boundary_count=0,
            actual_model_calls=sum(bool(event.fields.get("actual")) for event in calls),
            forecast_model_calls=sum(not bool(event.fields.get("actual")) for event in calls),
            logical_model_calls=len(calls),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            rendered_acceptance=False,
        )
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        _finish_capture(binding, error=error)
