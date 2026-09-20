"""OUTER_SAMPLE boundary for partitioned exact-prefix progressive continuation."""

from __future__ import annotations

import logging
import time
import uuid
from types import SimpleNamespace

from .audio_guided_overlap import (
    apply_audio_guided_overlap_mask,
    configured_audio_guided_overlap_ticks,
)
from .comfy_compat import _ProgressiveExactMaskExecutor, flow_outer_wrapper_with_exact_mask
from .handoff import ProgressiveTargetInputConfig
from .partitioned_scheduler import (
    PARTITIONED_PROGRESSIVE_KEY,
    PartitionedPreflightUnsupported,
    run_partitioned_progressive,
)
from .runtime import FLOW_BINDING_KEY, FLOW_REQUEST_ID_KEY, FlowBinding, _has_exact_video_protection

LOG = logging.getLogger(__name__)


def partitioned_outer_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    """Select the new exact-prefix path only for a supported protected prefix.

    Initial/no-prefix chunks continue through the released Progressive Target
    Input runtime.  An exact-prefix chunk that cannot satisfy the complete new
    contract falls back *before sampling* to the released full-target-grid exact
    path.  Once the partitioned scheduler begins a sampler lifetime, failures are
    candidate failures and are never restarted under a different numerical path.
    """
    guider = executor.class_obj
    model_options = getattr(guider, "model_options", None) or {}
    binding = model_options.get(FLOW_BINDING_KEY)
    progressive = model_options.get(PARTITIONED_PROGRESSIVE_KEY)
    if not isinstance(binding, FlowBinding) or not isinstance(progressive, ProgressiveTargetInputConfig):
        return flow_outer_wrapper_with_exact_mask(
            executor,
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
    if not isinstance(latent_shapes, list):
        raise RuntimeError("partitioned exact-prefix requires mutable packed latent-shape metadata")
    if not _has_exact_video_protection(denoise_mask, latent_shapes):
        return flow_outer_wrapper_with_exact_mask(
            executor,
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

    # Guided audio overlap is sampler-lifetime state only.  The original exact
    # mask remains the authority for caller-visible final restoration.
    runtime_denoise_mask = denoise_mask
    guided_ticks = configured_audio_guided_overlap_ticks()
    guided_report = None
    if guided_ticks:
        runtime_denoise_mask, guided_report = apply_audio_guided_overlap_mask(
            denoise_mask,
            latent_shapes,
            ticks=guided_ticks,
        )

    adapted = _ProgressiveExactMaskExecutor(
        executor,
        binding=binding,
        progressive=SimpleNamespace(exact_prefix_mode="partitioned_exact_prefix"),
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        sampler=sampler,
    )

    outer_started = time.perf_counter()
    binding.guidance_state.reset()
    binding.active_guidance_run = None

    # This wrapper intentionally bypasses runtime.flow_outer_wrapper for the
    # partitioned scheduler, so it must establish the same Flow request
    # correlation lifetime itself.  Without this, production partitioned runs
    # emit null request/evaluation IDs even though the ordinary path is fully
    # correlated, making stage/Sol/VDN attribution unverifiable.
    if binding.active_request_id is not None:
        raise RuntimeError("nested H3 Flow request correlation lifetimes are unsupported")
    request_transformer = model_options.setdefault("transformer_options", {})
    if not isinstance(request_transformer, dict):
        raise RuntimeError("H3 flow request tracking requires mutable transformer options")
    request_id = f"flow-{uuid.uuid4().hex}"
    previous_request_id = request_transformer.get(FLOW_REQUEST_ID_KEY)
    binding.active_request_id = request_id
    binding.evaluation_serial = 0
    request_transformer[FLOW_REQUEST_ID_KEY] = request_id

    error = None
    fallback_reason = None
    result = None
    try:
        result = run_partitioned_progressive(
            adapted,
            guider,
            binding,
            progressive,
            noise,
            latent_image,
            sampler,
            sigmas,
            runtime_denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes,
        )
    except PartitionedPreflightUnsupported as exc:
        fallback_reason = str(exc)
    except BaseException as exc:
        error = exc
        raise
    finally:
        binding.guidance_state.reset()
        binding.active_guidance_run = None
        if fallback_reason is None:
            binding.metrics.event(
                "sampler_wall",
                request_id=request_id,
                elapsed_ms=(time.perf_counter() - outer_started) * 1000.0,
                progressive=True,
                partitioned_exact_prefix=True,
                failed=error is not None,
            )
        if previous_request_id is None:
            request_transformer.pop(FLOW_REQUEST_ID_KEY, None)
        else:
            request_transformer[FLOW_REQUEST_ID_KEY] = previous_request_id
        binding.active_request_id = None
        binding.evaluation_serial = 0

    if fallback_reason is not None:
        binding.metrics.increment("partitioned_exact_prefix_fallbacks")
        binding.metrics.event(
            "partitioned_exact_prefix_fallback",
            reason=fallback_reason,
            before_sampler=True,
            target_grid_fallback=True,
        )
        return flow_outer_wrapper_with_exact_mask(
            executor,
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

    if isinstance(guided_report, dict):
        binding.metrics.event(
            "audio_guided_overlap",
            partitioned_exact_prefix=True,
            **guided_report,
        )
        if bool(guided_report.get("applied")):
            LOG.info(
                "partitioned audio guided overlap active ticks=%d exact_prefix=%d ramp=%s final_exact_restore=true",
                guided_ticks,
                int(guided_report.get("audio_prefix_ticks", 0)),
                guided_report.get("ramp_values"),
            )
    return result


__all__ = ["partitioned_outer_wrapper"]
