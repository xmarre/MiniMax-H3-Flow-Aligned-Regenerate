"""OUTER_SAMPLE boundary for partitioned exact-prefix progressive continuation."""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

from .audio_guided_overlap import (
    apply_audio_guided_overlap_mask,
    measure_audio_latent_boundary,
)
from .comfy_compat import _ProgressiveExactMaskExecutor, flow_outer_wrapper_with_exact_mask
from .geometry import unpack_streams
from .handoff import ProgressiveTargetInputConfig
from .partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY,
    PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY,
    PartitionedAudioModelTimestepContext,
    resolve_partitioned_audio_guided_overlap_mode,
    resolve_partitioned_audio_guided_overlap_ticks,
)
from .partitioned_scheduler import (
    PARTITIONED_PROGRESSIVE_KEY,
    PartitionedPreflightUnsupported,
    run_partitioned_progressive,
)
from .runtime import FLOW_BINDING_KEY, FlowBinding, _has_exact_video_protection

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

    # Guided audio overlap is sampler-lifetime state only. The original exact
    # mask remains the authority for caller-visible final restoration. The
    # diagnostic node may override the overlap width model-locally; the ordinary
    # node keeps the existing environment/default resolution path.
    diagnostic_audio_control = (
        PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY in model_options
        or PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY in model_options
    )
    guided_ticks, guided_configuration_source = resolve_partitioned_audio_guided_overlap_ticks(model_options)
    guided_mode, guided_mode_source = resolve_partitioned_audio_guided_overlap_mode(model_options)
    runtime_denoise_mask = denoise_mask
    guided_report = None
    audio_model_context = None
    if guided_ticks or diagnostic_audio_control:
        guided_mask, guided_report = apply_audio_guided_overlap_mask(
            denoise_mask,
            latent_shapes,
            ticks=guided_ticks,
        )
        if guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER:
            runtime_denoise_mask = guided_mask
        elif guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP:
            runtime_denoise_mask = denoise_mask
            if bool(guided_report.get("applied")):
                guided_audio_mask = unpack_streams(guided_mask, latent_shapes)[1].detach()
                audio_model_context = PartitionedAudioModelTimestepContext(
                    audio_mask=guided_audio_mask,
                    metrics=binding.metrics,
                    ticks=guided_ticks,
                    audio_prefix_ticks=int(guided_report.get("audio_prefix_ticks", 0)),
                )
        else:
            raise RuntimeError(f"unsupported partitioned audio guided-overlap mode {guided_mode!r}")
        guided_report.update(
            configuration_source=guided_configuration_source,
            mode=guided_mode,
            mode_source=guided_mode_source,
            sampler_mask_modified=bool(
                guided_report.get("applied") and guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER
            ),
            model_timestep_mask_modified=bool(guided_report.get("applied")),
            model_timestep_override_only=bool(
                guided_report.get("applied")
                and guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP
            ),
            sampler_exact_audio_prefix_preserved=not bool(
                guided_report.get("applied") and guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER
            ),
        )

    adapted = _ProgressiveExactMaskExecutor(
        executor,
        binding=binding,
        progressive=SimpleNamespace(exact_prefix_mode="partitioned_exact_prefix"),
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        sampler=sampler,
    )

    transformer_options = model_options.setdefault("transformer_options", {})
    if not isinstance(transformer_options, dict):
        raise RuntimeError("partitioned audio diagnostics require mutable transformer options")
    context_installed = False
    if audio_model_context is not None:
        if PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY in transformer_options:
            raise RuntimeError("nested partitioned audio model-timestep context is unsupported")
        transformer_options[PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY] = audio_model_context
        context_installed = True

    outer_started = time.perf_counter()
    binding.guidance_state.reset()
    binding.active_guidance_run = None
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
        if context_installed:
            transformer_options.pop(PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY, None)
        binding.guidance_state.reset()
        binding.active_guidance_run = None
        if fallback_reason is None:
            binding.metrics.event(
                "sampler_wall",
                elapsed_ms=(time.perf_counter() - outer_started) * 1000.0,
                progressive=True,
                partitioned_exact_prefix=True,
                failed=error is not None,
            )

    if fallback_reason is not None:
        binding.metrics.increment("partitioned_exact_prefix_fallbacks")
        binding.metrics.event(
            "partitioned_exact_prefix_fallback",
            reason=fallback_reason,
            before_sampler=True,
            target_grid_fallback=True,
        )
        if diagnostic_audio_control:
            raise RuntimeError(
                "partitioned diagnostic controls require supported partitioned exact-prefix "
                f"execution; refusing target-grid fallback: {fallback_reason}"
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
        LOG.info(
            "partitioned audio guided overlap mode=%s ticks=%d applied=%s source=%s "
            "reason=%s exact_prefix=%d ramp=%s final_exact_restore=true",
            guided_report.get("mode"),
            guided_ticks,
            bool(guided_report.get("applied")),
            guided_report.get("configuration_source"),
            guided_report.get("reason"),
            int(guided_report.get("audio_prefix_ticks", 0)),
            guided_report.get("ramp_values"),
        )

    if audio_model_context is not None:
        if audio_model_context.calls <= 0:
            raise RuntimeError(
                "model-timestep-only audio guidance was requested but zero MiniMax-H3 inner-forward overrides executed"
            )
        binding.metrics.event(
            "partitioned_audio_model_timestep_context",
            mode=guided_mode,
            ticks=guided_ticks,
            audio_prefix_ticks=audio_model_context.audio_prefix_ticks,
            override_calls=audio_model_context.calls,
            sampler_mask_modified=False,
            exact_sampler_prefix_preserved=True,
            fail_closed=True,
        )
        LOG.info(
            "partitioned audio model-timestep context verified ticks=%d prefix=%d calls=%d exact_sampler_prefix=true",
            guided_ticks,
            audio_model_context.audio_prefix_ticks,
            audio_model_context.calls,
        )

    if diagnostic_audio_control:
        latent_audio_report = measure_audio_latent_boundary(
            result,
            latent_shapes,
            denoise_mask,
            windows=(4, 20),
        )
        binding.metrics.event(
            "partitioned_audio_latent_boundary",
            partitioned_exact_prefix=True,
            audio_guided_overlap_ticks=guided_ticks,
            **latent_audio_report,
        )
        LOG.info(
            "partitioned audio latent boundary ticks=%d prefix=%d available=%s reason=%s windows=%s",
            guided_ticks,
            int(latent_audio_report.get("audio_prefix_ticks", 0)),
            bool(latent_audio_report.get("available")),
            latent_audio_report.get("reason"),
            latent_audio_report.get("windows"),
        )
    return result


__all__ = ["partitioned_outer_wrapper"]
