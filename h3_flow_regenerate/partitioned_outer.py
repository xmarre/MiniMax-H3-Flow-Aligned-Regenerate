"""OUTER_SAMPLE boundary for partitioned exact-prefix progressive continuation."""

from __future__ import annotations

import functools
import inspect
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
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_EXACT,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY,
    PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK,
    PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY,
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


def _claim_frame_gauge_invocation(binding: FlowBinding, *, enabled: bool) -> bool:
    """Claim the transient frame-gauge transaction for one outer invocation."""
    if not enabled:
        return False
    if binding.frame_gauge_invocation_active:
        raise RuntimeError("nested partitioned frame-gauge invocation is unsupported")
    binding.frame_gauge_invocation_active = True
    return True


def _release_frame_gauge_invocation(binding: FlowBinding, *, claimed: bool) -> None:
    if claimed:
        binding.frame_gauge_invocation_active = False


def _source_has_audio_velocity_mask_contract(source: str) -> bool:
    execute_index = source.find(").execute(")
    mask_index = source.find("out[1] = out[1] * audio_denoise_mask")
    return execute_index >= 0 and mask_index > execute_index


@functools.lru_cache(maxsize=1)
def _core_has_audio_velocity_mask_contract() -> bool:
    """Require Core #15988 semantics before decoupling inner and outer audio masks."""

    try:
        import comfy.ldm.minimax.model as native

        source = inspect.getsource(native.MiniMaxH3Model.forward)
    except (ImportError, OSError, TypeError):
        return False
    return _source_has_audio_velocity_mask_contract(source)


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

    # The original exact mask remains authoritative. sampler_mask reproduces
    # the existing sampler-lifetime ramp. model_timestep_only keeps sampler
    # ownership exact but publishes the ramp only to MiniMax-H3's inner timestep
    # labels. exact_mask keeps the native input, timestep and velocity masks
    # coherent. The old sampler_mask_exact_timestep selector aliases exact_mask:
    # releasing sampler rows while labeling them clean violates that contract.
    # The ordinary node keeps the existing environment/default path.
    diagnostic_audio_control = (
        PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY in model_options
        or PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY in model_options
    )
    diagnostic_video_overlap_control = PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY in model_options
    guided_ticks, guided_configuration_source = resolve_partitioned_audio_guided_overlap_ticks(model_options)
    guided_mode, guided_mode_source = resolve_partitioned_audio_guided_overlap_mode(model_options)
    exact_audio_mode = guided_mode in (
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_EXACT,
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    )
    # The historical video-overlap selector remains model-local for workflow
    # compatibility and execution provenance, but run 01093 retired its
    # target-high prefix-release mutation. Keep the caller/runtime video mask
    # exact here; the scheduler reports the requested width and verifies exact
    # target-high context without repainting carried video tokens.
    runtime_denoise_mask = denoise_mask

    guided_report = None
    audio_model_context = None
    core_audio_velocity_mask_contract = None
    if guided_ticks or diagnostic_audio_control:
        guided_mask, guided_report = apply_audio_guided_overlap_mask(
            runtime_denoise_mask,
            latent_shapes,
            ticks=0 if exact_audio_mode else guided_ticks,
        )
        if exact_audio_mode:
            audio_mask = unpack_streams(denoise_mask, latent_shapes)[1]
            minimum = audio_mask.amin(dim=(0, 1, 2))
            maximum = audio_mask.amax(dim=(0, 1, 2))
            protected = (minimum == 0) & (maximum == 0)
            generated = (minimum == 1) & (maximum == 1)
            audio_prefix_ticks = int(protected.sum().item())
            if not (
                bool(protected[:audio_prefix_ticks].all().item()) and bool(generated[audio_prefix_ticks:].all().item())
            ):
                raise RuntimeError("exact audio mask requires a contiguous fully protected prefix and generated suffix")
            guided_report.update(
                requested_ticks=guided_ticks,
                audio_prefix_ticks=audio_prefix_ticks,
                hard_prefix_ticks=audio_prefix_ticks,
                effective_ticks=0,
                reason="exact_audio_input_timestep_velocity_mask",
                policy="coherent_exact_audio_mask_v1",
                legacy_selector_alias=guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
            )
        sampler_mask_mode = guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER
        if sampler_mask_mode:
            runtime_denoise_mask = guided_mask
        elif guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP or exact_audio_mode:
            # Audio remains exact/model-timestep-only. The independent legacy
            # video-overlap selector is only provenance after 01093, so it does
            # not alter this runtime mask.
            pass
        else:
            raise RuntimeError(f"unsupported partitioned audio guided-overlap mode {guided_mode!r}")

        if exact_audio_mode:
            core_audio_velocity_mask_contract = _core_has_audio_velocity_mask_contract()
            if not core_audio_velocity_mask_contract:
                raise RuntimeError(
                    "exact audio masks require ComfyUI MiniMax-H3 denoise-mask velocity conversion fix #15988"
                )
            audio_model_context = PartitionedAudioModelTimestepContext(
                audio_mask=unpack_streams(denoise_mask, latent_shapes)[1][:1]
                .amax(dim=1, keepdim=True)
                .contiguous()
                .detach()
                .clone(),
                metrics=binding.metrics,
                ticks=0,
                audio_prefix_ticks=int(guided_report.get("audio_prefix_ticks", 0)),
                mask_kind="exact_authoritative",
            )
        elif bool(guided_report.get("applied")) and guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP:
            core_audio_velocity_mask_contract = _core_has_audio_velocity_mask_contract()
            if not core_audio_velocity_mask_contract:
                raise RuntimeError(
                    "partitioned inner/outer audio-mask decoupling requires ComfyUI MiniMax-H3 "
                    "denoise-mask velocity conversion fix #15988"
                )
            inner_audio_mask = (
                unpack_streams(guided_mask, latent_shapes)[1].amax(dim=1, keepdim=True).contiguous().detach()
            )
            audio_model_context = PartitionedAudioModelTimestepContext(
                audio_mask=inner_audio_mask,
                metrics=binding.metrics,
                ticks=int(guided_report["applied_ticks"]),
                audio_prefix_ticks=int(guided_report.get("audio_prefix_ticks", 0)),
                mask_kind="guided_overlap",
            )

        sampler_mask_modified = bool(guided_report.get("applied") and sampler_mask_mode)
        guided_report.update(
            configuration_source=guided_configuration_source,
            mode=guided_mode,
            mode_source=guided_mode_source,
            sampler_mask_modified=sampler_mask_modified,
            model_timestep_mask_modified=bool(
                guided_report.get("applied")
                and (audio_model_context is None or audio_model_context.mask_kind != "exact_authoritative")
            ),
            model_timestep_override_only=bool(
                guided_report.get("applied") and guided_mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP
            ),
            model_timestep_mask_kind=(
                audio_model_context.mask_kind if audio_model_context is not None else "runtime_sampler_mask"
            ),
            inner_exact_audio_prefix_preserved=bool(
                exact_audio_mode
                or (audio_model_context is not None and audio_model_context.mask_kind == "exact_authoritative")
            ),
            sampler_exact_audio_prefix_preserved=not sampler_mask_modified,
            core_audio_velocity_mask_contract=core_audio_velocity_mask_contract,
        )

    # Preserve sampler-owned guided overlap and the caller-owned exact-prefix
    # restore. Do not add a second post-sampler mutation to generated successor
    # audio; the returned suffix remains owned by the sampler trajectory.
    audio_exact_restore_successor_ticks = 0
    adapted = _ProgressiveExactMaskExecutor(
        executor,
        binding=binding,
        progressive=SimpleNamespace(exact_prefix_mode="partitioned_exact_prefix"),
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        sampler=sampler,
        latent_shapes=latent_shapes,
        audio_exact_restore_successor_ticks=audio_exact_restore_successor_ticks,
    )

    transformer_options = model_options.setdefault("transformer_options", {})
    if not isinstance(transformer_options, dict):
        raise RuntimeError("partitioned audio diagnostics require mutable transformer options")
    context_installed = False
    if audio_model_context is not None and PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY in transformer_options:
        raise RuntimeError("nested partitioned audio model-timestep context is unsupported")

    outer_started = time.perf_counter()
    if binding.registered_guidance_reference is not None:
        raise RuntimeError("stale or nested partitioned frame-gauge guidance context is unsupported")
    frame_gauge_claimed = _claim_frame_gauge_invocation(
        binding,
        enabled=bool(progressive.frame_gauge_repair),
    )
    error = None
    fallback_reason = None
    result = None
    try:
        if audio_model_context is not None:
            transformer_options[PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY] = audio_model_context
            context_installed = True
        binding.guidance_state.reset()
        binding.active_guidance_run = None
        binding.registered_guidance_reference = None
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
            exact_denoise_mask=denoise_mask,
        )
        if audio_model_context is not None and audio_model_context.calls <= 0:
            raise RuntimeError(
                "partitioned audio mask context was requested but zero MiniMax-H3 inner-forward calls executed"
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
        binding.registered_guidance_reference = None
        _release_frame_gauge_invocation(binding, claimed=frame_gauge_claimed)
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
        if (
            diagnostic_audio_control
            or diagnostic_video_overlap_control
            or transformer_options.get(PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY) == PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK
        ):
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
            "partitioned audio guided overlap mode=%s requested_ticks=%d applied_ticks=%d applied=%s source=%s "
            "reason=%s exact_prefix=%d ramp=%s final_exact_restore=true",
            guided_report.get("mode"),
            guided_ticks,
            int(guided_report["applied_ticks"]),
            bool(guided_report.get("applied")),
            guided_report.get("configuration_source"),
            guided_report.get("reason"),
            int(guided_report.get("audio_prefix_ticks", 0)),
            guided_report.get("ramp_values"),
        )

    if audio_model_context is not None:
        sampler_mask_modified = bool(guided_report and guided_report.get("sampler_mask_modified"))
        binding.metrics.event(
            "partitioned_audio_model_timestep_context",
            mode=guided_mode,
            ticks=guided_ticks,
            requested_ticks=guided_ticks,
            applied_ticks=audio_model_context.ticks,
            audio_prefix_ticks=audio_model_context.audio_prefix_ticks,
            override_calls=audio_model_context.calls - audio_model_context.verification_calls,
            verification_calls=audio_model_context.verification_calls,
            model_timestep_override_applied=not exact_audio_mode,
            mask_kind=audio_model_context.mask_kind,
            sampler_mask_modified=sampler_mask_modified,
            exact_sampler_prefix_preserved=not sampler_mask_modified,
            inner_exact_audio_prefix_preserved=audio_model_context.mask_kind == "exact_authoritative",
            core_audio_velocity_mask_contract=core_audio_velocity_mask_contract,
            fail_closed=True,
        )
        if exact_audio_mode:
            binding.metrics.event(
                "partitioned_exact_audio_mask_verified",
                policy="coherent_exact_audio_mask_v1",
                mode=guided_mode,
                requested_overlap_ticks=guided_ticks,
                effective_overlap_ticks=0,
                verified_model_entries=audio_model_context.verification_calls,
                sampler_input_mask_exact=True,
                model_timestep_mask_exact=True,
                model_velocity_mask_exact=True,
                timestep_override_applied=False,
                regenerated_prefix_restored=False,
                final_prefix_exact=True,
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                fail_closed=True,
            )
        LOG.info(
            "partitioned audio model-timestep context verified mode=%s requested_ticks=%d applied_ticks=%d "
            "prefix=%d calls=%d mask_kind=%s sampler_mask_modified=%s",
            guided_mode,
            guided_ticks,
            audio_model_context.ticks,
            audio_model_context.audio_prefix_ticks,
            audio_model_context.calls,
            audio_model_context.mask_kind,
            sampler_mask_modified,
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
    # sampler_wall autosaves before the outer audio receipts above are emitted.
    # Persist the completed invocation so file-based gates see those receipts.
    if binding.metrics.autosave_path is not None:
        binding.metrics.write_json(binding.metrics.autosave_path)
    return result


__all__ = ["partitioned_outer_wrapper"]
