from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .contracts import H3FlowTrajectory, TrajectorySample
from .geometry import geometry_from_video, pack_streams, resize_spatial_5d, unpack_streams
from .guidance import GuidanceConfig, GuidanceState, apply_guidance
from .handoff import (
    ProgressiveHandoffConfig,
    ProgressiveTargetInputConfig,
    build_handoff_state,
    deterministic_video_noise,
    select_handoff_index,
)
from .metrics import H3FlowMetrics
from .mixed_grid import MIXED_GRID_KEY, build_mixed_grid_plan
from .seam_diagnostics import (
    measure_exact_prefix_splice,
    measure_video_boundary,
    recover_conditional_clean_for_diagnostics,
)
from .sigma import H3_AUDIO_SHIFT, H3_VIDEO_SHIFT, audio_sigma, normalized_coordinate
from .target_sparse import TARGET_SPARSE_CONTRACT_KEY, build_target_sparse_plan, target_sparse_contract
from .tone_bridge import (
    apply_suffix_dc_bridge,
    disabled_suffix_dc_bridge_metrics,
    map_clean_bridge_to_conditional_state,
)

LOG = logging.getLogger(__name__)

FLOW_BINDING_KEY = "h3_flow_regenerate_binding"
PROGRESSIVE_KEY = "h3_flow_progressive_v1"
OUTER_WRAPPER_KEY = "h3_flow_regenerate.outer.v1"
PREDICT_WRAPPER_KEY = "h3_flow_regenerate.predict.v1"
CLONE_CALLBACK_KEY = "h3_flow_regenerate.clone.v1"
PROBE_MARKER = "_h3_flow_exact_probe"
PROBE_CONTEXT_KEY = "h3_flow_exact_probe_context"
FLOW_STAGE_KEY = "h3_flow_stage"
EXACT_PREFIX_BRIDGE_KEY = "h3_flow_exact_prefix_bridge_v1"
SPECTRUM_BINDING_KEY = "spectrum_h3_binding"
SPECTRUM_ACTUAL_KEY = "spectrum_h3_actual"
SPECTRUM_PHASE_KEY = "spectrum_h3_solver_phase"
SPECTRUM_OUTER_STEP_KEY = "spectrum_h3_outer_step_id"


@dataclass(slots=True)
class _ActiveCapture:
    run_id: str
    shapes: list[tuple[int, ...]]
    phases: tuple[tuple[int, str], ...]
    call_index: int = 0


@dataclass(slots=True)
class FlowBinding:
    trajectory: H3FlowTrajectory | None = None
    guidance: GuidanceConfig | None = None
    metrics: H3FlowMetrics = field(default_factory=H3FlowMetrics)
    capture_enabled: bool = False
    capture_forecasts: bool = False
    guidance_conditioning_signature: str | None = None
    captured_run_id: str | None = None
    guidance_run_id: str | None = None
    guidance_state: GuidanceState = field(default_factory=GuidanceState)
    active_capture: _ActiveCapture | None = None
    active_guidance_run: Any = None


def sampler_name(sampler: Any) -> str:
    function = getattr(sampler, "sampler_function", None)
    return str(getattr(function, "__name__", type(sampler).__name__))


def _validate_progressive_sampler_state(sampler: Any) -> None:
    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        raise TypeError("progressive handoff requires sampler extra_options to be a dictionary")
    if options.get("noise_sampler") is not None:
        raise ValueError(
            "progressive handoff does not support an explicit sampler noise_sampler; "
            "its mutable RNG/history cannot be proven reset across low/probe/high lifetimes"
        )


def _sampler_phases(sampler: Any, sigmas: torch.Tensor) -> tuple[tuple[int, str], ...]:
    name = sampler_name(sampler)
    outer_steps = max(0, int(sigmas.numel()) - 1)
    sa_solvers = {
        "sample_sa_solver",
        "sample_sa_solver_pece",
        "sample_refdelta_sa_solver",
        "sample_refdelta_sa_solver_pece",
    }
    if name in sa_solvers:
        options = getattr(sampler, "extra_options", {}) or {}
        pece = name.endswith("_pece") or options.get("use_pece") is True
        corrector_order = int(options.get("corrector_order", 4))
        phases: list[tuple[int, str]] = []
        for outer in range(outer_steps):
            phases.append((outer, "predicted"))
            if pece and corrector_order > 0 and outer > 0:
                phases.append((outer, "corrected"))
        return tuple(phases)
    stages = {
        "sample_seeds_2": 2,
        "sample_refdelta_seeds_2": 2,
        "sample_seeds_3": 3,
        "sample_refdelta_seeds_3": 3,
    }.get(name)
    if stages:
        phases = []
        for outer in range(outer_steps):
            count = 1 if float(sigmas[outer + 1]) == 0.0 else stages
            phases.extend((outer, f"stage_{stage + 1}") for stage in range(count))
        return tuple(phases)
    return tuple((outer, "single") for outer in range(outer_steps))


def _schedule_signature(sigmas: torch.Tensor) -> str:
    values = b"".join(
        struct.pack("!d", float(value)) for value in sigmas.detach().to(device="cpu", dtype=torch.float64)
    )
    return hashlib.sha256(values).hexdigest()[:16]


def _interop_identity(model_options: dict[str, Any] | None) -> tuple[str, str]:
    transformer = (model_options or {}).get("transformer_options") or {}
    continuum = transformer.get("h3_continuum")
    if isinstance(continuum, dict) and continuum.get("active") is True:
        chunk = str(continuum.get("chunk_index", "unknown"))
        session = str(continuum.get("session_id", "continuum"))
        return session, chunk
    return "standalone", "standalone"


def _tensor_signature(tensor: torch.Tensor, *, max_values: int = 128) -> bytes:
    """Return a bounded, device-independent content fingerprint.

    Conditioning can contain multi-megabyte Qwen embeddings and reference
    latents. Hashing every byte would introduce a full GPU readback. Sampling
    deterministic positions catches ordinary prompt/reference changes while
    keeping the identity check bounded; shape/dtype/layout are always included.
    """
    digest = hashlib.sha256()
    digest.update(f"{tuple(tensor.shape)}|{tensor.dtype}|{tensor.layout}".encode())
    if tensor.numel() == 0 or tensor.device.type == "meta" or tensor.layout != torch.strided:
        return digest.digest()

    count = min(int(max_values), int(tensor.numel()))
    if count == 1:
        linear = torch.zeros(1, dtype=torch.long, device=tensor.device)
    else:
        ordinal = torch.arange(count, dtype=torch.long, device=tensor.device)
        linear = torch.div(
            ordinal * (int(tensor.numel()) - 1),
            count - 1,
            rounding_mode="floor",
        )
    if tensor.ndim == 0:
        sampled = tensor.detach().reshape(1)
    else:
        remainder = linear
        reversed_coords: list[torch.Tensor] = []
        for size in reversed(tensor.shape):
            reversed_coords.append(remainder.remainder(int(size)))
            remainder = torch.div(remainder, int(size), rounding_mode="floor")
        sampled = tensor.detach()[tuple(reversed(reversed_coords))]
    raw = sampled.contiguous().view(torch.uint8).to(device="cpu")
    digest.update(bytes(raw.tolist()))
    return digest.digest()


def _update_conditioning_digest(digest, value: Any, *, depth: int = 0) -> None:
    if depth > 8:
        digest.update(f"<depth:{type(value).__module__}.{type(value).__qualname__}>".encode())
        return
    if torch.is_tensor(value):
        digest.update(b"tensor:")
        digest.update(_tensor_signature(value))
        return
    if isinstance(value, dict):
        # ComfyUI's convert_cond creates a fresh UUID on each conversion. It is
        # execution identity, not conditioning identity, so it must be excluded
        # from both the keyed payload *and the structural field count*.
        keys = [key for key in value if str(key) != "uuid"]
        digest.update(f"dict:{len(keys)}:".encode())
        for key in sorted(keys, key=lambda item: str(item)):
            digest.update(f"key:{key!s}:".encode())
            _update_conditioning_digest(digest, value[key], depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}:{len(value)}:".encode())
        for item in value:
            _update_conditioning_digest(digest, item, depth=depth + 1)
        return
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        digest.update(f"{type(value).__name__}:{value!r}:".encode())
        return
    digest.update(f"type:{type(value).__module__}.{type(value).__qualname__}:".encode())


def _conditioning_signature_from_original(original: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_conditioning_digest(digest, original)
    return digest.hexdigest()[:32]


def _convert_conditioning_for_signature(conditioning: list[Any]) -> list[dict[str, Any]]:
    converted = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            raise TypeError("conditioning entries must contain tensor/context plus metadata dictionary")
        metadata = entry[1].copy()
        model_conds = metadata.get("model_conds", {})
        if entry[0] is not None:
            metadata["cross_attn"] = entry[0]
        metadata["model_conds"] = model_conds
        converted.append(metadata)
    return converted


def conditioning_signature_from_conditioning(
    positive: list[Any],
    negative: list[Any] | None = None,
) -> str:
    """Match CFGGuider.convert_cond without its execution-only UUID."""
    original = {"positive": _convert_conditioning_for_signature(positive)}
    if negative is not None:
        original["negative"] = _convert_conditioning_for_signature(negative)
    return _conditioning_signature_from_original(original)


def _conditioning_signature(guider: Any) -> str:
    original = getattr(guider, "original_conds", {}) or {}
    if not isinstance(original, dict):
        raise TypeError("guider original_conds must be a dictionary")
    return _conditioning_signature_from_original(original)


def _resolve_binding(guider: Any) -> FlowBinding | None:
    value = (getattr(guider, "model_options", None) or {}).get(FLOW_BINDING_KEY)
    return value if isinstance(value, FlowBinding) else None


def flow_model_clone_callback(source_model: Any, cloned_model: Any) -> None:
    source_options = getattr(source_model, "model_options", None) or {}
    binding = source_options.get(FLOW_BINDING_KEY)
    if not isinstance(binding, FlowBinding):
        return
    cloned_options = getattr(cloned_model, "model_options", None)
    if not isinstance(cloned_options, dict):
        cloned_model.model_options = {}
        cloned_options = cloned_model.model_options
    cloned_options[FLOW_BINDING_KEY] = FlowBinding(
        trajectory=binding.trajectory,
        guidance=binding.guidance,
        metrics=binding.metrics,
        capture_enabled=binding.capture_enabled,
        capture_forecasts=binding.capture_forecasts,
        guidance_conditioning_signature=binding.guidance_conditioning_signature,
        captured_run_id=binding.captured_run_id,
        guidance_run_id=binding.guidance_run_id,
    )


def _begin_capture(
    binding: FlowBinding,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_shapes: list[tuple[int, ...]],
) -> None:
    trajectory = binding.trajectory
    if not binding.capture_enabled or trajectory is None or sampler_name(sampler) == PROBE_MARKER:
        return
    if len(latent_shapes) != 2:
        raise ValueError("H3 trajectory capture requires exactly video and audio latent shapes")
    video = torch.empty(latent_shapes[0], device="meta")
    geometry = geometry_from_video(video)
    session_id, chunk_id = _interop_identity(getattr(guider, "model_options", None))
    binding.captured_run_id = None
    run_id = trajectory.begin(
        session_id=session_id,
        chunk_id=chunk_id,
        sampler=sampler_name(sampler),
        scheduler=_schedule_signature(sigmas),
        geometry=geometry,
        audio_shape=tuple(latent_shapes[1]),
        layout_signature=f"{latent_shapes[0]}|{latent_shapes[1]}",
        conditioning_signature=_conditioning_signature(guider),
    )
    binding.active_capture = _ActiveCapture(
        run_id=run_id,
        shapes=list(latent_shapes),
        phases=_sampler_phases(sampler, sigmas),
    )
    binding.metrics.event(
        "trajectory_begin",
        run_id=run_id,
        sampler=sampler_name(sampler),
        chunk_id=chunk_id,
        geometry=tuple(latent_shapes[0]),
        pixels=(geometry.pixel_h, geometry.pixel_w),
        padded=(geometry.padded_h, geometry.padded_w),
        spatial_padding=not geometry.patch_safe,
        trajectory_bytes=binding.trajectory.bytes,
    )


def _finish_capture(binding: FlowBinding, *, error: BaseException | None = None):
    active = binding.active_capture
    if active is None or binding.trajectory is None:
        return None
    binding.active_capture = None
    if error is None:
        run = binding.trajectory.commit(active.run_id)
        binding.captured_run_id = run.run_id
        binding.metrics.event(
            "trajectory_commit",
            run_id=run.run_id,
            samples=len(run.samples),
            trajectory_bytes=binding.trajectory.bytes,
        )
        return run
    run = binding.trajectory.abort(active.run_id, f"{type(error).__name__}: {error}")
    binding.captured_run_id = None
    binding.metrics.event("trajectory_abort", run_id=active.run_id, error=type(error).__name__)
    return run


def _active_spectrum_runtime(guider: Any) -> Any | None:
    spectrum_binding = (getattr(guider, "model_options", None) or {}).get(SPECTRUM_BINDING_KEY)
    runtime = getattr(spectrum_binding, "runtime", None)
    if runtime is None or getattr(runtime, "active_run_id", None) is None:
        return None
    return runtime


def _active_spectrum_step(runtime: Any) -> tuple[str, str, int] | None:
    step_id = getattr(runtime, "active_step_id", None)
    if step_id is None:
        return None
    step = getattr(runtime, "_step", None)
    if step is None or int(getattr(step, "step_id", -1)) != int(step_id):
        raise RuntimeError("Spectrum H3 active-step provenance is internally inconsistent")
    mode = getattr(step, "mode", None)
    phase = getattr(runtime, "active_solver_phase", None)
    outer = getattr(runtime, "active_policy_step_id", None)
    if mode not in {"actual", "forecast"} or phase is None or outer is None:
        raise RuntimeError(
            "Spectrum H3 active-step provenance is incomplete; refusing to misclassify trajectory samples"
        )
    return str(mode), str(phase), int(outer)


def flow_predict_wrapper(executor, x, timestep, model_options=None, seed=None):
    guider = executor.class_obj
    binding = _resolve_binding(guider)
    progressive = (getattr(guider, "model_options", None) or {}).get(PROGRESSIVE_KEY)
    progressive_active = isinstance(progressive, (ProgressiveHandoffConfig, ProgressiveTargetInputConfig))
    if (
        binding is not None
        and "multigpu_clones" in (model_options or {})
        and (binding.active_capture is not None or binding.active_guidance_run is not None or progressive_active)
    ):
        raise RuntimeError(
            "H3 flow trajectory capture/guidance/progressive handoff does not support parallel multi-GPU model calls"
        )
    spectrum_runtime = _active_spectrum_runtime(guider) if binding is not None else None
    spectrum_completed_before = (
        getattr(spectrum_runtime, "last_completed_step_id", None) if spectrum_runtime is not None else None
    )
    started = time.perf_counter()
    result = executor(x, timestep, model_options, seed)
    if binding is None:
        return result
    transformer = (model_options or {}).get("transformer_options") or {}
    probe_context = transformer.get(PROBE_CONTEXT_KEY)
    stage = str(transformer.get(FLOW_STAGE_KEY, "single"))
    actual_value = transformer.get(SPECTRUM_ACTUAL_KEY)
    actual = True if actual_value is None else bool(actual_value)
    spectrum_active_step = _active_spectrum_step(spectrum_runtime) if spectrum_runtime is not None else None
    spectrum_completed_after = (
        getattr(spectrum_runtime, "last_completed_step_id", None) if spectrum_runtime is not None else None
    )
    spectrum_completed_here = (
        spectrum_runtime is not None
        and spectrum_active_step is None
        and spectrum_completed_after is not None
        and spectrum_completed_after != spectrum_completed_before
    )
    if spectrum_active_step is not None:
        active_mode, _, _ = spectrum_active_step
        actual = active_mode == "actual"
    elif spectrum_completed_here:
        completed_mode = getattr(spectrum_runtime, "last_completed_mode", None)
        if completed_mode == "actual":
            actual = True
        elif completed_mode == "forecast":
            actual = False
        else:
            raise RuntimeError(f"unreviewed completed Spectrum H3 mode {completed_mode!r}")
    if isinstance(probe_context, dict) and not actual:
        raise RuntimeError("progressive handoff exact probe was forecast instead of evaluated")
    binding.metrics.increment("transformer_actual_nfe" if actual else "spectrum_forecast_calls")
    binding.metrics.increment(f"transformer_actual_nfe_{stage}" if actual else f"spectrum_forecast_calls_{stage}")
    binding.metrics.increment("sampler_logical_calls")
    binding.metrics.increment(f"sampler_logical_calls_{stage}")
    sigma = float(timestep.detach().reshape(-1)[0].item())
    video_shift = float(transformer.get("minimax_h3_sigma_shift_video", H3_VIDEO_SHIFT))
    audio_shift = float(transformer.get("minimax_h3_sigma_shift_audio", H3_AUDIO_SHIFT))
    coordinate = float(normalized_coordinate(sigma, video_shift=video_shift))
    binding.metrics.event(
        "model_call",
        stage=stage,
        sigma=sigma,
        coordinate=coordinate,
        actual=actual,
        video_shift=video_shift,
        audio_shift=audio_shift,
        audio_sigma=float(audio_sigma(sigma, video_shift=video_shift, audio_shift=audio_shift)),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )

    active = binding.active_capture
    if active is not None:
        if active.call_index < len(active.phases):
            fallback_outer, fallback_phase = active.phases[active.call_index]
        else:
            fallback_outer, fallback_phase = active.call_index, "unclassified"
        if spectrum_active_step is not None:
            _, phase, outer = spectrum_active_step
        elif spectrum_completed_here:
            phase = fallback_phase
            outer = fallback_outer
        else:
            phase = str(transformer.get(SPECTRUM_PHASE_KEY, fallback_phase))
            outer = int(transformer.get(SPECTRUM_OUTER_STEP_KEY, fallback_outer))
        if isinstance(probe_context, dict):
            phase = "handoff_probe"
            outer = int(probe_context["outer_step"])
        if actual or binding.capture_forecasts:
            video_x0 = unpack_streams(result, active.shapes)[0]
            binding.trajectory.append(
                active.run_id,
                TrajectorySample(
                    coordinate=coordinate,
                    video_sigma=sigma,
                    audio_sigma=float(audio_sigma(sigma, video_shift=video_shift, audio_shift=audio_shift)),
                    outer_step=outer,
                    call_index=active.call_index,
                    phase=phase,
                    provenance="actual" if actual else "forecast",
                    video_x0=video_x0,
                ),
            )
        active.call_index += 1

    exact_bridge = transformer.get(EXACT_PREFIX_BRIDGE_KEY)
    if isinstance(exact_bridge, dict) and not bool(exact_bridge.get("applied")) and actual:
        exact_prefix = exact_bridge.get("exact_prefix")
        base_model = getattr(guider, "inner_model", None)
        bridge_shapes = getattr(base_model, "latent_shapes", None)
        if not isinstance(bridge_shapes, list) or len(bridge_shapes) != 2:
            raise RuntimeError("exact-prefix suffix bridge could not resolve H3 AV latent shapes")
        video_x0, audio_x0 = unpack_streams(result, bridge_shapes)
        bridged_video, bridge_metrics = apply_suffix_dc_bridge(
            video_x0,
            exact_prefix,
            weights=(1.0,),
        )
        result, _ = pack_streams((bridged_video, audio_x0))
        exact_bridge["applied"] = True
        binding.metrics.event(
            "exact_prefix_suffix_dc_bridge",
            source=str(exact_bridge.get("source", "unknown")),
            stage=stage,
            sigma=sigma,
            coordinate=coordinate,
            actual=True,
            suffix_dc_bridge_state_mapping="first_actual_model_x0",
            **bridge_metrics,
        )

    if binding.guidance is not None and binding.guidance.mode != "off":
        run = binding.active_guidance_run
        if run is None:
            return result
        base_model = getattr(guider, "inner_model", None)
        shapes = getattr(base_model, "latent_shapes", None)
        if not isinstance(shapes, list) or len(shapes) != 2:
            raise RuntimeError("flow guidance could not resolve H3 AV latent shapes")
        video_x0, audio_x0 = unpack_streams(result, shapes)
        video_state, _ = unpack_streams(x, shapes)
        guidance_started = time.perf_counter()
        guided_video = apply_guidance(
            video_x0,
            run=run,
            coordinate=coordinate,
            config=binding.guidance,
            state=binding.guidance_state,
            high_state=video_state,
            sigma=sigma,
        )
        guidance_elapsed_ms = (time.perf_counter() - guidance_started) * 1000.0
        result, _ = pack_streams((guided_video, audio_x0))
        binding.metrics.event(
            "guidance",
            coordinate=coordinate,
            elapsed_ms=guidance_elapsed_ms,
            mode=binding.guidance.mode,
            schedule=binding.guidance_state.last_schedule,
            correction_rms=binding.guidance_state.last_correction_rms,
            baseline_rms=binding.guidance_state.last_baseline_rms,
            correction_rms_ratio=binding.guidance_state.last_correction_rms_ratio,
            clamp_scale=binding.guidance_state.last_clamp_scale,
            direction_rms_ratio=binding.guidance_state.last_direction_rms_ratio,
            acceleration_rms_ratio=binding.guidance_state.last_acceleration_rms_ratio,
            acceleration_applied=binding.guidance_state.last_acceleration_applied,
            same_coordinate_refinement=binding.guidance_state.last_same_coordinate_refinement,
            acceleration_anchor_coordinate=binding.guidance_state.last_acceleration_anchor_coordinate,
            temporal_rms_ratio=binding.guidance_state.last_temporal_rms_ratio,
            temporal_confidence_mean=binding.guidance_state.last_temporal_confidence_mean,
            temporal_valid_fraction=binding.guidance_state.last_temporal_valid_fraction,
            temporal_disocclusion_fraction=binding.guidance_state.last_temporal_disocclusion_fraction,
            temporal_similarity_mean=binding.guidance_state.last_temporal_similarity_mean,
            temporal_margin_mean=binding.guidance_state.last_temporal_margin_mean,
            temporal_flow_magnitude_mean=binding.guidance_state.last_temporal_flow_magnitude_mean,
            temporal_flow_magnitude_max=binding.guidance_state.last_temporal_flow_magnitude_max,
            temporal_cache_hit=binding.guidance_state.last_temporal_cache_hit,
            temporal_reference_coordinate=binding.guidance_state.last_temporal_reference_coordinate,
            temporal_reference_clamped=binding.guidance_state.last_temporal_reference_clamped,
            actual=actual,
            solver_phase=(
                spectrum_active_step[1] if spectrum_active_step is not None else transformer.get(SPECTRUM_PHASE_KEY)
            ),
            solver_outer_step=spectrum_active_step[2]
            if spectrum_active_step is not None
            else transformer.get(SPECTRUM_OUTER_STEP_KEY),
        )
    return result


def flow_outer_wrapper(
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
    outer_started = time.perf_counter()
    guider = executor.class_obj
    binding = _resolve_binding(guider)
    if binding is None:
        return executor(
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
        raise RuntimeError("H3 flow wrapper requires mutable packed latent shape metadata")
    binding.guidance_state.reset()
    binding.active_guidance_run = None
    progressive = (getattr(guider, "model_options", None) or {}).get(PROGRESSIVE_KEY)
    is_progressive = isinstance(progressive, (ProgressiveHandoffConfig, ProgressiveTargetInputConfig))
    if not is_progressive and binding.guidance is not None and binding.guidance.mode != "off":
        if binding.trajectory is None:
            raise RuntimeError("flow guidance requires an H3_FLOW_TRAJECTORY")
        if binding.guidance_run_id is not None:
            run = binding.trajectory.select(run_id=binding.guidance_run_id)
        else:
            session_id, chunk_id = _interop_identity(getattr(guider, "model_options", None))
            expected_signature = binding.guidance_conditioning_signature or _conditioning_signature(guider)
            run = binding.trajectory.select(
                chunk_id=chunk_id,
                session_id=session_id,
                conditioning_signature=expected_signature,
            )
        if run.geometry.latent_t != int(latent_shapes[0][2]):
            raise ValueError("trajectory and target video temporal geometry differ")
        binding.active_guidance_run = run
    if not is_progressive:
        _begin_capture(binding, guider, sampler, sigmas, latent_shapes)
    error: BaseException | None = None
    try:
        if is_progressive:
            return _run_progressive(
                executor,
                guider,
                binding,
                progressive,
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes,
            )
        return executor(
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
    except BaseException as exc:
        error = exc
        raise
    finally:
        if not is_progressive:
            _finish_capture(binding, error=error)
        binding.guidance_state.reset()
        binding.active_guidance_run = None
        binding.metrics.event(
            "sampler_wall",
            elapsed_ms=(time.perf_counter() - outer_started) * 1000.0,
            progressive=is_progressive,
            failed=error is not None,
        )


def _exact_probe_function(model, x, sigmas, extra_args=None, callback=None, disable=False, **_options):
    if sigmas.numel() != 1 or not 0 < float(sigmas[0]) < 1:
        raise ValueError("H3 exact handoff probe requires exactly one non-terminal sigma")
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    denoised = model(x, sigmas[0] * s_in, **extra_args)
    if callback is not None:
        callback({"x": x, "i": 0, "sigma": sigmas[0], "sigma_hat": sigmas[0], "denoised": denoised})
    # Comfy's flow KSAMPLER applies inverse_noise_scaling at the terminal sigma.
    return denoised * (1.0 - sigmas[0])


_exact_probe_function.__name__ = PROBE_MARKER


def _make_probe_sampler(source_sampler: Any):
    import comfy.samplers

    inpaint_options = dict(getattr(source_sampler, "inpaint_options", {}) or {})
    return comfy.samplers.KSAMPLER(_exact_probe_function, inpaint_options=inpaint_options)


def _raw_sampler_state(
    base_model: Any,
    returned: torch.Tensor,
    shapes: list[tuple[int, ...]],
    sigma: float,
) -> torch.Tensor:
    previous = getattr(base_model, "latent_shapes", None)
    try:
        base_model.latent_shapes = shapes
        carried = base_model.process_latent_in(returned)
    finally:
        base_model.latent_shapes = previous
    return carried * (1.0 - sigma)


def _process_latent_in(base_model: Any, value: torch.Tensor, shapes: list[tuple[int, ...]]) -> torch.Tensor:
    previous = getattr(base_model, "latent_shapes", None)
    try:
        base_model.latent_shapes = shapes
        return base_model.process_latent_in(value)
    finally:
        base_model.latent_shapes = previous


def _noise_argument(
    base_model: Any,
    state: torch.Tensor,
    sigma: float,
    latent_image: torch.Tensor | None = None,
) -> torch.Tensor:
    model_sampling = getattr(base_model, "model_sampling", None)
    noise_scale = float(getattr(model_sampling, "noise_scale", 1.0))
    if not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("H3 model noise_scale must be finite and positive")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("H3 noise reconstruction requires a finite positive sigma")
    numerator = state
    if latent_image is not None:
        if latent_image.shape != state.shape:
            raise ValueError("H3 latent_image and sampler state shapes differ")
        # OUTER_SAMPLE wrappers run before CFGGuider.outer_sample moves the caller's
        # latent_image/noise onto the model load device. A split sampler result has
        # already passed through that boundary, so its raw state can be on CUDA while
        # the original/private latent_image retained here is still on CPU. Reconstruct
        # the sampler noise in the state domain, matching ComfyUI's inner-sampler
        # device/dtype contract rather than assuming wrapper inputs were preloaded.
        latent_image = latent_image.to(device=state.device, dtype=state.dtype)
        numerator = state - (1.0 - float(sigma)) * latent_image
    return numerator / (float(sigma) * noise_scale)


def _resize_packed_latent_image(
    latent_image: torch.Tensor,
    source_shapes: list[tuple[int, ...]],
    target_shapes: list[tuple[int, ...]],
) -> torch.Tensor:
    video, audio = unpack_streams(latent_image, source_shapes)
    target_h, target_w = target_shapes[0][-2:]
    video = resize_spatial_5d(video, target_h, target_w, mode="bicubic")
    return pack_streams((video, audio))[0]


def _resize_packed_mask(
    mask: torch.Tensor | None,
    source_shapes: list[tuple[int, ...]],
    target_shapes: list[tuple[int, ...]],
) -> torch.Tensor | None:
    """Resize ComfyUI's already-prepared packed AV denoise mask.

    CFGGuider expands each user mask to the corresponding latent channel count
    before OUTER_SAMPLE wrappers run, so this boundary sees the same packed shape
    as the AV sampler state rather than the original single-channel mask.
    """
    if mask is None:
        return None
    if tuple(mask.shape) != (source_shapes[0][0], 1, sum(math.prod(shape[1:]) for shape in source_shapes)):
        raise ValueError("prepared H3 denoise mask does not match source packed AV geometry")
    video_mask, audio_mask = unpack_streams(mask, source_shapes)
    target_h, target_w = target_shapes[0][-2:]
    video_mask = resize_spatial_5d(video_mask, target_h, target_w, mode="nearest")
    packed, packed_shapes = pack_streams((video_mask, audio_mask))
    if packed_shapes != target_shapes:
        raise RuntimeError("resized H3 denoise mask does not match target packed AV geometry")
    return packed


def _has_exact_video_protection(
    mask: torch.Tensor | None,
    shapes: list[tuple[int, ...]],
) -> bool:
    """Return whether a prepared packed H3 mask exactly protects video values.

    ComfyUI's inpaint contract uses mask value zero for exact preservation and
    supports fractional values as intentional blends.  Progressive target-input
    sampling may therefore resize fractional masks, but it must not resize any
    video value whose downstream sampler contract is exact.
    """
    if mask is None:
        return False
    if len(shapes) != 2:
        raise ValueError("H3 exact-protection detection requires video and audio shapes")
    expected = (shapes[0][0], 1, sum(math.prod(shape[1:]) for shape in shapes))
    if tuple(mask.shape) != expected:
        raise ValueError("prepared H3 denoise mask does not match packed AV geometry")
    video_mask, _audio_mask = unpack_streams(mask, shapes)
    return bool(torch.any(video_mask == 0).item())


def _merge_preserved_noise(
    generated_noise: torch.Tensor,
    preserved_noise: torch.Tensor,
    denoise_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Keep native inpaint noise in protected regions.

    MiniMax H3's scale_latent_inpaint uses the sampler's original noise for its
    0.999 visual-conditioning injection. Replacing that noise in mask==0 regions
    changes the protected context seen by the transformer even though ComfyUI
    restores the clean latent after every denoised prediction.
    """
    if denoise_mask is None:
        return generated_noise
    if generated_noise.shape != preserved_noise.shape or generated_noise.shape != denoise_mask.shape:
        raise ValueError("H3 handoff noise/mask shapes differ")
    mask = denoise_mask.to(device=generated_noise.device, dtype=generated_noise.dtype)
    return generated_noise * mask + preserved_noise.to(generated_noise) * (1.0 - mask)


def _resize_target_keyframes(entry: dict[str, Any], target_h: int, target_w: int) -> dict[str, Any]:
    out = entry.copy()
    keyframes = entry.get("minimax_keyframes")
    if keyframes is None:
        return out
    if not isinstance(keyframes, (list, tuple)):
        raise TypeError("minimax_keyframes must be a list or tuple")
    resized_keyframes = []
    for block in keyframes:
        if not isinstance(block, dict):
            raise TypeError("MiniMax H3 keyframe entries must be dictionaries")
        resized = block.copy()
        latent = block.get("latent")
        if latent is not None:
            if not torch.is_tensor(latent) or latent.ndim not in (4, 5) or latent.shape[1] != 24:
                raise ValueError("MiniMax H3 keyframe latent must be Bx24xHxW or Bx24xTxHxW")
            if latent.ndim == 4:
                latent = resize_spatial_5d(latent.unsqueeze(2), target_h, target_w, mode="bicubic").squeeze(2)
            else:
                latent = resize_spatial_5d(latent, target_h, target_w, mode="bicubic")
            resized["latent"] = latent
            if "latent_h" in resized:
                resized["latent_h"] = int(target_h)
            if "latent_w" in resized:
                resized["latent_w"] = int(target_w)
        resized_keyframes.append(resized)
    out["minimax_keyframes"] = tuple(resized_keyframes) if isinstance(keyframes, tuple) else resized_keyframes
    return out


def _reset_guider_conds(
    guider: Any,
    *,
    template: dict[str, list[Any]] | None = None,
    target_video_hw: tuple[int, int] | None = None,
) -> None:
    """Recreate conditioning before each independent geometry/sampler lifetime.

    ComfyUI resolves percentage areas, masks, and model conditions in-place on
    guider.conds. Reusing the processed low-resolution structure for the
    probe/high stage can therefore leak low-grid shape metadata across a
    progressive handoff. The progressive caller supplies a pristine snapshot of
    guider.conds taken *after* ComfyUI's hook preprocessing/filtering so those
    call-boundary hook semantics are not lost when geometry is rebuilt.
    """
    source = template if template is not None else getattr(guider, "original_conds", None)
    if not isinstance(source, dict):
        return
    target_h, target_w = target_video_hw if target_video_hw is not None else (None, None)
    rebuilt = {}
    for key, entries in source.items():
        copied_entries = []
        for entry in entries:
            copied = entry.copy() if isinstance(entry, dict) else copy.copy(entry)
            if target_video_hw is not None and isinstance(copied, dict):
                copied = _resize_target_keyframes(copied, int(target_h), int(target_w))
            copied_entries.append(copied)
        rebuilt[key] = copied_entries
    guider.conds = rebuilt


@contextlib.contextmanager
def _flow_stage_contract(guider: Any, stage: str):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("H3 flow stage tracking requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("H3 flow stage tracking requires mutable transformer options")
    previous = transformer.get(FLOW_STAGE_KEY)
    transformer[FLOW_STAGE_KEY] = str(stage)
    try:
        yield
    finally:
        if previous is None:
            transformer.pop(FLOW_STAGE_KEY, None)
        else:
            transformer[FLOW_STAGE_KEY] = previous


@contextlib.contextmanager
def _target_sparse_stage_contract(guider: Any, contract: dict[str, Any]):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("target-sparse progressive handoff requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("target-sparse progressive handoff requires mutable transformer options")
    previous = transformer.get(TARGET_SPARSE_CONTRACT_KEY)
    transformer[TARGET_SPARSE_CONTRACT_KEY] = contract
    try:
        yield
    finally:
        if previous is None:
            transformer.pop(TARGET_SPARSE_CONTRACT_KEY, None)
        else:
            transformer[TARGET_SPARSE_CONTRACT_KEY] = previous


@contextlib.contextmanager
def _mixed_grid_stage_contract(guider, plan, metrics):
    if plan is None:
        yield
        return
    transformer = guider.model_options.setdefault("transformer_options", {})
    if MIXED_GRID_KEY in transformer:
        raise RuntimeError("nested mixed-grid stage is unsupported")
    transformer[MIXED_GRID_KEY] = {"plan": plan, "metrics": metrics}
    try:
        yield
    finally:
        transformer.pop(MIXED_GRID_KEY, None)


def _contiguous_exact_video_prefix(base_model, latent_image, denoise_mask, shapes):
    """Return a canonical whole-frame Continuum prefix in model-domain latents.

    Arbitrary masks keep their existing behavior; the bridge is only defined for
    a contiguous all-zero video prefix followed by an all-one generated suffix.
    """
    if denoise_mask is None:
        return None
    if len(shapes) != 2:
        raise ValueError("exact-prefix suffix bridge requires native video/audio shapes")
    expected = (shapes[0][0], 1, sum(math.prod(shape[1:]) for shape in shapes))
    if tuple(denoise_mask.shape) != expected:
        raise ValueError("prepared H3 denoise mask does not match packed AV geometry")
    video_mask, _audio_mask = unpack_streams(denoise_mask, shapes)
    if not bool(torch.isfinite(video_mask).all().item()):
        raise ValueError("exact-prefix suffix bridge requires a finite video mask")
    temporal = int(shapes[0][2])
    frames = video_mask.permute(2, 0, 1, 3, 4).reshape(temporal, -1)
    protected = (frames == 0).all(1)
    generated = (frames == 1).all(1)
    prefix_t = int(protected.sum().item())
    if not 0 < prefix_t < temporal:
        return None
    if not bool(protected[:prefix_t].all().item() and generated[prefix_t:].all().item()):
        return None
    internal = _process_latent_in(base_model, latent_image, shapes)
    video, _audio = unpack_streams(internal, shapes)
    return video[:, :, :prefix_t].detach().clone()


@contextlib.contextmanager
def _exact_prefix_suffix_bridge_contract(guider, exact_prefix, *, source):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("exact-prefix suffix bridge requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("exact-prefix suffix bridge requires mutable transformer options")
    if EXACT_PREFIX_BRIDGE_KEY in transformer:
        raise RuntimeError("nested exact-prefix suffix bridge contract is unsupported")
    contract = {
        "exact_prefix": exact_prefix,
        "source": str(source),
        "applied": False,
    }
    transformer[EXACT_PREFIX_BRIDGE_KEY] = contract
    try:
        yield contract
    finally:
        transformer.pop(EXACT_PREFIX_BRIDGE_KEY, None)


@contextlib.contextmanager
def _high_stage_contract(guider: Any):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("progressive handoff requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    previous = transformer.get("h3_refinement")
    request = {
        "api": 1,
        "active": True,
        "min_actual_prefix_steps": 1,
        "sigma_reference": 1.0,
        "source": "h3_flow_progressive_handoff",
    }
    if previous is not None:
        if not isinstance(previous, dict):
            raise RuntimeError("existing h3_refinement contract is not a dictionary")
        conflicts = {
            key: (previous[key], value) for key, value in request.items() if key in previous and previous[key] != value
        }
        if conflicts:
            raise RuntimeError(f"existing h3_refinement contract conflicts with progressive handoff: {conflicts}")
        request = {**previous, **request}
    transformer["h3_refinement"] = request
    try:
        yield
    finally:
        if previous is None:
            transformer.pop("h3_refinement", None)
        else:
            transformer["h3_refinement"] = previous


def _run_target_sparse_exact_prefix(
    executor,
    guider,
    binding: FlowBinding,
    config: ProgressiveTargetInputConfig,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask,
    callback,
    disable_pbar,
    seed,
    latent_shapes,
):
    """Progressively reduce H3 transformer tokens without resizing exact context.

    This path is intentionally different from the normal low-grid handoff. The
    sampler state, latent image, denoise mask, conditioning layout and RoPE all
    remain on the caller's target grid. During the early sampler lifetime only,
    H3 block wrappers keep every non-video row, every exact-protected target-video
    row and a regular coarse lattice of generated video rows. The last block
    lifts the coarse hidden field to the full target grid. A fresh full-transformer
    sampler lifetime begins at the configured handoff coordinate.
    """

    if denoise_mask is None:
        raise ValueError("target-sparse exact-prefix mode requires a prepared H3 denoise mask")
    _validate_progressive_sampler_state(sampler)
    if sigmas.ndim != 1 or sigmas.numel() < 4:
        raise ValueError("progressive handoff requires a full H3 sigma schedule")
    if not math.isclose(float(sigmas[0]), 1.0, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
        float(sigmas[-1]), 0.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("progressive handoff requires a full 1-to-0 H3 sigma schedule")

    target_shapes = list(latent_shapes)
    target_h, target_w = map(int, target_shapes[0][-2:])
    if target_h % 2 or target_w % 2:
        raise ValueError("target-sparse input H3 geometry is not patch-safe")
    source_h, source_w = config.resolve_source(target_h, target_w)

    model_options = getattr(guider, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("target-sparse progressive handoff requires mutable model options")
    transformer = model_options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("target-sparse progressive handoff requires mutable transformer options")
    current_conds = getattr(guider, "conds", None)
    if not isinstance(current_conds, dict):
        raise RuntimeError("target-sparse progressive handoff requires ComfyUI guider conditioning state")
    conditioning_template = {
        key: [entry.copy() if isinstance(entry, dict) else copy.copy(entry) for entry in entries]
        for key, entries in current_conds.items()
    }
    video_shift = float(transformer.get("minimax_h3_sigma_shift_video", H3_VIDEO_SHIFT))
    selected_coordinate = config.resolve_coordinate(source_h, source_w, target_h, target_w)
    index = select_handoff_index(
        sigmas,
        selected_coordinate,
        min_high_steps=config.min_high_steps,
        video_shift=video_shift,
    )
    sigma = float(sigmas[index].item())
    low_sigmas = sigmas[: index + 1]
    high_sigmas = sigmas[index:]
    plan = build_target_sparse_plan(
        denoise_mask,
        target_shapes,
        source_h=source_h,
        source_w=source_w,
    )
    sparse_contract = target_sparse_contract(plan)

    binding.metrics.increment("progressive_target_sparse_runs")
    binding.metrics.event(
        "handoff_plan",
        index=index,
        sigma=sigma,
        coordinate=float(normalized_coordinate(sigma, video_shift=video_shift)),
        requested_coordinate=config.handoff_coordinate,
        selected_coordinate=selected_coordinate,
        selection=config.handoff_selection,
        transfer_mode="target_sparse_lifter",
        input_mode="target_grid_sparse",
        source_shape=(*target_shapes[0][:-2], source_h, source_w),
        target_hw=(target_h, target_w),
        protected_video_rows=plan.protected_video_row_count,
        anchor_video_rows=plan.anchor_video_row_count,
        selected_video_rows=plan.selected_video_row_count,
        target_video_rows=plan.target_video_rows,
        video_row_fraction=plan.video_row_fraction,
        exact_target_latent_resized=False,
        exact_target_mask_resized=False,
    )

    sampler_invocation_count = 0
    history_boundary_count = 0
    _begin_capture(binding, guider, sampler, low_sigmas, target_shapes)
    try:
        _reset_guider_conds(guider, template=conditioning_template)
        low_started = time.perf_counter()
        try:
            sampler_invocation_count += 1
            binding.metrics.increment("progressive_sampler_invocations")
            with _flow_stage_contract(guider, "low"), _target_sparse_stage_contract(guider, sparse_contract):
                low_result = executor(
                    noise,
                    latent_image,
                    sampler,
                    low_sigmas,
                    denoise_mask,
                    callback,
                    disable_pbar,
                    seed,
                    latent_shapes=target_shapes,
                )
        finally:
            binding.metrics.event(
                "low_stage_wall",
                elapsed_ms=(time.perf_counter() - low_started) * 1000.0,
                target_sparse=True,
            )
        base_model = guider.model_patcher.model
        target_raw = _raw_sampler_state(base_model, low_result, target_shapes, sigma)
    except BaseException as exc:
        _finish_capture(binding, error=exc)
        raise

    committed_low_run = _finish_capture(binding)
    try:
        target_latent_internal = _process_latent_in(base_model, latent_image, target_shapes)
        target_noise = _noise_argument(base_model, target_raw, sigma, target_latent_internal)
        target_noise = _merge_preserved_noise(target_noise, noise, denoise_mask)

        if binding.guidance is not None and binding.guidance.mode != "off":
            if committed_low_run is None:
                raise RuntimeError("target-sparse Flow guidance requires low-stage trajectory capture")
            if committed_low_run.geometry.latent_t != int(target_shapes[0][2]):
                raise ValueError("target-sparse low trajectory and target video temporal geometry differ")
            binding.active_guidance_run = committed_low_run

        def high_callback(step, x0, x, _total):
            if callback is not None:
                return callback(index + step, x0, x, len(sigmas) - 1)
            return None

        bridge_prefix = None
        if config.suffix_dc_bridge:
            bridge_prefix = _contiguous_exact_video_prefix(
                base_model,
                latent_image,
                denoise_mask,
                target_shapes,
            )
            if bridge_prefix is None:
                binding.metrics.event(
                    "exact_prefix_suffix_dc_bridge_skipped",
                    source="target_sparse_high",
                    reason="noncanonical_exact_mask",
                )
        bridge_context = (
            _exact_prefix_suffix_bridge_contract(
                guider,
                bridge_prefix,
                source="target_sparse_high",
            )
            if bridge_prefix is not None
            else contextlib.nullcontext(None)
        )

        _reset_guider_conds(guider, template=conditioning_template)
        high_started = time.perf_counter()
        high_event_start = len(binding.metrics.events)
        sampler_invocation_count += 1
        history_boundary_count += 1
        binding.metrics.increment("progressive_sampler_invocations")
        binding.metrics.increment("progressive_history_boundaries")
        with _flow_stage_contract(guider, "high"), _high_stage_contract(guider), bridge_context as bridge_contract:
            result = executor(
                target_noise,
                latent_image,
                sampler,
                high_sigmas,
                denoise_mask,
                high_callback,
                disable_pbar,
                seed,
                latent_shapes=target_shapes,
            )
        if isinstance(bridge_contract, dict) and not bool(bridge_contract.get("applied")):
            raise RuntimeError(
                "target-sparse exact-prefix suffix bridge did not observe the required actual high-stage H3 evaluation"
            )
        binding.metrics.event(
            "high_stage_wall",
            elapsed_ms=(time.perf_counter() - high_started) * 1000.0,
            target_sparse=False,
        )
        high_model_calls = [event for event in binding.metrics.events[high_event_start:] if event.kind == "model_call"]
        if not high_model_calls:
            raise RuntimeError("target-sparse progressive high stage produced no H3 model evaluations")
        first_high_actual = bool(high_model_calls[0].fields.get("actual"))
        if not first_high_actual:
            raise RuntimeError("target-sparse progressive high stage did not begin with an exact H3 model evaluation")
        binding.metrics.event(
            "handoff_complete",
            sigma=sigma,
            target_shape=target_shapes[0],
            audio_state_copied=False,
            separate_sampler_invocations=True,
            sampler_invocation_count=sampler_invocation_count,
            history_boundary_count=history_boundary_count,
            exact_probe_performed=False,
            high_stage_exact_prefix_requested=1,
            high_stage_first_call_actual=first_high_actual,
            high_stage_model_calls=len(high_model_calls),
            conditioning_rebuilt_for_high_grid=False,
            transfer_mode="target_sparse_lifter",
            input_mode="target_grid_sparse",
            exact_target_inputs_forwarded=True,
            exact_protected_rows_retained=True,
            source_latent_resize_performed=False,
            learned_transfer_performed=False,
        )
        return result
    except BaseException as exc:
        if committed_low_run is not None and binding.trajectory is not None:
            invalid = binding.trajectory.invalidate(
                committed_low_run.run_id,
                f"target-sparse progressive continuation failed: {type(exc).__name__}: {exc}",
            )
            binding.metrics.event("trajectory_invalidate", run_id=invalid.run_id, error=type(exc).__name__)
        raise


def _run_progressive(
    executor,
    guider,
    binding: FlowBinding,
    config: ProgressiveHandoffConfig | ProgressiveTargetInputConfig,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask,
    callback,
    disable_pbar,
    seed,
    latent_shapes,
):
    chunk_started = time.perf_counter()
    if len(latent_shapes) != 2:
        raise ValueError("progressive handoff supports native packed H3 AV latents only")
    if isinstance(config, ProgressiveTargetInputConfig) and config.exact_prefix_mode == "mixed_grid_low_suffix":
        # Recheck the assembled workflow, including VDN applied after Flow.
        from .comfy_compat import _validate_vdn_target_sparse_compat

        patcher = guider.model_patcher
        _validate_vdn_target_sparse_compat(patcher, len(patcher.model.diffusion_model.blocks), minimum_api=2)
    input_shapes = list(latent_shapes)
    mixed = (
        isinstance(config, ProgressiveTargetInputConfig)
        and config.exact_prefix_mode == "mixed_grid_low_suffix"
        and _has_exact_video_protection(denoise_mask, input_shapes)
    )
    mixed_plan = None
    if (
        not mixed
        and isinstance(config, ProgressiveTargetInputConfig)
        and _has_exact_video_protection(denoise_mask, input_shapes)
    ):
        if config.exact_prefix_mode == "target_sparse_lifter":
            return _run_target_sparse_exact_prefix(
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
            )
        # Native masked continuation promises exact video-prefix preservation.
        # A private low-grid lifetime would resize those clean prefix latents and
        # expose the changed values to every generated row through H3's dense
        # attention.  A later mask/noise merge can restore the returned prefix,
        # but it cannot undo that altered low-stage context.  Preserve the exact
        # contract by executing the untouched target-grid sampler once.
        binding.metrics.increment("progressive_target_fallbacks")
        binding.metrics.increment("progressive_sampler_invocations")
        binding.metrics.event(
            "progressive_target_fallback",
            reason="exact_video_protection",
            input_mode="target_grid",
            target_shape=input_shapes[0],
            sampler_invocation_count=1,
            history_boundary_count=0,
            exact_target_inputs_forwarded=True,
            progressive_guidance_applied=False,
        )
        _begin_capture(binding, guider, sampler, sigmas, input_shapes)
        error: BaseException | None = None
        try:
            return executor(
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
        except BaseException as exc:
            error = exc
            raise
        finally:
            _finish_capture(binding, error=error)
    _validate_progressive_sampler_state(sampler)
    if sigmas.ndim != 1 or sigmas.numel() < 4:
        raise ValueError("progressive handoff requires a full H3 sigma schedule")
    if not math.isclose(float(sigmas[0]), 1.0, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
        float(sigmas[-1]), 0.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("progressive handoff requires a full 1-to-0 H3 sigma schedule")
    if input_shapes[0][-2] % 2 or input_shapes[0][-1] % 2:
        raise ValueError("input H3 geometry is not patch-safe")
    model_options = getattr(guider, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("progressive handoff requires mutable model options")
    transformer = model_options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("progressive handoff requires mutable transformer options")
    current_conds = getattr(guider, "conds", None)
    if not isinstance(current_conds, dict):
        raise RuntimeError("progressive handoff requires ComfyUI guider conditioning state")
    conditioning_template = {
        key: [entry.copy() if isinstance(entry, dict) else copy.copy(entry) for entry in entries]
        for key, entries in current_conds.items()
    }
    video_shift = float(transformer.get("minimax_h3_sigma_shift_video", H3_VIDEO_SHIFT))

    target_input = isinstance(config, ProgressiveTargetInputConfig)
    if target_input:
        target_shapes = input_shapes
        target_h, target_w = target_shapes[0][-2:]
        source_h, source_w = config.resolve_source(target_h, target_w)
        if mixed:
            mixed_plan = build_mixed_grid_plan(
                denoise_mask,
                input_shapes,
                _process_latent_in(guider.model_patcher.model, latent_image, input_shapes),
                noise,
                source_h=source_h,
                source_w=source_w,
            )
            binding.metrics.event(
                "mixed_grid_plan",
                input_mode="mixed_grid_low_suffix",
                transfer_mode="learned_3d_suffix",
                prefix_temporal_length=mixed_plan.prefix_t,
                suffix_temporal_length=mixed_plan.temporal - mixed_plan.prefix_t,
                prefix_target_hw=mixed_plan.target_hw,
                suffix_source_hw=(source_h, source_w),
                target_hw=mixed_plan.target_hw,
                full_target_video_rows=mixed_plan.temporal * mixed_plan.target_rows,
                mixed_video_rows=mixed_plan.mixed_rows,
                prefix_video_rows=mixed_plan.prefix_rows,
                suffix_video_rows=(mixed_plan.temporal - mixed_plan.prefix_t) * mixed_plan.source_rows,
                prefix_exact_latent_resized=False,
                prefix_target_grid_rope=True,
                suffix_source_grid_rope=True,
                continuous_temporal_rope=True,
                low_suffix_real_latent=True,
            )
        source_shapes = list(target_shapes)
        source_shapes[0] = (*source_shapes[0][:-2], source_h, source_w)
        target_video_noise, target_audio_noise = unpack_streams(noise, target_shapes)
        source_video_noise = deterministic_video_noise(
            (*target_video_noise.shape[:-2], source_h, source_w),
            seed=int(seed or 0) + config.source_noise_offset,
            device=target_video_noise.device,
            dtype=target_video_noise.dtype,
        )
        low_noise = pack_streams((source_video_noise, target_audio_noise))[0]
        low_latent_image = _resize_packed_latent_image(latent_image, target_shapes, source_shapes)
        low_mask = _resize_packed_mask(denoise_mask, target_shapes, source_shapes)
    else:
        source_shapes = input_shapes
        source_h, source_w = source_shapes[0][-2:]
        target_h, target_w = config.resolve_target(source_h, source_w)
        if target_h * target_w <= source_h * source_w:
            raise ValueError("progressive handoff target must increase the video latent area")
        target_shapes = None
        low_noise = noise
        low_latent_image = latent_image
        low_mask = denoise_mask
    selected_coordinate = config.resolve_coordinate(
        source_shapes[0][-2],
        source_shapes[0][-1],
        target_h,
        target_w,
    )
    index = select_handoff_index(
        sigmas,
        selected_coordinate,
        min_high_steps=config.min_high_steps,
        video_shift=video_shift,
    )
    sigma = float(sigmas[index].item())
    low_sigmas = sigmas[: index + 1]
    high_sigmas = sigmas[index:]
    binding.metrics.event(
        "handoff_plan",
        index=index,
        sigma=sigma,
        coordinate=float(normalized_coordinate(sigma, video_shift=video_shift)),
        requested_coordinate=config.handoff_coordinate,
        selected_coordinate=selected_coordinate,
        selection=config.handoff_selection,
        transfer_mode=config.transfer_mode,
        input_mode="mixed_grid_low_suffix" if mixed else ("target_grid" if target_input else "source_grid"),
        source_shape=source_shapes[0],
        target_hw=(target_h, target_w),
    )

    sampler_invocation_count = 0
    history_boundary_count = 0

    def low_callback(step, x0, x, _total):
        if callback is None:
            return None
        if target_input:
            # CFGGuider's packed callback closure was created against the caller's
            # target latent_shapes. Feed it target-shaped preview/state tensors even
            # though this private sampler lifetime runs on source_shapes.
            x0 = _resize_packed_latent_image(x0, source_shapes, target_shapes)
            x = _resize_packed_latent_image(x, source_shapes, target_shapes)
        return callback(step, x0, x, len(sigmas) - 1)

    _begin_capture(binding, guider, sampler, low_sigmas, source_shapes)
    try:
        _reset_guider_conds(
            guider,
            template=conditioning_template,
            target_video_hw=(source_h, source_w) if target_input and not mixed else None,
        )
        low_started = time.perf_counter()
        try:
            sampler_invocation_count += 1
            binding.metrics.increment("progressive_sampler_invocations")
            with _flow_stage_contract(guider, "low"), _mixed_grid_stage_contract(guider, mixed_plan, binding.metrics):
                low_result = executor(
                    low_noise,
                    low_latent_image,
                    sampler,
                    low_sigmas,
                    low_mask,
                    low_callback,
                    disable_pbar,
                    seed,
                    latent_shapes=source_shapes,
                )
        finally:
            binding.metrics.event("low_stage_wall", elapsed_ms=(time.perf_counter() - low_started) * 1000.0)
        base_model = guider.model_patcher.model
        source_raw = _raw_sampler_state(base_model, low_result, source_shapes, sigma)
        source_latent_internal = _process_latent_in(base_model, low_latent_image, source_shapes)
        active = binding.active_capture
        if active is not None:
            active.phases = (*active.phases, (index, "handoff_probe"))
        probe_started = time.perf_counter()
        probe_noise = _noise_argument(base_model, source_raw, sigma, source_latent_internal)
        previous_probe = transformer.get(PROBE_CONTEXT_KEY)
        transformer[PROBE_CONTEXT_KEY] = {"outer_step": index}
        try:
            _reset_guider_conds(
                guider,
                template=conditioning_template,
                target_video_hw=(source_h, source_w) if target_input and not mixed else None,
            )
            # The probe is a one-call sampler lifetime, but model-level patches such
            # as DiffAid must still see the full H3 sigma reference. The explicit
            # refinement contract provides that reference without carrying solver or
            # Spectrum history across the split.
            sampler_invocation_count += 1
            history_boundary_count += 1
            binding.metrics.increment("progressive_sampler_invocations")
            binding.metrics.increment("progressive_history_boundaries")
            with (
                _flow_stage_contract(guider, "probe"),
                _high_stage_contract(guider),
                _mixed_grid_stage_contract(guider, mixed_plan, binding.metrics),
            ):
                source_x0 = executor(
                    probe_noise,
                    low_latent_image,
                    _make_probe_sampler(sampler),
                    sigmas[index : index + 1],
                    low_mask,
                    None,
                    disable_pbar,
                    seed,
                    latent_shapes=source_shapes,
                )
        finally:
            if previous_probe is None:
                transformer.pop(PROBE_CONTEXT_KEY, None)
            else:
                transformer[PROBE_CONTEXT_KEY] = previous_probe
            binding.metrics.event("handoff_probe_wall", elapsed_ms=(time.perf_counter() - probe_started) * 1000.0)
    except BaseException as exc:
        _finish_capture(binding, error=exc)
        raise
    committed_low_run = _finish_capture(binding)
    binding.metrics.increment("handoff_exact_probe_nfe")
    try:
        source_x0 = _process_latent_in(base_model, source_x0, source_shapes)
        if mixed_plan is not None:
            # Use all prefix frames as transient upscaler context. Its 3D attention
            # has no proven finite temporal receptive field permitting truncation.
            clean_video, clean_audio = unpack_streams(source_x0, source_shapes)
            clean_video = clean_video.clone()
            clean_video[:, :, : mixed_plan.prefix_t] = resize_spatial_5d(
                mixed_plan.prefix.to(clean_video), source_h, source_w, mode="bicubic"
            )
            source_x0 = pack_streams((clean_video, clean_audio))[0]
        transfer_started = time.perf_counter()
        transfer_metrics: dict[str, Any] = {}
        splice_diagnostics: dict[str, Any] = {}
        target_raw, target_shapes = build_handoff_state(
            source_packed_state=source_raw,
            source_x0_packed=source_x0,
            source_shapes=source_shapes,
            sigma=sigma,
            target_h=target_h,
            target_w=target_w,
            seed=int(seed or 0) + config.seed_offset,
            transfer_mode=config.transfer_mode,
            learned_upscaler=getattr(config, "learned_upscaler", None),
            transfer_metrics=transfer_metrics,
        )
        if mixed_plan is not None:
            target_video, target_audio = unpack_streams(target_raw, target_shapes)
            diagnostic_started = time.perf_counter()
            diagnostic_noise = deterministic_video_noise(
                tuple(target_video.shape),
                seed=int(seed or 0) + config.seed_offset,
                device=target_video.device,
                dtype=target_video.dtype,
            )
            learned_clean = recover_conditional_clean_for_diagnostics(
                target_video,
                diagnostic_noise,
                sigma=sigma,
            )
            exact_prefix = mixed_plan.prefix.to(device=learned_clean.device, dtype=learned_clean.dtype)
            bridge_enabled = bool(getattr(config, "suffix_dc_bridge", False))
            if bridge_enabled:
                corrected_clean, bridge_metrics = apply_suffix_dc_bridge(
                    learned_clean,
                    exact_prefix,
                    weights=(1.0,),
                )
                target_video = map_clean_bridge_to_conditional_state(
                    target_video,
                    learned_clean,
                    corrected_clean,
                    sigma=sigma,
                    prefix_t=mixed_plan.prefix_t,
                    corrected_tokens=int(bridge_metrics["suffix_dc_bridge_corrected_tokens"]),
                )
            else:
                corrected_clean = learned_clean
                bridge_metrics = disabled_suffix_dc_bridge_metrics(prefix_t=mixed_plan.prefix_t)
            splice_diagnostics = measure_exact_prefix_splice(
                learned_clean,
                exact_prefix,
                corrected_clean_video=corrected_clean,
            )
            splice_diagnostics["splice_diagnostic_elapsed_ms"] = (time.perf_counter() - diagnostic_started) * 1000.0
            splice_diagnostics["splice_recovery"] = "inverse_conditional_renoise"
            splice_diagnostics["suffix_dc_bridge_state_mapping"] = (
                "affine_equivalent_pre_renoise" if bridge_enabled else "disabled"
            )
            binding.metrics.increment("mixed_grid_splice_diagnostic_runs")
            del diagnostic_noise, learned_clean, corrected_clean

            if not bridge_enabled:
                target_video = target_video.clone()
            target_video[:, :, : mixed_plan.prefix_t] = mixed_plan.prefix.to(target_video)
            target_raw = pack_streams((target_video, target_audio))[0]
            binding.metrics.event(
                "mixed_grid_transfer",
                learned_transfer_performed=True,
                upscaler_prefix_context_used=True,
                upscaler_prefix_output_discarded=True,
                final_original_prefix_restored=True,
                transfer_mode="learned_3d_suffix",
                **bridge_metrics,
                **splice_diagnostics,
            )
        if config.transfer_mode == "learned_3d":
            binding.metrics.event(
                "handoff_learned_upscale_wall",
                elapsed_ms=transfer_metrics["learned_upscale_elapsed_ms"],
                provider_api_version=transfer_metrics["provider_api_version"],
                provider_kind=transfer_metrics["provider_kind"],
                model_name=transfer_metrics["model_name"],
                source_hw=transfer_metrics["source_hw"],
                target_hw=transfer_metrics["target_hw"],
                temporal_length=transfer_metrics["temporal_length"],
                input_dtype=transfer_metrics["input_dtype"],
                input_device=transfer_metrics["input_device"],
                inference_precision=transfer_metrics["inference_precision"],
                configured_device=transfer_metrics["configured_device"],
                inference_device=transfer_metrics["inference_device"],
                offload_after_upscale=transfer_metrics["offload_after_upscale"],
                offloaded_after_upscale=transfer_metrics["offloaded_after_upscale"],
                output_dtype=transfer_metrics["output_dtype"],
                output_device=transfer_metrics["output_device"],
            )
        if target_input and target_shapes != input_shapes:
            raise RuntimeError("target-input progressive handoff changed the caller-visible AV geometry")
        if target_input:
            target_latent_image = latent_image
            target_mask = denoise_mask
        else:
            target_latent_image = _resize_packed_latent_image(latent_image, source_shapes, target_shapes)
            target_mask = _resize_packed_mask(denoise_mask, source_shapes, target_shapes)
            latent_shapes[:] = target_shapes
        target_latent_internal = _process_latent_in(base_model, target_latent_image, target_shapes)
        target_noise = _noise_argument(base_model, target_raw, sigma, target_latent_internal)
        if target_input:
            target_noise = _merge_preserved_noise(target_noise, noise, target_mask)
        binding.metrics.event("handoff_transfer_wall", elapsed_ms=(time.perf_counter() - transfer_started) * 1000.0)

        if binding.guidance is not None and binding.guidance.mode != "off":
            if binding.trajectory is None:
                raise RuntimeError("flow guidance requires an H3_FLOW_TRAJECTORY")
            session_id, chunk_id = _interop_identity(getattr(guider, "model_options", None))
            expected_signature = binding.guidance_conditioning_signature or _conditioning_signature(guider)
            run = binding.trajectory.select(
                chunk_id=chunk_id,
                session_id=session_id,
                conditioning_signature=expected_signature,
            )
            if run.geometry.latent_t != int(target_shapes[0][2]):
                raise ValueError("trajectory and target video temporal geometry differ")
            binding.active_guidance_run = run

        def high_callback(step, x0, x, _total):
            if callback is not None:
                return callback(index + step, x0, x, len(sigmas) - 1)
            return None

        high_started = time.perf_counter()
        high_event_start = len(binding.metrics.events)
        _reset_guider_conds(
            guider,
            template=conditioning_template,
            target_video_hw=None if target_input else (target_h, target_w),
        )
        sampler_invocation_count += 1
        history_boundary_count += 1
        binding.metrics.increment("progressive_sampler_invocations")
        binding.metrics.increment("progressive_history_boundaries")
        with _flow_stage_contract(guider, "high"), _high_stage_contract(guider):
            result = executor(
                target_noise,
                target_latent_image,
                sampler,
                high_sigmas,
                target_mask,
                high_callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        binding.metrics.event("high_stage_wall", elapsed_ms=(time.perf_counter() - high_started) * 1000.0)
        high_model_calls = [event for event in binding.metrics.events[high_event_start:] if event.kind == "model_call"]
        if not high_model_calls:
            raise RuntimeError("progressive high stage produced no H3 model evaluations")
        first_high_actual = bool(high_model_calls[0].fields.get("actual"))
        if not first_high_actual:
            raise RuntimeError("progressive high stage did not begin with the required exact H3 model evaluation")
        if mixed_plan is not None:
            final_video, _ = unpack_streams(result, target_shapes)
            original_video, _ = unpack_streams(latent_image, target_shapes)
            if not torch.equal(
                final_video[:, :, : mixed_plan.prefix_t], original_video[:, :, : mixed_plan.prefix_t].to(final_video)
            ):
                raise RuntimeError("mixed-grid high stage violated exact original-prefix preservation")
            final_boundary = measure_video_boundary(final_video, mixed_plan.prefix_t)
            if not splice_diagnostics:
                raise RuntimeError("mixed-grid splice diagnostics were not recorded before high-stage sampling")
            binding.metrics.event(
                "mixed_grid_complete",
                final_prefix_exact=True,
                high_stage_first_call_actual=first_high_actual,
                total_chunk_sampler_elapsed_ms=(time.perf_counter() - chunk_started) * 1000.0,
                final_seam_lowpass_kernel=final_boundary["lowpass_kernel"],
                final_seam_rms=final_boundary["seam_rms"],
                final_seam_lowpass_rms=final_boundary["seam_lowpass_rms"],
                final_seam_spatial_mean_rms=final_boundary["seam_spatial_mean_rms"],
                final_over_transfer_exact_seam_rms_ratio=final_boundary["seam_rms"]
                / max(float(splice_diagnostics["exact_restored_seam_rms"]), 1e-12),
                final_over_transfer_exact_seam_lowpass_ratio=final_boundary["seam_lowpass_rms"]
                / max(float(splice_diagnostics["exact_restored_seam_lowpass_rms"]), 1e-12),
                final_over_transfer_exact_seam_spatial_mean_ratio=final_boundary["seam_spatial_mean_rms"]
                / max(float(splice_diagnostics["exact_restored_seam_spatial_mean_rms"]), 1e-12),
                final_over_transfer_corrected_seam_rms_ratio=final_boundary["seam_rms"]
                / max(float(splice_diagnostics["corrected_exact_seam_rms"]), 1e-12),
                final_over_transfer_corrected_seam_lowpass_ratio=final_boundary["seam_lowpass_rms"]
                / max(float(splice_diagnostics["corrected_exact_seam_lowpass_rms"]), 1e-12),
                final_over_transfer_corrected_seam_spatial_mean_ratio=final_boundary["seam_spatial_mean_rms"]
                / max(float(splice_diagnostics["corrected_exact_seam_spatial_mean_rms"]), 1e-12),
            )
        binding.metrics.event(
            "handoff_complete",
            sigma=sigma,
            target_shape=target_shapes[0],
            audio_state_copied=True,
            separate_sampler_invocations=True,
            sampler_invocation_count=sampler_invocation_count,
            history_boundary_count=history_boundary_count,
            exact_probe_performed=True,
            high_stage_exact_prefix_requested=1,
            high_stage_first_call_actual=first_high_actual,
            high_stage_model_calls=len(high_model_calls),
            conditioning_rebuilt_for_high_grid=True,
            transfer_mode=config.transfer_mode,
            input_mode="mixed_grid_low_suffix" if mixed else ("target_grid" if target_input else "source_grid"),
        )
        return result

    except BaseException as exc:
        if committed_low_run is not None and binding.trajectory is not None:
            invalid = binding.trajectory.invalidate(
                committed_low_run.run_id,
                f"progressive continuation failed: {type(exc).__name__}: {exc}",
            )
            binding.metrics.event(
                "trajectory_invalidate",
                run_id=invalid.run_id,
                error=type(exc).__name__,
            )
        raise


def clone_sampler(sampler: Any) -> Any:
    cloned = copy.copy(sampler)
    if hasattr(sampler, "extra_options"):
        cloned.extra_options = dict(getattr(sampler, "extra_options", {}) or {})
    return cloned
