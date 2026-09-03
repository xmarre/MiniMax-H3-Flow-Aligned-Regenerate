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
from .geometry import geometry_from_video, pack_streams, unpack_streams
from .guidance import GuidanceConfig, GuidanceState, apply_guidance
from .handoff import ProgressiveHandoffConfig, build_handoff_state, select_handoff_index
from .metrics import H3FlowMetrics
from .sigma import H3_AUDIO_SHIFT, H3_VIDEO_SHIFT, audio_sigma, normalized_coordinate

LOG = logging.getLogger(__name__)

FLOW_BINDING_KEY = "h3_flow_regenerate_binding"
PROGRESSIVE_KEY = "h3_flow_progressive_v1"
OUTER_WRAPPER_KEY = "h3_flow_regenerate.outer.v1"
PREDICT_WRAPPER_KEY = "h3_flow_regenerate.predict.v1"
PROBE_MARKER = "_h3_flow_exact_probe"
PROBE_CONTEXT_KEY = "h3_flow_exact_probe_context"
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
    guidance_state: GuidanceState = field(default_factory=GuidanceState)
    active_capture: _ActiveCapture | None = None
    active_guidance_run: Any = None


def sampler_name(sampler: Any) -> str:
    function = getattr(sampler, "sampler_function", None)
    return str(getattr(function, "__name__", type(sampler).__name__))


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


def _update_keyframe_digest(digest, value: Any, *, depth: int) -> None:
    """Hash keyframe semantics while ignoring expected target-grid resize fields."""
    if not isinstance(value, (list, tuple)):
        _update_conditioning_digest(digest, value, depth=depth + 1)
        return
    digest.update(f"keyframes:{len(value)}:".encode())
    for block in value:
        if not isinstance(block, dict):
            _update_conditioning_digest(digest, block, depth=depth + 1)
            continue
        latent = block.get("latent")
        if torch.is_tensor(latent):
            leading_shape = tuple(int(size) for size in latent.shape[:-2])
            digest.update(f"latent-leading:{leading_shape}|{latent.dtype}|{latent.layout}:".encode())
        for key in sorted(block, key=lambda item: str(item)):
            if str(key) in {"latent", "latent_h", "latent_w"}:
                continue
            digest.update(f"kf-key:{key!s}:".encode())
            _update_conditioning_digest(digest, block[key], depth=depth + 1)

def _update_conditioning_digest(digest, value: Any, *, depth: int = 0) -> None:
    if depth > 8:
        digest.update(f"<depth:{type(value).__module__}.{type(value).__qualname__}>".encode())
        return
    if torch.is_tensor(value):
        digest.update(b"tensor:")
        digest.update(_tensor_signature(value))
        return
    if isinstance(value, dict):
        digest.update(f"dict:{len(value)}:".encode())
        for key in sorted(value, key=lambda item: str(item)):
            # ComfyUI's convert_cond creates a fresh UUID on each conversion.
            # It is execution identity, not conditioning identity.
            if str(key) == "uuid":
                continue
            digest.update(f"key:{key!s}:".encode())
            if str(key) == "minimax_keyframes":
                _update_keyframe_digest(digest, value[key], depth=depth)
            else:
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


def _conditioning_signature(guider: Any) -> str:
    digest = hashlib.sha256()
    original = getattr(guider, "original_conds", {}) or {}
    _update_conditioning_digest(digest, original)
    return digest.hexdigest()[:32]


def _resolve_binding(guider: Any) -> FlowBinding | None:
    value = (getattr(guider, "model_options", None) or {}).get(FLOW_BINDING_KEY)
    return value if isinstance(value, FlowBinding) else None


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


def _finish_capture(binding: FlowBinding, *, error: BaseException | None = None) -> None:
    active = binding.active_capture
    if active is None or binding.trajectory is None:
        return
    binding.active_capture = None
    if error is None:
        run = binding.trajectory.commit(active.run_id)
        binding.metrics.event(
            "trajectory_commit",
            run_id=run.run_id,
            samples=len(run.samples),
            trajectory_bytes=binding.trajectory.bytes,
        )
    else:
        binding.trajectory.abort(active.run_id, f"{type(error).__name__}: {error}")
        binding.metrics.event("trajectory_abort", run_id=active.run_id, error=type(error).__name__)


def flow_predict_wrapper(executor, x, timestep, model_options=None, seed=None):
    started = time.perf_counter()
    result = executor(x, timestep, model_options, seed)
    guider = executor.class_obj
    binding = _resolve_binding(guider)
    if binding is None:
        return result
    transformer = (model_options or {}).get("transformer_options") or {}
    probe_context = transformer.get(PROBE_CONTEXT_KEY)
    actual_value = transformer.get(SPECTRUM_ACTUAL_KEY)
    actual = True if actual_value is None or isinstance(probe_context, dict) else bool(actual_value)
    binding.metrics.increment("transformer_actual_nfe" if actual else "spectrum_forecast_calls")
    binding.metrics.increment("sampler_logical_calls")
    sigma = float(timestep.detach().reshape(-1)[0].item())
    video_shift = float(transformer.get("minimax_h3_sigma_shift_video", H3_VIDEO_SHIFT))
    audio_shift = float(transformer.get("minimax_h3_sigma_shift_audio", H3_AUDIO_SHIFT))
    coordinate = float(normalized_coordinate(sigma, video_shift=video_shift))
    binding.metrics.event(
        "model_call",
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

    if binding.guidance is not None and binding.guidance.mode != "off":
        run = binding.active_guidance_run
        if run is None:
            return result
        base_model = getattr(guider, "inner_model", None)
        shapes = getattr(base_model, "latent_shapes", None)
        if not isinstance(shapes, list) or len(shapes) != 2:
            raise RuntimeError("flow guidance could not resolve H3 AV latent shapes")
        video_x0, audio_x0 = unpack_streams(result, shapes)
        guided_video = apply_guidance(
            video_x0,
            run=run,
            coordinate=coordinate,
            config=binding.guidance,
            state=binding.guidance_state,
        )
        result, _ = pack_streams((guided_video, audio_x0))
        binding.metrics.event("guidance", coordinate=coordinate, mode=binding.guidance.mode)
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
    is_progressive = isinstance(progressive, ProgressiveHandoffConfig)
    if not is_progressive and binding.guidance is not None and binding.guidance.mode != "off":
        if binding.trajectory is None:
            raise RuntimeError("flow guidance requires an H3_FLOW_TRAJECTORY")
        session_id, chunk_id = _interop_identity(getattr(guider, "model_options", None))
        run = binding.trajectory.select(chunk_id=chunk_id, session_id=session_id)
        if run.geometry.latent_t != int(latent_shapes[0][2]):
            raise ValueError("trajectory and target video temporal geometry differ")
        if run.conditioning_signature != _conditioning_signature(guider):
            raise ValueError("trajectory conditioning signature does not match the regeneration run")
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


def _make_probe_sampler():
    import comfy.samplers

    return comfy.samplers.KSAMPLER(_exact_probe_function)


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


def _noise_argument(base_model: Any, state: torch.Tensor, sigma: float) -> torch.Tensor:
    model_sampling = getattr(base_model, "model_sampling", None)
    noise_scale = float(getattr(model_sampling, "noise_scale", 1.0))
    if not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("H3 model noise_scale must be finite and positive")
    return state / (sigma * noise_scale)


def _reset_guider_conds(guider: Any) -> None:
    """Recreate raw conditioning before each independent geometry/sampler lifetime.

    ComfyUI resolves percentage areas, masks, and model conditions in-place on
    guider.conds. Reusing the processed low-resolution structure for the
    probe/high stage can therefore leak low-grid shape metadata across a
    progressive handoff.
    """
    original = getattr(guider, "original_conds", None)
    if not isinstance(original, dict):
        return
    guider.conds = {
        key: [entry.copy() if isinstance(entry, dict) else copy.copy(entry) for entry in entries]
        for key, entries in original.items()
    }


@contextlib.contextmanager
def _high_stage_contract(guider: Any):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("progressive handoff requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    previous = transformer.get("h3_refinement")
    transformer["h3_refinement"] = {
        "api": 1,
        "active": True,
        "min_actual_prefix_steps": 1,
        "sigma_reference": 1.0,
        "source": "h3_flow_progressive_handoff",
    }
    try:
        yield
    finally:
        if previous is None:
            transformer.pop("h3_refinement", None)
        else:
            transformer["h3_refinement"] = previous


def _run_progressive(
    executor,
    guider,
    binding: FlowBinding,
    config: ProgressiveHandoffConfig,
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
    if len(latent_shapes) != 2:
        raise ValueError("progressive handoff supports native packed H3 AV latents only")
    if denoise_mask is not None:
        raise RuntimeError(
            "progressive handoff does not yet support denoise masks; preserving masked H3 regions "
            "requires carrying and spatially transforming the sampler latent_image across the grid transition"
        )
    if sigmas.ndim != 1 or sigmas.numel() < 4:
        raise ValueError("progressive handoff requires a full H3 sigma schedule")
    if not math.isclose(float(sigmas[0]), 1.0, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
        float(sigmas[-1]), 0.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("progressive handoff requires a full 1-to-0 H3 sigma schedule")
    source_shapes = list(latent_shapes)
    if source_shapes[0][-2] % 2 or source_shapes[0][-1] % 2:
        raise ValueError("source H3 geometry is not patch-safe")
    model_options = getattr(guider, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("progressive handoff requires mutable model options")
    transformer = model_options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("progressive handoff requires mutable transformer options")
    video_shift = float(transformer.get("minimax_h3_sigma_shift_video", H3_VIDEO_SHIFT))
    source_h, source_w = source_shapes[0][-2:]
    target_h, target_w = config.resolve_target(source_h, source_w)
    if target_h * target_w <= source_h * source_w:
        raise ValueError("progressive handoff target must increase the video latent area")
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
        source_shape=source_shapes[0],
        target_hw=(target_h, target_w),
    )

    def low_callback(step, x0, x, _total):
        if callback is not None:
            return callback(step, x0, x, len(sigmas) - 1)
        return None

    _begin_capture(binding, guider, sampler, low_sigmas, source_shapes)
    try:
        _reset_guider_conds(guider)
        low_started = time.perf_counter()
        try:
            low_result = executor(
                noise,
                latent_image,
                sampler,
                low_sigmas,
                denoise_mask,
                low_callback,
                disable_pbar,
                seed,
                latent_shapes=source_shapes,
            )
        finally:
            binding.metrics.event("low_stage_wall", elapsed_ms=(time.perf_counter() - low_started) * 1000.0)
        base_model = guider.model_patcher.model
        source_raw = _raw_sampler_state(base_model, low_result, source_shapes, sigma)
        active = binding.active_capture
        if active is not None:
            active.phases = (*active.phases, (index, "handoff_probe"))
        probe_started = time.perf_counter()
        probe_noise = _noise_argument(base_model, source_raw, sigma)
        previous_probe = transformer.get(PROBE_CONTEXT_KEY)
        transformer[PROBE_CONTEXT_KEY] = {"outer_step": index}
        try:
            _reset_guider_conds(guider)
            # The probe is a one-call sampler lifetime, but model-level patches such
            # as DiffAid must still see the full H3 sigma reference. The explicit
            # refinement contract provides that reference without carrying solver or
            # Spectrum history across the split.
            with _high_stage_contract(guider):
                source_x0 = executor(
                    probe_noise,
                    torch.zeros_like(source_raw),
                    _make_probe_sampler(),
                    sigmas[index : index + 1],
                    denoise_mask,
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
    _finish_capture(binding)
    binding.metrics.increment("handoff_exact_probe_nfe")
    source_x0 = _process_latent_in(base_model, source_x0, source_shapes)
    transfer_started = time.perf_counter()
    target_raw, target_shapes = build_handoff_state(
        source_packed_state=source_raw,
        source_x0_packed=source_x0,
        source_shapes=source_shapes,
        sigma=sigma,
        target_h=target_h,
        target_w=target_w,
        seed=int(seed or 0) + config.seed_offset,
        transfer_mode=config.transfer_mode,
    )
    binding.metrics.event("handoff_transfer_wall", elapsed_ms=(time.perf_counter() - transfer_started) * 1000.0)
    latent_shapes[:] = target_shapes

    if binding.guidance is not None and binding.guidance.mode != "off":
        if binding.trajectory is None:
            raise RuntimeError("flow guidance requires an H3_FLOW_TRAJECTORY")
        session_id, chunk_id = _interop_identity(getattr(guider, "model_options", None))
        run = binding.trajectory.select(chunk_id=chunk_id, session_id=session_id)
        if run.geometry.latent_t != int(target_shapes[0][2]):
            raise ValueError("trajectory and target video temporal geometry differ")
        if run.conditioning_signature != _conditioning_signature(guider):
            raise ValueError("trajectory conditioning signature does not match the regeneration run")
        binding.active_guidance_run = run

    def high_callback(step, x0, x, _total):
        if callback is not None:
            return callback(index + step, x0, x, len(sigmas) - 1)
        return None

    high_started = time.perf_counter()
    _reset_guider_conds(guider)
    with _high_stage_contract(guider):
        result = executor(
            _noise_argument(base_model, target_raw, sigma),
            torch.zeros_like(target_raw),
            sampler,
            high_sigmas,
            None,
            high_callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    binding.metrics.event("high_stage_wall", elapsed_ms=(time.perf_counter() - high_started) * 1000.0)
    binding.metrics.event(
        "handoff_complete",
        sigma=sigma,
        target_shape=target_shapes[0],
        audio_state_copied=True,
        separate_sampler_invocations=True,
        exact_probe_performed=True,
        high_stage_exact_prefix_requested=1,
        conditioning_rebuilt_for_high_grid=True,
    )
    return result


def clone_sampler(sampler: Any) -> Any:
    cloned = copy.copy(sampler)
    if hasattr(sampler, "extra_options"):
        cloned.extra_options = dict(getattr(sampler, "extra_options", {}) or {})
    return cloned
