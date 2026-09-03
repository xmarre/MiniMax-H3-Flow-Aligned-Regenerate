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
        binding.metrics.event(
            "guidance",
            coordinate=coordinate,
            mode=binding.guidance.mode,
            schedule=binding.guidance_state.last_schedule,
            correction_rms=binding.guidance_state.last_correction_rms,
            reference_rms=binding.guidance_state.last_reference_rms,
            correction_rms_ratio=binding.guidance_state.last_correction_rms_ratio,
            clamp_scale=binding.guidance_state.last_clamp_scale,
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


def _mask_shapes(latent_shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    if len(latent_shapes) != 2:
        raise ValueError("H3 denoise-mask geometry requires video and audio latent shapes")
    return [(shape[0], 1, *shape[2:]) for shape in latent_shapes]


def _resize_packed_mask(
    mask: torch.Tensor | None,
    source_shapes: list[tuple[int, ...]],
    target_shapes: list[tuple[int, ...]],
) -> torch.Tensor | None:
    if mask is None:
        return None
    source_mask_shapes = _mask_shapes(source_shapes)
    video_mask, audio_mask = unpack_streams(mask, source_mask_shapes)
    target_h, target_w = target_shapes[0][-2:]
    video_mask = resize_spatial_5d(video_mask, target_h, target_w, mode="nearest")
    packed, packed_shapes = pack_streams((video_mask, audio_mask))
    if packed_shapes != _mask_shapes(target_shapes):
        raise RuntimeError("resized H3 denoise mask does not match target AV geometry")
    return packed


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


def _reset_guider_conds(guider: Any, *, target_video_hw: tuple[int, int] | None = None) -> None:
    """Recreate raw conditioning before each independent geometry/sampler lifetime.

    ComfyUI resolves percentage areas, masks, and model conditions in-place on
    guider.conds. Reusing the processed low-resolution structure for the
    probe/high stage can therefore leak low-grid shape metadata across a
    progressive handoff.
    """
    original = getattr(guider, "original_conds", None)
    if not isinstance(original, dict):
        return
    target_h, target_w = target_video_hw if target_video_hw is not None else (None, None)
    rebuilt = {}
    for key, entries in original.items():
        copied_entries = []
        for entry in entries:
            copied = entry.copy() if isinstance(entry, dict) else copy.copy(entry)
            if target_video_hw is not None and isinstance(copied, dict):
                copied = _resize_target_keyframes(copied, int(target_h), int(target_w))
            copied_entries.append(copied)
        rebuilt[key] = copied_entries
    guider.conds = rebuilt


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
    if len(latent_shapes) != 2:
        raise ValueError("progressive handoff supports native packed H3 AV latents only")
    if sigmas.ndim != 1 or sigmas.numel() < 4:
        raise ValueError("progressive handoff requires a full H3 sigma schedule")
    if not math.isclose(float(sigmas[0]), 1.0, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
        float(sigmas[-1]), 0.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("progressive handoff requires a full 1-to-0 H3 sigma schedule")
    input_shapes = list(latent_shapes)
    if input_shapes[0][-2] % 2 or input_shapes[0][-1] % 2:
        raise ValueError("input H3 geometry is not patch-safe")
    model_options = getattr(guider, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("progressive handoff requires mutable model options")
    transformer = model_options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("progressive handoff requires mutable transformer options")
    video_shift = float(transformer.get("minimax_h3_sigma_shift_video", H3_VIDEO_SHIFT))

    target_input = isinstance(config, ProgressiveTargetInputConfig)
    if target_input:
        target_shapes = input_shapes
        target_h, target_w = target_shapes[0][-2:]
        source_h, source_w = config.resolve_source(target_h, target_w)
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
        input_mode="target_grid" if target_input else "source_grid",
        source_shape=source_shapes[0],
        target_hw=(target_h, target_w),
    )

    def low_callback(step, x0, x, _total):
        if callback is not None:
            return callback(step, x0, x, len(sigmas) - 1)
        return None

    _begin_capture(binding, guider, sampler, low_sigmas, source_shapes)
    try:
        _reset_guider_conds(
            guider,
            target_video_hw=(source_h, source_w) if target_input else None,
        )
        low_started = time.perf_counter()
        try:
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
                target_video_hw=(source_h, source_w) if target_input else None,
            )
            # The probe is a one-call sampler lifetime, but model-level patches such
            # as DiffAid must still see the full H3 sigma reference. The explicit
            # refinement contract provides that reference without carrying solver or
            # Spectrum history across the split.
            with _high_stage_contract(guider):
                source_x0 = executor(
                    probe_noise,
                    low_latent_image,
                    _make_probe_sampler(),
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
    if target_input:
        target_latent_image = latent_image
        target_mask = denoise_mask
    else:
        target_latent_image = _resize_packed_latent_image(latent_image, source_shapes, target_shapes)
        target_mask = _resize_packed_mask(denoise_mask, source_shapes, target_shapes)
        latent_shapes[:] = target_shapes
    target_latent_internal = _process_latent_in(base_model, target_latent_image, target_shapes)
    target_noise = _noise_argument(base_model, target_raw, sigma, target_latent_internal)
    binding.metrics.event("handoff_transfer_wall", elapsed_ms=(time.perf_counter() - transfer_started) * 1000.0)

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
    high_event_start = len(binding.metrics.events)
    _reset_guider_conds(
        guider,
        target_video_hw=None if target_input else (target_h, target_w),
    )
    with _high_stage_contract(guider):
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
    high_model_calls = [
        event
        for event in binding.metrics.events[high_event_start:]
        if event.kind == "model_call"
    ]
    first_high_actual = None
    if high_model_calls:
        first_high_actual = bool(high_model_calls[0].fields.get("actual"))
        if not first_high_actual:
            raise RuntimeError(
                "progressive high stage did not begin with the required exact H3 model evaluation"
            )
    binding.metrics.event(
        "handoff_complete",
        sigma=sigma,
        target_shape=target_shapes[0],
        audio_state_copied=True,
        separate_sampler_invocations=True,
        exact_probe_performed=True,
        high_stage_exact_prefix_requested=1,
        high_stage_first_call_actual=first_high_actual,
        high_stage_model_calls=len(high_model_calls),
        conditioning_rebuilt_for_high_grid=True,
        input_mode="target_grid" if target_input else "source_grid",
    )
    return result


def clone_sampler(sampler: Any) -> Any:
    cloned = copy.copy(sampler)
    if hasattr(sampler, "extra_options"):
        cloned.extra_options = dict(getattr(sampler, "extra_options", {}) or {})
    return cloned
