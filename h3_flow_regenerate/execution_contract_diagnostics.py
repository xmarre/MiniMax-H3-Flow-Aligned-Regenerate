"""Native bounded O/C diagnostics for the first-high progressive H3 contract.

The implementation composes ordinary ModelPatcher wrappers around Flow. It does
not replace process-global functions, add an H3 evaluation, invoke the learned
upscaler, or modify sampler state. PR #35 remains the owner of decode checkpoint
outputs and PR #36 remains the owner of learned-anchor guidance validation.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from . import comfy_compat as _comfy_compat
from . import runtime as _runtime
from .execution_contract_provenance import alias_graph, collect_manifest, fingerprint, safe_value
from .geometry import unpack_streams
from .handoff import ProgressiveTargetInputConfig, deterministic_video_noise
from .seam_diagnostics import recover_conditional_clean_for_diagnostics

DIAGNOSTIC_KEY = "h3_flow_execution_contract_diagnostic_v1"
_INVOCATION_KEY = "h3_flow_regenerate.exec_contract.invocation.v1"
_STAGE_KEY = "h3_flow_regenerate.exec_contract.stage.v1"
_PREDICT_POST_KEY = "h3_flow_regenerate.exec_contract.predict_post.v1"
_PREDICT_RAW_KEY = "h3_flow_regenerate.exec_contract.predict_raw.v1"
_SAMPLER_KEY = "h3_flow_regenerate.exec_contract.sampler.v1"
_APPLY_KEY = "h3_flow_regenerate.exec_contract.apply.v1"
_DIFFUSION_KEY = "h3_flow_regenerate.exec_contract.diffusion.v1"
_MAX_REPORTS = 4
_RESEARCH_TOKENS = ("weighted", "mixed_grid", "attention_measure")

_ACTIVE: contextvars.ContextVar[_Record | None] = contextvars.ContextVar(
    "h3_flow_execution_contract_record", default=None
)
_STAGE: contextvars.ContextVar[str | None] = contextvars.ContextVar("h3_flow_execution_contract_stage", default=None)


@dataclass(slots=True)
class _Snapshot:
    tensor: torch.Tensor
    original_device: str
    original_stride: tuple[int, ...]


@dataclass(slots=True)
class _State:
    max_bytes: int
    strict_provenance: bool
    manifest: dict[str, Any]
    complete: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_MAX_REPORTS))


@dataclass(slots=True)
class _Record:
    state: _State
    capture_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_ns: int = field(default_factory=time.time_ns)
    completed_ns: int | None = None
    seed: int = 0
    config: ProgressiveTargetInputConfig | None = None
    active_model_options: dict[str, Any] | None = field(default=None, repr=False)
    original_sigmas: tuple[float, ...] = ()
    original_schedule_digest: str = ""
    pristine_conds: dict[str, list[Any]] | None = None
    pristine_cond_digest: str | None = None
    pristine_cond_summary: Any = None
    pristine_cond_alias_graph: list[dict[str, Any]] = field(default_factory=list)
    high_pre_core_digest: str | None = None
    high_pre_core_summary: Any = None
    high_pre_core_alias_graph: list[dict[str, Any]] = field(default_factory=list)
    condition_compare: dict[str, Any] | None = None
    metric_start: dict[str, int] = field(default_factory=dict)
    event_start: int = 0
    stage_calls: list[dict[str, Any]] = field(default_factory=list)
    first_high_contract: dict[str, Any] | None = None
    first_high_runtime: dict[str, Any] | None = None
    first_high_h3_contract: dict[str, Any] | None = None
    source_shapes: list[tuple[int, ...]] | None = None
    target_shapes: list[tuple[int, ...]] | None = None
    handoff_sigma: float | None = None
    byte_count: int = 0
    incomplete: list[str] = field(default_factory=list)
    snapshots: dict[str, _Snapshot] = field(default_factory=dict)
    high_predict_count: int = 0
    error: str | None = None

    def capture(self, label: str, value: torch.Tensor | None, *, replace: bool = False) -> None:
        if value is None:
            return
        previous = self.snapshots.get(label)
        if previous is not None and not replace:
            return
        size = int(value.numel()) * int(value.element_size())
        previous_size = 0 if previous is None else int(previous.tensor.numel()) * int(previous.tensor.element_size())
        next_total = self.byte_count - previous_size + size
        if next_total > self.state.max_bytes:
            message = f"{label}: capture budget exceeded ({size} bytes)"
            if message not in self.incomplete:
                self.incomplete.append(message)
            return
        self.snapshots[label] = _Snapshot(
            tensor=value.detach().clone(),
            original_device=str(value.device),
            original_stride=tuple(int(item) for item in value.stride()),
        )
        self.byte_count = next_total

    def mark_incomplete(self, message: str) -> None:
        if message not in self.incomplete:
            self.incomplete.append(message)


def _eligible(config: Any) -> bool:
    return (
        isinstance(config, ProgressiveTargetInputConfig)
        and config.transfer_mode == "learned_3d"
        and config.exact_prefix_mode == "fallback"
    )


def _clone_conds(conds: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {
        key: [entry.copy() if isinstance(entry, dict) else entry for entry in entries] for key, entries in conds.items()
    }


def _runtime_snapshot(model_options: dict[str, Any] | None) -> dict[str, Any]:
    transformer = (model_options or {}).get("transformer_options") or {}
    interesting = (
        "h3_",
        "minimax",
        "spectrum",
        "sol",
        "vdn",
        "untwist",
        "diffaid",
        "sample_sigmas",
    )
    out = {}
    for key, value in transformer.items():
        name = str(key).lower()
        if any(token in name for token in interesting):
            out[str(key)] = safe_value(value)
    return out


def _insert_relative(
    model: Any,
    wrapper_type: str,
    anchor: str,
    key: str,
    wrapper: Any,
    *,
    before: bool,
) -> None:
    model.remove_wrappers_with_key(wrapper_type, key)
    existing = model.wrappers.get(wrapper_type, {})
    if anchor not in existing:
        raise RuntimeError(f"execution-contract diagnostic requires wrapper {anchor!r} in {wrapper_type!r}")
    rebuilt = {}
    for existing_key, wrappers in existing.items():
        if before and existing_key == anchor:
            rebuilt[key] = [wrapper]
        rebuilt[existing_key] = wrappers
        if not before and existing_key == anchor:
            rebuilt[key] = [wrapper]
    model.wrappers[wrapper_type] = rebuilt


def _append_wrapper(model: Any, wrapper_type: str, key: str, wrapper: Any) -> None:
    model.remove_wrappers_with_key(wrapper_type, key)
    model.wrappers[wrapper_type] = {**model.wrappers.get(wrapper_type, {}), key: [wrapper]}


def _state_for_guider(guider: Any) -> _State | None:
    value = (getattr(guider, "model_options", None) or {}).get(DIAGNOSTIC_KEY)
    return value if isinstance(value, _State) else None


def _stage_name(model_options: dict[str, Any] | None) -> str:
    transformer = (model_options or {}).get("transformer_options") or {}
    return str(transformer.get(_runtime.FLOW_STAGE_KEY, "single"))


def _invocation_wrapper(
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
    guider = executor.class_obj
    state = _state_for_guider(guider)
    config = (getattr(guider, "model_options", None) or {}).get(_runtime.PROGRESSIVE_KEY)
    if state is None or not _eligible(config):
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
    if _ACTIVE.get() is not None:
        raise RuntimeError("nested execution-contract diagnostic invocation is unsupported")
    state.manifest = collect_manifest(
        guider.model_patcher,
        model_options=getattr(guider, "model_options", None),
        phase="outer_sample_runtime",
    )
    if state.strict_provenance and not state.manifest.get("gate_complete", False):
        raise RuntimeError(
            "execution-contract provenance gate is incomplete: " + ", ".join(state.manifest.get("unresolved", []))
        )
    conds = getattr(guider, "conds", None)
    if not isinstance(conds, dict):
        raise RuntimeError("execution-contract diagnostic requires active CFGGuider conditioning")
    schedule = tuple(float(item) for item in sigmas.detach().to(device="cpu", dtype=torch.float64).tolist())
    pristine = _clone_conds(conds)
    pristine_digest, pristine_summary = fingerprint(pristine)
    record = _Record(
        state=state,
        seed=int(seed or 0),
        config=config,
        active_model_options=getattr(guider, "model_options", None),
        original_sigmas=schedule,
        original_schedule_digest=_runtime._schedule_signature(sigmas),
        pristine_conds=pristine,
        pristine_cond_digest=pristine_digest,
        pristine_cond_summary=pristine_summary,
        pristine_cond_alias_graph=alias_graph(pristine),
    )
    binding = _runtime._resolve_binding(guider)
    if binding is not None:
        record.metric_start = binding.metrics.counters
        record.event_start = len(binding.metrics.events)
    token = _ACTIVE.set(record)
    sampling_error: BaseException | None = None
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
        sampling_error = exc
        record.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _ACTIVE.reset(token)
        _finalize_record_guarded(record, guider, sampling_error=sampling_error)


def _stage_wrapper(
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
    record = _ACTIVE.get()
    if record is None:
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
    guider = executor.class_obj
    stage = _stage_name(getattr(guider, "model_options", None))
    shapes = [tuple(int(dim) for dim in shape) for shape in (latent_shapes or [])]
    record.stage_calls.append(
        {
            "stage": stage,
            "sigmas": [float(item) for item in sigmas.detach().to(device="cpu", dtype=torch.float64).tolist()],
            "latent_shapes": shapes,
            "sampler": _runtime.sampler_name(sampler),
            "runtime": _runtime_snapshot(getattr(guider, "model_options", None)),
        }
    )
    if stage == "low":
        record.source_shapes = shapes
    elif stage == "high":
        record.target_shapes = shapes
        if record.high_pre_core_digest is None:
            current_conds = getattr(guider, "conds", None)
            if isinstance(current_conds, dict):
                record.high_pre_core_digest, record.high_pre_core_summary = fingerprint(current_conds)
                record.high_pre_core_alias_graph = alias_graph(current_conds)
        if sigmas.numel():
            record.handoff_sigma = float(sigmas[0].item())
        record.capture("high_noise_argument", noise)
        record.capture("high_latent_image", latent_image)
        if denoise_mask is not None:
            record.capture("high_denoise_mask", denoise_mask)

    stage_token = _STAGE.set(stage)
    try:
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
    finally:
        _STAGE.reset(stage_token)

    base_model = getattr(guider.model_patcher, "model", None)
    if base_model is not None and len(shapes) == 2 and sigmas.numel():
        if stage == "low":
            raw = _runtime._raw_sampler_state(base_model, result, shapes, float(sigmas[-1].item()))
            video, audio = unpack_streams(raw, shapes)
            record.capture("low_sampler_state_video", video)
            record.capture("low_sampler_state_audio", audio)
        elif stage == "probe":
            internal = _runtime._process_latent_in(base_model, result, shapes)
            video, audio = unpack_streams(internal, shapes)
            record.capture("exact_probe_internal_clean_video", video)
            record.capture("exact_probe_internal_clean_audio", audio)
        elif stage == "high":
            video, audio = unpack_streams(result, shapes)
            record.capture("final_high_video", video)
            record.capture("final_high_audio", audio)
    return result


def _capture_first_high_sampler_state(
    record: _Record,
    model_wrap: Any,
    sigmas: torch.Tensor,
    noise: torch.Tensor,
    latent_image: torch.Tensor | None,
) -> None:
    """Capture raw H3 sampler X before KSamplerX0Inpaint can rewrite model input."""

    inner = getattr(model_wrap, "inner_model", None)
    shapes = getattr(inner, "latent_shapes", None)
    if not isinstance(shapes, list) or len(shapes) != 2:
        record.mark_incomplete("first_high_sampler_input: packed AV shapes unavailable at SAMPLER_SAMPLE")
        return
    if not torch.is_tensor(noise) or not torch.is_tensor(latent_image):
        record.mark_incomplete("first_high_sampler_input: noise/latent_image unavailable at SAMPLER_SAMPLE")
        return
    if tuple(noise.shape) != tuple(latent_image.shape):
        record.mark_incomplete("first_high_sampler_input: noise/latent_image shapes differ")
        return
    if sigmas.numel() == 0:
        record.mark_incomplete("first_high_sampler_input: high sigma suffix is empty")
        return
    model_sampling = getattr(inner, "model_sampling", None)
    if model_sampling is None:
        record.mark_incomplete("first_high_sampler_input: H3 model_sampling unavailable")
        return
    noise_scale = float(getattr(model_sampling, "noise_scale", 1.0))
    if not math.isfinite(noise_scale) or noise_scale <= 0.0:
        record.mark_incomplete("first_high_sampler_input: invalid H3 noise_scale")
        return
    sigma = sigmas[0].detach().to(device=noise.device, dtype=noise.dtype)
    if sigma.numel() != 1 or not bool(torch.isfinite(sigma).all().item()):
        record.mark_incomplete("first_high_sampler_input: invalid first high sigma")
        return
    latent = latent_image.to(device=noise.device, dtype=noise.dtype)
    raw_state = sigma * (noise_scale * noise) + (1.0 - sigma) * latent
    normalized_shapes = [tuple(int(dim) for dim in shape) for shape in shapes]
    video, audio = unpack_streams(raw_state, normalized_shapes)
    record.capture("first_high_sampler_input_video", video)
    record.capture("first_high_sampler_input_audio", audio)


def _sampler_wrapper(
    executor,
    model_wrap,
    sigmas,
    extra_args,
    callback,
    noise,
    latent_image=None,
    denoise_mask=None,
    disable_pbar=False,
):
    record = _ACTIVE.get()
    is_first_high = (
        record is not None
        and _stage_name(extra_args.get("model_options")) == "high"
        and record.condition_compare is None
    )
    if is_first_high:
        _capture_first_high_sampler_state(record, model_wrap, sigmas, noise, latent_image)
        actual_conds = getattr(model_wrap, "conds", {})
        actual_digest, actual_summary = fingerprint(actual_conds)
        record.condition_compare = {
            "scope": (
                "exact pristine-target vs Flow high pre-core comparison plus observation of the already-processed "
                "MiniMaxH3 conditioning; extra_conds is not replayed because it executes H3 text preprocessing"
            ),
            "pristine_target_digest": record.pristine_cond_digest,
            "pristine_target_alias_graph": record.pristine_cond_alias_graph,
            "high_pre_core_digest": record.high_pre_core_digest,
            "high_pre_core_alias_graph": record.high_pre_core_alias_graph,
            "pre_core_equal": record.pristine_cond_digest == record.high_pre_core_digest,
            "actual_processed_digest": actual_digest,
            "actual_processed": actual_summary,
            "actual_processed_alias_graph": alias_graph(actual_conds),
            "no_extra_condition_preprocess": True,
            "no_extra_h3_evaluation": True,
            "no_extra_upscaler_call": True,
        }
    return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)


def _predict_raw_wrapper(executor, x, timestep, model_options=None, seed=None):
    result = executor(x, timestep, model_options, seed)
    record = _ACTIVE.get()
    if record is None:
        return result
    stage = _stage_name(model_options)
    guider = executor.class_obj
    shapes = getattr(getattr(guider, "inner_model", None), "latent_shapes", None)
    if not isinstance(shapes, list) or len(shapes) != 2:
        return result
    shapes = [tuple(int(dim) for dim in shape) for shape in shapes]
    video, audio = unpack_streams(result, shapes)
    if stage == "low":
        record.capture("low_last_model_clean_video", video, replace=True)
        return result
    if stage != "high":
        return result

    transformer = (model_options or {}).get("transformer_options") or {}
    bridge_present = _runtime.EXACT_PREFIX_BRIDGE_KEY in transformer
    record.high_predict_count += 1
    if record.high_predict_count == 1:
        record.target_shapes = shapes
        sigma = float(timestep.detach().reshape(-1)[0].item())
        record.handoff_sigma = sigma
        model_input_video, model_input_audio = unpack_streams(x, shapes)
        record.capture("first_high_model_input_video", model_input_video)
        record.capture("first_high_model_input_audio", model_input_audio)
        record.capture("first_high_model_raw_video", video)
        record.capture("first_high_model_raw_audio", audio)
        if bridge_present:
            record.mark_incomplete("first_high_pre_guidance: exact-prefix bridge present; raw equivalence is invalid")
        else:
            record.capture("first_high_pre_guidance_video", video)
        sampler_state = record.snapshots.get("first_high_sampler_input_video")
        if sampler_state is None:
            record.mark_incomplete("first_high_sampler_input: missing raw SAMPLER_SAMPLE capture")
        elif record.config is not None:
            state_video = sampler_state.tensor
            fresh = deterministic_video_noise(
                tuple(int(dim) for dim in state_video.shape),
                seed=record.seed + int(record.config.seed_offset),
                device=state_video.device,
                dtype=state_video.dtype,
            )
            record.capture("fresh_target_video_noise", fresh)
            clean = recover_conditional_clean_for_diagnostics(state_video, fresh, sigma=sigma)
            record.capture("learned_transfer_clean_video", clean)
        video_shift = float(transformer.get("minimax_h3_sigma_shift_video", _runtime.H3_VIDEO_SHIFT))
        record.first_high_contract = {
            "coordinate": float(_runtime.normalized_coordinate(sigma, video_shift=video_shift)),
            "refinement": safe_value(transformer.get("h3_refinement")),
            "model_options_alias_graph": alias_graph(model_options or {}),
            "exact_prefix_bridge_present": bridge_present,
            "guidance_split_owned_by_pr35_pr36": True,
        }
        record.first_high_runtime = _runtime_snapshot(model_options)
    record.capture("last_high_model_raw_video", video, replace=True)
    if not bridge_present:
        record.capture("last_high_pre_guidance_video", video, replace=True)
    return result


def _predict_post_wrapper(executor, x, timestep, model_options=None, seed=None):
    result = executor(x, timestep, model_options, seed)
    record = _ACTIVE.get()
    if record is None or _stage_name(model_options) != "high":
        return result
    guider = executor.class_obj
    shapes = getattr(getattr(guider, "inner_model", None), "latent_shapes", None)
    if isinstance(shapes, list) and len(shapes) == 2:
        video, _audio = unpack_streams(result, shapes)
        record.capture("first_high_full_guidance_video", video)
        record.capture("last_high_full_guidance_video", video, replace=True)
    return result


def _apply_wrapper(executor, *args, **kwargs):
    record = _ACTIVE.get()
    capture = record is not None and _STAGE.get() == "high" and "first_high_core_denoised" not in record.snapshots
    result = executor(*args, **kwargs)
    if capture and torch.is_tensor(result):
        record.capture("first_high_core_denoised", result)
    return result


def _capture_h3_streams(record: _Record, prefix: str, value: Any) -> None:
    if torch.is_tensor(value):
        record.capture(prefix, value)
        return
    if isinstance(value, (list, tuple)) and len(value) >= 2 and torch.is_tensor(value[0]) and torch.is_tensor(value[1]):
        record.capture(f"{prefix}_video", value[0])
        record.capture(f"{prefix}_audio", value[1])


def _diffusion_wrapper(executor, *args, **kwargs):
    record = _ACTIVE.get()
    capture = record is not None and _STAGE.get() == "high" and record.first_high_h3_contract is None
    if capture:
        if args:
            _capture_h3_streams(record, "first_high_h3_input", args[0])
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        transformer = kwargs.get("transformer_options")
        if transformer is None:
            for candidate in reversed(args):
                if isinstance(candidate, dict) and _runtime.FLOW_STAGE_KEY in candidate:
                    transformer = candidate
                    break
        record.first_high_h3_contract = {
            "timestep": safe_value(timestep),
            "runtime": _runtime_snapshot({"transformer_options": transformer or {}}),
            "transformer_options_alias_graph": alias_graph(transformer or {}),
        }
    result = executor(*args, **kwargs)
    if capture:
        _capture_h3_streams(record, "first_high_h3_velocity", result)
    return result


def _tensor_summary(snapshot: _Snapshot) -> tuple[dict[str, Any], torch.Tensor]:
    value = snapshot.tensor.detach().to(device="cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    finite = torch.isfinite(value) if value.is_floating_point() else torch.ones_like(value, dtype=torch.bool)
    report: dict[str, Any] = {
        "shape": [int(dim) for dim in value.shape],
        "stride": list(snapshot.original_stride),
        "dtype": str(value.dtype),
        "device": snapshot.original_device,
        "numel": int(value.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite_count": int(finite.sum().item()),
    }
    if value.numel() and value.is_floating_point():
        floating = value.float()
        report["mean"] = float(floating.mean().item())
        report["rms"] = float(floating.square().mean().sqrt().item())
        if floating.ndim >= 2 and int(floating.shape[1]) <= 64:
            axes = tuple(index for index in range(floating.ndim) if index != 1)
            report["channel_mean"] = [float(item) for item in floating.mean(dim=axes).tolist()]
            report["channel_rms"] = [float(item) for item in floating.square().mean(dim=axes).sqrt().tolist()]
        if floating.ndim == 5:
            correlations = {}
            for name, dimension in (("temporal", 2), ("height", 3), ("width", 4)):
                if int(floating.shape[dimension]) <= 1:
                    continue
                left = floating.narrow(dimension, 0, int(floating.shape[dimension]) - 1)
                right = floating.narrow(dimension, 1, int(floating.shape[dimension]) - 1)
                denom = float(left.square().mean().sqrt().item() * right.square().mean().sqrt().item())
                correlations[name] = float((left * right).mean().item() / max(denom, 1e-20))
            report["adjacent_correlation"] = correlations
    return report, value


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in sorted(set(before) | set(after))}


def _matches_controlled_o(topology: dict[str, Any]) -> bool:
    expected = {
        "low": {"logical": 5, "actual": 4, "forecast": 1},
        "probe": {"logical": 1, "actual": 1, "forecast": 0},
        "high": {"logical": 3, "actual": 2, "forecast": 1},
    }
    return (
        topology.get("logical") == 9
        and topology.get("actual") == 7
        and topology.get("forecast") == 2
        and topology.get("learned_upscale_events") == 1
        and topology.get("per_stage") == expected
    )


def _research_receipts(value: Any, *, path: str = "$") -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            child = f"{path}.{name}"
            if any(token in name.lower() for token in _RESEARCH_TOKENS):
                receipts.append({"path": child, "value": safe_value(item)})
            receipts.extend(_research_receipts(item, path=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            receipts.extend(_research_receipts(item, path=f"{path}[{index}]"))
    return receipts


def _receipt_zero(value: Any) -> bool | None:
    primitives: list[bool] = []

    def visit(item: Any) -> None:
        if isinstance(item, bool):
            primitives.append(not item)
        elif isinstance(item, (int, float)):
            primitives.append(float(item) == 0.0)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return all(primitives) if primitives else None


def _all_research_receipts_zero(receipts: list[dict[str, Any]]) -> bool | None:
    states = [_receipt_zero(receipt["value"]) for receipt in receipts]
    observed = [state for state in states if state is not None]
    return all(observed) if observed else None


def _append_finalize_failure(record: _Record, exc: BaseException) -> None:
    record.state.complete.append(
        {
            "schema_version": 2,
            "capture_id": record.capture_id,
            "started_ns": record.started_ns,
            "completed_ns": time.time_ns(),
            "error": record.error,
            "diagnostic_finalize_error": f"{type(exc).__name__}: {exc}",
            "incomplete": [*record.incomplete, "diagnostic finalization failed"],
            "provenance": record.state.manifest,
            "promotion": {"production_fix_authorized": False},
        }
    )


def _finalize_record_guarded(
    record: _Record,
    guider: Any,
    *,
    sampling_error: BaseException | None,
) -> None:
    try:
        _finalize_record(record, guider)
    except BaseException as exc:
        _append_finalize_failure(record, exc)
        if sampling_error is None:
            raise


def _finalize_record(record: _Record, guider: Any) -> None:
    record.completed_ns = time.time_ns()
    summaries: dict[str, Any] = {}
    values: dict[str, torch.Tensor] = {}
    for label, snapshot in record.snapshots.items():
        summary, value = _tensor_summary(snapshot)
        summaries[label] = summary
        values[label] = value

    exact_checks: dict[str, Any] = {}
    low_audio = values.get("low_sampler_state_audio")
    high_audio = values.get("first_high_sampler_input_audio")
    if low_audio is not None and high_audio is not None:
        exact_checks["carried_audio_exact"] = bool(torch.equal(low_audio, high_audio))
    raw = values.get("first_high_model_raw_video")
    pre = values.get("first_high_pre_guidance_video")
    if raw is not None and pre is not None:
        exact_checks["raw_equals_pre_guidance"] = bool(torch.equal(raw, pre))

    binding = _runtime._resolve_binding(guider)
    metric_end = binding.metrics.counters if binding is not None else {}
    metric_delta = _counter_delta(record.metric_start, metric_end)
    logical = metric_delta.get("sampler_logical_calls", 0)
    actual = metric_delta.get("transformer_actual_nfe", 0)
    forecast = metric_delta.get("spectrum_forecast_calls", 0)
    events = () if binding is None else binding.metrics.events[record.event_start :]
    upscale_events = sum(1 for event in events if event.kind == "handoff_learned_upscale_wall")
    model_calls = [event.fields for event in events if event.kind == "model_call"]
    topology = {
        "logical": logical,
        "actual": actual,
        "forecast": forecast,
        "learned_upscale_events": upscale_events,
        "per_stage": {
            stage: {
                "logical": metric_delta.get(f"sampler_logical_calls_{stage}", 0),
                "actual": metric_delta.get(f"transformer_actual_nfe_{stage}", 0),
                "forecast": metric_delta.get(f"spectrum_forecast_calls_{stage}", 0),
            }
            for stage in ("low", "probe", "high")
        },
    }
    topology["matches_controlled_O"] = _matches_controlled_o(topology)

    flow_research_activity = {
        key: value
        for key, value in metric_delta.items()
        if value and any(token in key.lower() for token in _RESEARCH_TOKENS)
    }
    companion_receipts = _research_receipts(record.first_high_runtime or {})
    companion_receipts_zero = _all_research_receipts_zero(companion_receipts)
    refinement = (record.first_high_contract or {}).get("refinement")
    refinement_prefix_one = isinstance(refinement, dict) and refinement.get("min_actual_prefix_steps") == 1
    condition_equal = bool((record.condition_compare or {}).get("pre_core_equal"))
    structural_o_candidate = (
        record.error is None
        and bool(record.state.manifest.get("gate_complete"))
        and not record.incomplete
        and bool(topology["matches_controlled_O"])
        and exact_checks.get("carried_audio_exact") is True
        and condition_equal
        and refinement_prefix_one
        and not flow_research_activity
        and companion_receipts_zero is True
    )

    report = {
        "schema_version": 2,
        "capture_id": record.capture_id,
        "started_ns": record.started_ns,
        "completed_ns": record.completed_ns,
        "error": record.error,
        "path": "target_input_unprotected_progressive_learned_3d",
        "original_sigmas": list(record.original_sigmas),
        "original_schedule_digest": record.original_schedule_digest,
        "source_shapes": record.source_shapes,
        "target_shapes": record.target_shapes,
        "handoff_sigma": record.handoff_sigma,
        "metric_delta": metric_delta,
        "model_calls": model_calls,
        "topology": topology,
        "research_activity": {
            "flow_metric_nonzero": flow_research_activity,
            "companion_receipts": companion_receipts,
            "companion_receipts_zero": companion_receipts_zero,
            "receipt_gate_note": (
                "None means the active runtime exposed no machine-readable weighted/Mixed-Grid receipt at this boundary; "
                "absence is not treated as zero activity."
            ),
        },
        "condition_rebuild_compare": record.condition_compare,
        "first_high_contract": record.first_high_contract,
        "first_high_runtime": record.first_high_runtime,
        "first_high_h3_contract": record.first_high_h3_contract,
        "stage_calls": record.stage_calls,
        "snapshots": summaries,
        "exact_checks": exact_checks,
        "capture_bytes": record.byte_count,
        "capture_budget_bytes": record.state.max_bytes,
        "incomplete": record.incomplete,
        "provenance": record.state.manifest,
        "observation_gate": {
            "structural_candidate": structural_o_candidate,
            "paired_output_transparency_still_required": True,
            "cuda_execution_still_required": True,
            "media_comparison_still_required": True,
        },
        "promotion": {
            "cpu_or_algebra_sufficient": False,
            "cuda_media_required": True,
            "original_baseline_media_required": True,
            "00442_backend_receipt_must_be_verified_separately": True,
            "replay_checkpoint_identity_exact_required": True,
            "production_fix_authorized": False,
        },
    }
    record.state.complete.append(report)
    record.snapshots.clear()


def patch_execution_contract_diagnostics(
    model: Any,
    *,
    capture_mib: int = 256,
    strict_provenance: bool = True,
) -> tuple[Any, _State]:
    if not 16 <= int(capture_mib) <= 1024:
        raise ValueError("capture_mib must be between 16 and 1024")
    patched = model.clone()
    _comfy_compat._copy_model_options(patched)
    progressive = patched.model_options.get(_runtime.PROGRESSIVE_KEY)
    if not _eligible(progressive):
        raise ValueError(
            "execution-contract diagnostic requires learned_3d Target Input with exact_prefix_mode=fallback"
        )
    manifest = collect_manifest(patched, phase="node_apply_preflight")
    state = _State(
        max_bytes=int(capture_mib) * 1024 * 1024,
        strict_provenance=bool(strict_provenance),
        manifest=manifest,
    )
    patched.model_options[DIAGNOSTIC_KEY] = state

    import comfy.patcher_extension

    outer = comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
    predict = comfy.patcher_extension.WrappersMP.PREDICT_NOISE
    _insert_relative(
        patched,
        outer,
        _runtime.OUTER_WRAPPER_KEY,
        _INVOCATION_KEY,
        _invocation_wrapper,
        before=True,
    )
    _insert_relative(
        patched,
        outer,
        _runtime.OUTER_WRAPPER_KEY,
        _STAGE_KEY,
        _stage_wrapper,
        before=False,
    )
    _insert_relative(
        patched,
        predict,
        _runtime.PREDICT_WRAPPER_KEY,
        _PREDICT_POST_KEY,
        _predict_post_wrapper,
        before=True,
    )
    _insert_relative(
        patched,
        predict,
        _runtime.PREDICT_WRAPPER_KEY,
        _PREDICT_RAW_KEY,
        _predict_raw_wrapper,
        before=False,
    )
    _append_wrapper(
        patched,
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        _SAMPLER_KEY,
        _sampler_wrapper,
    )
    _append_wrapper(
        patched,
        comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        _APPLY_KEY,
        _apply_wrapper,
    )
    _append_wrapper(
        patched,
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        _DIFFUSION_KEY,
        _diffusion_wrapper,
    )
    state.manifest["observer_wrapper_keys"] = {
        "outer": [_INVOCATION_KEY, _STAGE_KEY],
        "predict": [_PREDICT_POST_KEY, _PREDICT_RAW_KEY],
        "sampler": [_SAMPLER_KEY],
        "apply": [_APPLY_KEY],
        "diffusion": [_DIFFUSION_KEY],
    }
    return patched, state


class H3ExecutionContractDiagnostics:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Native bounded O/C recorder for the first-high learned_3d contract. It adds no H3 evaluation or upscaler "
        "call and can fail before sampling when installed-source provenance is incomplete."
    )
    RETURN_TYPES = ("MODEL", "H3_FLOW_EXECUTION_CONTRACT", "STRING")
    RETURN_NAMES = ("model", "diagnostic", "preflight_provenance_manifest")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "capture_mib": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 16}),
                "strict_provenance": ("BOOLEAN", {"default": True}),
            }
        }

    def apply(self, model, capture_mib, strict_provenance):
        patched, state = patch_execution_contract_diagnostics(
            model,
            capture_mib=int(capture_mib),
            strict_provenance=bool(strict_provenance),
        )
        return patched, state, json.dumps(state.manifest, indent=2, sort_keys=True, default=str)


class H3ExecutionContractReport:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Emit the latest completed execution-contract report. Connect the sampled LATENT as trigger so extraction "
        "runs after sampling."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"diagnostic": ("H3_FLOW_EXECUTION_CONTRACT",), "trigger": ("LATENT",)}}

    def extract(self, diagnostic, trigger):
        _ = trigger
        if not isinstance(diagnostic, _State):
            raise TypeError("invalid H3 execution-contract diagnostic handle")
        if not diagnostic.complete:
            raise RuntimeError("no completed execution-contract capture is available")
        return (json.dumps(diagnostic.complete.pop(), indent=2, sort_keys=True, default=str),)


NODE_CLASS_MAPPINGS = {
    "H3ExecutionContractDiagnostics": H3ExecutionContractDiagnostics,
    "H3ExecutionContractReport": H3ExecutionContractReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ExecutionContractDiagnostics": "MiniMax H3 Execution Contract Diagnostics",
    "H3ExecutionContractReport": "MiniMax H3 Execution Contract Report",
}
