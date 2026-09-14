"""Bounded U experiment for the progressive first-high artifact.

This diagnostic publishes the original Flow sampling trajectory to Untwist RoPE
without changing the sampler-owned ``sample_sigmas`` option, adding an H3 call,
or invoking the learned upscaler.  It requires the execution-contract diagnostic
to be applied upstream so the original schedule and invocation identity come
from the already-audited O/C recorder.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import torch

from . import comfy_compat as _comfy_compat
from . import execution_contract_diagnostics as _diag
from . import runtime as _runtime

FLOW_SAMPLING_CONTEXT_KEY = "h3_flow_sampling_context"
TRIAL_REQUEST_KEY = "h3_flow_untwist_clock_trial_v1"
_TRIAL_WRAPPER_KEY = "h3_flow_regenerate.untwist_clock_trial.v1"
_CONTEXT_API = 1
_MISSING = object()


def _validated_nonzero_schedule(record: _diag._Record) -> tuple[float, ...]:
    values = tuple(float(value) for value in record.original_sigmas)
    if len(values) < 4:
        raise RuntimeError("Untwist clock trial requires the complete progressive sigma schedule")
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Untwist clock trial requires a finite sigma schedule")
    if not math.isclose(values[0], 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("Untwist clock trial requires the controlled full schedule to start at sigma=1")
    if not math.isclose(values[-1], 0.0, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("Untwist clock trial requires the controlled full schedule to end at sigma=0")
    nonzero = values[:-1]
    if any(value <= 0.0 for value in nonzero):
        raise RuntimeError("Untwist clock trial requires positive denoiser coordinates before the terminal zero")
    if any(left <= right for left, right in pairwise(nonzero)):
        raise RuntimeError("Untwist clock trial requires a strictly descending controlled Euler schedule")
    return nonzero


def _nearest_schedule_index(schedule: tuple[float, ...], value: float) -> int:
    if not math.isfinite(float(value)):
        raise RuntimeError("Untwist clock trial received a non-finite child sampling coordinate")
    index = min(range(len(schedule)), key=lambda candidate: abs(schedule[candidate] - float(value)))
    delta = abs(schedule[index] - float(value))
    tolerance = max(1e-7, abs(float(value)) * 1e-6)
    if delta > tolerance:
        raise RuntimeError(
            "Untwist clock trial child schedule is not a subset of the original Flow trajectory: "
            f"coordinate={float(value):.9g} nearest={schedule[index]:.9g} delta={delta:.3g}"
        )
    return int(index)


def _build_sampling_context(
    record: _diag._Record,
    *,
    stage: str,
    child_sigmas: torch.Tensor,
) -> dict[str, Any]:
    if stage not in {"low", "probe", "high"}:
        raise RuntimeError(f"Untwist clock trial received unsupported Flow stage {stage!r}")
    if not torch.is_tensor(child_sigmas) or child_sigmas.ndim != 1 or child_sigmas.numel() == 0:
        raise RuntimeError("Untwist clock trial requires a non-empty one-dimensional child sigma schedule")
    schedule = _validated_nonzero_schedule(record)
    child = tuple(float(value) for value in child_sigmas.detach().to(device="cpu", dtype=torch.float64).tolist())
    start_index = _nearest_schedule_index(schedule, child[0])
    if stage == "low" and start_index != 0:
        raise RuntimeError("Untwist clock trial low stage does not begin at the original trajectory start")
    if stage == "probe" and len(child) != 1:
        raise RuntimeError("Untwist clock trial probe must remain the existing one-coordinate exact probe")
    expected_progress = start_index / float(len(schedule) - 1)
    return {
        "api": _CONTEXT_API,
        "source": "h3_flow_execution_contract_untwist_clock_trial",
        "units": "comfy_sigma",
        "stage": stage,
        "original_nonzero_sigmas": list(schedule),
        "original_schedule_digest": str(record.original_schedule_digest),
        "original_stage_start_index": int(start_index),
        "original_stage_start_progress": float(expected_progress),
        "invocation_generation": int(record.started_ns),
        "capture_id": str(record.capture_id),
        "child_sigmas": list(child),
        "sample_sigmas_ownership": "sampler_unchanged",
    }


def _trial_stage_wrapper(
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
    record = _diag._ACTIVE.get()
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
    stage = _diag._stage_name(getattr(guider, "model_options", None))
    if stage not in {"low", "probe", "high"}:
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
    if _runtime.sampler_name(sampler) != "sample_euler" and stage != "probe":
        raise RuntimeError("Untwist clock trial is intentionally bounded to the controlled Euler reproduction")
    if denoise_mask is not None:
        raise RuntimeError("Untwist clock trial is intentionally bounded to the unprotected controlled reproduction")

    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("Untwist clock trial requires mutable guider model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("Untwist clock trial requires mutable transformer options")
    existing = transformer.get(FLOW_SAMPLING_CONTEXT_KEY, _MISSING)
    if existing is not _MISSING:
        raise RuntimeError("Untwist clock trial refuses to replace an existing Flow sampling context")

    context = _build_sampling_context(record, stage=stage, child_sigmas=sigmas)
    transformer[FLOW_SAMPLING_CONTEXT_KEY] = context
    binding = _runtime._resolve_binding(guider)
    if binding is not None:
        binding.metrics.event(
            "untwist_full_trajectory_clock_trial",
            stage=stage,
            schedule_digest=context["original_schedule_digest"],
            original_stage_start_index=context["original_stage_start_index"],
            original_stage_start_progress=context["original_stage_start_progress"],
            sample_sigmas_unchanged=True,
            no_extra_h3_evaluation=True,
            no_extra_upscaler_call=True,
        )
    child_error: BaseException | None = None
    ownership_changed = False
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
        child_error = exc
        raise
    finally:
        ownership_changed = transformer.get(FLOW_SAMPLING_CONTEXT_KEY) is not context
        transformer.pop(FLOW_SAMPLING_CONTEXT_KEY, None)
        if ownership_changed and binding is not None:
            binding.metrics.event(
                "untwist_full_trajectory_clock_trial_cleanup_error",
                stage=stage,
                child_error=type(child_error).__name__ if child_error is not None else None,
            )
        if ownership_changed and child_error is None:
            raise RuntimeError("Untwist clock trial Flow sampling context ownership changed during the child sampler")


class H3UntwistFullTrajectoryClockTrial:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Diagnostic U experiment selected by the O/C evidence. Apply after MiniMax H3 Execution Contract Diagnostics. "
        "It publishes the original progressive schedule to compatible Untwist RoPE builds only; sampler sample_sigmas, "
        "H3 NFE, noise, Flow state, guidance, Spectrum, Sol and VDN ownership are otherwise unchanged."
    )
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def apply(self, model):
        state = (getattr(model, "model_options", None) or {}).get(_diag.DIAGNOSTIC_KEY)
        if not isinstance(state, _diag._State):
            raise RuntimeError(
                "Untwist clock trial must be applied after MiniMax H3 Execution Contract Diagnostics so the original "
                "schedule and installed-runtime provenance remain owned by the O/C recorder"
            )
        patched = model.clone()
        _comfy_compat._copy_model_options(patched)
        transformer = patched.model_options.setdefault("transformer_options", {})
        if FLOW_SAMPLING_CONTEXT_KEY in transformer:
            raise RuntimeError("Untwist clock trial found a pre-existing Flow sampling context")
        transformer[TRIAL_REQUEST_KEY] = {
            "api": _CONTEXT_API,
            "mode": "untwist_full_trajectory_clock",
            "diagnostic_only": True,
        }

        import comfy.patcher_extension

        _diag._append_wrapper(
            patched,
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            _TRIAL_WRAPPER_KEY,
            _trial_stage_wrapper,
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {"H3UntwistFullTrajectoryClockTrial": H3UntwistFullTrajectoryClockTrial}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3UntwistFullTrajectoryClockTrial": "MiniMax H3 Untwist Full-Trajectory Clock Trial"
}
