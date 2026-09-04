from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from .attention import AttentionConfig, make_attention_override, make_layout_block_wrapper, mark_layout_wrapper
from .contracts import H3FlowTrajectory
from .guidance import GuidanceConfig
from .handoff import ProgressiveHandoffConfig, ProgressiveTargetInputConfig
from .metrics import H3FlowMetrics
from .runtime import (
    CLONE_CALLBACK_KEY,
    FLOW_BINDING_KEY,
    OUTER_WRAPPER_KEY,
    PREDICT_WRAPPER_KEY,
    PROGRESSIVE_KEY,
    FlowBinding,
    flow_model_clone_callback,
    flow_outer_wrapper,
    flow_predict_wrapper,
)
from .target_sparse import make_target_sparse_block_wrapper


def validate_h3_model(model: Any) -> Any:
    try:
        diffusion = model.model.diffusion_model
        base = model.model
    except AttributeError as exc:
        raise TypeError("expected a ComfyUI MODEL containing native MiniMax H3") from exc
    facts = {
        "patch_size": tuple(getattr(diffusion, "patch_size", ())),
        "latents_dim": int(getattr(diffusion, "latents_dim", -1)),
        "audio_latents_dim": int(getattr(diffusion, "audio_latents_dim", -1)),
        "sigma_shift_video": float(getattr(diffusion, "sigma_shift_video", -1.0)),
        "sigma_shift_audio": float(getattr(diffusion, "sigma_shift_audio", -1.0)),
    }
    expected = {
        "patch_size": (1, 2, 2),
        "latents_dim": 24,
        "audio_latents_dim": 32,
        "sigma_shift_video": 12.0,
        "sigma_shift_audio": 3.0,
    }
    if facts != expected or base.__class__.__name__ != "MiniMaxH3":
        raise TypeError(f"model does not match the supported native MiniMax H3 contract: {facts}")
    return diffusion


def _copy_model_options(model: Any) -> None:
    model.model_options = dict(model.model_options)
    transformer = dict(model.model_options.get("transformer_options") or {})
    model.model_options["transformer_options"] = transformer


def _put_wrapper_first(model: Any, wrapper_type: str, key: str, wrapper) -> None:
    model.remove_wrappers_with_key(wrapper_type, key)
    existing = model.wrappers.get(wrapper_type, {})
    model.wrappers[wrapper_type] = {key: [wrapper], **existing}


def _put_wrapper_last(model: Any, wrapper_type: str, key: str, wrapper) -> None:
    model.remove_wrappers_with_key(wrapper_type, key)
    existing = model.wrappers.get(wrapper_type, {})
    model.wrappers[wrapper_type] = {**existing, key: [wrapper]}


def patch_flow_model(
    model: Any,
    *,
    trajectory: H3FlowTrajectory | None = None,
    clear_trajectory: bool = False,
    guidance: GuidanceConfig | None = None,
    progressive: ProgressiveHandoffConfig | ProgressiveTargetInputConfig | None = None,
    clear_progressive: bool = False,
    attention: AttentionConfig | None = None,
    capture_enabled: bool | None = None,
    capture_forecasts: bool | None = None,
    guidance_conditioning_signature: str | None = None,
    clear_guidance_conditioning_signature: bool = False,
    guidance_run_id: str | None = None,
    clear_guidance_run_id: bool = False,
    metrics: H3FlowMetrics | None = None,
) -> tuple[Any, FlowBinding]:
    validate_h3_model(model)
    prior = model.model_options.get(FLOW_BINDING_KEY)
    if not isinstance(prior, FlowBinding):
        prior = None
    if clear_trajectory and trajectory is not None:
        raise ValueError("cannot set and clear trajectory in the same model patch")
    patched = model.clone()
    _copy_model_options(patched)
    binding = FlowBinding(
        trajectory=(
            None
            if clear_trajectory
            else (trajectory if trajectory is not None else (prior.trajectory if prior else None))
        ),
        guidance=guidance if guidance is not None else (prior.guidance if prior else None),
        metrics=metrics or (prior.metrics if prior else H3FlowMetrics()),
        capture_enabled=(
            prior.capture_enabled if capture_enabled is None and prior is not None else bool(capture_enabled)
        ),
        capture_forecasts=(
            prior.capture_forecasts if capture_forecasts is None and prior is not None else bool(capture_forecasts)
        ),
        guidance_conditioning_signature=(
            None
            if clear_guidance_conditioning_signature
            else (
                guidance_conditioning_signature
                if guidance_conditioning_signature is not None
                else (prior.guidance_conditioning_signature if prior else None)
            )
        ),
        captured_run_id=prior.captured_run_id if prior else None,
        guidance_run_id=(
            None
            if clear_guidance_run_id
            else (guidance_run_id if guidance_run_id is not None else (prior.guidance_run_id if prior else None))
        ),
    )
    if clear_guidance_conditioning_signature and guidance_conditioning_signature is not None:
        raise ValueError("cannot set and clear guidance conditioning signature in the same model patch")
    if clear_guidance_run_id and guidance_run_id is not None:
        raise ValueError("cannot set and clear guidance run id in the same model patch")
    patched.model_options[FLOW_BINDING_KEY] = binding
    if clear_progressive and progressive is not None:
        raise ValueError("cannot set and clear progressive handoff in the same model patch")
    if clear_progressive:
        patched.model_options.pop(PROGRESSIVE_KEY, None)
    elif progressive is not None:
        patched.model_options[PROGRESSIVE_KEY] = progressive

    import comfy.patcher_extension

    _put_wrapper_first(patched, comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, OUTER_WRAPPER_KEY, flow_outer_wrapper)
    # OUTER_SAMPLE must remain outside Spectrum so progressive low/probe/high
    # invocations each traverse Spectrum independently. PREDICT_NOISE must be
    # inside Spectrum so it receives Spectrum's per-call copied model_options
    # containing exact/forecast provenance.
    _put_wrapper_last(
        patched,
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        PREDICT_WRAPPER_KEY,
        flow_predict_wrapper,
    )
    callback_type = comfy.patcher_extension.CallbacksMP.ON_CLONE
    patched.remove_callbacks_with_key(callback_type, CLONE_CALLBACK_KEY)
    patched.add_callback_with_key(callback_type, CLONE_CALLBACK_KEY, flow_model_clone_callback)
    _install_layout_metrics(patched, binding.metrics)
    if attention is not None and attention.mode != "native":
        _install_attention(patched, attention, binding.metrics)
    if (
        isinstance(progressive, ProgressiveTargetInputConfig)
        and progressive.exact_prefix_mode == "target_sparse_lifter"
    ):
        _install_target_sparse(patched, binding.metrics)
    return patched, binding


def _install_attention(model: Any, config: AttentionConfig, metrics: H3FlowMetrics) -> None:
    diffusion = validate_h3_model(model)
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None:
        raise TypeError("native MiniMax H3 diffusion model does not expose transformer blocks")
    num_layers = len(blocks)
    invalid = tuple(layer for layer in config.layers if layer >= num_layers)
    if invalid:
        raise ValueError(f"H3 attention layers are outside the loaded model's {num_layers} blocks: {invalid}")
    transformer = model.model_options["transformer_options"]
    previous_override = transformer.get("optimized_attention_override")
    while getattr(previous_override, "_h3_flow_attention_override", False):
        previous_override = getattr(previous_override, "_h3_flow_previous_override", None)
    transformer["optimized_attention_override"] = make_attention_override(
        config,
        metrics,
        previous_override=previous_override,
    )
    existing = ((transformer.get("patches_replace") or {}).get("dit") or {}).copy()
    for layer in range(num_layers):
        previous = existing.get(("double_block", layer))
        while getattr(previous, "_h3_flow_layout_wrapper", False):
            previous = getattr(previous, "_h3_flow_previous", None)
        wrapper = make_layout_block_wrapper(
            layer,
            metrics,
            previous,
            record_layout=layer == 0,
        )
        wrapper = mark_layout_wrapper(
            wrapper,
            metrics=metrics,
            previous=previous,
            scope="attention",
        )
        model.set_model_patch_replace(
            wrapper,
            "dit",
            "double_block",
            layer,
        )


def _contains_target_sparse_wrapper(wrapper: Any) -> bool:
    """Walk Flow-owned wrapper links without depending on third-party internals."""
    seen: set[int] = set()
    current = wrapper
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "_h3_flow_target_sparse_wrapper", False):
            return True
        if getattr(current, "_h3_flow_layout_wrapper", False):
            current = getattr(current, "_h3_flow_previous", None)
            continue
        current = None
    return False


def _install_target_sparse(model: Any, metrics: H3FlowMetrics) -> None:
    diffusion = validate_h3_model(model)
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None:
        raise TypeError("native MiniMax H3 diffusion model does not expose transformer blocks")
    num_layers = len(blocks)
    transformer = model.model_options["transformer_options"]
    existing = ((transformer.get("patches_replace") or {}).get("dit") or {}).copy()
    for layer in range(num_layers):
        previous = existing.get(("double_block", layer))
        if _contains_target_sparse_wrapper(previous):
            continue
        model.set_model_patch_replace(
            make_target_sparse_block_wrapper(layer, num_layers, metrics, previous=previous),
            "dit",
            "double_block",
            layer,
        )


def _install_layout_metrics(model: Any, metrics: H3FlowMetrics) -> None:
    transformer = model.model_options["transformer_options"]
    existing = ((transformer.get("patches_replace") or {}).get("dit") or {}).get(("double_block", 0))
    if (
        getattr(existing, "_h3_flow_layout_wrapper", False)
        and getattr(existing, "_h3_flow_layout_scope", "layout") == "layout"
        and getattr(existing, "_h3_flow_metrics", None) is metrics
    ):
        return
    while (
        getattr(existing, "_h3_flow_layout_wrapper", False)
        and getattr(existing, "_h3_flow_layout_scope", "layout") == "layout"
    ):
        existing = getattr(existing, "_h3_flow_previous", None)
    wrapper = make_layout_block_wrapper(0, metrics, existing)
    marked = mark_layout_wrapper(
        wrapper,
        metrics=metrics,
        previous=existing,
        scope="layout",
    )
    model.set_model_patch_replace(marked, "dit", "double_block", 0)


def reconfigure_binding(binding: FlowBinding, **changes: Any) -> FlowBinding:
    allowed = {
        "trajectory",
        "guidance",
        "capture_enabled",
        "capture_forecasts",
        "guidance_conditioning_signature",
        "captured_run_id",
        "guidance_run_id",
    }
    unknown = set(changes) - allowed
    if unknown:
        raise TypeError(f"unknown binding fields: {sorted(unknown)}")
    values = {
        "trajectory": binding.trajectory,
        "guidance": binding.guidance,
        "metrics": binding.metrics,
        "capture_enabled": binding.capture_enabled,
        "capture_forecasts": binding.capture_forecasts,
        "guidance_conditioning_signature": binding.guidance_conditioning_signature,
        "captured_run_id": binding.captured_run_id,
        "guidance_run_id": binding.guidance_run_id,
    }
    values.update(changes)
    return FlowBinding(**values)


def clone_config(config: Any, **changes: Any) -> Any:
    if hasattr(config, "__dataclass_fields__"):
        return replace(config, **changes)
    return copy.copy(config)
