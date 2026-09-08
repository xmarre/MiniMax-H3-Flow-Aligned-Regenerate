from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import torch

from .attention import AttentionConfig, make_attention_override, make_layout_block_wrapper, mark_layout_wrapper
from .contracts import H3FlowTrajectory
from .final_geometry import close_final_mixed_grid_residual
from .geometry import pack_streams, unpack_streams
from .guidance import GuidanceConfig
from .handoff import ProgressiveHandoffConfig, ProgressiveTargetInputConfig
from .metrics import H3FlowMetrics
from .mixed_grid import MIXED_WRAPPER_KEY, mixed_diffusion_wrapper
from .runtime import (
    CLONE_CALLBACK_KEY,
    FLOW_BINDING_KEY,
    FLOW_STAGE_KEY,
    OUTER_WRAPPER_KEY,
    PREDICT_WRAPPER_KEY,
    PROGRESSIVE_KEY,
    FlowBinding,
    flow_model_clone_callback,
    flow_outer_wrapper,
    flow_predict_wrapper,
    sampler_name,
)
from .target_sparse import VDN_EXTERNAL_SEQUENCE_API_VERSION, make_target_sparse_block_wrapper


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


def _canonicalize_exact_masked_output(
    result: torch.Tensor,
    latent_image: torch.Tensor,
    denoise_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Restore exact mask==0 sampler outputs without touching generated values.

    Comfy's inpaint wrapper restores protected values for every model evaluation,
    but a solver may perform one final arithmetic update after the last evaluation.
    Algebraically equivalent endpoint formulas (notably res_multistep's final Euler
    update) can therefore return a few-ULP difference in protected values. The
    progressive exact-prefix contract is stronger: those caller-owned values must
    be exact in the returned packed latent as well.
    """

    stats: dict[str, Any] = {
        "protected_elements": 0,
        "pre_restore_exact": True,
        "canonicalized": False,
        "changed_elements": 0,
        "nonfinite_changed_elements": 0,
        "max_abs_delta": 0.0,
        "rms_delta": 0.0,
        "final_exact": True,
    }
    if denoise_mask is None:
        return result, stats
    if result.shape != latent_image.shape or result.shape != denoise_mask.shape:
        raise ValueError("exact-mask output canonicalization requires matching packed sampler tensors")

    exact_mask = denoise_mask.to(device=result.device) == 0
    protected_elements = int(torch.count_nonzero(exact_mask).item())
    stats["protected_elements"] = protected_elements
    if protected_elements == 0:
        return result, stats

    reference = latent_image.to(device=result.device, dtype=result.dtype)
    mismatch = exact_mask & torch.ne(result, reference)
    changed_elements = int(torch.count_nonzero(mismatch).item())
    stats["pre_restore_exact"] = changed_elements == 0
    stats["changed_elements"] = changed_elements

    if changed_elements:
        delta = result[mismatch].to(torch.float32) - reference[mismatch].to(torch.float32)
        finite = torch.isfinite(delta)
        finite_count = int(torch.count_nonzero(finite).item())
        stats["nonfinite_changed_elements"] = changed_elements - finite_count
        if finite_count:
            finite_delta = delta[finite]
            stats["max_abs_delta"] = float(finite_delta.abs().max().item())
            stats["rms_delta"] = float(finite_delta.square().mean().sqrt().item())
        else:
            stats["max_abs_delta"] = None
            stats["rms_delta"] = None

        result = result.clone()
        result[exact_mask] = reference[exact_mask]
        stats["canonicalized"] = True

    if bool(torch.any(exact_mask & torch.ne(result, reference)).item()):
        raise RuntimeError("exact-mask output canonicalization failed")
    return result, stats


def _record_exact_mask_output(
    binding: FlowBinding,
    sampler: Any,
    *,
    source: str,
    stats: dict[str, Any],
) -> None:
    if int(stats.get("protected_elements", 0)) <= 0:
        return
    if bool(stats.get("canonicalized")):
        binding.metrics.increment("exact_mask_output_canonicalizations")
    binding.metrics.event(
        "exact_mask_output",
        source=str(source),
        sampler=sampler_name(sampler),
        **stats,
    )


def _latest_mixed_grid_geometry(metrics: H3FlowMetrics) -> dict | None:
    for event in reversed(metrics.events):
        if event.kind == "mixed_grid_geometry":
            return event.fields
    return None


def _mixed_grid_prefix_t(video_mask: torch.Tensor) -> int:
    if video_mask.ndim != 5:
        raise ValueError("mixed-grid final geometry requires a BxCxTxHxW video mask")
    frames = video_mask.permute(2, 0, 1, 3, 4).reshape(video_mask.shape[2], -1)
    protected = (frames == 0).all(1)
    generated = (frames == 1).all(1)
    prefix_t = int(protected.sum().item())
    if not 0 < prefix_t < int(video_mask.shape[2]):
        raise ValueError("mixed-grid final geometry requires a nonempty protected prefix and generated suffix")
    if not bool(protected[:prefix_t].all() and generated[prefix_t:].all()):
        raise ValueError("mixed-grid final geometry requires contiguous whole-frame mask protection")
    return prefix_t


def _assert_protected_output_exact(
    result: torch.Tensor,
    latent_image: torch.Tensor,
    denoise_mask: torch.Tensor | None,
) -> None:
    if denoise_mask is None:
        return
    exact_mask = denoise_mask.to(device=result.device) == 0
    reference = latent_image.to(device=result.device, dtype=result.dtype)
    if bool(torch.any(exact_mask & torch.ne(result, reference)).item()):
        raise RuntimeError("final mixed-grid geometry violated the authoritative exact mask")


class _ProgressiveExactMaskExecutor:
    """Adapt Comfy sampler returns before runtime exact-prefix postconditions."""

    def __init__(
        self,
        executor: Any,
        *,
        binding: FlowBinding,
        progressive: ProgressiveTargetInputConfig,
        latent_image: torch.Tensor,
        denoise_mask: torch.Tensor | None,
        sampler: Any,
        latent_shapes: list[tuple[int, ...]],
    ) -> None:
        self._executor = executor
        self.class_obj = executor.class_obj
        self._binding = binding
        self._progressive = progressive
        self._latent_image = latent_image
        self._denoise_mask = denoise_mask
        self._sampler = sampler
        self._latent_shapes = [tuple(int(value) for value in shape) for shape in latent_shapes]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)

    def __call__(self, *args: Any, **kwargs: Any):
        result = self._executor(*args, **kwargs)
        transformer = (getattr(self.class_obj, "model_options", None) or {}).get("transformer_options") or {}
        stage = str(transformer.get(FLOW_STAGE_KEY, ""))
        if stage == "high":
            source = f"{self._progressive.exact_prefix_mode}_high"
        elif stage == "" and self._progressive.exact_prefix_mode == "fallback":
            # Conservative exact-prefix fallback is one ordinary target-grid
            # sampler lifetime and therefore has no Flow stage marker.
            source = "target_input_fallback_return"
        else:
            return result

        result, stats = _canonicalize_exact_masked_output(
            result,
            self._latent_image,
            self._denoise_mask,
        )
        _record_exact_mask_output(
            self._binding,
            self._sampler,
            source=source,
            stats=stats,
        )

        if (
            stage == "high"
            and self._progressive.exact_prefix_mode == "mixed_grid_low_suffix"
            and bool(self._progressive.suffix_geometric_bridge)
        ):
            initial_geometry = _latest_mixed_grid_geometry(self._binding.metrics)
            if initial_geometry is None:
                self._binding.metrics.event(
                    "mixed_grid_final_geometry",
                    requested=True,
                    accepted=False,
                    reason="initial_geometry_metrics_unavailable",
                    policy="final_target_residual_closure_from_initial_cross_grid_authorization",
                )
            elif self._denoise_mask is None:
                raise RuntimeError("mixed-grid final geometry requires the exact target denoise mask")
            else:
                video, audio = unpack_streams(result, self._latent_shapes)
                video_mask, _ = unpack_streams(self._denoise_mask.to(device=result.device), self._latent_shapes)
                prefix_t = _mixed_grid_prefix_t(video_mask)
                corrected_video, closure = close_final_mixed_grid_residual(video, prefix_t, initial_geometry)
                if corrected_video is not video:
                    result, packed_shapes = pack_streams((corrected_video, audio))
                    if packed_shapes != self._latent_shapes:
                        raise RuntimeError("final mixed-grid geometry changed packed H3 stream shapes")
                    _assert_protected_output_exact(result, self._latent_image, self._denoise_mask)
                self._binding.metrics.event("mixed_grid_final_geometry", **closure)
        return result


def flow_outer_wrapper_with_exact_mask(
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
    """Comfy adapter around the sampler-agnostic Flow outer runtime.

    Target-input exact masks are canonicalized at the framework boundary after
    each final target-grid sampler lifetime returns. Progressive high stages are
    fixed before runtime's hard exact-prefix check and seam diagnostics; the
    conservative fallback is fixed inside its single stage-less target-grid call.
    """

    guider = executor.class_obj
    model_options = getattr(guider, "model_options", None) or {}
    binding = model_options.get(FLOW_BINDING_KEY)
    progressive = model_options.get(PROGRESSIVE_KEY)
    if not isinstance(binding, FlowBinding) or not isinstance(progressive, ProgressiveTargetInputConfig):
        return flow_outer_wrapper(
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
    if not isinstance(latent_shapes, list) or len(latent_shapes) != 2:
        raise RuntimeError("progressive exact-mask adapter requires native H3 AV latent shapes")

    adapted = _ProgressiveExactMaskExecutor(
        executor,
        binding=binding,
        progressive=progressive,
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        sampler=sampler,
        latent_shapes=latent_shapes,
    )
    return flow_outer_wrapper(
        adapted,
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

    _put_wrapper_first(
        patched,
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        OUTER_WRAPPER_KEY,
        flow_outer_wrapper_with_exact_mask,
    )
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
    if (
        isinstance(progressive, ProgressiveTargetInputConfig)
        and progressive.exact_prefix_mode == "mixed_grid_low_suffix"
    ):
        _validate_vdn_target_sparse_compat(patched, len(validate_h3_model(patched).blocks), minimum_api=2)
        _put_wrapper_first(
            patched, comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MIXED_WRAPPER_KEY, mixed_diffusion_wrapper
        )
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


def _validate_vdn_target_sparse_compat(
    model: Any, num_layers: int, minimum_api=VDN_EXTERNAL_SEQUENCE_API_VERSION
) -> None:
    object_patches = getattr(model, "object_patches", None)
    if not isinstance(object_patches, dict):
        return
    for layer in range(num_layers):
        key = f"diffusion_model.blocks.{layer}.attn.forward"
        owner = object_patches.get(key)
        if owner is None or not getattr(owner, "_vdn_forward", False):
            continue
        api = int(getattr(owner, "_vdn_external_sequence_api", 0))
        if api < minimum_api:
            raise RuntimeError(
                "target-sparse Continuum detected VDN-H3 attention without the required "
                f"external sequence API v{minimum_api}; "
                "update ComfyUI-VDN-H3 or disable VDN for this experimental path"
            )


def _install_target_sparse(model: Any, metrics: H3FlowMetrics) -> None:
    diffusion = validate_h3_model(model)
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None:
        raise TypeError("native MiniMax H3 diffusion model does not expose transformer blocks")
    num_layers = len(blocks)
    _validate_vdn_target_sparse_compat(model, num_layers)
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