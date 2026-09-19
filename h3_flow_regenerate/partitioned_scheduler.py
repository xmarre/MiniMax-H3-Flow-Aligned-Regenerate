"""Low/probe/high scheduler for partitioned exact-prefix continuation.

The released Flow runtime remains untouched.  This module is selected only by the
experimental partitioned node and publishes a new stage contract rather than
reinterpreting the retired Mixed-Grid mode.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from typing import Any

import torch

from .geometry import pack_streams, resize_spatial_5d, unpack_streams
from .handoff import ProgressiveTargetInputConfig, build_handoff_state, deterministic_video_noise
from .partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    PartitionedStageRuntime,
    build_partitioned_stage_plan,
)
from .partitioned_transformer import VDN_PARTITIONED_SEQUENCE_API
from .runtime import (
    PROBE_CONTEXT_KEY,
    _begin_capture,
    _conditioning_signature,
    _finish_capture,
    _flow_stage_contract,
    _has_exact_video_protection,
    _high_stage_contract,
    _interop_identity,
    _make_probe_sampler,
    _merge_preserved_noise,
    _noise_argument,
    _process_latent_in,
    _raw_sampler_state,
    _reset_guider_conds,
    _resize_packed_latent_image,
    _resize_packed_mask,
    _validate_progressive_sampler_state,
)
from .seam_diagnostics import (
    measure_exact_prefix_splice,
    measure_translation_trajectory,
    measure_video_boundary,
    project_translation_trajectory_to_grid,
    recover_conditional_clean_for_diagnostics,
)
from .sigma import H3_VIDEO_SHIFT, normalized_coordinate

PARTITIONED_PROGRESSIVE_KEY = "h3_flow_partitioned_progressive_v1"
SOL_RUNTIME_KEY = "sol_h3_runtime_v1"
PARTITIONED_SOL_REQUIRED_METADATA = {
    "api": 1,
    "owner": "comfyui_sol_h3",
    "exact": True,
    "backend": "sol",
    "attention_ownership": "sol",
    "kernel_contract": "sana-sol-engine-sol-attn-64-rect-sm120-mapped-neighbor-v4",
    "history_policy": "attention_backend_history_v1",
}


class PartitionedPreflightUnsupported(RuntimeError):
    """A condition detected before sampling that must use the exact target fallback."""


@contextlib.contextmanager
def _partitioned_stage_contract(guider: Any, plan, metrics):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("partitioned exact-prefix requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("partitioned exact-prefix requires mutable transformer options")
    if PARTITIONED_STAGE_KEY in transformer:
        raise RuntimeError("nested partitioned exact-prefix stage is unsupported")
    if "h3_flow_mixed_grid_v1" in transformer or "h3_flow_mixed_grid_attention_measure_v1" in transformer:
        raise RuntimeError("partitioned exact-prefix refuses deprecated Mixed-Grid stage state")

    # This must remain a non-dict leaf. ComfyUI recursively copies nested option
    # dictionaries between model calls, while scalar/object leaves retain their
    # identity. The runtime object therefore owns provider identity for exactly
    # this low/probe sampler-stage lifetime.
    transformer[PARTITIONED_STAGE_KEY] = PartitionedStageRuntime(plan=plan, metrics=metrics)
    try:
        yield
    finally:
        transformer.pop(PARTITIONED_STAGE_KEY, None)


def _validate_partitioned_vdn_compat(patcher: Any) -> None:
    object_patches = getattr(patcher, "object_patches", None)
    if not isinstance(object_patches, dict):
        raise PartitionedPreflightUnsupported("VDN object-patch ownership is unavailable")
    matched = 0
    for key, owner in object_patches.items():
        if not key.startswith("diffusion_model.blocks.") or not key.endswith(".attn.forward"):
            continue
        if not getattr(owner, "_vdn_forward", False):
            continue
        matched += 1
        if int(getattr(owner, "_vdn_external_sequence_api", 0)) != VDN_PARTITIONED_SEQUENCE_API:
            raise PartitionedPreflightUnsupported(
                f"VDN partitioned external-sequence API {VDN_PARTITIONED_SEQUENCE_API} is unavailable"
            )
    if matched == 0:
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires active VDN-H3 ownership")


def _validate_partitioned_sol_compat(guider: Any) -> None:
    """Require the Sol request owner before any partitioned sampler lifetime.

    VDN API-4 delegates partitioned sparse attention to Sol's request-owned
    backend.  Sol intentionally refuses those calls outside its native
    OUTER_SAMPLE lifecycle, so missing or stale Sol metadata is a preflight
    fallback condition rather than a mid-sampler runtime failure.
    """
    options = getattr(guider, "model_options", None)
    transformer = options.get("transformer_options") if isinstance(options, dict) else None
    metadata = transformer.get(SOL_RUNTIME_KEY) if isinstance(transformer, dict) else None
    if not isinstance(metadata, dict):
        raise PartitionedPreflightUnsupported(
            "partitioned exact-prefix requires active Sol-H3 native runtime ownership"
        )

    mismatches = [
        f"{name}={metadata.get(name)!r}"
        for name, expected in PARTITIONED_SOL_REQUIRED_METADATA.items()
        if metadata.get(name) != expected
    ]
    if mismatches:
        raise PartitionedPreflightUnsupported(
            "partitioned exact-prefix requires compatible Sol-H3 native runtime metadata: " + ", ".join(mismatches)
        )


def _preflight(
    guider: Any,
    config: ProgressiveTargetInputConfig,
    noise: torch.Tensor,
    latent_image: torch.Tensor,
    denoise_mask: torch.Tensor | None,
    latent_shapes: list[tuple[int, ...]],
):
    if len(latent_shapes) != 2:
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires native packed H3 AV latents")
    if config.transfer_mode != "learned_3d":
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires learned_3d handoff transfer")
    if config.suffix_dc_bridge or config.suffix_geometric_bridge:
        raise PartitionedPreflightUnsupported(
            "partitioned exact-prefix does not inherit retired seam-repair heuristics"
        )
    if denoise_mask is None or not _has_exact_video_protection(denoise_mask, latent_shapes):
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires an exact protected video prefix")
    target_h, target_w = map(int, latent_shapes[0][-2:])
    if target_h % 2 or target_w % 2:
        raise PartitionedPreflightUnsupported("target H3 geometry is not patch-safe")

    # These two owners are structural prerequisites for the heterogeneous
    # attention path. Validate them before any sampler lifetime is committed so
    # unsupported saved workflows use the released exact target-grid fallback.
    _validate_partitioned_vdn_compat(guider.model_patcher)
    _validate_partitioned_sol_compat(guider)

    try:
        source_h, source_w = config.resolve_source(target_h, target_w)
        internal = _process_latent_in(guider.model_patcher.model, latent_image, latent_shapes)
        stage_plan = build_partitioned_stage_plan(
            denoise_mask,
            latent_shapes,
            internal,
            noise,
            source_h=source_h,
            source_w=source_w,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PartitionedPreflightUnsupported(str(exc)) from exc
    return source_h, source_w, stage_plan


def _recover_partitioned_transfer_clean(
    target_video: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> torch.Tensor:
    diagnostic_noise = deterministic_video_noise(
        tuple(target_video.shape),
        seed=int(seed),
        device=target_video.device,
        dtype=target_video.dtype,
    )
    return recover_conditional_clean_for_diagnostics(
        target_video,
        diagnostic_noise,
        sigma=float(sigma),
    )


def _measure_partitioned_transfer_splice(
    target_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> dict[str, Any]:
    """Measure the learned-transfer seam immediately before exact-prefix restore.

    target_video is the target-grid conditional state returned by
    build_handoff_state. Reconstructing the clean learned state with the same
    deterministic handoff noise lets us compare two clean-domain boundaries:
    the learned upscaler's own prefix-last -> suffix-first boundary and the
    authoritative exact-prefix-last -> learned suffix-first boundary created
    when Flow discards the learned prefix.

    The tensors are diagnostics-only and never feed back into sampling.
    """

    diagnostic_started = time.perf_counter()
    learned_clean = _recover_partitioned_transfer_clean(
        target_video,
        sigma=float(sigma),
        seed=int(seed),
    )
    exact = exact_prefix.to(device=learned_clean.device, dtype=learned_clean.dtype)
    fields = measure_exact_prefix_splice(learned_clean, exact)
    fields.update(
        splice_diagnostic_elapsed_ms=(time.perf_counter() - diagnostic_started) * 1000.0,
        splice_recovery="inverse_conditional_renoise",
        splice_scope="learned_clean_before_exact_prefix_restore",
    )
    return fields


def run_partitioned_progressive(
    executor,
    guider,
    binding,
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
    """Execute exact-prefix low/probe/high continuation with a physical two-grid H3 stage."""
    chunk_started = time.perf_counter()
    if not isinstance(latent_shapes, list):
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires mutable latent-shape metadata")
    source_h, source_w, stage_plan = _preflight(
        guider,
        config,
        noise,
        latent_image,
        denoise_mask,
        latent_shapes,
    )

    # All unsupported conditions above are checked before any sampler lifetime.
    # From this point onward a failure is a real candidate failure and must not
    # silently restart the chunk under a different numerical path.
    _validate_progressive_sampler_state(sampler)
    if sigmas.ndim != 1 or sigmas.numel() < 4:
        raise RuntimeError("partitioned exact-prefix requires a full H3 sigma schedule")
    if not math.isclose(float(sigmas[0]), 1.0, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
        float(sigmas[-1]), 0.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise RuntimeError("partitioned exact-prefix requires a full 1-to-0 H3 sigma schedule")

    target_shapes = list(latent_shapes)
    target_h, target_w = map(int, target_shapes[0][-2:])
    model_options = getattr(guider, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("partitioned exact-prefix requires mutable model options")
    transformer = model_options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("partitioned exact-prefix requires mutable transformer options")
    current_conds = getattr(guider, "conds", None)
    if not isinstance(current_conds, dict):
        raise RuntimeError("partitioned exact-prefix requires ComfyUI guider conditioning state")
    conditioning_template = {
        key: [entry.copy() if isinstance(entry, dict) else copy.copy(entry) for entry in entries]
        for key, entries in current_conds.items()
    }
    video_shift = float(transformer.get("minimax_h3_sigma_shift_video", H3_VIDEO_SHIFT))
    selected_coordinate = config.resolve_coordinate(source_h, source_w, target_h, target_w)
    from .handoff import select_handoff_index

    index = select_handoff_index(
        sigmas,
        selected_coordinate,
        min_high_steps=config.min_high_steps,
        video_shift=video_shift,
    )
    sigma = float(sigmas[index].item())
    low_sigmas = sigmas[: index + 1]
    high_sigmas = sigmas[index:]

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

    binding.metrics.increment("progressive_partitioned_exact_prefix_runs")
    binding.metrics.event(
        "partitioned_stage_plan",
        index=index,
        sigma=sigma,
        coordinate=float(normalized_coordinate(sigma, video_shift=video_shift)),
        requested_coordinate=config.handoff_coordinate,
        selected_coordinate=selected_coordinate,
        selection=config.handoff_selection,
        input_mode="partitioned_exact_prefix",
        transfer_mode="learned_3d_suffix_context",
        prefix_temporal_length=stage_plan.prefix_t,
        suffix_temporal_length=stage_plan.temporal - stage_plan.prefix_t,
        prefix_target_hw=stage_plan.target_hw,
        suffix_source_hw=(source_h, source_w),
        source_shape=source_shapes[0],
        target_hw=(target_h, target_w),
        prefix_video_rows=stage_plan.prefix_rows,
        suffix_video_rows=stage_plan.suffix_rows,
        partitioned_video_rows=stage_plan.partitioned_rows,
        prefix_exact_latent_resized_for_transformer=False,
        deprecated_mixed_grid_contract_active=False,
    )

    sampler_invocation_count = 0
    history_boundary_count = 0

    def low_callback(step, x0, x, _total):
        if callback is None:
            return None
        x0 = _resize_packed_latent_image(x0, source_shapes, target_shapes)
        x = _resize_packed_latent_image(x, source_shapes, target_shapes)
        return callback(step, x0, x, len(sigmas) - 1)

    _begin_capture(binding, guider, sampler, low_sigmas, source_shapes)
    try:
        _reset_guider_conds(guider, template=conditioning_template)
        low_started = time.perf_counter()
        try:
            sampler_invocation_count += 1
            binding.metrics.increment("progressive_sampler_invocations")
            with _flow_stage_contract(guider, "low"), _partitioned_stage_contract(guider, stage_plan, binding.metrics):
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
            binding.metrics.event(
                "low_stage_wall",
                elapsed_ms=(time.perf_counter() - low_started) * 1000.0,
                partitioned_exact_prefix=True,
            )

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
            _reset_guider_conds(guider, template=conditioning_template)
            sampler_invocation_count += 1
            history_boundary_count += 1
            binding.metrics.increment("progressive_sampler_invocations")
            binding.metrics.increment("progressive_history_boundaries")
            with (
                _flow_stage_contract(guider, "probe"),
                _high_stage_contract(guider),
                _partitioned_stage_contract(guider, stage_plan, binding.metrics),
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
            binding.metrics.event(
                "handoff_probe_wall",
                elapsed_ms=(time.perf_counter() - probe_started) * 1000.0,
                partitioned_exact_prefix=True,
            )
    except BaseException as exc:
        _finish_capture(binding, error=exc)
        raise

    committed_low_run = _finish_capture(binding)
    binding.metrics.increment("handoff_exact_probe_nfe")
    try:
        source_x0 = _process_latent_in(base_model, source_x0, source_shapes)

        # The learned 3D upscaler may use all prefix frames as transient temporal
        # context.  This resized copy never enters H3 attention and its upscaled
        # prefix output is discarded below in favor of the authoritative target
        # prefix captured before the low stage.
        clean_video, clean_audio = unpack_streams(source_x0, source_shapes)
        for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
            source_native_trajectory = measure_translation_trajectory(
                clean_video,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="source_low_native",
                roi=roi_name,
                source_hw=(source_h, source_w),
                target_hw=(target_h, target_w),
                **source_native_trajectory,
                **project_translation_trajectory_to_grid(
                    source_native_trajectory,
                    source_hw=(source_h, source_w),
                    target_hw=(target_h, target_w),
                ),
            )
        clean_video = clean_video.clone()
        clean_video[:, :, : stage_plan.prefix_t] = resize_spatial_5d(
            stage_plan.prefix.to(clean_video),
            source_h,
            source_w,
            mode="bicubic",
        )
        for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
            source_exact_trajectory = measure_translation_trajectory(
                clean_video,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="source_low_exact_context",
                roi=roi_name,
                source_hw=(source_h, source_w),
                target_hw=(target_h, target_w),
                **source_exact_trajectory,
                **project_translation_trajectory_to_grid(
                    source_exact_trajectory,
                    source_hw=(source_h, source_w),
                    target_hw=(target_h, target_w),
                ),
            )
        source_x0 = pack_streams((clean_video, clean_audio))[0]

        transfer_started = time.perf_counter()
        transfer_metrics: dict[str, Any] = {}
        target_raw, rebuilt_shapes = build_handoff_state(
            source_packed_state=source_raw,
            source_x0_packed=source_x0,
            source_shapes=source_shapes,
            sigma=sigma,
            target_h=target_h,
            target_w=target_w,
            seed=int(seed or 0) + config.seed_offset,
            transfer_mode="learned_3d",
            learned_upscaler=config.learned_upscaler,
            transfer_metrics=transfer_metrics,
        )
        if rebuilt_shapes != target_shapes:
            raise RuntimeError("partitioned exact-prefix handoff changed caller-visible AV geometry")
        target_video, target_audio = unpack_streams(target_raw, target_shapes)
        diagnostic_seed = int(seed or 0) + config.seed_offset
        learned_clean = _recover_partitioned_transfer_clean(
            target_video,
            sigma=sigma,
            seed=diagnostic_seed,
        )
        splice_started = time.perf_counter()
        exact_prefix = stage_plan.prefix.to(
            device=learned_clean.device,
            dtype=learned_clean.dtype,
        )
        splice_diagnostics = measure_exact_prefix_splice(
            learned_clean,
            exact_prefix,
        )
        splice_diagnostics.update(
            splice_diagnostic_elapsed_ms=(time.perf_counter() - splice_started) * 1000.0,
            splice_recovery="inverse_conditional_renoise",
            splice_scope="learned_clean_before_exact_prefix_restore",
        )
        restored_clean = learned_clean.clone()
        restored_clean[:, :, : stage_plan.prefix_t] = exact_prefix
        for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
            native_trajectory = measure_translation_trajectory(
                learned_clean,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            restored_trajectory = measure_translation_trajectory(
                restored_clean,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="learned_native",
                roi=roi_name,
                **native_trajectory,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="exact_restored_pre_high",
                roi=roi_name,
                **restored_trajectory,
            )
        binding.metrics.increment("partitioned_splice_diagnostic_runs")
        binding.metrics.increment("partitioned_multiframe_trajectory_runs")
        del restored_clean, learned_clean
        target_video = target_video.clone()
        target_video[:, :, : stage_plan.prefix_t] = stage_plan.prefix.to(target_video)
        target_raw = pack_streams((target_video, target_audio))[0]
        binding.metrics.event(
            "partitioned_transfer",
            learned_transfer_performed=True,
            upscaler_prefix_context_used=True,
            upscaler_prefix_output_discarded=True,
            authoritative_target_prefix_restored=True,
            target_prefix_resized_for_transformer=False,
            deprecated_mixed_grid_repairs_applied=False,
            provider_api_version=transfer_metrics.get("provider_api_version"),
            provider_kind=transfer_metrics.get("provider_kind"),
            model_name=transfer_metrics.get("model_name"),
            source_hw=transfer_metrics.get("source_hw"),
            target_hw=transfer_metrics.get("target_hw"),
            temporal_length=transfer_metrics.get("temporal_length"),
            learned_upscale_elapsed_ms=transfer_metrics.get("learned_upscale_elapsed_ms"),
            **splice_diagnostics,
        )

        target_latent_internal = _process_latent_in(base_model, latent_image, target_shapes)
        target_noise = _noise_argument(base_model, target_raw, sigma, target_latent_internal)
        target_noise = _merge_preserved_noise(target_noise, noise, denoise_mask)
        binding.metrics.event(
            "handoff_transfer_wall",
            elapsed_ms=(time.perf_counter() - transfer_started) * 1000.0,
            partitioned_exact_prefix=True,
        )

        if binding.guidance is not None and binding.guidance.mode != "off":
            if binding.trajectory is None:
                raise RuntimeError("partitioned exact-prefix Flow guidance requires an H3_FLOW_TRAJECTORY")
            session_id, chunk_id = _interop_identity(getattr(guider, "model_options", None))
            expected_signature = binding.guidance_conditioning_signature or _conditioning_signature(guider)
            run = binding.trajectory.select(
                chunk_id=chunk_id,
                session_id=session_id,
                conditioning_signature=expected_signature,
            )
            if run.geometry.latent_t != int(target_shapes[0][2]):
                raise RuntimeError("partitioned low trajectory and target video temporal geometry differ")
            binding.active_guidance_run = run

        def high_callback(step, x0, x, _total):
            if callback is not None:
                return callback(index + step, x0, x, len(sigmas) - 1)
            return None

        _reset_guider_conds(guider, template=conditioning_template)
        high_started = time.perf_counter()
        high_event_start = len(binding.metrics.events)
        sampler_invocation_count += 1
        history_boundary_count += 1
        binding.metrics.increment("progressive_sampler_invocations")
        binding.metrics.increment("progressive_history_boundaries")
        with _flow_stage_contract(guider, "high"), _high_stage_contract(guider):
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
        binding.metrics.event(
            "high_stage_wall",
            elapsed_ms=(time.perf_counter() - high_started) * 1000.0,
            partitioned_exact_prefix=True,
        )
        high_model_calls = [event for event in binding.metrics.events[high_event_start:] if event.kind == "model_call"]
        if not high_model_calls:
            raise RuntimeError("partitioned exact-prefix high stage produced no H3 model evaluations")
        first_high_actual = bool(high_model_calls[0].fields.get("actual"))
        if not first_high_actual:
            raise RuntimeError("partitioned exact-prefix high stage did not begin with an exact H3 evaluation")

        final_video, _ = unpack_streams(result, target_shapes)
        original_video, _ = unpack_streams(latent_image, target_shapes)
        if not torch.equal(
            final_video[:, :, : stage_plan.prefix_t],
            original_video[:, :, : stage_plan.prefix_t].to(final_video),
        ):
            raise RuntimeError("partitioned exact-prefix high stage violated exact original-prefix preservation")
        boundary = measure_video_boundary(final_video, stage_plan.prefix_t)
        for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
            final_trajectory = measure_translation_trajectory(
                final_video,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="final_post_high",
                roi=roi_name,
                **final_trajectory,
            )
        if not splice_diagnostics:
            raise RuntimeError(
                "partitioned exact-prefix splice diagnostics were not recorded before high-stage sampling"
            )
        binding.metrics.event(
            "partitioned_exact_prefix_complete",
            final_prefix_exact=True,
            high_stage_first_call_actual=first_high_actual,
            total_chunk_sampler_elapsed_ms=(time.perf_counter() - chunk_started) * 1000.0,
            final_seam_lowpass_kernel=boundary["lowpass_kernel"],
            final_seam_rms=boundary["seam_rms"],
            final_seam_lowpass_rms=boundary["seam_lowpass_rms"],
            final_seam_spatial_mean_rms=boundary["seam_spatial_mean_rms"],
            final_over_transfer_native_seam_rms_ratio=boundary["seam_rms"]
            / max(float(splice_diagnostics["upscaler_native_seam_rms"]), 1e-12),
            final_over_transfer_native_seam_lowpass_ratio=boundary["seam_lowpass_rms"]
            / max(float(splice_diagnostics["upscaler_native_seam_lowpass_rms"]), 1e-12),
            final_over_transfer_native_seam_spatial_mean_ratio=boundary["seam_spatial_mean_rms"]
            / max(float(splice_diagnostics["upscaler_native_seam_spatial_mean_rms"]), 1e-12),
            final_over_transfer_exact_seam_rms_ratio=boundary["seam_rms"]
            / max(float(splice_diagnostics["exact_restored_seam_rms"]), 1e-12),
            final_over_transfer_exact_seam_lowpass_ratio=boundary["seam_lowpass_rms"]
            / max(float(splice_diagnostics["exact_restored_seam_lowpass_rms"]), 1e-12),
            final_over_transfer_exact_seam_spatial_mean_ratio=boundary["seam_spatial_mean_rms"]
            / max(float(splice_diagnostics["exact_restored_seam_spatial_mean_rms"]), 1e-12),
            deprecated_mixed_grid_contract_active=False,
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
            transfer_mode="learned_3d_suffix_context",
            input_mode="partitioned_exact_prefix",
        )
        return result
    except BaseException as exc:
        if committed_low_run is not None and binding.trajectory is not None:
            invalid = binding.trajectory.invalidate(
                committed_low_run.run_id,
                f"partitioned exact-prefix continuation failed: {type(exc).__name__}: {exc}",
            )
            binding.metrics.event(
                "trajectory_invalidate",
                run_id=invalid.run_id,
                error=type(exc).__name__,
                partitioned_exact_prefix=True,
            )
        raise


__all__ = [
    "PARTITIONED_PROGRESSIVE_KEY",
    "PARTITIONED_SOL_REQUIRED_METADATA",
    "SOL_RUNTIME_KEY",
    "PartitionedPreflightUnsupported",
    "run_partitioned_progressive",
]
