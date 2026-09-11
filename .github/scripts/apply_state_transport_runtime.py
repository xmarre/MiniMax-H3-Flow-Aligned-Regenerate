from __future__ import annotations

from pathlib import Path

PATH = Path("h3_flow_regenerate/runtime.py")
text = PATH.read_text()


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one runtime match, found {count}: {old[:120]!r}")
    text = text.replace(old, new, 1)


replace_once(
    """    build_handoff_state,\n    deterministic_video_noise,\n    select_handoff_index,\n)\n""",
    """    build_handoff_state,\n    deterministic_video_noise,\n    select_handoff_index,\n    upscale_learned_clean_video,\n)\n""",
)
replace_once(
    """from .source_trajectory_bridge import disabled_source_trajectory_bridge_metrics\n""",
    """from .source_trajectory_bridge import disabled_source_trajectory_bridge_metrics\nfrom .state_transport import (\n    HANDOFF_STATE_POLICY_LEGACY,\n    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n    resolve_handoff_state_policy,\n    transport_velocity_bicubic_v1,\n)\n""",
)
replace_once(
    """    mixed_plan = None\n    if (\n""",
    """    mixed_plan = None\n    requested_state_policy = HANDOFF_STATE_POLICY_LEGACY\n    if isinstance(config, ProgressiveTargetInputConfig) and config.exact_prefix_mode == \"mixed_grid_low_suffix\":\n        requested_state_policy = resolve_handoff_state_policy(getattr(config, \"handoff_state_policy\", None))\n    effective_state_policy = requested_state_policy if mixed else HANDOFF_STATE_POLICY_LEGACY\n    if (\n""",
)
replace_once(
    """                low_suffix_real_latent=True,\n                attention_measure_requested=measure_profile != MIXED_GRID_MEASURE_PROFILE_OFF,\n""",
    """                low_suffix_real_latent=True,\n                handoff_state_policy_requested=requested_state_policy,\n                handoff_state_policy=effective_state_policy,\n                attention_measure_requested=measure_profile != MIXED_GRID_MEASURE_PROFILE_OFF,\n""",
)
replace_once(
    """        source_shape=source_shapes[0],\n        target_hw=(target_h, target_w),\n    )\n""",
    """        source_shape=source_shapes[0],\n        target_hw=(target_h, target_w),\n        handoff_state_policy_requested=(requested_state_policy if target_input else None),\n        handoff_state_policy=(effective_state_policy if target_input else None),\n    )\n""",
)
replace_once(
    """    try:\n        source_x0 = _process_latent_in(base_model, source_x0, source_shapes)\n        if mixed_plan is not None:\n""",
    """    try:\n        source_x0 = _process_latent_in(base_model, source_x0, source_shapes)\n        accepted_source_x0 = (\n            source_x0.detach().clone()\n            if mixed_plan is not None and effective_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1\n            else None\n        )\n        if mixed_plan is not None:\n""",
)

start = text.index("        transfer_started = time.perf_counter()\n")
end = text.index("        if config.transfer_mode == \"learned_3d\":\n", start)
replacement = '''        transfer_started = time.perf_counter()
        transfer_metrics: dict[str, Any] = {}
        splice_diagnostics: dict[str, Any] = {}
        state_transport_metrics: dict[str, Any] = {
            "handoff_state_policy": effective_state_policy,
            "state_transport_applied": False,
        }
        diagnostic_noise = None

        if mixed_plan is not None and effective_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1:
            if accepted_source_x0 is None:
                raise RuntimeError("velocity state transport lost the immutable accepted probe clean")
            provider_video, _provider_audio = unpack_streams(source_x0, source_shapes)
            learned_clean, transfer_metrics = upscale_learned_clean_video(
                provider_video,
                target_h=target_h,
                target_w=target_w,
                learned_upscaler=getattr(config, "learned_upscaler", None),
            )
            source_state_video, source_audio = unpack_streams(source_raw, source_shapes)
            accepted_clean_video, _accepted_clean_audio = unpack_streams(accepted_source_x0, source_shapes)
            target_audio = source_audio.clone()
        else:
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

        if mixed_plan is not None:
            diagnostic_started = time.perf_counter()
            exact_prefix = mixed_plan.prefix.to(device=learned_clean.device, dtype=learned_clean.dtype)
            representation_requested = bool(getattr(config, "suffix_geometric_bridge", False))
            if representation_requested:
                corrected_clean, representation_metrics = apply_suffix_representation_bridge(
                    learned_clean,
                    exact_prefix,
                    requested=True,
                )
            else:
                corrected_clean = learned_clean
                representation_metrics = disabled_suffix_representation_bridge_metrics(
                    prefix_t=mixed_plan.prefix_t,
                    requested=False,
                )

            dc_enabled = bool(getattr(config, "suffix_dc_bridge", False))
            if dc_enabled:
                corrected_clean, bridge_metrics = apply_suffix_dc_bridge(
                    corrected_clean,
                    exact_prefix,
                    weights=(1.0,),
                )
            else:
                bridge_metrics = disabled_suffix_dc_bridge_metrics(prefix_t=mixed_plan.prefix_t)

            corrected_tokens = max(
                int(representation_metrics["suffix_representation_bridge_corrected_tokens"]),
                int(bridge_metrics["suffix_dc_bridge_corrected_tokens"]),
            )
            splice_diagnostics = measure_exact_prefix_splice(
                learned_clean,
                exact_prefix,
                corrected_clean_video=corrected_clean,
            )
            splice_diagnostics["splice_diagnostic_elapsed_ms"] = (time.perf_counter() - diagnostic_started) * 1000.0
            if effective_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1:
                splice_diagnostics["splice_recovery"] = "retained_learned_clean"
                splice_diagnostics["suffix_dc_bridge_state_mapping"] = (
                    "clean_reanchor_velocity_bicubic_v1" if dc_enabled else "disabled"
                )
                splice_diagnostics["suffix_representation_bridge_state_mapping"] = (
                    "clean_reanchor_velocity_bicubic_v1"
                    if representation_metrics["suffix_representation_bridge_accepted"]
                    else "disabled_or_noop"
                )
            else:
                splice_diagnostics["splice_recovery"] = "inverse_conditional_renoise"
                splice_diagnostics["suffix_dc_bridge_state_mapping"] = (
                    "affine_equivalent_pre_renoise" if dc_enabled else "disabled"
                )
                splice_diagnostics["suffix_representation_bridge_state_mapping"] = (
                    "affine_equivalent_pre_renoise"
                    if representation_metrics["suffix_representation_bridge_accepted"]
                    else "disabled_or_noop"
                )
            binding.metrics.increment("mixed_grid_splice_diagnostic_runs")
            binding.metrics.event(
                "mixed_grid_representation_bridge",
                legacy_option_name="suffix_geometric_bridge",
                authoritative_prefix_modified=False,
                later_suffix_extrapolated=False,
                **representation_metrics,
            )

            if effective_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1:
                target_video, state_transport_metrics = transport_velocity_bicubic_v1(
                    source_state_video,
                    accepted_clean_video,
                    corrected_clean,
                    prefix_t=mixed_plan.prefix_t,
                )
                binding.metrics.increment("mixed_grid_state_transport_runs")
                binding.metrics.event("mixed_grid_state_transport", **state_transport_metrics)
            else:
                if corrected_tokens:
                    target_video = map_clean_bridge_to_conditional_state(
                        target_video,
                        learned_clean,
                        corrected_clean,
                        sigma=sigma,
                        prefix_t=mixed_plan.prefix_t,
                        corrected_tokens=corrected_tokens,
                    )
                else:
                    target_video = target_video.clone()

            target_video[:, :, : mixed_plan.prefix_t] = mixed_plan.prefix.to(target_video)
            target_raw, target_shapes = pack_streams((target_video, target_audio))
            binding.metrics.event(
                "mixed_grid_transfer",
                learned_transfer_performed=True,
                upscaler_prefix_context_used=True,
                upscaler_prefix_output_discarded=True,
                final_original_prefix_restored=True,
                transfer_mode="learned_3d_suffix",
                handoff_state_policy_requested=requested_state_policy,
                handoff_state_policy=effective_state_policy,
                source_trajectory_bridge_requested=source_trajectory_metrics["source_trajectory_bridge_requested"],
                source_trajectory_bridge_accepted=source_trajectory_metrics["source_trajectory_bridge_accepted"],
                source_trajectory_bridge_reason=source_trajectory_metrics["source_trajectory_bridge_reason"],
                source_trajectory_bridge_tokens_corrected=source_trajectory_metrics[
                    "source_trajectory_bridge_tokens_corrected"
                ],
                source_trajectory_axis_authorized=source_trajectory_metrics.get(
                    "source_trajectory_axis_authorized", [False] * 4
                ),
                source_trajectory_residual_reduction_ratio=source_trajectory_metrics.get(
                    "source_trajectory_residual_reduction_ratio", [None] * 4
                ),
                **state_transport_metrics,
                **representation_metrics,
                **bridge_metrics,
                **splice_diagnostics,
            )
            if diagnostic_noise is not None:
                del diagnostic_noise
            del learned_clean, corrected_clean
        if accepted_source_x0 is not None:
            del accepted_source_x0
'''
text = text[:start] + replacement + text[end:]

replace_once(
    """                final_prefix_exact=True,\n                high_stage_first_call_actual=first_high_actual,\n""",
    """                final_prefix_exact=True,\n                high_stage_first_call_actual=first_high_actual,\n                handoff_state_policy=effective_state_policy,\n""",
)
replace_once(
    """            transfer_mode=config.transfer_mode,\n            input_mode=\"mixed_grid_low_suffix\" if mixed else (\"target_grid\" if target_input else \"source_grid\"),\n        )\n""",
    """            transfer_mode=config.transfer_mode,\n            input_mode=\"mixed_grid_low_suffix\" if mixed else (\"target_grid\" if target_input else \"source_grid\"),\n            handoff_state_policy_requested=(requested_state_policy if target_input else None),\n            handoff_state_policy=(effective_state_policy if target_input else None),\n        )\n""",
)

PATH.write_text(text)
