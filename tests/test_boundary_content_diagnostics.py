from __future__ import annotations

import copy

import pytest
import torch

from h3_flow_regenerate.boundary_content_diagnostics import (
    BOUNDARY_CONTENT_DIAGNOSTIC_POLICY,
    PROVIDER_BOUNDARY_CALIBRATION_POLICY,
    PROVIDER_BOUNDARY_POST_HIGH_SHADOW_POLICY,
    PROVIDER_BOUNDARY_PREDICTOR_POLICY,
    PROVIDER_BOUNDARY_SOFT_SUPPORT_SHADOW_POLICY,
    PROVIDER_BOUNDARY_STABILIZATION_POLICY,
    PROVIDER_BOUNDARY_STABILIZATION_SHADOW_POLICY,
    apply_provider_boundary_soft_support_stabilization,
    compare_boundary_content_stages,
    measure_boundary_content_continuity,
    measure_provider_boundary_post_high_shadow,
    measure_provider_boundary_soft_support_shadow,
    measure_provider_boundary_stabilization_shadow,
    measure_provider_boundary_temporal_calibration,
    measure_provider_boundary_temporal_predictor,
)
from h3_flow_regenerate.partitioned_runtime_gate import (
    RuntimeGateError,
    _validate_boundary_content_diagnostics,
    _validate_provider_boundary_post_high_shadow,
    _validate_provider_boundary_predictor,
    _validate_provider_boundary_predictor_calibration,
    _validate_provider_boundary_soft_support_shadow,
    _validate_provider_boundary_stabilization,
    _validate_provider_boundary_stabilization_shadow,
)


def _video_with_local_boundary_change(*, scale: float) -> torch.Tensor:
    torch.manual_seed(7001)
    base = torch.randn(1, 24, 6, 32, 32, dtype=torch.float32)
    anchor = torch.randn(1, 24, 32, 32, dtype=torch.float32)
    drift = torch.randn(1, 24, 32, 32, dtype=torch.float32)
    for index in range(6):
        base[:, :, index] = anchor + 0.01 * float(index) * drift
    base[:, :, 4] = anchor + 0.04 * drift
    base[:, :, 5] = anchor + 0.05 * drift
    patch = torch.randn(1, 24, 8, 8, dtype=torch.float32)
    base[:, :, 5, :8, :8] += float(scale) * patch
    return base


def test_boundary_content_diagnostic_localizes_structural_change():
    video = _video_with_local_boundary_change(scale=2.0)

    receipt = measure_boundary_content_continuity(
        video,
        5,
        correspondence_radius=2,
    )

    assert receipt["policy"] == BOUNDARY_CONTENT_DIAGNOSTIC_POLICY
    assert receipt["diagnostic_only"] is True
    assert receipt["production_gate"] is False
    assert receipt["pre_steps"] == 3
    assert len(receipt["tiles"]) == 16
    assert receipt["tiles_by_centered_structural_change"][0] == "r0c0"

    changed = receipt["tiles"]["r0c0"]
    quiet = receipt["tiles"]["r3c3"]
    assert (
        changed["boundary_vs_prefix"]["centered_lowpass_rms_over_prefix_median"]
        > quiet["boundary_vs_prefix"]["centered_lowpass_rms_over_prefix_median"]
    )
    assert changed["boundary"]["centered_lowpass_rms"] > quiet["boundary"]["centered_lowpass_rms"]
    assert changed["boundary"]["ncc"] < quiet["boundary"]["ncc"]


def test_boundary_content_diagnostic_reports_correspondence_support_without_mutation():
    torch.manual_seed(7002)
    frame = torch.randn(1, 24, 32, 32, dtype=torch.float32)
    video = frame[:, :, None].repeat(1, 1, 6, 1, 1)
    before = video.clone()

    receipt = measure_boundary_content_continuity(
        video,
        5,
        correspondence_radius=2,
    )

    assert torch.equal(video, before)
    assert receipt["global_boundary"]["cycle_support_fraction"] > 0.95
    assert receipt["global_boundary"]["ncc"] == pytest.approx(1.0, abs=1e-6)
    assert receipt["global_boundary"]["centered_lowpass_rms"] == pytest.approx(0.0, abs=1e-7)


def test_boundary_content_stage_delta_localizes_post_high_amplification():
    pre = measure_boundary_content_continuity(
        _video_with_local_boundary_change(scale=0.4),
        5,
        correspondence_radius=2,
    )
    post = measure_boundary_content_continuity(
        _video_with_local_boundary_change(scale=2.0),
        5,
        correspondence_radius=2,
    )

    delta = compare_boundary_content_stages(pre, post)

    assert delta["policy"] == BOUNDARY_CONTENT_DIAGNOSTIC_POLICY
    assert delta["diagnostic_only"] is True
    assert delta["production_gate"] is False
    assert delta["tiles_by_post_high_structural_amplification"][0] == "r0c0"
    assert delta["tiles"]["r0c0"]["centered_lowpass_rms_post_over_pre"] > 1.0


def test_provider_boundary_predictor_matches_smooth_temporal_trend():
    video = _video_with_local_boundary_change(scale=0.0)
    before = video.clone()

    receipt = measure_provider_boundary_temporal_predictor(video, 5)

    assert torch.equal(video, before)
    assert receipt["policy"] == PROVIDER_BOUNDARY_PREDICTOR_POLICY
    assert receipt["diagnostic_only"] is True
    assert receipt["production_gate"] is False
    assert receipt["predictor"] == "elementwise_median_centered_lowpass_delta_v1"
    assert receipt["global"]["prediction_error_rms"] == pytest.approx(0.0, abs=1e-6)
    assert receipt["global"]["actual_vs_predictor_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_provider_boundary_predictor_localizes_unexpected_first_suffix_change():
    receipt = measure_provider_boundary_temporal_predictor(
        _video_with_local_boundary_change(scale=2.0),
        5,
    )

    assert receipt["tiles_by_prediction_error_rms"][0] == "r0c0"
    changed = receipt["tiles"]["r0c0"]
    quiet = receipt["tiles"]["r3c3"]
    assert changed["prediction_error_rms"] > quiet["prediction_error_rms"]
    assert changed["prediction_error_over_actual_delta"] > quiet["prediction_error_over_actual_delta"]
    assert changed["actual_vs_predictor_cosine"] < quiet["actual_vs_predictor_cosine"]


def _predictor_event(receipt: dict) -> dict:
    return {
        "kind": "partitioned_provider_boundary_predictor",
        "fields": {
            "domain": "model_internal_clean",
            "owner_before": "learned_provider_prefix",
            "owner_after": "learned_provider_suffix",
            "elapsed_ms": 1.0,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
            **receipt,
        },
    }


def test_runtime_gate_accepts_provider_boundary_predictor_receipt():
    receipt = measure_provider_boundary_temporal_predictor(
        _video_with_local_boundary_change(scale=0.7),
        5,
    )

    _validate_provider_boundary_predictor([_predictor_event(receipt)])


def test_runtime_gate_rejects_provider_boundary_predictor_that_claims_production_control():
    receipt = measure_provider_boundary_temporal_predictor(
        _video_with_local_boundary_change(scale=0.7),
        5,
    )
    bad = _predictor_event(copy.deepcopy(receipt))
    bad["fields"]["production_gate"] = True

    with pytest.raises(RuntimeGateError, match="production gate"):
        _validate_provider_boundary_predictor([bad])


def test_provider_boundary_calibration_matches_smooth_temporal_trend():
    video = _video_with_local_boundary_change(scale=0.0)
    before = video.clone()

    receipt = measure_provider_boundary_temporal_calibration(
        video,
        5,
        calibration_targets=1,
    )

    assert torch.equal(video, before)
    assert receipt["policy"] == PROVIDER_BOUNDARY_CALIBRATION_POLICY
    assert receipt["diagnostic_only"] is True
    assert receipt["production_gate"] is False
    assert receipt["calibration"] == "rolling_held_out_prefix_transitions_v1"
    assert receipt["calibration_target_count"] == 1
    assert receipt["global"]["boundary_prediction_error_rms"] == pytest.approx(0.0, abs=1e-6)
    assert receipt["global"]["boundary_error_over_historical_max"] <= 1.05


def test_provider_boundary_calibration_localizes_held_out_boundary_surprise():
    receipt = measure_provider_boundary_temporal_calibration(
        _video_with_local_boundary_change(scale=2.0),
        5,
        calibration_targets=1,
    )

    assert receipt["tiles_by_boundary_error_over_historical_max"][0] == "r0c0"
    changed = receipt["tiles"]["r0c0"]
    quiet = receipt["tiles"]["r3c3"]
    assert changed["boundary_prediction_error_rms"] > quiet["boundary_prediction_error_rms"]
    assert changed["boundary_error_over_historical_max"] > quiet["boundary_error_over_historical_max"]


def _calibration_event(receipt: dict) -> dict:
    return {
        "kind": "partitioned_provider_boundary_predictor_calibration",
        "fields": {
            "domain": "model_internal_clean",
            "owner_before": "learned_provider_prefix",
            "owner_after": "learned_provider_suffix",
            "elapsed_ms": 1.0,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
            **receipt,
        },
    }


def test_runtime_gate_accepts_provider_boundary_calibration_receipt():
    torch.manual_seed(7003)
    video = torch.randn(1, 24, 10, 32, 32, dtype=torch.float32)
    receipt = measure_provider_boundary_temporal_calibration(video, 9)

    _validate_provider_boundary_predictor_calibration([_calibration_event(receipt)])


def test_runtime_gate_rejects_provider_boundary_calibration_that_claims_production_control():
    torch.manual_seed(7004)
    video = torch.randn(1, 24, 10, 32, 32, dtype=torch.float32)
    receipt = measure_provider_boundary_temporal_calibration(video, 9)
    bad = _calibration_event(copy.deepcopy(receipt))
    bad["fields"]["production_gate"] = True

    with pytest.raises(RuntimeGateError, match="production gate"):
        _validate_provider_boundary_predictor_calibration([bad])


def _shadow_receipt(video: torch.Tensor, prefix_t: int, *, calibration_targets: int = 1) -> dict:
    provider = measure_boundary_content_continuity(video, prefix_t)
    calibration = measure_provider_boundary_temporal_calibration(
        video,
        prefix_t,
        calibration_targets=calibration_targets,
    )
    return measure_provider_boundary_stabilization_shadow(
        video,
        prefix_t,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
    )


def test_provider_boundary_stabilization_shadow_is_non_mutating_and_localized():
    video = _video_with_local_boundary_change(scale=4.0)
    torch.manual_seed(7005)
    video[:, :, :5] += 0.02 * torch.randn_like(video[:, :, :5])
    before = video.clone()

    receipt = _shadow_receipt(video, 5)

    assert torch.equal(video, before)
    assert receipt["policy"] == PROVIDER_BOUNDARY_STABILIZATION_SHADOW_POLICY
    assert receipt["diagnostic_only"] is True
    assert receipt["production_gate"] is False
    assert receipt["production_applied"] is False
    assert receipt["output_mutated"] is False
    assert "r0c0" in receipt["eligible_tiles"]
    changed = receipt["tiles"]["r0c0"]
    assert changed["eligible"] is True
    assert 0.0 < changed["residual_scale"] < 1.0
    assert changed["predicted_error_over_historical_max_after"] <= 1.0 + 1e-9
    assert changed["predicted_dispersion_ratio_over_historical_max_after"] <= 1.0 + 1e-9
    assert changed["shadow_correction_rms"] > 0.0


def test_provider_boundary_stabilization_shadow_leaves_smooth_trend_unselected():
    video = _video_with_local_boundary_change(scale=0.0)

    receipt = _shadow_receipt(video, 5)

    assert receipt["eligible_tile_count"] == 0
    assert receipt["eligible_tiles"] == []
    assert receipt["correction_rms"] == pytest.approx(0.0, abs=1e-8)
    assert receipt["correction_abs_max"] == pytest.approx(0.0, abs=1e-8)
    for tile in receipt["tiles"].values():
        assert tile["eligible"] is False
        assert tile["residual_scale"] == pytest.approx(1.0)
        assert tile["shadow_correction_rms"] == pytest.approx(0.0, abs=1e-8)


def _soft_shadow_receipt(video: torch.Tensor, prefix_t: int, *, calibration_targets: int = 1) -> dict:
    provider = measure_boundary_content_continuity(video, prefix_t)
    calibration = measure_provider_boundary_temporal_calibration(
        video,
        prefix_t,
        calibration_targets=calibration_targets,
    )
    hard = measure_provider_boundary_stabilization_shadow(
        video,
        prefix_t,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
    )
    return measure_provider_boundary_soft_support_shadow(
        video,
        prefix_t,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
        hard_shadow_receipt=hard,
    )


def test_provider_boundary_soft_support_shadow_reduces_frontier_jump_without_mutation():
    video = _video_with_local_boundary_change(scale=4.0)
    torch.manual_seed(7006)
    video[:, :, :5] += 0.02 * torch.randn_like(video[:, :, :5])
    before = video.clone()

    receipt = _soft_shadow_receipt(video, 5)

    assert torch.equal(video, before)
    assert receipt["policy"] == PROVIDER_BOUNDARY_SOFT_SUPPORT_SHADOW_POLICY
    assert receipt["diagnostic_only"] is True
    assert receipt["production_applied"] is False
    assert receipt["output_mutated"] is False
    assert receipt["eligible_tiles"]
    assert receipt["feather_width"] == 3
    assert receipt["soft_frontier_edge_jump_rms"] <= receipt["hard_frontier_edge_jump_rms"] + 1e-9
    assert receipt["soft_frontier_edge_jump_abs_max"] <= receipt["hard_frontier_edge_jump_abs_max"] + 1e-9
    frontier_tiles = [
        receipt["tiles"][tile_id]
        for tile_id in receipt["eligible_tiles"]
        if receipt["tiles"][tile_id]["support_min"] < 1.0
    ]
    assert frontier_tiles
    changed = frontier_tiles[0]
    assert changed["support_min"] == pytest.approx(0.0, abs=1e-8)
    assert 0.0 < changed["support_mean"] < 1.0
    assert changed["support_max"] == pytest.approx(1.0)
    assert 0.0 < changed["soft_direct_correction_rms"] <= changed["hard_direct_correction_rms"]


def test_provider_boundary_soft_support_shadow_leaves_ineligible_tiles_directly_unmodified():
    video = _video_with_local_boundary_change(scale=4.0)
    torch.manual_seed(7007)
    video[:, :, :5] += 0.02 * torch.randn_like(video[:, :, :5])

    receipt = _soft_shadow_receipt(video, 5)

    eligible = set(receipt["eligible_tiles"])
    for tile_id, tile in receipt["tiles"].items():
        if tile_id in eligible:
            continue
        assert tile["hard_direct_correction_rms"] == pytest.approx(0.0, abs=1e-8)
        assert tile["soft_direct_correction_rms"] == pytest.approx(0.0, abs=1e-8)
        assert tile["support_max"] == pytest.approx(0.0, abs=1e-8)


def _soft_shadow_event(receipt: dict) -> dict:
    return {
        "kind": "partitioned_provider_boundary_soft_support_shadow",
        "fields": {
            "domain": "model_internal_clean",
            "owner_before": "learned_provider_prefix",
            "owner_after": "learned_provider_suffix",
            "elapsed_ms": 1.0,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
            **receipt,
        },
    }


def test_runtime_gate_accepts_provider_boundary_soft_support_shadow():
    video = _video_with_local_boundary_change(scale=4.0)
    torch.manual_seed(7008)
    video[:, :, :5] += 0.02 * torch.randn_like(video[:, :, :5])
    provider = measure_boundary_content_continuity(video, 5)
    calibration = measure_provider_boundary_temporal_calibration(video, 5, calibration_targets=1)
    hard = measure_provider_boundary_stabilization_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
    )
    soft = measure_provider_boundary_soft_support_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
        hard_shadow_receipt=hard,
    )

    _validate_provider_boundary_soft_support_shadow([_shadow_event(hard), _soft_shadow_event(soft)])


def test_runtime_gate_rejects_provider_boundary_soft_support_production_mutation():
    video = _video_with_local_boundary_change(scale=4.0)
    torch.manual_seed(7009)
    video[:, :, :5] += 0.02 * torch.randn_like(video[:, :, :5])
    provider = measure_boundary_content_continuity(video, 5)
    calibration = measure_provider_boundary_temporal_calibration(video, 5, calibration_targets=1)
    hard = measure_provider_boundary_stabilization_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
    )
    soft = measure_provider_boundary_soft_support_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
        hard_shadow_receipt=hard,
    )
    bad = _soft_shadow_event(copy.deepcopy(soft))
    bad["fields"]["production_applied"] = True

    with pytest.raises(RuntimeGateError, match="production mutation"):
        _validate_provider_boundary_soft_support_shadow([_shadow_event(hard), bad])


def test_provider_boundary_soft_support_production_candidate_changes_only_first_suffix():
    video = _video_with_local_boundary_change(scale=4.0)
    torch.manual_seed(7010)
    video[:, :, :5] += 0.02 * torch.randn_like(video[:, :, :5])
    provider = measure_boundary_content_continuity(video, 5)
    calibration = measure_provider_boundary_temporal_calibration(video, 5, calibration_targets=1)
    hard = measure_provider_boundary_stabilization_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
    )
    soft = measure_provider_boundary_soft_support_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
        hard_shadow_receipt=hard,
    )
    before = video.clone()

    stabilized, receipt = apply_provider_boundary_soft_support_stabilization(
        video,
        5,
        soft_shadow_receipt=soft,
    )

    assert torch.equal(video, before)
    assert receipt["policy"] == PROVIDER_BOUNDARY_STABILIZATION_POLICY
    assert receipt["source_shadow_policy"] == PROVIDER_BOUNDARY_SOFT_SUPPORT_SHADOW_POLICY
    assert receipt["applied"] is True
    assert receipt["corrected_tokens"] == 1
    assert receipt["authoritative_prefix_modified"] is False
    assert receipt["later_suffix_extrapolated"] is False
    assert torch.equal(stabilized[:, :, :5], before[:, :, :5])
    assert not torch.equal(stabilized[:, :, 5], before[:, :, 5])
    assert torch.equal(stabilized[:, :, 6:], before[:, :, 6:])
    assert receipt["correction_rms"] == pytest.approx(soft["soft_correction_rms"], rel=1e-6, abs=1e-8)
    assert receipt["correction_abs_max"] == pytest.approx(soft["soft_correction_abs_max"], rel=1e-6, abs=1e-8)


def test_provider_boundary_soft_support_production_candidate_noops_without_eligible_region():
    video = _video_with_local_boundary_change(scale=0.0)
    provider = measure_boundary_content_continuity(video, 5)
    calibration = measure_provider_boundary_temporal_calibration(video, 5, calibration_targets=1)
    hard = measure_provider_boundary_stabilization_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
    )
    soft = measure_provider_boundary_soft_support_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
        hard_shadow_receipt=hard,
    )

    stabilized, receipt = apply_provider_boundary_soft_support_stabilization(
        video,
        5,
        soft_shadow_receipt=soft,
    )

    assert torch.equal(stabilized, video)
    assert receipt["applied"] is False
    assert receipt["corrected_tokens"] == 0
    assert receipt["eligible_tiles"] == []


def test_runtime_gate_rejects_applied_provider_boundary_stabilization_after_post_high_regression():
    video = _video_with_local_boundary_change(scale=4.0)
    torch.manual_seed(7011)
    video[:, :, :5] += 0.02 * torch.randn_like(video[:, :, :5])
    provider = measure_boundary_content_continuity(video, 5)
    calibration = measure_provider_boundary_temporal_calibration(video, 5, calibration_targets=1)
    hard = measure_provider_boundary_stabilization_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
    )
    soft = measure_provider_boundary_soft_support_shadow(
        video,
        5,
        calibration_receipt=calibration,
        provider_content_receipt=provider,
        hard_shadow_receipt=hard,
    )
    _, production = apply_provider_boundary_soft_support_stabilization(
        video,
        5,
        soft_shadow_receipt=soft,
    )
    production_event = {
        "kind": "partitioned_provider_boundary_stabilization",
        "fields": {
            "requested": True,
            "mode": "soft_support_v1",
            "reason": "applied",
            "exact_overlap_fallback_required": True,
            "exact_overlap_fallback_requested": True,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
            **production,
        },
    }

    with pytest.raises(RuntimeGateError, match="production mutation must stay disabled"):
        _validate_provider_boundary_stabilization([production_event])


def test_post_high_provider_boundary_shadow_is_non_mutating_and_reuses_final_domain():
    video = _video_with_local_boundary_change(scale=4.0)
    video = torch.cat([video, video[:, :, -1:].clone()], dim=2)
    torch.manual_seed(7012)
    video[:, :, :5] += 0.02 * torch.randn_like(video[:, :, :5])
    content = measure_boundary_content_continuity(video, 5)
    before = video.clone()

    receipt = measure_provider_boundary_post_high_shadow(
        video,
        5,
        post_high_content_receipt=content,
    )

    assert torch.equal(video, before)
    assert receipt["policy"] == PROVIDER_BOUNDARY_POST_HIGH_SHADOW_POLICY
    assert receipt["diagnostic_only"] is True
    assert receipt["production_gate"] is False
    assert receipt["production_application_permitted"] is False
    assert receipt["output_mutated"] is False
    assert receipt["source_stage"] == "post_high_internal_clean"
    assert receipt["hard_shadow"]["output_mutated"] is False
    assert receipt["soft_shadow"]["output_mutated"] is False
    assert receipt["eligible_tiles"] == receipt["soft_shadow"]["eligible_tiles"]
    successor = receipt["successor_transition"]
    assert successor["available"] is True
    assert successor["left_index"] == 5
    assert successor["right_index"] == 6
    assert successor["candidate_output_discarded"] is True
    assert set(successor["tiles"]) == {f"r{row}c{col}" for row in range(4) for col in range(4)}


def test_runtime_gate_accepts_fail_closed_receipt_and_post_high_shadow():
    video = _video_with_local_boundary_change(scale=4.0)
    video = torch.cat([video, video[:, :, -1:].clone()], dim=2)
    torch.manual_seed(7013)
    video[:, :, :5] += 0.02 * torch.randn_like(video[:, :, :5])
    content = measure_boundary_content_continuity(video, 5)
    shadow = measure_provider_boundary_post_high_shadow(
        video,
        5,
        post_high_content_receipt=content,
    )
    stabilization = {
        "kind": "partitioned_provider_boundary_stabilization",
        "fields": {
            "requested": True,
            "mode": "soft_support_v1",
            "applied": False,
            "reason": "disabled_pending_post_high_validation",
            "authoritative_prefix_modified": False,
            "later_suffix_extrapolated": False,
            "corrected_tokens": 0,
            "exact_overlap_fallback_required": True,
            "exact_overlap_fallback_requested": True,
            "historical_candidate_policy": PROVIDER_BOUNDARY_STABILIZATION_POLICY,
            "historical_candidate_mutation_disabled": True,
            "production_mutation_allowed": False,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
        },
    }
    post_event = {
        "kind": "partitioned_provider_boundary_post_high_shadow",
        "fields": {
            "mode": "soft_support_v1",
            "domain": "model_internal_clean",
            "owner_before": "authoritative_exact_prefix",
            "owner_after": "post_high_generated_suffix",
            "legacy_pre_high_mutation_disabled": True,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
            "elapsed_ms": 1.0,
            **shadow,
        },
    }
    # The post-high validator is coupled to the existing boundary-content path.
    boundary_event = _event("post_high_internal_clean", content)
    window = [stabilization, boundary_event, post_event]

    _validate_provider_boundary_stabilization(window)
    _validate_provider_boundary_post_high_shadow(window)


def test_runtime_gate_rejects_post_high_shadow_that_claims_output_mutation():
    video = _video_with_local_boundary_change(scale=4.0)
    content = measure_boundary_content_continuity(video, 5)
    shadow = measure_provider_boundary_post_high_shadow(
        video,
        5,
        post_high_content_receipt=content,
    )
    shadow["output_mutated"] = True
    window = [
        {
            "kind": "partitioned_provider_boundary_stabilization",
            "fields": {
                "requested": True,
                "mode": "soft_support_v1",
                "applied": False,
                "reason": "disabled_pending_post_high_validation",
                "authoritative_prefix_modified": False,
                "later_suffix_extrapolated": False,
                "corrected_tokens": 0,
                "exact_overlap_fallback_required": True,
                "exact_overlap_fallback_requested": True,
                "historical_candidate_policy": PROVIDER_BOUNDARY_STABILIZATION_POLICY,
                "historical_candidate_mutation_disabled": True,
                "production_mutation_allowed": False,
                "extra_h3_nfe": 0,
                "extra_sampler_lifetimes": 0,
                "extra_history_boundaries": 0,
                "extra_provider_calls": 0,
                "extra_vae_calls": 0,
            },
        },
        _event("post_high_internal_clean", content),
        {
            "kind": "partitioned_provider_boundary_post_high_shadow",
            "fields": {
                "mode": "soft_support_v1",
                "legacy_pre_high_mutation_disabled": True,
                "extra_h3_nfe": 0,
                "extra_sampler_lifetimes": 0,
                "extra_history_boundaries": 0,
                "extra_provider_calls": 0,
                "extra_vae_calls": 0,
                "elapsed_ms": 1.0,
                **shadow,
            },
        },
    ]

    with pytest.raises(RuntimeGateError, match="mutated output"):
        _validate_provider_boundary_post_high_shadow(window)


def _shadow_event(receipt: dict) -> dict:
    return {
        "kind": "partitioned_provider_boundary_stabilization_shadow",
        "fields": {
            "domain": "model_internal_clean",
            "owner_before": "learned_provider_prefix",
            "owner_after": "learned_provider_suffix",
            "elapsed_ms": 1.0,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
            **receipt,
        },
    }


def test_runtime_gate_accepts_provider_boundary_stabilization_shadow():
    receipt = _shadow_receipt(_video_with_local_boundary_change(scale=2.0), 5)

    _validate_provider_boundary_stabilization_shadow([_shadow_event(receipt)])


def test_runtime_gate_rejects_provider_boundary_stabilization_shadow_production_mutation():
    receipt = _shadow_receipt(_video_with_local_boundary_change(scale=2.0), 5)
    bad = _shadow_event(copy.deepcopy(receipt))
    bad["fields"]["production_applied"] = True

    with pytest.raises(RuntimeGateError, match="production mutation"):
        _validate_provider_boundary_stabilization_shadow([bad])


def _event(stage: str, receipt: dict) -> dict:
    return {
        "kind": "partitioned_boundary_content_continuity",
        "fields": {
            "stage": stage,
            "domain": "model_internal_clean",
            "owner_before": "test",
            "owner_after": "test",
            "elapsed_ms": 1.0,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
            **receipt,
        },
    }


def test_runtime_gate_accepts_observation_only_boundary_content_receipts():
    receipt = measure_boundary_content_continuity(
        _video_with_local_boundary_change(scale=0.7),
        5,
    )
    post = measure_boundary_content_continuity(
        _video_with_local_boundary_change(scale=1.2),
        5,
    )
    delta = compare_boundary_content_stages(receipt, post)
    window = [
        _event("provider_native", receipt),
        _event("pre_high_exact_restored", receipt),
        _event("post_high_internal_clean", post),
        {
            "kind": "partitioned_boundary_content_stage_delta",
            "fields": {
                "pre_stage": "pre_high_exact_restored",
                "post_stage": "post_high_internal_clean",
                "extra_h3_nfe": 0,
                "extra_sampler_lifetimes": 0,
                "extra_history_boundaries": 0,
                "extra_provider_calls": 0,
                "extra_vae_calls": 0,
                **delta,
            },
        },
    ]

    _validate_boundary_content_diagnostics(window)


def test_runtime_gate_rejects_boundary_content_receipt_that_claims_production_control():
    receipt = measure_boundary_content_continuity(
        _video_with_local_boundary_change(scale=0.7),
        5,
    )
    post = measure_boundary_content_continuity(
        _video_with_local_boundary_change(scale=1.2),
        5,
    )
    delta = compare_boundary_content_stages(receipt, post)
    window = [
        _event("provider_native", receipt),
        _event("pre_high_exact_restored", receipt),
        _event("post_high_internal_clean", post),
        {
            "kind": "partitioned_boundary_content_stage_delta",
            "fields": {
                "pre_stage": "pre_high_exact_restored",
                "post_stage": "post_high_internal_clean",
                "extra_h3_nfe": 0,
                "extra_sampler_lifetimes": 0,
                "extra_history_boundaries": 0,
                "extra_provider_calls": 0,
                "extra_vae_calls": 0,
                **delta,
            },
        },
    ]
    bad = copy.deepcopy(window)
    bad[1]["fields"]["production_gate"] = True

    with pytest.raises(RuntimeGateError, match="production gate"):
        _validate_boundary_content_diagnostics(bad)
