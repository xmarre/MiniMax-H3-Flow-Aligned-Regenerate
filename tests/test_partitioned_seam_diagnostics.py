from __future__ import annotations

import math

import pytest
import torch

from h3_flow_regenerate.guidance import conditional_renoise_target
from h3_flow_regenerate.handoff import (
    H3_HANDOFF_NOISE_INDEPENDENT,
    H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
    deterministic_video_noise,
)
from h3_flow_regenerate.partitioned_scheduler import (
    _apply_partitioned_exact_overlap_bridge,
    _apply_partitioned_suffix_dc_bridge,
    _measure_partitioned_transfer_splice,
    _partitioned_exact_overlap_fallback_eligibility,
    _resolve_partitioned_transfer_clean,
)
from h3_flow_regenerate.seam_diagnostics import project_translation_trajectory_to_grid


def test_partitioned_transfer_clean_uses_captured_actual_clean_for_source_residual():
    torch.manual_seed(1201)
    target = torch.randn(1, 24, 5, 8, 8, dtype=torch.float32)
    actual = torch.randn_like(target)

    resolved, source, recovery = _resolve_partitioned_transfer_clean(
        target,
        actual,
        handoff_noise_mode=H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
        sigma=0.4,
        seed=123,
    )

    assert resolved.data_ptr() == actual.data_ptr()
    assert source == "actual_clean_postprocess"
    assert recovery == "actual_clean_postprocess_no_inverse"


def test_partitioned_transfer_clean_refuses_gaussian_inverse_for_source_residual_without_capture():
    target = torch.randn(1, 24, 5, 8, 8, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="refusing deterministic-noise inverse recovery"):
        _resolve_partitioned_transfer_clean(
            target,
            None,
            handoff_noise_mode=H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
            sigma=0.4,
            seed=123,
        )


def test_partitioned_transfer_clean_keeps_independent_noise_inverse_contract():
    torch.manual_seed(1202)
    learned = torch.randn(1, 24, 5, 8, 8, dtype=torch.float32)
    sigma = 0.4
    seed = 123
    noise = deterministic_video_noise(
        tuple(learned.shape),
        seed=seed,
        device=learned.device,
        dtype=learned.dtype,
    )
    target = conditional_renoise_target(learned, sigma=sigma, noise=noise)

    resolved, source, recovery = _resolve_partitioned_transfer_clean(
        target,
        None,
        handoff_noise_mode=H3_HANDOFF_NOISE_INDEPENDENT,
        sigma=sigma,
        seed=seed,
    )

    torch.testing.assert_close(resolved, learned, rtol=0.0, atol=2e-6)
    assert source == "inverse_recovered"
    assert recovery == "inverse_conditional_renoise"


def test_partitioned_transfer_splice_measures_before_and_after_exact_prefix_restore():
    learned = torch.zeros(1, 24, 5, 8, 8, dtype=torch.float32)
    learned[:, :, 1] = 1.0
    learned[:, :, 2] = 2.0
    learned[:, :, 3] = 3.0
    learned[:, :, 4] = 4.0
    exact = learned[:, :, :2].clone()
    exact[:, :, 1] = 10.0

    sigma = 0.4
    seed = 12345
    noise = deterministic_video_noise(
        tuple(learned.shape),
        seed=seed,
        device=learned.device,
        dtype=learned.dtype,
    )
    state = conditional_renoise_target(learned, sigma=sigma, noise=noise)

    fields = _measure_partitioned_transfer_splice(
        state,
        exact,
        sigma=sigma,
        seed=seed,
    )

    assert fields["splice_diagnostic_version"] == 1
    assert fields["splice_prefix_temporal_length"] == 2
    assert fields["splice_recovery"] == "inverse_conditional_renoise"
    assert fields["splice_clean_source"] == "inverse_recovered"
    assert fields["splice_scope"] == "learned_clean_before_exact_prefix_restore"
    assert fields["upscaler_native_seam_rms"] == pytest.approx(1.0, abs=1e-5)
    assert fields["exact_restored_seam_rms"] == pytest.approx(8.0, abs=1e-5)
    assert fields["seam_rms_amplification"] == pytest.approx(8.0, rel=1e-5)
    assert fields["prefix_restore_rms_tail_1"] == pytest.approx(9.0, abs=1e-5)
    assert fields["prefix_restore_rms_tail_2"] == pytest.approx(9.0 / math.sqrt(2.0), rel=1e-5)

    required = {
        "prefix_restore_lowpass_rms_tail_1",
        "prefix_restore_spatial_mean_rms_tail_1",
        "upscaler_native_seam_lowpass_rms",
        "exact_restored_seam_lowpass_rms",
        "seam_lowpass_amplification",
        "upscaler_native_seam_spatial_mean_rms",
        "exact_restored_seam_spatial_mean_rms",
        "seam_spatial_mean_amplification",
        "splice_diagnostic_elapsed_ms",
    }
    assert required <= fields.keys()
    assert all(math.isfinite(float(fields[key])) for key in required)


def test_multiframe_trajectory_recovers_bounded_translation_direction():
    torch.manual_seed(123)
    base = torch.randn(1, 8, 6, 32, 40, dtype=torch.float32)
    for index in range(1, 6):
        base[:, :, index] = torch.roll(
            base[:, :, index - 1],
            shifts=(1, -1),
            dims=(-2, -1),
        )

    from h3_flow_regenerate.seam_diagnostics import measure_translation_trajectory

    fields = measure_translation_trajectory(
        base,
        1,
        forward_steps=4,
        roi_fraction=1.0,
        max_shift=3,
    )

    assert fields["trajectory_diagnostic_version"] == 1
    assert fields["trajectory_boundary_t"] == 1
    assert fields["trajectory_forward_steps"] == 4
    assert fields["trajectory_backward_steps"] == 0
    assert fields["pre_pairwise_dy"] == []
    assert fields["pairwise_dy"] == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=0.05)
    assert fields["pairwise_dx"] == pytest.approx([-1.0, -1.0, -1.0, -1.0], abs=0.05)
    assert fields["pairwise_net_dy"] == pytest.approx(4.0, abs=0.1)
    assert fields["pairwise_net_dx"] == pytest.approx(-4.0, abs=0.1)
    assert fields["anchor_final_dy"] == pytest.approx(4.0, abs=0.1)
    assert fields["anchor_final_dx"] == pytest.approx(-4.0, abs=0.1)
    assert all(value > 1.0 for value in fields["pairwise_response"])


def test_multiframe_trajectory_reports_prefix_motion_before_boundary():
    torch.manual_seed(456)
    video = torch.randn(1, 8, 8, 32, 40, dtype=torch.float32)
    for index in range(1, 8):
        video[:, :, index] = torch.roll(
            video[:, :, index - 1],
            shifts=(1, 0),
            dims=(-2, -1),
        )

    from h3_flow_regenerate.seam_diagnostics import measure_translation_trajectory

    fields = measure_translation_trajectory(
        video,
        4,
        forward_steps=3,
        backward_steps=3,
        roi_fraction=1.0,
        max_shift=3,
    )

    assert fields["trajectory_backward_steps"] == 3
    assert fields["pre_pairwise_dy"] == pytest.approx([1.0, 1.0, 1.0], abs=0.05)
    assert fields["pre_pairwise_median_dy"] == pytest.approx(1.0, abs=0.05)
    assert fields["pairwise_dy"] == pytest.approx([1.0, 1.0, 1.0], abs=0.05)


def test_trajectory_grid_projection_preserves_source_receipt_and_scales_axes_independently():
    fields = {
        "pre_pairwise_dx": [-1.0, 2.0],
        "pre_pairwise_dy": [0.5, -1.0],
        "pairwise_dx": [-3.0],
        "pairwise_dy": [4.0],
        "pairwise_cumulative_dx": [-3.0],
        "pairwise_cumulative_dy": [4.0],
        "anchor_dx": [-2.0],
        "anchor_dy": [1.0],
        "pre_pairwise_median_dx": -1.0,
        "pre_pairwise_median_dy": 0.5,
        "pairwise_net_dx": -3.0,
        "pairwise_net_dy": 4.0,
        "anchor_final_dx": -2.0,
        "anchor_final_dy": 1.0,
        "pairwise_response": [9.0],
    }
    original = {key: value[:] if isinstance(value, list) else value for key, value in fields.items()}

    projected = project_translation_trajectory_to_grid(
        fields,
        source_hw=(40, 50),
        target_hw=(60, 100),
    )

    assert fields == original
    assert projected["trajectory_measurement_hw"] == (40, 50)
    assert projected["trajectory_target_equivalent_hw"] == (60, 100)
    assert projected["trajectory_target_equivalent_scale_x"] == pytest.approx(2.0)
    assert projected["trajectory_target_equivalent_scale_y"] == pytest.approx(1.5)
    assert projected["target_equivalent_pairwise_dx"] == pytest.approx([-6.0])
    assert projected["target_equivalent_pairwise_dy"] == pytest.approx([6.0])
    assert projected["target_equivalent_pairwise_net_dx"] == pytest.approx(-6.0)
    assert projected["target_equivalent_pairwise_net_dy"] == pytest.approx(6.0)
    assert projected["target_equivalent_anchor_final_dx"] == pytest.approx(-4.0)
    assert projected["target_equivalent_anchor_final_dy"] == pytest.approx(1.5)
    assert "target_equivalent_pairwise_response" not in projected


def test_partitioned_suffix_dc_bridge_preserves_learned_native_dc_relation_and_scope():
    learned = torch.zeros(1, 24, 5, 4, 4, dtype=torch.float32)
    learned[:, :, 1] = 1.0
    learned[:, :, 2] = 3.0
    learned[:, :, 3] = 5.0
    learned[:, :, 4] = 7.0
    exact = learned[:, :, :2].clone()
    exact[:, :, 1] = 10.0

    sigma = 0.4
    seed = 321
    noise = deterministic_video_noise(
        tuple(learned.shape),
        seed=seed,
        device=learned.device,
        dtype=learned.dtype,
    )
    state = conditional_renoise_target(learned, sigma=sigma, noise=noise)

    mapped, corrected, metrics = _apply_partitioned_suffix_dc_bridge(
        state,
        learned,
        exact,
        sigma=sigma,
        enabled=True,
    )

    learned_native = learned[:, :, 2].mean((-2, -1)) - learned[:, :, 1].mean((-2, -1))
    corrected_exact = corrected[:, :, 2].mean((-2, -1)) - exact[:, :, 1].mean((-2, -1))
    assert torch.allclose(corrected_exact, learned_native, rtol=0.0, atol=1e-6)
    assert torch.equal(corrected[:, :, :2], learned[:, :, :2])
    assert torch.equal(corrected[:, :, 3:], learned[:, :, 3:])
    assert torch.equal(mapped[:, :, :2], state[:, :, :2])
    assert torch.equal(mapped[:, :, 3:], state[:, :, 3:])

    expected = state.clone()
    expected[:, :, 2] += (1.0 - sigma) * (corrected[:, :, 2] - learned[:, :, 2])
    assert torch.allclose(mapped, expected, rtol=1e-6, atol=1e-6)
    assert metrics["suffix_dc_bridge_enabled"] is True
    assert metrics["suffix_dc_bridge_corrected_tokens"] == 1
    assert metrics["suffix_dc_bridge_first_weight"] == 1.0


def test_partitioned_suffix_dc_bridge_disabled_is_state_preserving():
    learned = torch.randn(1, 24, 4, 4, 4, dtype=torch.float32)
    exact = learned[:, :, :2].clone()
    state = torch.randn_like(learned)

    mapped, corrected, metrics = _apply_partitioned_suffix_dc_bridge(
        state,
        learned,
        exact,
        sigma=0.5,
        enabled=False,
    )

    assert torch.equal(mapped, state)
    assert torch.equal(corrected, learned)
    assert mapped.data_ptr() != state.data_ptr()
    assert corrected.data_ptr() != learned.data_ptr()
    assert metrics["suffix_dc_bridge_enabled"] is False
    assert metrics["suffix_dc_bridge_corrected_tokens"] == 0


def test_partitioned_exact_overlap_bridge_preserves_provider_native_transition_and_scope():
    torch.manual_seed(77)
    learned = torch.randn(1, 24, 5, 8, 10, dtype=torch.float32)
    exact = learned[:, :, :2].clone()
    yy = torch.linspace(-1.0, 1.0, 8).view(1, 1, 8, 1)
    xx = torch.linspace(-1.0, 1.0, 10).view(1, 1, 1, 10)
    exact[:, :, 1] += 0.15 + 0.09 * yy - 0.06 * xx

    sigma = 0.4
    seed = 991
    noise = deterministic_video_noise(
        tuple(learned.shape),
        seed=seed,
        device=learned.device,
        dtype=learned.dtype,
    )
    state = conditional_renoise_target(learned, sigma=sigma, noise=noise)

    mapped, corrected, representation, dc = _apply_partitioned_exact_overlap_bridge(
        state,
        learned,
        exact,
        sigma=sigma,
    )

    assert representation["suffix_representation_bridge_accepted"] is True
    assert representation["suffix_representation_bridge_corrected_tokens"] == 1
    assert dc["suffix_dc_bridge_corrected_tokens"] == 1

    native_transition = learned[:, :, 2].float() - learned[:, :, 1].float()
    restored_transition = corrected[:, :, 2].float() - exact[:, :, 1].float()
    assert torch.allclose(restored_transition, native_transition, rtol=0.0, atol=2e-6)

    assert torch.equal(corrected[:, :, :2], learned[:, :, :2])
    assert torch.equal(corrected[:, :, 3:], learned[:, :, 3:])
    assert torch.equal(mapped[:, :, :2], state[:, :, :2])
    assert torch.equal(mapped[:, :, 3:], state[:, :, 3:])

    expected = state.clone()
    expected[:, :, 2] += (1.0 - sigma) * (corrected[:, :, 2] - learned[:, :, 2])
    assert torch.allclose(mapped, expected, rtol=1e-6, atol=1e-6)


def test_partitioned_exact_overlap_successor_support_bounds_relocated_residual():
    torch.manual_seed(78)
    learned = torch.randn(1, 24, 8, 8, 10, dtype=torch.float32)
    exact = learned[:, :, :2].clone()
    yy = torch.linspace(-1.0, 1.0, 8).view(1, 1, 8, 1)
    xx = torch.linspace(-1.0, 1.0, 10).view(1, 1, 1, 10)
    exact[:, :, 1] += 0.15 + 0.09 * yy - 0.06 * xx
    sigma = 0.4
    noise = deterministic_video_noise(tuple(learned.shape), seed=992, device=learned.device, dtype=learned.dtype)
    state = conditional_renoise_target(learned, sigma=sigma, noise=noise)
    weights = (1.0, 0.75, 0.5, 0.25)

    mapped, corrected, representation, dc = _apply_partitioned_exact_overlap_bridge(
        state,
        learned,
        exact,
        sigma=sigma,
        weights=weights,
    )

    delta = exact[:, :, 1].float() - learned[:, :, 1].float()
    native_boundary = learned[:, :, 2].float() - learned[:, :, 1].float()
    restored_boundary = corrected[:, :, 2].float() - exact[:, :, 1].float()
    torch.testing.assert_close(restored_boundary, native_boundary, rtol=0.0, atol=2e-6)
    expected_step = -0.25 * delta
    for offset in range(1, 4):
        corrected_step = corrected[:, :, 2 + offset].float() - corrected[:, :, 1 + offset].float()
        native_step = learned[:, :, 2 + offset].float() - learned[:, :, 1 + offset].float()
        torch.testing.assert_close(corrected_step - native_step, expected_step, rtol=0.0, atol=2e-6)
    exit_step = corrected[:, :, 6].float() - corrected[:, :, 5].float()
    native_exit = learned[:, :, 6].float() - learned[:, :, 5].float()
    torch.testing.assert_close(exit_step - native_exit, expected_step, rtol=0.0, atol=2e-6)

    assert representation["suffix_representation_bridge_corrected_tokens"] == 4
    assert representation["suffix_representation_bridge_successor_safe"] is True
    assert dc["suffix_dc_bridge_corrected_tokens"] == 4
    assert torch.equal(corrected[:, :, 6:], learned[:, :, 6:])
    assert torch.equal(mapped[:, :, 6:], state[:, :, 6:])


def _boundary_rejection_transaction(reason="boundary_upper45_insufficient_improvement"):
    receipt = {"dx": 0.0, "dy": 0.0, "response": 5.0, "clipped": False}
    checks = {
        roi: {variant: dict(receipt) for variant in ("native", "transformed_native", "exact_restored", "candidate")}
        for roi in ("upper45", "full")
    }
    return {
        "result": "rejected",
        "reason": reason,
        "video_registration": {"status": "accepted"},
        "boundary_motion": {
            "policy": "native_boundary_motion_preservation_v2",
            "status": "rejected",
            "reason": reason,
            "checks": checks,
        },
    }


def test_exact_overlap_fallback_requires_unambiguous_rigid_boundary_veto():
    eligible, trigger = _partitioned_exact_overlap_fallback_eligibility(_boundary_rejection_transaction())
    assert eligible is True
    assert trigger == "boundary_upper45_insufficient_improvement"

    ambiguous = _boundary_rejection_transaction()
    ambiguous["boundary_motion"]["checks"]["upper45"]["transformed_native"]["response"] = 2.0
    eligible, trigger = _partitioned_exact_overlap_fallback_eligibility(ambiguous)
    assert eligible is False
    assert trigger == "boundary_upper45_transformed_native_ambiguous"

    wrong_failure = _boundary_rejection_transaction("guidance_insufficient_validation_improvement")
    eligible, trigger = _partitioned_exact_overlap_fallback_eligibility(wrong_failure)
    assert eligible is False
    assert trigger == "frame_gauge_rejection_not_structural_overlap_eligible"


def _hardware_invalidated_shadow_transaction():
    transaction = _boundary_rejection_transaction()
    transaction.update(
        result="shadow_only",
        reason="hardware_invalidated_global_rigid_application_00687",
        candidate_accepted=True,
        production_mutation_allowed=False,
        spatial_warp_applied=False,
    )
    transaction["boundary_motion"]["status"] = "accepted"
    transaction["boundary_motion"]["reason"] = "accepted"
    transaction["boundary_motion"]["policy"] = "native_boundary_motion_preservation_v2"
    return transaction


def test_exact_overlap_fallback_accepts_hardware_invalidated_rigid_shadow():
    transaction = _hardware_invalidated_shadow_transaction()

    eligible, trigger = _partitioned_exact_overlap_fallback_eligibility(transaction)

    assert eligible is True
    assert trigger == "hardware_invalidated_global_rigid_application_00687"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("candidate_accepted", False, "frame_gauge_shadow_candidate_not_accepted"),
        ("production_mutation_allowed", True, "frame_gauge_shadow_production_mutation_not_disabled"),
        ("spatial_warp_applied", True, "frame_gauge_shadow_spatial_warp_was_applied"),
    ],
)
def test_exact_overlap_fallback_rejects_invalid_hardware_shadow_ownership(field, value, reason):
    transaction = _hardware_invalidated_shadow_transaction()
    transaction[field] = value

    eligible, trigger = _partitioned_exact_overlap_fallback_eligibility(transaction)

    assert eligible is False
    assert trigger == reason


def test_exact_overlap_fallback_rejects_ambiguous_hardware_shadow_boundary():
    transaction = _hardware_invalidated_shadow_transaction()
    transaction["boundary_motion"]["checks"]["full"]["candidate"]["clipped"] = True

    eligible, trigger = _partitioned_exact_overlap_fallback_eligibility(transaction)

    assert eligible is False
    assert trigger == "boundary_full_candidate_clipped"
