from __future__ import annotations

import math

import pytest
import torch

from h3_flow_regenerate.contracts import TrajectoryRun, TrajectorySample
from h3_flow_regenerate.geometry import geometry_from_video
from h3_flow_regenerate.guidance import conditional_renoise_target
from h3_flow_regenerate.handoff import deterministic_video_noise
from h3_flow_regenerate.partitioned_scheduler import (
    _align_guidance_run_suffix_gauge,
    _apply_partitioned_suffix_dc_bridge,
    _measure_partitioned_transfer_splice,
)
from h3_flow_regenerate.seam_diagnostics import (
    estimate_prefix_rigid_alignment,
    project_translation_trajectory_to_grid,
)


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


def test_prefix_rigid_alignment_recovers_same_frame_gauge_offset():
    torch.manual_seed(812)
    exact = torch.randn(1, 8, 5, 32, 40, dtype=torch.float32)
    learned = torch.roll(exact, shifts=(1, -2), dims=(-2, -1))

    fields = estimate_prefix_rigid_alignment(
        learned,
        exact,
        frames=4,
        max_shift=3,
    )

    assert fields["prefix_alignment_version"] == 1
    assert fields["prefix_alignment_frames"] == 4
    assert fields["learned_relative_median_dx"] == pytest.approx(-2.0, abs=0.05)
    assert fields["learned_relative_median_dy"] == pytest.approx(1.0, abs=0.05)
    assert fields["correction_dx"] == pytest.approx(2.0, abs=0.05)
    assert fields["correction_dy"] == pytest.approx(-1.0, abs=0.05)
    assert fields["learned_relative_mad_dx"] < 0.05
    assert fields["learned_relative_mad_dy"] < 0.05
    assert fields["median_response"] > 1.0

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


def test_guidance_run_suffix_gauge_alignment_scales_target_offset_to_source_grid():
    video = torch.zeros(1, 24, 5, 20, 26, dtype=torch.float32)
    video[:, :, :2] = torch.randn(1, 24, 2, 20, 26)
    video[:, :, 2:, 10, 13] = 1.0
    sample = TrajectorySample(
        coordinate=0.5,
        video_sigma=0.8,
        audio_sigma=0.7,
        outer_step=1,
        call_index=2,
        phase="single",
        provenance="actual",
        video_x0=video,
    )
    run = TrajectoryRun(
        schema_version=1,
        run_id="run",
        session_id="session",
        chunk_id="chunk",
        sampler="sampler",
        scheduler="scheduler",
        geometry=geometry_from_video(video),
        audio_shape=(1, 32, 2, 10),
        layout_signature="layout",
        conditioning_signature="conditioning",
        storage="system_ram",
        samples=(sample,),
        started_ns=1,
        completed_ns=2,
        complete=True,
    )

    aligned, source_dx, source_dy = _align_guidance_run_suffix_gauge(
        run,
        prefix_t=2,
        target_hw=(40, 52),
        correction_dx=2.0,
        correction_dy=-2.0,
    )

    assert source_dx == pytest.approx(1.0)
    assert source_dy == pytest.approx(-1.0)
    assert torch.equal(aligned.samples[0].video_x0[:, :, :2], video[:, :, :2])
    assert aligned.samples[0].video_x0[0, 0, 2, 9, 14] == pytest.approx(1.0, abs=1e-6)
    assert run.samples[0].video_x0[0, 0, 2, 10, 13] == pytest.approx(1.0)

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
