from __future__ import annotations

import math

import pytest
import torch

from h3_flow_regenerate.guidance import conditional_renoise_target
from h3_flow_regenerate.handoff import deterministic_video_noise
from h3_flow_regenerate.partitioned_scheduler import _measure_partitioned_transfer_splice


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
    assert fields["pairwise_dy"] == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=0.05)
    assert fields["pairwise_dx"] == pytest.approx([-1.0, -1.0, -1.0, -1.0], abs=0.05)
    assert fields["pairwise_net_dy"] == pytest.approx(4.0, abs=0.1)
    assert fields["pairwise_net_dx"] == pytest.approx(-4.0, abs=0.1)
    assert fields["anchor_final_dy"] == pytest.approx(4.0, abs=0.1)
    assert fields["anchor_final_dx"] == pytest.approx(-4.0, abs=0.1)
    assert all(value > 1.0 for value in fields["pairwise_response"])
