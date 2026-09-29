from __future__ import annotations

import pytest
import torch

import h3_flow_regenerate.vae_boundary_video as vae_boundary_video
from h3_flow_regenerate.vae_boundary_video import (
    VAE_WINDOW_RELEASE_TAIL,
    apply_vae_window_vertical_translation,
    h3_vae_boundary_window,
    plan_vae_window_vertical_residual,
    repair_vae_window_vertical_residual,
)


def _trajectory(*, first_dx, first_dy, pre_dx, pre_dy, pairs=9, response=12.0):
    return {
        "pairwise_dx": [float(first_dx)] + [0.0] * (pairs - 1),
        "pairwise_dy": [float(first_dy)] + [0.0] * (pairs - 1),
        "pairwise_response": [float(response)] * pairs,
        "pairwise_clipped": [False] * pairs,
        "pre_pairwise_median_dx": float(pre_dx),
        "pre_pairwise_median_dy": float(pre_dy),
    }


def _00715_receipts():
    pre_high = {
        "upper45": _trajectory(
            first_dx=-0.0075787891,
            first_dy=0.0555063963,
            pre_dx=-0.0076364889,
            pre_dy=0.0348370224,
            response=25.48,
        ),
        "full": _trajectory(
            first_dx=-0.0227513474,
            first_dy=-0.0023144214,
            pre_dx=-0.0573308133,
            pre_dy=-0.0120560369,
            response=29.59,
        ),
    }
    post_high = {
        "upper45": _trajectory(
            first_dx=0.0744942290,
            first_dy=0.1118459395,
            pre_dx=-0.0076364889,
            pre_dy=0.0348370224,
            response=11.0,
        ),
        "full": _trajectory(
            first_dx=-0.0399231186,
            first_dy=0.1931790173,
            pre_dx=-0.0573308133,
            pre_dy=-0.0120560369,
            response=10.0,
        ),
    }
    return pre_high, post_high


def test_phase_aligned_prefix_maps_to_native_boundary_decoder_window():
    window = h3_vae_boundary_window(prefix_t=12, temporal=62)

    assert window["chunk_index"] == 2
    assert window["window_start_t"] == 10
    assert window["window_stop_t"] == 17
    assert window["generated_window_start_t"] == 12
    assert window["generated_window_stop_t"] == 17
    assert window["generated_window_tokens"] == 5
    assert window["decoded_trim_frames"] == 39
    assert window["decoder_chunk_output_start_frame"] == 34
    assert window["first_retained_local_frame"] == 5
    assert window["decoder_internal_overlap_frames"] == 5
    assert window["weights"] == pytest.approx(
        [1.0] * 5 + list(VAE_WINDOW_RELEASE_TAIL)
    )
    assert window["release_frontier_t"] == 20


@pytest.mark.parametrize("prefix_t", [3, 6, 8, 11])
def test_non_native_prefix_phase_fails_closed(prefix_t):
    with pytest.raises(ValueError, match="native 5k\+2 latent phase"):
        h3_vae_boundary_window(prefix_t=prefix_t, temporal=62)


def test_00715_receipts_select_decoder_window_coherent_vertical_plan():
    pre_high, post_high = _00715_receipts()
    plan = plan_vae_window_vertical_residual(
        pre_high,
        post_high,
        prefix_t=12,
        temporal=62,
    )

    assert plan["eligible"] is True
    assert plan["reason"] == "coherent_target_high_vertical_residual_inside_vae_boundary_window"
    assert plan["horizontal_application_enabled"] is False
    assert plan["correction_magnitude_cells"] == pytest.approx(0.1259164909, abs=1e-6)
    assert plan["plateau_tokens"] == 5
    assert plan["support_tokens"] == 8
    assert plan["max_induced_step_cells"] < 0.0625


def test_window_translation_preserves_prefix_and_holds_constant_through_boundary_window():
    torch.manual_seed(123)
    video = torch.randn(1, 4, 30, 12, 12)
    prefix_t = 12
    dy = -0.125
    corrected = apply_vae_window_vertical_translation(video, prefix_t=prefix_t, dy=dy)
    window = h3_vae_boundary_window(prefix_t, video.shape[2])

    assert torch.equal(corrected[:, :, :prefix_t], video[:, :, :prefix_t])
    assert not torch.equal(
        corrected[:, :, prefix_t : window["window_stop_t"]],
        video[:, :, prefix_t : window["window_stop_t"]],
    )
    assert torch.equal(
        corrected[:, :, window["support_stop_t"] :],
        video[:, :, window["support_stop_t"] :],
    )

    # Every generated token in the active decoder window receives the same
    # spatial operator. This is the invariant the rejected 00713 release lacked.
    expected = []
    for index in range(prefix_t, window["window_stop_t"]):
        one = vae_boundary_video.translate_video_cells(
            video[:, :, index : index + 1],
            dx=0.0,
            dy=dy,
            start_frame=0,
            batch_frames=1,
        ).video
        expected.append(one)
    torch.testing.assert_close(
        corrected[:, :, prefix_t : window["window_stop_t"]],
        torch.cat(expected, dim=2),
        rtol=0,
        atol=0,
    )


def test_plan_fails_closed_when_target_high_y_delta_disagrees_between_rois():
    pre_high, post_high = _00715_receipts()
    post_high["upper45"]["pairwise_dy"][0] = 0.02

    plan = plan_vae_window_vertical_residual(
        pre_high,
        post_high,
        prefix_t=12,
        temporal=62,
    )

    assert plan["eligible"] is False
    assert plan["reason"] == "target_high_vertical_delta_not_coherent"


def test_repair_trials_both_signs_and_accepts_only_pre_high_restoration(monkeypatch):
    pre_high, post_high = _00715_receipts()
    improved = {
        "upper45": _trajectory(
            first_dx=0.074,
            first_dy=0.060,
            pre_dx=-0.0076364889,
            pre_dy=0.0348370224,
            response=11.0,
        ),
        "full": _trajectory(
            first_dx=-0.040,
            first_dy=0.002,
            pre_dx=-0.0573308133,
            pre_dy=-0.0120560369,
            response=10.0,
        ),
    }
    worsened = {
        "upper45": _trajectory(
            first_dx=0.074,
            first_dy=0.25,
            pre_dx=-0.0076364889,
            pre_dy=0.0348370224,
            response=11.0,
        ),
        "full": _trajectory(
            first_dx=-0.040,
            first_dy=0.31,
            pre_dx=-0.0573308133,
            pre_dy=-0.0120560369,
            response=10.0,
        ),
    }
    receipts = iter((post_high, improved, worsened))
    monkeypatch.setattr(
        vae_boundary_video,
        "measure_vae_window_video_trajectory",
        lambda _video, _prefix_t: next(receipts),
    )

    torch.manual_seed(42)
    video = torch.randn(1, 4, 30, 12, 12)
    corrected, receipt = repair_vae_window_vertical_residual(
        pre_high,
        video,
        prefix_t=12,
    )

    assert receipt["applied"] is True
    assert receipt["accepted"] is True
    assert receipt["selected_dy_cells"] < 0.0
    assert receipt["selected_score"]["mean_improvement_ratio"] > 0.25
    assert receipt["selected_score"]["vae_window_plateau_stable"] is True
    assert receipt["selected_score"]["post_window_release_stable"] is True
    assert torch.equal(corrected[:, :, :12], video[:, :, :12])
    assert not torch.equal(corrected[:, :, 12:20], video[:, :, 12:20])
    assert torch.equal(corrected[:, :, 20:], video[:, :, 20:])
