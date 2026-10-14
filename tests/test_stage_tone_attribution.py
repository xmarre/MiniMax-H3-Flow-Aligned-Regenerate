"""Contract tests for the read-only full-stage photometric ledger."""

import pytest
import torch

from h3_flow_regenerate.stage_tone_attribution import STAGE_ORDER, tone_attribution_ledger


def _stages():
    labels = list(range(173, 190))
    base = torch.full((len(labels), 12, 16, 3), 0.60)
    pixels = {name: base.clone() for name in STAGE_ORDER}
    # Uniform provider shift changes absolute brightness, not within-stage slope.
    pixels["provider_native"] += 0.04
    for name in STAGE_ORDER[2:]:
        pixels[name] += 0.04
    # Source already falls after join; provider transfer adds no extra fall.
    for name in STAGE_ORDER:
        pixels[name][3:] -= 0.020
    # Only first high prediction introduces additional late darkening.
    for name in STAGE_ORDER[4:]:
        pixels[name][4:] -= 0.030
    return pixels, labels


def test_ledger_tracks_absolute_offsets_and_temporal_change_separately():
    stages, labels = _stages()
    snapshots = {name: value.clone() for name, value in stages.items()}
    result = tone_attribution_ledger(stages, labels, 175, {"background": (0, 0, 0.5, 0.5)})
    assert result["policy"] == "full_stage_boundary_tone_ledger_v1"
    assert result["anchor_is_discarded_current_chunk_context"] is True
    assert result["raw_production_output_provided"] is False
    assert result["global_frame_labels"] == labels
    rows = result["regions"]["full_frame"]
    pairs = rows["sequential_pairs"]
    absolute = pairs["source_grid_to_provider_native"]["same_time_luma_difference"]
    relative = pairs["source_grid_to_provider_native"]["change_in_join_relative_luma"]
    assert absolute[4] == pytest.approx(0.04, abs=1e-6)
    assert all(abs(value) < 1e-6 for value in relative)
    onset = pairs["pre_high_to_first_high_before_flow"]["change_in_join_relative_luma"]
    assert onset[3] == pytest.approx(0, abs=1e-6)
    assert onset[4] == pytest.approx(-0.03, abs=1e-6)
    assert rows["stages"]["source_grid"]["luma_change_from_anchor"][4] == pytest.approx(-0.02, abs=1e-6)
    assert pairs["source_grid_to_provider_native"]["representation_domain"] == (
        "different_latent_and_decoder_spatial_grids"
    )
    assert pairs["provider_to_pre_high"]["representation_domain"] == "shared_target_decoder_grid"
    assert all(torch.equal(stages[name], snapshots[name]) for name in STAGE_ORDER)


def test_production_video_adds_real_prior_chunk_and_output_rgb_comparison():
    stages, labels = _stages()
    # Node input is the globally indexed final video, including prior chunk.
    actual = torch.full((195, 12, 16, 3), 0.6)
    actual[labels[0] : labels[-1] + 1] = stages["final"]
    actual[174] += 0.02  # preceding assembled frame differs from current discarded context
    actual[175] -= 0.015  # color correction only on first retained frame
    result = tone_attribution_ledger(stages, labels, 175, {}, production_images=actual)
    compared = result["production_image_comparison"]
    assert compared["previous_chunk_reference_is_real_assembled_frame"] is True
    row = compared["regions"]["full_frame"]
    assert row["same_frame_assembled_minus_replayed_final_luma"][labels.index(174)] == pytest.approx(0.02, abs=1e-6)
    assert row["same_frame_assembled_minus_replayed_final_luma"][labels.index(175)] == pytest.approx(-0.015, abs=1e-6)
    assert row["assembled_minus_replay_join_step"] == pytest.approx(-0.035, abs=1e-6)
    assert torch.equal(actual[173], stages["final"][0])


@pytest.mark.parametrize(
    "bad",
    [
        lambda t: t[:175],
        lambda t: t[:, :5],
        lambda t: t.unsqueeze(0),
        lambda t: t.fill_(float("nan")),
        lambda t: t.fill_(1.25),
    ],
)
def test_reject_mismatched_or_nonfinite_production_images(bad):
    stages, labels = _stages()
    full = torch.full((195, 12, 16, 3), 0.6)
    with pytest.raises((ValueError, TypeError)):
        tone_attribution_ledger(stages, labels, 175, {}, production_images=bad(full))


def test_reject_missing_stage_or_wrong_label_timeline():
    stages, labels = _stages()
    stages.pop("final")
    with pytest.raises(ValueError, match="seven"):
        tone_attribution_ledger(stages, labels, 175, {})
    stages, labels = _stages()
    with pytest.raises(ValueError, match="contiguous"):
        tone_attribution_ledger(stages, [*labels[:-1], 190], 175, {})


def test_custom_roi_separates_background_and_foreground_drift():
    stages, labels = _stages()
    stages["final"][5:, :, 8:] -= 0.06
    result = tone_attribution_ledger(
        stages,
        labels,
        175,
        {"left": (0, 0, 0.5, 1), "right": (0.5, 0, 1, 1)},
    )
    left = result["regions"]["left"]["stages"]["final"]["luma_mean"]
    right = result["regions"]["right"]["stages"]["final"]["luma_mean"]
    assert right[5] < left[5] - 0.05
    assert result["regions"]["full_frame"]["stages"]["final"]["luma_mean"][5] == pytest.approx(
        (right[5] + left[5]) / 2, abs=1e-6
    )
