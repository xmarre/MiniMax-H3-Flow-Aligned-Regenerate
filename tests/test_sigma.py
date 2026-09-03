import pytest
import torch

from h3_flow_regenerate.sigma import (
    audio_sigma,
    flow_shift,
    inverse_flow_shift,
    normalized_coordinate,
    remap_shift,
    resolution_aware_sigmas,
    resolution_shift_factor,
)


def test_shift_inverse_and_composition():
    base = torch.linspace(0, 1, 101)
    shifted = flow_shift(base, 12.0)
    assert torch.allclose(inverse_flow_shift(shifted, 12.0), base, atol=1e-6)
    assert torch.allclose(remap_shift(shifted, 12.0, 3.0), flow_shift(base, 3.0), atol=1e-6)


def test_joint_av_mapping_preserves_common_base_coordinate():
    video = torch.linspace(1, 0, 101)
    audio = audio_sigma(video)
    assert torch.allclose(inverse_flow_shift(video, 12.0), inverse_flow_shift(audio, 3.0), atol=1e-6)


def test_resolution_node_reports_effective_shift_factor():
    from h3_flow_regenerate.nodes import H3ResolutionAwareSigmas

    sigmas = torch.tensor([1.0, 0.7, 0.0])
    node = H3ResolutionAwareSigmas()
    _, off = node.map(sigmas, "off", 864, 640, 1024, 768, 1.0, 2.5)
    _, calibrated = node.map(sigmas, "calibrated", 864, 640, 1024, 768, 1.0, 2.5)
    _, derived = node.map(sigmas, "resolution_aware", 864, 640, 1024, 768, 1.0, 2.5)

    assert off["extra_shift_factor"] == 1.0
    assert calibrated["extra_shift_factor"] == 2.5
    assert derived["extra_shift_factor"] == pytest.approx(
        resolution_shift_factor(864 * 640, 1024 * 768)
    )


def test_resolution_mapping_is_ordered_finite_and_exact_at_endpoints():
    sigmas = flow_shift(torch.linspace(1, 0, 21), 12.0)
    mapped = resolution_aware_sigmas(sigmas, source_area=864 * 640, target_area=1024 * 768)
    assert mapped[0] == 1
    assert mapped[-1] == 0
    assert torch.isfinite(mapped).all()
    assert torch.all(mapped[1:] <= mapped[:-1])


def test_resolution_factor_matches_sd3_area_derivation():
    assert resolution_shift_factor(100, 400) == pytest.approx(2.0)
    assert resolution_shift_factor(100, 400, strength=0.0) == pytest.approx(1.0)


def test_off_is_exact_parity_and_does_not_alias():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    mapped = resolution_aware_sigmas(sigmas, source_area=1, target_area=4, mode="off")
    assert torch.equal(mapped, sigmas)
    assert mapped.data_ptr() != sigmas.data_ptr()


def test_normalized_coordinate_is_full_unshifted_trajectory():
    assert normalized_coordinate(0.0) == pytest.approx(0.0)
    assert normalized_coordinate(1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [torch.tensor([0.0, 1.0]), torch.tensor([1.0, float("nan")])])
def test_bad_schedule_fails_closed(bad):
    with pytest.raises(ValueError):
        resolution_aware_sigmas(bad, source_area=1, target_area=2)
